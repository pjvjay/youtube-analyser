"""Searching a whole YouTube channel: download its videos at the lowest resolution, search them,
and report every time someone does an action, or a described person does it, with a link to
each moment on YouTube (`yta channel`).

    yta channel "https://www.youtube.com/@SomeShow" "people jumping"
    yta channel "https://www.youtube.com/@SomeShow" "woman with purple hair talking" --max-videos 50
    yta channel ./my_videos "someone smiling"                     (videos already downloaded)

How it is quick:
- Videos are downloaded at the lowest resolution (144p), without the sound when the query needs
  none (talking, silent, voices, gender and names do), several at a time, while earlier ones are
  being searched: downloading and searching overlap, and searching is the slow part.
- Only what the query needs is computed (search.Searcher works cheapest first and stops when
  nobody can match). With a colour in the query, each video first gets a quick look at its
  keyframes only: a video where nobody has purple hair is done in seconds. speed="thorough"
  skips the quick look (which can miss someone on screen for under about 10 s); "fastest" also
  runs the pose model at a smaller size (about 4x faster; it finds fewer small, distant people).
- Everything is saved in the work folder, so an interrupted run carries on where it stopped, and
  a second query on the same channel reuses the videos and the model output.

The per-video step (process_video) and the report (write_report) work on a cloud.Store, so the
same search runs on one machine (a folder) or spread over cloud workers (enqueue_channel: one
queue message per video, handled by `yta worker`; collect_channel writes the report).
"""
import contextlib
import csv
import hashlib
import html
import json
import queue
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import intro, media, people
from .analysis import Settings
from .cloud import FileStore
from .query import PARTS, describe_query, parse
from .search import Searcher, slug, thumbnail
from .youtube import YtDlp, in_dates

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows: runs of one search should not share a work folder there
    fcntl = None

SPEEDS = {  # speed: (quick look first, pose model input size)
    "thorough": (False, 640),
    "quick": (True, 640),
    "fastest": (True, 320),
}
INVENTORY_FIELDS = ["file", "upload_date", "video_id", "title", "kind", "duration_min", "asked_for_sound"]
MATCH_FIELDS = ["upload_date", "title", "video_id", "video", "person", "start_s", "end_s", "start", "end",
                "duration_s", "action", "match", "in_intro", "evidence", "youtube", "thumb"]
KEEP = ("video", "person", "start_s", "end_s", "start", "end", "duration_s", "action", "match", "evidence",
        "youtube", "thumb", "in_intro")
BANK_VIDEOS = 8          # the intro bank is built from the first videos of 4 minutes or more
TOO_OLD_IN_A_ROW = 5     # a channel is listed newest first: this many too old means the rest are too


@dataclass
class ChannelQuery:
    """What a channel search looks for, the same on one machine and in the cloud."""
    query: str
    action: str = None
    speed: str = "quick"
    possible: bool = False
    intros: str = "label"        # label: results in intros are set apart; keep: each video is one whole
    config_text: str = ""        # the user's people.toml, laid over the defaults

    def __post_init__(self):
        if self.speed not in SPEEDS:
            raise ValueError(f"speed must be one of {', '.join(SPEEDS)}")
        if self.intros not in ("label", "keep"):
            raise ValueError("intros must be label or keep")

    def config(self):
        return people.load_config(text=self.config_text)

    def request(self, config=None):
        config = config or self.config()
        req = parse(self.query, config, [p["name"] for p in config.get("people", [])])
        req["action"] = self.action or req["action"]
        if not any(req[k] for k in PARTS) and req["action"] in (None, "visible"):
            raise ValueError("the query names nobody and no action: nothing to search for")
        return req

    @property
    def scout(self):
        return SPEEDS[self.speed][0]

    @property
    def imgsz(self):
        return SPEEDS[self.speed][1]

    def name(self):
        return slug(f"{self.query} {self.action or ''}" + (" fastest" if self.speed == "fastest" else ""))

    def key(self):
        """What the results depend on: a search folder made for another key starts again."""
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def needs_sound(req):
    return bool(req["voice"] or req["name"] or req["gender"]) or req["action"] in ("talking", "silent")


def output_folder(work, imgsz):
    """The model output of another pose input size is kept apart."""
    return Path(work) / ("output" if imgsz == 640 else f"output-{imgsz}")


# ---- one video ---------------------------------------------------------------------------------

def process_video(searcher, path, meta, req, scout, store, prefix):
    """Search one video (a quick look first when asked), save its thumbnails and its result
    (prefix/done/<id>.json) in the store, and return the result. meta: {id, title,
    upload_date, duration_s}. A video that fails is not saved as done (it is tried again)."""
    t0 = time.time()
    before = dict(searcher.timings)
    rows, note = [], ""
    try:
        if scout and not searcher.scout(path, req):
            note = searcher.note
        else:
            rows = searcher.search(path, req, req["action"])
            note = searcher.note
            with _thumbs_folder(store, prefix) as folder:
                for r in rows:
                    r["thumb"] = thumbnail(path, r, folder)
    except Exception as x:  # one bad file should not stop the run
        note = f"failed: {type(x).__name__}: {x}"
    result = {"video": path.name, "video_id": meta["id"], "title": meta.get("title", ""),
              "upload_date": meta.get("upload_date", ""), "duration_s": float(meta.get("duration_s") or 0),
              "note": note, "seconds": round(time.time() - t0, 1),
              "steps": {k: round(v - before.get(k, 0.0), 1) for k, v in searcher.timings.items()
                        if v - before.get(k, 0.0) >= 0.05},
              "rows": [{k: _plain(r.get(k)) for k in KEEP} for r in rows]}
    if not note.startswith("failed:"):
        store.put_json(f"{prefix}/done/{meta['id']}.json", result)
    return result


@contextlib.contextmanager
def _thumbs_folder(store, prefix):
    """A folder to write thumbnails into: the store's own folder, or a temporary one uploaded after."""
    if isinstance(store, FileStore):
        folder = store.path(f"{prefix}/thumbs")
        folder.mkdir(parents=True, exist_ok=True)
        yield folder
        return
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)
        for p in Path(tmp).iterdir():
            store.put_file(f"{prefix}/thumbs/{p.name}", p)


def _plain(v):
    return v.item() if isinstance(v, np.generic) else v


def done_results(store, prefix):
    """{video id: result} of every video searched so far (by any run of the search)."""
    out = {}
    for key in store.list(f"{prefix}/done/"):
        try:
            out[Path(key).stem] = store.get_json(key)
        except (ValueError, KeyError):   # being written by another run
            pass
    return out


# ---- the report --------------------------------------------------------------------------------

def write_report(store, prefix, query, req, source, total, started, notes=()):
    """report.html, report.md and matches.csv (in the store, under prefix) from every video done
    so far. Results in an intro are listed apart, at the end."""
    done = done_results(store, prefix)
    total = max(total, len(done))
    videos = sorted(done.values(), key=lambda d: (d["upload_date"], d["video"]), reverse=True)
    rows = [dict(r, upload_date=d["upload_date"], title=d["title"], video_id=d["video_id"]) for d in videos for r in d["rows"]]
    out = _Csv(MATCH_FIELDS)
    out.rows(rows)
    store.put(f"{prefix}/matches.csv", out.bytes())
    hours = sum(d.get("duration_s") or 0 for d in videos) / 3600
    main = [(d, rs) for d, rs in ((d, [r for r in d["rows"] if not r.get("in_intro")]) for d in videos) if rs]
    intros = [(d, rs) for d, rs in ((d, [r for r in d["rows"] if r.get("in_intro")]) for d in videos) if rs]
    n_main, n_intro = sum(len(rs) for _, rs in main), sum(len(rs) for _, rs in intros)
    action = req["action"]
    understood = f"{describe_query(req)}; action: {action or 'any (in view)'}"
    length = f"{hours:.1f} hours" if hours >= 1 else f"{hours * 60:.0f} minutes"
    summary = (f"{_n(n_main, 'result')} in {_n(len(main), 'video')}"
               + (f" (and {n_intro} in the intros, listed at the end)" if n_intro else "")
               + f"; {len(videos)} of {_n(total, 'video')} searched ({length} of video)"
               + (f" in {_took(time.time() - started)}" if started else ""))
    name = lambda d: " ".join(x for x in (d["upload_date"], d["title"] or d["video"]) if x)

    def md_list(groups):
        lines = []
        for d, rs in groups:
            lines += [f"## {name(d)}", ""]
            for r in rs:
                link = f"[{_when(r)}]({r['youtube']})" if r["youtube"] else _when(r)
                lines.append(f"- {link} {r['action']}{' (possible)' if r['match'] == 'possible' else ''}: {r['evidence']}")
            lines.append("")
        return lines
    md = [f"# {query}", "", f"Looking for: {understood}. Source: {source}", "", summary, ""] + md_list(main)
    if intros:
        md += ["# In the intros (title sequences, teasers)", ""] + md_list(intros)
    store.put(f"{prefix}/report.md", "\n".join(md).encode())

    e = html.escape

    def sections(groups):
        parts = []
        for d, rs in groups:
            url = media.youtube_url(d["video"])
            head = f'<a href="{e(url)}">{e(d["title"] or d["video"])}</a>' if url else e(d["title"] or d["video"])
            cards = "".join(
                f'<figure><a href="{e(r["youtube"] or "thumbs/" + r["thumb"])}"><img src="thumbs/{e(r["thumb"])}" '
                f'loading="lazy"></a><figcaption><b>{e(_when(r))}</b> ({r["duration_s"]}s) {e(r["action"])}'
                + (' <span class="tag">possible</span>' if r["match"] == "possible" else "")
                + (f' &middot; <a href="{e(r["youtube"])}">watch</a>' if r["youtube"] else "")
                + f'<br><small>{e(r["evidence"])}</small></figcaption></figure>' for r in rs)
            parts.append(f'<section><h2>{" &middot; ".join(x for x in (e(d["upload_date"]), head) if x)}</h2>'
                         f'<main>{cards}</main></section>')
        return "".join(parts)
    with_main = {d["video"] for d, _ in main}
    empty = [d for d in videos if d["video"] not in with_main]
    nothing = "".join(f"<li>{e(name(d))}: {e(d['note'] or ('only in the intro' if d['rows'] else 'nothing found'))}</li>"
                      for d in empty)
    problems = "".join(f"<li>{e(n)}</li>" for n in notes)
    store.put(f"{prefix}/report.html", (
        f"<!doctype html><meta charset=utf-8><title>{e(query)}</title>"
        "<style>body{font:14px system-ui;margin:16px;max-width:1400px}main{display:grid;"
        "grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}figure{margin:0}"
        "img{width:100%;border-radius:4px}h2{font-size:16px;margin:24px 0 8px}.tag{background:#eee;"
        "border-radius:3px;padding:0 4px}summary{margin:16px 0;font-weight:600}</style>"
        f"<h1>{e(query)}</h1><p>Looking for: {e(understood)}"
        + (f" (ignored words: {e(', '.join(req['ignored']))})" if req["ignored"] else "")
        + f". Source: {e(source)}.</p><p><b>{e(summary)}</b></p>{sections(main)}"
        + (f"<details><summary>{n_intro} results in the intros (title sequences, teasers)</summary>{sections(intros)}</details>"
           if intros else "")
        + (f"<details><summary>{len(empty)} videos with nothing found</summary><ul>{nothing}</ul></details>" if empty else "")
        + (f"<details open><summary>Problems</summary><ul>{problems}</ul></details>" if notes else "")).encode())
    return done


class _Csv:
    def __init__(self, fields):
        import io
        self.buf = io.StringIO()
        self.w = csv.DictWriter(self.buf, fields, extrasaction="ignore", lineterminator="\n")
        self.w.writeheader()

    def rows(self, rows):
        self.w.writerows(rows)

    def bytes(self):
        return self.buf.getvalue().encode()


def _n(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def _when(r):
    """0:03:35-0:03:48, or one time for a moment shorter than a second (a jump)."""
    return r["start"] if r["start"] == r["end"] else f"{r['start']}-{r['end']}"


def _took(s):
    return f"{s:.0f}s" if s < 90 else f"{s / 60:.1f} min" if s < 5400 else f"{s / 3600:.1f} h"


def _eta(searched_s, n, took_s, left):
    """'; 12x real time, about 40 min left' from the video time of the n videos searched so far
    in took_s (videos of unknown length taken as long as the average so far)."""
    if not searched_s or not took_s:
        return ""
    rate = searched_s / took_s
    known = [e["duration"] for e in left if e.get("duration")]
    left_s = sum(known) + (len(left) - len(known)) * searched_s / max(n, 1)
    return f"; {rate:.0f}x real time" + (f", about {_took(left_s / rate)} left" if left else "")


# ---- the inventory (videos.csv) ----------------------------------------------------------------

class Inventory:
    """videos.csv in the work folder: one row per video, in `yta report`'s format (plus whether
    the download asked for the sound)."""

    def __init__(self, path):
        self.path, self.rows, self.lock = Path(path), {}, threading.Lock()
        if self.path.exists():
            with open(self.path, newline="") as f:
                self.rows = {r["video_id"]: r for r in csv.DictReader(f)}

    def asked_for_sound(self, vid):
        with self.lock:
            return self.rows.get(vid, {}).get("asked_for_sound") == "yes"

    def add(self, vid, video, info, entry, sound=None):
        with self.lock, _file_lock(self.path.with_suffix(".lock")):
            if self.path.exists():  # other runs (--shard) may have added videos
                with open(self.path, newline="") as f:
                    self.rows.update({r["video_id"]: r for r in csv.DictReader(f) if r["video_id"] not in self.rows})
            old = self.rows.get(vid, {})
            date = info.get("upload_date") or media.upload_date(video) or entry.get("date") or ""
            duration = info.get("duration") or entry.get("duration") or media.duration(video) or None
            self.rows[vid] = {"file": Path(video).name, "video_id": vid,
                              "upload_date": (f"{date[:4]}-{date[4:6]}-{date[6:]}" if re.fullmatch(r"\d{8}", date)
                                              else date or old.get("upload_date", "")),
                              "title": info.get("title") or entry.get("title") or old.get("title", ""),
                              "kind": "short" if duration and float(duration) < 240 else "long",
                              "duration_min": round(float(duration) / 60, 1) if duration else old.get("duration_min", ""),
                              "asked_for_sound": ("yes" if sound else "no") if sound is not None else old.get("asked_for_sound", "")}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, INVENTORY_FIELDS, lineterminator="\n")
                w.writeheader()
                w.writerows(sorted(self.rows.values(), key=lambda r: r["upload_date"], reverse=True))
            tmp.replace(self.path)
            return self.rows[vid]


@contextlib.contextmanager
def _file_lock(path):
    """An exclusive lock on path between processes (where the system has one)."""
    with open(path, "a") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)


def local_entries(folder, since=None, until=None):
    """The videos in a folder, newest first by name (yt-dlp's YYYYMMDD_<id> names sort by date),
    as listing entries; --since and --until use the date in the name where there is one."""
    out = []
    for p in sorted(Path(folder).iterdir(), reverse=True):
        if media.is_video(p) and in_dates(media.upload_date(p), since, until):
            out.append({"id": media.youtube_id(p) or p.stem, "title": p.stem, "duration": None,
                        "date": media.upload_date(p), "path": p})
    return out


# ---- one machine -------------------------------------------------------------------------------

def run_channel(source, cq, work=None, yt=None, max_videos=None, since=None, until=None, shorts=False, downloads=3,
                ahead=6, delete_videos=False, shard=None, runtime="auto", threads=0, explain=False, log=print):
    """Search a channel (or a folder of videos) on this machine. Returns (results by video id,
    the search folder)."""
    source = str(source)
    config = cq.config()
    req = cq.request(config)
    log(f"looking for: {describe_query(req)}; action: {req['action'] or 'any (in view)'}"
        + (f"; ignored words: {', '.join(req['ignored'])}" if req["ignored"] else ""), flush=True)
    sound = needs_sound(req)
    src_dir = Path(source) if Path(source).is_dir() else None
    name = src_dir.name if src_dir else slug(re.sub(r"^https?://(www\.)?youtube\.com/", "", source))
    work = Path(work) if work else Path("channels") / name
    videos_dir = src_dir or work / "videos"
    store, prefix = FileStore(work), f"searches/{cq.name()}"
    found_dir = store.path(prefix)
    if store.exists(f"{prefix}/settings") and store.get(f"{prefix}/settings").decode() != cq.key():
        log("the settings changed since this search was last run: starting it again", flush=True)
        for key in store.list(f"{prefix}/done/"):
            store.delete(key)
    store.put(f"{prefix}/settings", cq.key().encode())
    inventory = Inventory(work / "videos.csv")
    yt = yt or YtDlp()

    entries = local_entries(src_dir, since, until) if src_dir else \
        [e for e in yt.list(source, shorts) if in_dates(e.get("date"), since, until)]
    if max_videos:
        entries = entries[:max_videos]
    log(f"{len(entries)} videos {'in ' + str(src_dir) if src_dir else 'listed'}; work folder {work}", flush=True)
    done = {k: v for k, v in done_results(store, prefix).items() if k in {e["id"] for e in entries}}
    todo = [e for e in entries if e["id"] not in done]
    if shard:
        k, n = (int(x) for x in shard.split("/"))
        if not 1 <= k <= n:
            raise ValueError("--shard K/N needs 1 <= K <= N")
        mine = {e["id"] for i, e in enumerate(entries) if i % n == k - 1}
        todo = [e for e in todo if e["id"] in mine]
    if done:
        log(f"{len(done)} already searched; {len(todo)} to go", flush=True)

    settings = Settings(imgsz=cq.imgsz, runtime=runtime, threads=threads)
    existing = [p for p in videos_dir.glob("*") if media.is_video(p)] if videos_dir.exists() else []
    searcher = Searcher(config, settings, output_folder(work, cq.imgsz), possible=cq.possible, explain=explain,
                        label_intros=cq.intros == "label", videos=existing, log=log)
    bank_path = work / ("intro_bank.npz" if not shard else f"intro_bank-{shard.replace('/', 'of')}.npz")
    started, notes = time.time(), []
    write_report(store, prefix, cq.query, req, source, len(entries), started, notes)

    # downloads run ahead of the search, in threads; the search runs here, one video at a time
    ready, room, stop = queue.Queue(), threading.Semaphore(max(1, ahead)), threading.Event()

    def fetch(e):
        room.acquire()
        if stop.is_set():
            return
        t0 = time.time()
        try:
            if "path" in e:
                path, info, note = e["path"], {}, ""
            else:
                path, info, note = yt.download(e, videos_dir, sound, since, until, inventory.asked_for_sound(e["id"]))
        except Exception as x:  # report it and carry on with the others
            path, info, note = None, {}, f"download failed: {x}"
        ready.put((e, path, info, note, time.time() - t0))

    pool = ThreadPoolExecutor(max(1, downloads))
    for e in todo:
        pool.submit(fetch, e)
    bank_needed = cq.intros == "label" and not bank_path.exists()
    if cq.intros == "label" and bank_path.exists():
        searcher.bank = _usable(intro.load_bank(bank_path))
    waiting = []        # downloaded videos held back until the intro bank is built
    counts = {"analysed": 0, "searched_s": 0.0}
    processed = set()

    def build_bank():
        long_ones = [w[1] for w in waiting if w[2]["kind"] == "long"]
        if len(long_ones) >= intro.SHARED_BY + 1:
            bank, _ = intro.build_bank_from(long_ones)
            intro.save_bank(bank_path, bank)
            searcher.bank = _usable(bank)
            log(f"intro bank built from {len(long_ones)} videos: {intro.title_frames(bank)} "
                "title frames shared by the episodes", flush=True)

    def search_batch(batch):
        for e2, path2, row2, took2, holds_room in batch:
            counts["analysed"] += 1
            meta = {"id": e2["id"], "title": row2["title"], "upload_date": row2["upload_date"],
                    "duration_s": float(row2["duration_min"] or 0) * 60}
            result = process_video(searcher, path2, meta, req, cq.scout, store, prefix)
            if result["note"].startswith("failed:"):
                notes.append(f"{path2.name}: {result['note']}")
                log(f"{path2.name}: {result['note']}", file=sys.stderr, flush=True)
            else:
                done[e2["id"]] = result
                if delete_videos and not src_dir:
                    path2.unlink(missing_ok=True)
            if holds_room:
                room.release()
            counts["searched_s"] += result["duration_s"]
            processed.add(e2["id"])
            log(f"[{len(done)}/{len(entries)}] {row2['upload_date']} {path2.name} ({row2['duration_min']} min): "
                f"{len(result['rows'])} results" + (f" ({result['note']})" if result["note"] and not result["rows"] else "")
                + (f"; downloaded in {took2:.0f}s" if not src_dir else "") + f", searched in {_took(result['seconds'])}"
                + (_eta(counts["searched_s"], counts["analysed"], time.time() - started,
                        [x for x in todo if x["id"] not in processed]) if counts["analysed"] > 1 else ""), flush=True)
            write_report(store, prefix, cq.query, req, source, len(entries), started, notes)

    too_old = 0
    try:
        for i in range(len(todo)):
            e, path, info, note, took_dl = ready.get()
            if path is None:
                room.release()
                log(f"[{i + 1}/{len(todo)}] {e['id']}: {note}", flush=True)
                if note == "outside the dates asked for":
                    too_old += 1
                    if too_old >= TOO_OLD_IN_A_ROW and since and not src_dir:
                        log(f"{TOO_OLD_IN_A_ROW} videos in a row older than --since: the rest are older still; stopping",
                            flush=True)
                        break
                else:
                    notes.append(f"{e['id']} {e.get('title', '')}: {note}")
                continue   # not saved as done: a later run tries again
            too_old = 0
            row = inventory.add(e["id"], path, info, e, None if note == "already downloaded" or src_dir else sound)
            if bank_needed:  # the intro bank is built from the first long videos, before they are searched
                room.release()   # (held-back videos do not hold up the downloads)
                waiting.append((e, path, row, took_dl, False))
                if sum(w[2]["kind"] == "long" for w in waiting) >= BANK_VIDEOS:
                    bank_needed = False
                    build_bank()
                    batch, waiting = waiting, []
                    search_batch(batch)
                continue
            search_batch([(e, path, row, took_dl, True)])
        if waiting:   # fewer long videos than a bank is built from: build it from those there are
            build_bank()
            batch, waiting = waiting, []
            search_batch(batch)
    finally:
        stop.set()
        for _ in range(len(todo)):
            room.release()
        pool.shutdown(wait=False, cancel_futures=True)
    done = write_report(store, prefix, cq.query, req, source, len(entries), started, notes)
    total = sum(len(d["rows"]) for d in done.values())
    log(f"{total} results in {sum(bool(d['rows']) for d in done.values())} videos -> {found_dir / 'report.html'}, "
        f"{found_dir / 'report.md'}, {found_dir / 'matches.csv'}", flush=True)
    if searcher.timings:
        log("time spent: " + ", ".join(f"{k} {v:.0f}s" for k, v in sorted(searcher.timings.items(), key=lambda x: -x[1])),
            flush=True)
    if searcher.unsure:
        log(f"{searcher.unsure} more people might match (not seen well enough to tell); --possible lists them", flush=True)
    return done, found_dir


def _usable(bank):
    """A bank with title frames, or None (a channel without a shared intro: nothing to set apart)."""
    return bank if intro.title_frames(bank) else None


# ---- the cloud: one queue message per video ----------------------------------------------------

def enqueue_channel(source, cq, store, task_queue, yt=None, max_videos=None, since=None, until=None, shorts=False,
                    log=print):
    """Start a channel search on cloud workers (`yta worker`): the search's description goes to
    the store (searches/<id>/spec.json), with the voices of people enrolled by name already
    fingerprinted and, when intros are labelled, the intro bank built here from the first long
    videos; then one queue message per video. Returns the search id; collect_channel writes the
    report (again whenever called)."""
    config = cq.config()
    req = cq.request(config)
    yt = yt or YtDlp()
    entries = [e for e in yt.list(source, shorts) if in_dates(e.get("date"), since, until)]
    if max_videos:
        entries = entries[:max_videos]
    search_id = f"{cq.name()}-{cq.key()}"
    prefix = f"searches/{search_id}"
    spec = {"query": asdict(cq), "source": source, "since": since, "until": until, "total": len(entries),
            "sound": needs_sound(req), "started": time.time(), "enrolled": {}, "bank": None}
    if req["name"] or any(p.get("voice_file") or p.get("voice_from") for p in config.get("people", [])):
        enrolled = Searcher(config, log=log).enrol()
        spec["enrolled"] = {n: [c.tolist(), float(f0)] for n, (c, f0) in enrolled.items()}
    if cq.intros == "label":
        with tempfile.TemporaryDirectory() as tmp:
            long_ones = []
            for e in entries:
                if len(long_ones) >= BANK_VIDEOS:
                    break
                if e.get("duration") and e["duration"] < intro.OPENING_S:
                    continue
                path, _, _ = yt.download(e, tmp, False, since, until)
                if path is not None and media.duration(path) >= intro.OPENING_S:
                    long_ones.append(path)
            bank = _usable(intro.build_bank_from(long_ones)[0]) if len(long_ones) > intro.SHARED_BY else None
            if bank is not None:
                with tempfile.TemporaryDirectory() as t2:
                    intro.save_bank(Path(t2) / "bank.npz", bank)
                    store.put_file(f"{prefix}/intro_bank.npz", Path(t2) / "bank.npz")
                spec["bank"] = f"{prefix}/intro_bank.npz"
                log(f"intro bank built from {len(long_ones)} videos: {intro.title_frames(bank)} title frames", flush=True)
    store.put_json(f"{prefix}/spec.json", spec)
    done = done_results(store, prefix)
    sent = 0
    for e in entries:
        if e["id"] not in done:
            task_queue.send({"kind": "channel-video", "search": search_id, "entry": {k: e[k] for k in ("id", "title", "duration", "date", "url")}})
            sent += 1
    log(f"search {search_id}: {len(entries)} videos, {sent} sent to the workers", flush=True)
    return search_id


class ChannelTasks:
    """The worker's side of a channel search: handles "channel-video" messages, keeping one
    Searcher (models loaded once) per search."""

    def __init__(self, store, runtime="auto", threads=0, yt=None, log=print):
        self.store, self.runtime, self.threads, self.log = store, runtime, threads, log
        self.yt = yt or YtDlp()
        self._searches = {}

    def _search(self, search_id):
        if search_id not in self._searches:
            prefix = f"searches/{search_id}"
            spec = self.store.get_json(f"{prefix}/spec.json")
            cq = ChannelQuery(**spec["query"])
            config = cq.config()
            req = cq.request(config)
            bank = None
            if spec.get("bank"):
                with tempfile.TemporaryDirectory() as tmp:
                    self.store.get_file(spec["bank"], Path(tmp) / "bank.npz")
                    bank = intro.load_bank(Path(tmp) / "bank.npz")
            searcher = Searcher(config, Settings(imgsz=cq.imgsz, runtime=self.runtime, threads=self.threads), cache=None,
                                bank=bank, possible=cq.possible, label_intros=cq.intros == "label", log=self.log)
            searcher.enrolled = {n: (np.array(c, np.float32), f0) for n, (c, f0) in spec.get("enrolled", {}).items()}
            self._searches[search_id] = (spec, cq, req, searcher)
        return self._searches[search_id]

    def handle(self, task):
        """Download the video, search it, save the result; the video is deleted in any case."""
        spec, cq, req, searcher = self._search(task["search"])
        entry, prefix = task["entry"], f"searches/{task['search']}"
        with tempfile.TemporaryDirectory() as tmp:
            path, info, note = self.yt.download(entry, tmp, spec["sound"], spec.get("since"), spec.get("until"))
            if path is None:
                if note == "outside the dates asked for":
                    return {"video_id": entry["id"], "note": note}
                raise RuntimeError(note)
            meta = {"id": entry["id"], "title": info.get("title") or entry.get("title", ""),
                    "upload_date": _dashes(info.get("upload_date") or media.upload_date(path) or entry.get("date") or ""),
                    "duration_s": info.get("duration") or entry.get("duration") or media.duration(path)}
            result = process_video(searcher, path, meta, req, cq.scout, self.store, prefix)
        if result["note"].startswith("failed:"):
            raise RuntimeError(result["note"])
        return result


def _dashes(date):
    return f"{date[:4]}-{date[4:6]}-{date[6:]}" if re.fullmatch(r"\d{8}", date or "") else date or ""


def collect_channel(store, search_id, folder=None, log=print):
    """Write the report of a cloud channel search (in the store, and into folder with the
    thumbnails when given). Returns (videos done, videos listed)."""
    prefix = f"searches/{search_id}"
    spec = store.get_json(f"{prefix}/spec.json")
    cq = ChannelQuery(**spec["query"])
    req = cq.request()
    done = write_report(store, prefix, cq.query, req, spec["source"], spec["total"], None)
    if folder is not None:
        from .cloud import copy_tree
        copy_tree(store, prefix, folder)
    log(f"{len(done)} of {spec['total']} videos searched; {sum(len(d['rows']) for d in done.values())} results"
        + (f" -> {Path(folder) / 'report.html'}" if folder is not None else ""), flush=True)
    return len(done), spec["total"]
