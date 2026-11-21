"""Search a whole YouTube channel: download its videos at the lowest resolution, analyse them and
report every time someone does an action, or a described person does it, with timestamped links.

    .venv/bin/python standing/channel.py "https://www.youtube.com/@SomeShow" "people jumping"
    .venv/bin/python standing/channel.py "https://www.youtube.com/@SomeShow" "woman with purple hair talking"
    .venv/bin/python standing/channel.py "https://www.youtube.com/@SomeShow" "man in a blue coat walking" \\
        --max-videos 50 --speed fastest
    .venv/bin/python standing/channel.py ./my_videos "someone smiling"      (videos already downloaded)

The query is search.py's: an action alone ("people jumping", "anyone standing") finds everyone
doing it, as detect_standing.py does; a description with or without an action ("woman with
purple hair talking", "man in a blue coat") finds that person. Words it does not understand are
listed as ignored before anything starts.

How it is quick:
- Videos are downloaded at the lowest resolution (144p), video only when the query needs no
  sound (talking, silent, voices, gender and names do), several at a time, while earlier ones
  are being analysed: the download and the analysis overlap, and the analysis is the slow part.
- Only what the query needs is computed (search.py works cheapest first and stops when nobody
  can match). With a colour in the query, each video first gets a quick look at its keyframes
  only (a few seconds per hour of video): a video where nobody has purple hair is done there.
  --speed thorough skips the quick look (which can miss someone on screen for under about 10 s);
  --speed fastest also runs the pose model at a smaller size (about 4x faster; it passed the same
  end-to-end tests, but finds fewer small, distant people).
- Everything computed is saved (--work), so an interrupted run carries on where it stopped, and a
  second search of the same channel reuses the first's model output (and videos, unless
  --delete-videos).

Writes, in --work/searches/<query>/: report.html (every result with a thumbnail and a link to the
moment on YouTube, grouped by video, newest first), report.md (the same as a list of links),
matches.csv, and thumbs/. They are rewritten after each video, so the report can be opened while
the run goes on. --work also holds videos/, videos.csv (the inventory: upload date, title,
duration) and the model output (output/).
"""
import argparse
import contextlib
import csv
import hashlib
import html
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: runs of one search should not share a work folder there
    fcntl = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import detect_standing as ds  # noqa: E402
import intro  # noqa: E402
import people  # noqa: E402
import search  # noqa: E402

VIDEO_EXT = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
# The lowest resolution that is still 144p (YouTube's smallest is 144p; storyboards are smaller
# and are not videos), with the smallest sound when the query needs sound.
FORMAT_SOUND = "worstvideo[height>=144]+worstaudio/worst[height>=144]/worst"
FORMAT_SILENT = "worstvideo[height>=144]/worst[height>=144]/worst"
SPEEDS = {  # --speed: (quick look first, pose model input size)
    "thorough": (False, 640),
    "quick": (True, 640),
    "fastest": (True, 320),
}
INVENTORY_FIELDS = ["file", "upload_date", "video_id", "title", "kind", "duration_min", "asked_for_sound"]
MATCH_FIELDS = ["upload_date", "title", "video_id", "video", "person", "start_s", "end_s", "start", "end",
                "duration_s", "action", "match", "in_intro", "evidence", "youtube", "thumb"]
BANK_VIDEOS = 8          # intros: the title-frame bank is built from the first videos of 4 minutes or more


# ---- listing and downloading -----------------------------------------------------------------

def yt_dlp_command(args):
    """yt-dlp as installed (on the PATH, or the Python package), with the user's options."""
    if args.yt_dlp:
        base = shlex.split(args.yt_dlp)
    elif shutil.which("yt-dlp"):
        base = [shutil.which("yt-dlp")]
    else:
        base = [sys.executable, "-m", "yt_dlp"]
    extra = []
    if args.cookies:
        extra += ["--cookies", str(args.cookies)]
    if args.cookies_from_browser:
        extra += ["--cookies-from-browser", args.cookies_from_browser]
    for a in args.yt_dlp_arg or []:
        extra += shlex.split(a)
    return base, extra


def channel_tabs(url, shorts=False):
    """A channel's address turned into its uploads: /videos (and /shorts); other addresses
    (a playlist, a channel tab, a single video) are used as they are."""
    m = re.match(r"^(https?://(?:www\.|m\.)?youtube\.com/(?:@[^/?#]+|channel/[^/?#]+|c/[^/?#]+|user/[^/?#]+))/?(?:featured)?/?$", url)
    if not m:
        return [url]
    return [m.group(1) + "/videos"] + ([m.group(1) + "/shorts"] if shorts else [])


def list_videos(args):
    """[{id, title, duration, url}] for the channel (newest first, as YouTube lists them), or for
    the video files in a folder."""
    src = Path(args.source)
    if src.is_dir():
        return [{"id": video_id(p), "title": p.stem, "duration": None, "path": p}
                for p in sorted(src.iterdir(), reverse=True)
                if p.suffix.lower() in VIDEO_EXT and not re.search(r"\.f\d+$", p.stem)]
    base, extra = yt_dlp_command(args)
    out, seen = [], set()
    for tab in channel_tabs(args.source, args.shorts):
        r = subprocess.run(base + extra + ["--flat-playlist", "--dump-single-json", "--no-warnings", tab],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"yt-dlp could not list {tab}:\n{r.stderr.strip()[-2000:]}")
        for e in _entries(json.loads(r.stdout)):
            if e.get("id") and e["id"] not in seen:
                seen.add(e["id"])
                out.append({"id": e["id"], "title": e.get("title") or "", "duration": e.get("duration"),
                            "url": e.get("url") or f"https://www.youtube.com/watch?v={e['id']}"})
    return out


def _entries(info):
    """The videos in yt-dlp's JSON for a playlist, a channel (whose entries can be its tabs) or a video."""
    if info.get("_type") in ("playlist", "multi_video"):
        for e in info.get("entries") or []:
            yield from _entries(e) if e.get("_type") in ("playlist", "multi_video") else [e]
    else:
        yield info


def video_id(path):
    m = ds.NAME_RE.match(path.stem)
    return m.group(2) if m else path.stem


def find_download(folder, vid):
    """A finished download of this video in folder (yt-dlp's YYYYMMDD_<id>.<ext>), or None."""
    for p in folder.glob(f"*_{ds.glob_escape(vid)}.*"):
        if p.suffix.lower() in VIDEO_EXT and not re.search(r"\.f\d+$", p.stem) and ds.NAME_RE.match(p.stem):
            return p
    return None


def has_sound(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def download(entry, folder, args, sound, out=None, had_sound=False):
    """Download one video at the lowest resolution into folder (again, with its sound, when a
    copy downloaded without it is there and the sound is needed now; had_sound: the copy was
    asked for with its sound; out: the model output folder, whose saved sound analysis of the
    silent copy is then removed). Returns (path or None, info, note)."""
    folder.mkdir(parents=True, exist_ok=True)
    have = find_download(folder, entry["id"])
    if have is not None and sound and not had_sound and not has_sound(have):
        have.unlink()
        if out is not None:
            (out / "audio" / f"{have.stem}.npz").unlink(missing_ok=True)
        have = None
    if have is not None:
        return have, {}, "already downloaded"
    base, extra = yt_dlp_command(args)
    cmd = base + extra + ["-f", FORMAT_SOUND if sound else FORMAT_SILENT, "-o", str(folder / "%(upload_date)s_%(id)s.%(ext)s"),
                          "--no-playlist", "--no-progress", "--no-warnings", "--concurrent-fragments", "4",
                          "--no-simulate", "--print", "after_move:%(filepath)s\t%(upload_date)s\t%(duration)s\t%(title)s"]
    if args.since:
        cmd += ["--dateafter", args.since]
    if args.until:
        cmd += ["--datebefore", args.until]
    r = subprocess.run(cmd + [entry["url"]], capture_output=True, text=True)
    lines = [x for x in r.stdout.splitlines() if x.count("\t") >= 3]
    if lines:
        path, date, duration, title = lines[-1].split("\t", 3)
        info = {"upload_date": date, "title": title, "duration": float(duration) if duration not in ("NA", "") else None}
        return Path(path), info, ""
    have = find_download(folder, entry["id"])
    if have is not None:
        return have, {}, ""
    why = (r.stderr.strip().splitlines() or ["no file"])[-1]
    if "upload date is not in range" in r.stdout + r.stderr or "not in range" in why:
        return None, {}, "outside the dates asked for"
    return None, {}, f"download failed: {why[:300]}"


# ---- the inventory (videos.csv) ---------------------------------------------------------------

class Inventory:
    """videos.csv in the work folder: one row per video, in make_report.py's format."""

    def __init__(self, path):
        self.path, self.rows, self.lock = path, {}, threading.Lock()
        if path.exists():
            with open(path, newline="") as f:
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
            date = info.get("upload_date") or (ds.NAME_RE.match(video.stem).group(1) if ds.NAME_RE.match(video.stem) else "")
            duration = info.get("duration") or entry.get("duration") or _duration(video)
            self.rows[vid] = {"file": video.name, "video_id": vid,
                              "upload_date": f"{date[:4]}-{date[4:6]}-{date[6:]}" if re.fullmatch(r"\d{8}", date) else date or old.get("upload_date", ""),
                              "title": info.get("title") or entry.get("title") or old.get("title", ""),
                              "kind": "short" if duration and float(duration) < 240 else "long",
                              "duration_min": round(float(duration) / 60, 1) if duration else old.get("duration_min", ""),
                              "asked_for_sound": ("yes" if sound else "no") if sound is not None else old.get("asked_for_sound", "")}
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


def _duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


# ---- the report ------------------------------------------------------------------------------

def write_report(folder, query, req, action, source, done, total, started, notes):
    """report.html, report.md and matches.csv from every video done so far (including those
    done by other runs of the same search, --shard)."""
    done = dict(done)
    for f in (folder / "done").glob("*.json"):
        if f.stem not in done:
            try:
                done[f.stem] = json.loads(f.read_text())
            except ValueError:   # being written by another run
                pass
    total = max(total, len(done))
    videos = sorted(done.values(), key=lambda d: (d["upload_date"], d["video"]), reverse=True)
    rows = [dict(r, upload_date=d["upload_date"], title=d["title"], video_id=d["video_id"])
            for d in videos for r in d["rows"]]
    tmp = folder / "matches.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, MATCH_FIELDS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(folder / "matches.csv")
    hours = sum(d.get("duration_s") or 0 for d in videos) / 3600
    main = [(d, [r for r in d["rows"] if not r.get("in_intro")]) for d in videos]
    intros = [(d, [r for r in d["rows"] if r.get("in_intro")]) for d in videos]
    main, intros = [(d, rs) for d, rs in main if rs], [(d, rs) for d, rs in intros if rs]
    n_main, n_intro = sum(len(rs) for _, rs in main), sum(len(rs) for _, rs in intros)
    understood = f"{search.describe_query(req)}; action: {action or 'any (in view)'}"
    length = f"{hours:.1f} hours" if hours >= 1 else f"{hours * 60:.0f} minutes"
    summary = (f"{_n(n_main, 'result')} in {_n(len(main), 'video')}" + (f" (and {n_intro} in the intros, listed at the end)" if n_intro else "")
               + f"; {len(videos)} of {_n(total, 'video')} searched ({length} of video) in {_took(time.time() - started)}")
    name = lambda d: " ".join(x for x in (d["upload_date"], d["title"] or d["video"]) if x)

    # markdown: a list of links
    def md_list(groups):
        out = []
        for d, rs in groups:
            out += [f"## {name(d)}", ""]
            for r in rs:
                when = _when(r)
                link = f"[{when}]({r['youtube']})" if r["youtube"] else when
                out.append(f"- {link} {r['action']}{' (possible)' if r['match'] == 'possible' else ''}: {r['evidence']}")
            out.append("")
        return out
    md = [f"# {query}", "", f"Looking for: {understood}. Source: {source}", "", summary, ""] + md_list(main)
    if intros:
        md += ["# In the intros (title sequences, teasers)", ""] + md_list(intros)
    _write(folder / "report.md", "\n".join(md))

    # html: thumbnails, grouped by video
    e = html.escape

    def sections(groups):
        parts = []
        for d, rs in groups:
            url = f"https://www.youtube.com/watch?v={d['video_id']}" if ds.NAME_RE.match(Path(d["video"]).stem) else ""
            head = f'<a href="{e(url)}">{e(d["title"] or d["video"])}</a>' if url else e(d["title"] or d["video"])
            cards = "".join(
                f'<figure><a href="{e(r["youtube"] or "thumbs/" + r["thumb"])}"><img src="thumbs/{e(r["thumb"])}" loading="lazy"></a>'
                f'<figcaption><b>{e(_when(r))}</b> ({r["duration_s"]}s) {e(r["action"])}'
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
    _write(folder / "report.html",
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
           + (f"<details open><summary>Problems</summary><ul>{problems}</ul></details>" if notes else ""))


def _n(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def _when(r):
    """0:03:35-0:03:48, or one time for a moment shorter than a second (a jump)."""
    return r["start"] if r["start"] == r["end"] else f"{r['start']}-{r['end']}"


def _write(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _took(s):
    return f"{s:.0f}s" if s < 90 else f"{s / 60:.1f} min" if s < 5400 else f"{s / 3600:.1f} h"


# ---- the run ---------------------------------------------------------------------------------

def needs_sound(req, action):
    return bool(req["voice"] or req["name"] or req["gender"]) or action in ds.SOUND_ACTIONS


def settings_key(args, req, action, config_path):
    """What the results depend on; a search folder made with other settings starts again."""
    cfg = Path(config_path).read_bytes() if config_path else b""
    return hashlib.sha1(json.dumps([args.query, action, args.possible, args.speed, args.intros,
                                    sorted(req.get("ignored", [])), cfg.hex()]).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a YouTube channel, playlist or video address, or a folder of downloaded videos")
    ap.add_argument("query", help='what to look for, e.g. "people jumping" or "woman with purple hair talking"')
    ap.add_argument("--action", choices=list(ds.ACTIONS) + ["visible"], help="instead of an action in the query")
    ap.add_argument("--work", type=Path, help="where videos, model output and reports go (default: channels/<channel>)")
    ap.add_argument("--speed", choices=list(SPEEDS), default="quick",
                    help="quick (default): a quick look at keyframes first when the query has a colour; "
                         "thorough: every second of every video; fastest: quick, and a smaller pose model input")
    ap.add_argument("--max-videos", type=int, help="only the newest N videos")
    ap.add_argument("--since", help="only videos uploaded on or after this date (YYYYMMDD)")
    ap.add_argument("--until", help="only videos uploaded on or before this date (YYYYMMDD)")
    ap.add_argument("--shorts", action="store_true", help="include the channel's Shorts")
    ap.add_argument("--downloads", type=int, default=3, help="videos downloaded at once (default 3)")
    ap.add_argument("--ahead", type=int, default=6, help="at most this many downloaded videos waiting to be analysed")
    ap.add_argument("--delete-videos", action="store_true", help="delete each video once it has been searched")
    ap.add_argument("--shard", help="K/N: search only every Nth video, starting with the Kth, so N runs (on one big "
                                    "machine, or several sharing the work folder) share the channel; each report shows all")
    ap.add_argument("--intros", choices=["label", "keep"], default="label",
                    help="label (default): find each episode's intro (title sequence, teaser) from the title "
                         "footage the episodes share, and mark results there; keep: treat every video as one whole")
    ap.add_argument("--config", type=Path, help="your people.toml (merged over the defaults)")
    ap.add_argument("--possible", action="store_true", help="also list people who might match (some part not seen)")
    ap.add_argument("--explain", action="store_true", help="print what was seen of every person")
    ap.add_argument("--cookies", type=Path, help="cookies.txt for yt-dlp (if YouTube asks you to sign in)")
    ap.add_argument("--cookies-from-browser", help="e.g. firefox or chrome: yt-dlp reads the browser's cookies")
    ap.add_argument("--yt-dlp", help="the yt-dlp command to use (default: yt-dlp on the PATH, else python -m yt_dlp)")
    ap.add_argument("--yt-dlp-arg", action="append", help="extra options for yt-dlp, e.g. '--sleep-requests 1'")
    ap.add_argument("--runtime", choices=["auto", "openvino", "onnxruntime"], default="auto")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    config = people.load_config(args.config)
    names = [p["name"] for p in config.get("people", [])]
    req = search.parse(args.query, config, names)
    action = args.action or req["action"]
    req["action"] = action
    understood = f"looking for: {search.describe_query(req)}; action: {action or 'any (in view)'}"
    print(understood + (f"; ignored words: {', '.join(req['ignored'])}" if req["ignored"] else ""), flush=True)
    if not any(req[k] for k in search.PARTS) and action in (None, "visible"):
        sys.exit("the query names nobody and no action: nothing to search for")
    scout, imgsz = SPEEDS[args.speed]
    sound = needs_sound(req, action)

    src_dir = Path(args.source) if Path(args.source).is_dir() else None
    name = src_dir.name if src_dir else search.slug(re.sub(r"^https?://(www\.)?youtube\.com/", "", args.source))
    work = args.work or HERE.parent / "channels" / name
    videos_dir = src_dir or work / "videos"
    out = work / ("output" if imgsz == 640 else f"output-{imgsz}")   # model output of another input size is kept apart
    found_dir = work / "searches" / search.slug(f"{args.query} {args.action or ''}" + ("" if args.speed != "fastest" else " fastest"))
    (found_dir / "done").mkdir(parents=True, exist_ok=True)
    key = settings_key(args, req, action, args.config)
    key_file = found_dir / "settings"
    if key_file.exists() and key_file.read_text() != key:
        print("the settings changed since this search was last run: starting it again", flush=True)
        shutil.rmtree(found_dir / "done")
        (found_dir / "done").mkdir()
    key_file.write_text(key)
    inventory = Inventory(work / "videos.csv")

    entries = list_videos(args)
    if args.max_videos:
        entries = entries[:args.max_videos]
    print(f"{len(entries)} videos {'in ' + str(src_dir) if src_dir else 'listed'}; work folder {work}", flush=True)
    done = {}
    for e in entries:
        f = found_dir / "done" / f"{e['id']}.json"
        if f.exists():
            done[e["id"]] = json.loads(f.read_text())
    todo = [e for e in entries if e["id"] not in done]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not 1 <= k <= n:
            sys.exit("--shard K/N needs 1 <= K <= N")
        mine = {e["id"] for i, e in enumerate(entries) if i % n == k - 1}
        todo = [e for e in todo if e["id"] in mine]
    if done:
        print(f"{len(done)} already searched; {len(todo)} to go", flush=True)

    sargs = search.options(out, imgsz=imgsz, possible=args.possible, explain=args.explain, runtime=args.runtime,
                           threads=args.threads, keep_intro=True,
                           videos=[p for p in videos_dir.glob("*") if p.suffix.lower() in VIDEO_EXT] if videos_dir.exists() else [])
    bank_path = work / ("intro_bank.npz" if not args.shard else f"intro_bank-{args.shard.replace('/', 'of')}.npz")
    searcher = search.Searcher(sargs, config, bank=None)
    searcher.label_intros = args.intros == "label"
    started, notes = time.time(), []
    write_report(found_dir, args.query, req, action, args.source, done, len(entries), started, notes)

    # downloads run ahead of the analysis, in threads; the analysis runs here, one video at a time
    ready = queue.Queue()
    room = threading.Semaphore(max(1, args.ahead))
    stop = threading.Event()

    def fetch(e):
        room.acquire()
        if stop.is_set():
            return
        t0 = time.time()
        try:
            if "path" in e:
                path, info, note = e["path"], {}, ""
            else:
                path, info, note = download(e, videos_dir, args, sound, out, inventory.asked_for_sound(e["id"]))
        except Exception as x:  # report it and carry on with the others
            path, info, note = None, {}, f"download failed: {x}"
        ready.put((e, path, info, note, time.time() - t0))

    pool = ThreadPoolExecutor(max(1, args.downloads))
    for e in todo:
        pool.submit(fetch, e)

    bank_needed = args.intros == "label" and not bank_path.exists()
    if args.intros == "label" and bank_path.exists():
        searcher.bank = intro.load_bank(bank_path)
    waiting, analysed, too_old, searched_s, processed = [], 0, 0, 0.0, set()   # waiting: downloaded videos held back until the intro bank is built
    try:
        for i in range(len(todo)):
            e, path, info, note, took_dl = ready.get()
            if path is None:
                room.release()
                print(f"[{i + 1}/{len(todo)}] {e['id']}: {note}", flush=True)
                if note == "outside the dates asked for":
                    too_old += 1
                    if too_old >= 5 and args.since and not src_dir:
                        print("5 videos in a row older than --since: the rest are older still; stopping", flush=True)
                        break
                else:
                    notes.append(f"{e['id']} {e.get('title', '')}: {note}")
                continue   # not saved as done: a later run tries again
            too_old = 0
            row = inventory.add(e["id"], path, info, e, None if note == "already downloaded" or src_dir else sound)
            if bank_needed:  # the intro bank is built from the first long videos, before they are searched
                room.release()   # (held-back videos do not hold up the downloads)
                waiting.append((e, path, row, took_dl, False))
                long_ones = [w for w in waiting if w[2]["kind"] == "long"]
                if len(long_ones) < BANK_VIDEOS and i < len(todo) - 1:
                    continue
                bank_needed = False
                if len(long_ones) >= intro.SHARED_BY + 1:
                    bank = intro.build_bank({w[1].name: intro.opening(w[1]) for w in long_ones})
                    ds.save_npz(bank_path, bank)
                    searcher.bank = bank
                    print(f"intro bank built from {len(long_ones)} videos: {(bank['count'] >= intro.SHARED_BY).sum()} "
                          "title frames shared by the episodes", flush=True)
                batch, waiting = waiting, []
            else:
                batch = [(e, path, row, took_dl, True)]
            for e2, path2, row2, took2, holds_room in batch:
                analysed += 1
                result = search_one(searcher, path2, e2, row2, req, action, scout, found_dir, args)
                if result["note"].startswith("failed:"):   # not saved as done: tried again next time
                    notes.append(f"{path2.name}: {result['note']}")
                else:
                    done[e2["id"]] = result
                    (found_dir / "done" / f"{e2['id']}.json").write_text(json.dumps(result))
                    if args.delete_videos and not src_dir:
                        path2.unlink(missing_ok=True)
                if holds_room:
                    room.release()
                searched_s += result["duration_s"]
                processed.add(e2["id"])
                print(f"[{len(done)}/{len(entries)}] {row2['upload_date']} {path2.name} ({row2['duration_min']} min): "
                      f"{len(result['rows'])} results" + (f" ({result['note']})" if result["note"] and not result["rows"] else "")
                      + (f"; downloaded in {took2:.0f}s" if not src_dir else "") + f", searched in {_took(result['seconds'])}"
                      + (_eta(searched_s, analysed, time.time() - started, [x for x in todo if x["id"] not in processed])
                         if analysed > 1 else ""), flush=True)
                write_report(found_dir, args.query, req, action, args.source, done, len(entries), started, notes)
    finally:
        stop.set()
        for _ in range(len(todo)):
            room.release()
        pool.shutdown(wait=False, cancel_futures=True)
    write_report(found_dir, args.query, req, action, args.source, done, len(entries), started, notes)
    total = sum(len(d["rows"]) for d in done.values())
    print(f"{total} results in {sum(bool(d['rows']) for d in done.values())} videos -> {found_dir / 'report.html'}, "
          f"{found_dir / 'report.md'}, {found_dir / 'matches.csv'}", flush=True)
    if searcher.timings:
        print("time spent: " + ", ".join(f"{k} {v:.0f}s" for k, v in sorted(searcher.timings.items(), key=lambda x: -x[1])),
              flush=True)
    if searcher.unsure:
        print(f"{searcher.unsure} more people might match (not seen well enough to tell); --possible lists them", flush=True)


def _eta(searched_s, n, took_s, left):
    """'; 12x real time, about 40 min left' from the video time of the n videos searched so far
    in took_s (videos of unknown length taken as long as the average so far)."""
    if not searched_s or not took_s:
        return ""
    rate = searched_s / took_s
    known = [e["duration"] for e in left if e.get("duration")]
    left_s = sum(known) + (len(left) - len(known)) * searched_s / max(n, 1)
    return f"; {rate:.0f}x real time" + (f", about {_took(left_s / rate)} left" if left else "")


def search_one(searcher, path, entry, row, req, action, scout, found_dir, args):
    """Search one video (a quick look first when asked); thumbnails are saved before the video
    can be deleted. Returns what done/<id>.json keeps."""
    t0 = time.time()
    before = dict(searcher.timings)
    rows, note = [], ""
    try:
        if scout and not searcher.scout(path, req):
            note = searcher.note
        else:
            rows = searcher.search(path, req, action)
            note = searcher.note
            for r in rows:
                r["thumb"] = search.thumbnail(path, r, found_dir / "thumbs")
    except Exception as x:  # one bad file should not stop the run
        note = f"failed: {type(x).__name__}: {x}"
        print(f"{path.name}: {note}", file=sys.stderr, flush=True)
    keep = ("video", "person", "start_s", "end_s", "start", "end", "duration_s", "action", "match", "evidence",
            "youtube", "thumb", "in_intro")
    return {"video": path.name, "video_id": entry["id"], "title": row["title"], "upload_date": row["upload_date"],
            "duration_s": float(row["duration_min"] or 0) * 60, "note": note, "seconds": round(time.time() - t0, 1),
            "steps": {k: round(v - before.get(k, 0.0), 1) for k, v in searcher.timings.items() if v - before.get(k, 0.0) >= 0.05},
            "rows": [{k: (r.get(k).item() if hasattr(r.get(k), "item") else r.get(k)) for k in keep} for r in rows]}


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
