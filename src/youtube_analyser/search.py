"""Searching videos for a described person doing an action (`yta search`).

    yta search "woman with purple hair talking" *.mp4
    yta search "man in a blue coat" *.mp4 --action walking
    yta search "Jane talking" *.mp4 --config my_people.toml       (Jane enrolled in people.toml)
    yta search "people jumping" *.mp4                             (an action alone: anyone doing it)

For each video, each person followed through a shot (people.py) is checked against every part of
the query (query.py): yes, no, or unknown (not seen well enough). People matching every part are
found ("match"); with possible=True, so are people for whom nothing is a "no" but something is
unknown ("possible"). Then the times they do the action (the detector's rules) become the
results. The work is done cheapest first and stops as soon as nobody in the video can match (see
Searcher.search). Everything computed is saved in the cache folder, shared with `yta detect`.
"""
import contextlib
import csv
import html
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import audio, media, people
from .actions import ACTIONS, DENSE_ACTIONS, SOUND_ACTIONS, label_people
from .analysis import Settings, VideoAnalysis, infer_images
from .query import LOOKS, PARTS, check, describe_person, describe_query, parse

FIELDS = ["video", "person", "start_s", "end_s", "start", "end", "duration_s", "action", "match", "evidence",
          "youtube", "thumb"]


class Searcher:
    """Searches videos, one at a time, with one set of models.

    cache: the model output folder (shared with `yta detect`; None keeps it in memory); bank: the
    intro bank (None: each whole video is searched); label_intros: keep results in the intros,
    marked in_intro (else they are left out); possible: also return people who might match;
    explain: log what was seen of every person."""

    def __init__(self, config=None, settings=None, cache=None, models=None, bank=None, possible=False,
                 explain=False, label_intros=False, videos=(), log=print):
        self.config = config or people.load_config()
        self.settings = settings or Settings()
        self.cache = cache
        self.models = models or self.settings.models(self.config["models"]["gender_age"])
        self.bank, self.possible, self.explain, self.label_intros = bank, possible, explain, label_intros
        self.videos = [Path(v) for v in videos]   # where enrolment's voice_from videos may be
        self.log = log
        self.enrolled = None
        self.unsure = 0          # people left out who might match (see possible)
        self.timings = {}        # seconds spent on each step, added up over the videos searched
        self.note = ""           # why the last video searched gave no results

    def analysis(self, video):
        return VideoAnalysis(video, self.settings, self.models, self.cache)

    @contextlib.contextmanager
    def timed(self, step):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[step] = self.timings.get(step, 0.0) + time.perf_counter() - t0

    # ---- people by name -----------------------------------------------------------------------

    def enrol(self):
        """{name: (voice fingerprint, pitch)} of the people in the config with a voice sample."""
        if self.enrolled is not None:
            return self.enrolled
        self.enrolled = {}
        for p in self.config.get("people", []):
            src = p.get("voice_from") or {}
            if "voice_file" in p:
                path = Path(p["voice_file"])
            elif src:
                path = Path(src.get("video", ""))
                if not path.exists():
                    path = next((v for v in self.videos if v.name == path.name), path)
            else:
                continue
            wav = media.load_audio(path) if path.exists() else None
            if wav is None:
                self.log(f"{p['name']}: no sound in {path}, not enrolled", file=sys.stderr)
                continue
            a, b = int(src.get("start", 0) * media.SR), int(src.get("end", len(wav) / media.SR) * media.SR)
            voice = audio.voice_print(wav[a:b], self.models.voices)
            if voice is None:
                self.log(f"{p['name']}: too little speech in the voice sample (at least 2 s needed)", file=sys.stderr)
                continue
            self.enrolled[p["name"]] = voice
        return self.enrolled

    # ---- one video ----------------------------------------------------------------------------

    def search(self, video, req, action=None):
        """Results for one video: [row dict], each with best_t, best_j (the 1-fps detection) and
        best_box(es) for the thumbnail. self.note says why there were none.

        Cheapest first, stopping as soon as nobody can match: the pose model; then hair and
        clothes colours, if asked for (nobody with purple hair: done); then following people and
        checking their looks; then their voices, if needed; then the face model for gender and
        age, on each remaining person's few clearest faces; then the action. Without a
        description ("anyone jumping"), the detector's own sections are the results."""
        cfg, va = self.config, self.analysis(video)
        action = action or req.get("action")
        self.note = ""
        with self.timed("pose model"):
            poses = va.poses
        if not len(poses["frame"]):
            self.note = "nobody in view"
            return []
        if not any(req[k] for k in PARTS):
            return self.anyone(va, action)
        enrolled_req = None
        if req["name"]:
            described = next((p.get("describe") for p in cfg.get("people", []) if p["name"] == req["name"]), None)
            enrolled_req = parse(described, cfg) if described else None
        either = [r for r in (req, enrolled_req) if r]
        looks = any(r[k] for r in either for k in LOOKS)
        by_face = any(r[k] for r in either for k in ("gender", "age"))
        with self.timed("hair and clothes"):
            look = va.appearance(cfg, looks=looks)
        if not self.possible and not req["name"] and not self.could_match(look, req):
            self.note = "nobody looks right: " + self.note
            return []
        with self.timed("camera"):
            camera = va.camera
        track = people.tracks(poses, camera, camera["look"], look["hair"])
        persons = people.summarise(poses, track, look, cfg)
        alive = {tr for tr, p in persons.items() if check(p, req, cfg, parts=[k for k in LOOKS if req[k]])[0] != "no"}
        if not alive:
            self.note = "nobody looks right"
            return []
        own, voices = {}, None
        if req["voice"] or req["name"] or req["gender"] or action in SOUND_ACTIONS:
            with self.timed("faces and voices"):
                own, voices = self.voices(va)
            persons = people.summarise(poses, track, look, cfg, own, voices)

            def heard(tr):
                return ["voice"] * bool(req["voice"]) + ["gender"] * bool(req["gender"] and persons[tr]["gender_by"] == "voice")
            alive = {tr for tr in alive if check(persons[tr], req, cfg, parts=heard(tr))[0] != "no"}
        if by_face and alive:
            undecided = [tr for tr in alive if persons[tr]["gender_by"] != "voice" or any(r["age"] for r in either)]
            with self.timed("gender and age"):
                look = va.appearance(cfg, looks=looks, faces_to_judge=people.face_picks(look, track, undecided, cfg))
            persons = people.summarise(poses, track, look, cfg, own, voices)
        wanted = {}
        for tr, person in persons.items():
            verdict, why = check(person, req, cfg, enrolled_req) if tr in alive else ("no", "ruled out by their looks")
            if self.explain:
                self.log(f"  {video.name} person {tr}: {describe_person(person, cfg)} -> {verdict}: {why}", flush=True)
            if verdict == "match" or (verdict == "possible" and self.possible):
                wanted[tr] = (verdict, why)
            elif verdict == "possible":
                self.unsure += 1
        if not wanted:
            self.note = "nobody matches"
            return []
        with self.timed(action or "visible"):
            timeline, states = self.states(va, track, action)
        rows = self.sections(va, timeline, states, wanted, action)
        if not rows:
            self.note = f"found, but not {action}"
        return rows

    def scout(self, video, req):
        """A quick look before the full search: the pose model and the hair and clothes colours
        on the video's keyframes only (media.keyframes). False when nobody there has the colours
        asked for (self.note says which); True when someone might, or when the query names no
        colours (nothing to look for)."""
        if not any(req[k] for k in LOOKS):
            return True
        with self.timed("quick look"):
            w, h = media.probe(video)
            # two passes over the keyframes (reading only keyframes is quick), so they are never all in memory
            poses = infer_images((img for _, img in media.keyframes(video, w, h)), self.models.pose, self.settings.conf, w, h)
            n = len(poses["t"])
            if not len(poses["frame"]):
                self.note = f"quick look: nobody in {n} keyframes"
                return False
            look = people.describe(media.keyframes(video, w, h), poses, self.models.appearance, self.config, looks=True)
            if self.could_match(look, req):
                return True
            self.note = f"quick look ({n} keyframes): {self.note}"
            return False

    def could_match(self, look, req):
        """Whether anyone in the video could have the hair and clothes colours asked for: each
        colour must be the name of at least one detection's (a person's colour is the most
        common of theirs). Sets self.note to what was missing."""
        palettes = self.config["colours"]
        names = {}
        for part, palette in (("hair", "hair"), ("top", "clothes"), ("bottom", "clothes")):
            names[part] = {people.colour_name(x, palettes[palette])[0] for x in look[part] if not np.isnan(x).any()}
        names["clothes"] = names["top"] | names["bottom"]
        for part in LOOKS:
            if req[part] and not (req[part] & names[part]):
                self.note = f"no {'/'.join(sorted(req[part]))} {part} seen"
                return False
        return True

    def voices(self, va):
        """({detection: its own voice}, {"f0", "names"}) from the talking rules and the sound."""
        own, voices = {}, None
        extra = va.inputs("talking")
        _, heard = label_people(va.poses, "talking", extra)
        for frame in heard:
            for p in frame:
                voice = p["m"].get("voice", "")
                if p["state"] == "talking" and len(voice) == 1:
                    own[p["j"]] = ord(voice) - ord("A")
        sound = extra["audio"]
        if sound is not None and "voice_f0" in sound:
            names = people.name_voices(sound["voice_centroid"], sound["voice_f0"], self.enrol(), self.config)
            voices = {"f0": sound["voice_f0"], "names": names}
        return own, voices

    def anyone(self, va, action):
        """Anyone doing the action: the detector's own sections (still figures dropped), one row
        each with everyone doing it boxed."""
        from .detector import detect
        if action in (None, "visible"):
            raise ValueError("a search needs a person or an action")
        with self.timed(action):
            extra = va.inputs(action)
        rules = ACTIONS[action]
        _, sections, in_intro, _ = detect(va.path, va.poses, action, rules["min_frames"], rules["max_gap"],
                                          va.interview_start(self.bank)["interview_start_s"], extra)
        for s in in_intro:
            s["in_intro"] = True
        rows = []
        for s in sorted((in_intro if self.label_intros else []) + sections, key=lambda s: s["start_s"]):
            boxes = [p["box"] for p in s["best_people"] if p["state"] == action]
            n = s[f"max_people_{action}"]
            rows.append({"video": va.path.name, "person": "", "start_s": s["start_s"], "end_s": s["end_s"],
                         "start": s["start"], "end": s["end"], "duration_s": s["duration_s"], "action": action,
                         "match": "match", "evidence": f"{n} {'person' if n == 1 else 'people'} {action}",
                         "youtube": media.youtube_url(va.path, s["start_s"]), "best_t": s["best_t"], "best_j": None,
                         "fps": float(extra["fps"]) if action in DENSE_ACTIONS else float(va.poses["fps"]),
                         "best_box": [float(x) for x in max(boxes, key=lambda b: float(b[2] * b[3]))],
                         "best_boxes": [[float(x) for x in b] for b in boxes], "in_intro": s.get("in_intro", False)})
        if not rows:
            self.note = f"nobody {action}"
        return rows

    def states(self, va, track, action):
        """(timeline, [(t, track, detection, doing it, box)] for every sample of a tracked person;
        detection indexes the 1-fps poses, box is where they are in that sample)."""
        poses = va.poses
        if action in (None, "visible"):
            return poses, [(float(poses["t"][f]), int(track[j]), j, True, poses["box"][j]) for j, f in enumerate(poses["frame"])]
        timeline, by_frame = va.label(action)
        out = []
        if action not in DENSE_ACTIONS:
            for t, frame in zip(timeline["t"].tolist(), by_frame):
                out += [(t, int(track[p["j"]]), p["j"], p["state"] == action, p["box"]) for p in frame]
            return timeline, out
        # the dense samples: each detection belongs to the track of the nearest 1-fps sample's
        # box (the one the sample showing it rounds to)
        fps = float(poses["fps"])
        for t, frame in zip(timeline["t"].tolist(), by_frame):
            near = np.flatnonzero(poses["frame"] == min(int(round(t * fps)), len(poses["t"]) - 1))
            for p in frame:
                overlaps = [(people.iou(people.xyxy(poses["box"][j]), people.xyxy(p["box"])), j) for j in near]
                best = max(overlaps, key=lambda x: x[0], default=(0.0, None))
                if best[0] >= 0.2:
                    out.append((t, int(track[best[1]]), best[1], p["state"] == action, p["box"]))
        return timeline, out

    def sections(self, va, timeline, states, wanted, action):
        """The times each wanted person does the action, joined as the detector joins frames:
        gaps of up to max_gap samples, at least min_frames."""
        rules = ACTIONS[action] if action not in (None, "visible") else {"min_frames": 1, "max_gap": 2}
        fps = float(timeline["fps"])
        start = va.interview_start(self.bank)["interview_start_s"]
        since = 0 if self.label_intros else start
        runs = []
        for tr, (verdict, why) in wanted.items():
            cur = None
            for t, j, box in sorted(((t, j, box) for t, k, j, doing, box in states if k == tr and doing and t >= since),
                                    key=lambda x: x[0]):
                if cur and t - cur["last"] <= (rules["max_gap"] + 1) / fps:
                    cur["last"], cur["n"] = t, cur["n"] + 1
                    cur["samples"].append((t, j, box))
                else:
                    cur = {"first": t, "last": t, "n": 1, "samples": [(t, j, box)]}
                    runs.append((tr, verdict, why, cur))
        out = []
        for tr, verdict, why, s in runs:
            if s["n"] < rules["min_frames"]:
                continue
            end = s["last"] + 1 / fps
            best_t, best_j, best_box = max(s["samples"], key=lambda x: float(x[2][2] * x[2][3]))  # the largest view
            out.append({"video": va.path.name, "person": tr, "start_s": round(s["first"], 3), "end_s": round(end, 3),
                        "start": media.hms(s["first"]), "end": media.hms(end), "duration_s": round(end - s["first"], 1),
                        "action": action or "visible", "match": verdict, "evidence": why,
                        "youtube": media.youtube_url(va.path, s["first"]), "best_t": best_t, "best_j": best_j, "fps": fps,
                        "best_box": [float(x) for x in best_box], "in_intro": bool(s["first"] < start)})
        return sorted(out, key=lambda r: (r["start_s"], str(r["person"])))


# ---- output ----------------------------------------------------------------------------------

def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "search"


def thumbnail(video, r, folder):
    """Save a result's thumbnail (its best sample, 3x enlarged, the person boxed) in folder;
    returns its file name."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    w, h = media.probe(video)
    big = cv2.resize(media.grab_frame(video, r["best_t"], r.get("fps") or 1.0, w, h), None, fx=3, fy=3,
                     interpolation=cv2.INTER_CUBIC)
    for x, y, bw, bh in r.get("best_boxes") or [r["best_box"]]:
        cv2.rectangle(big, (int(3 * x), int(3 * y)), (int(3 * (x + bw)), int(3 * (y + bh))), (60, 200, 60), 2)
    name = f"{Path(video).stem}_{int(r['start_s']):05d}_{r['person']}.jpg"
    cv2.imwrite(str(folder / name), big)
    return name


def write_results(rows, query, req, out_dir):
    """matches.csv and index.html in out_dir (the thumbnails are already in out_dir/thumbs)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "matches.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, FIELDS, extrasaction="ignore", lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    e = html.escape
    cards = "\n".join(
        f'<figure><a href="thumbs/{e(r["thumb"])}"><img src="thumbs/{e(r["thumb"])}" loading="lazy"></a>'
        f'<figcaption>{e(r["video"])} person {r["person"]}<br>{r["start"]} to {r["end"]} ({r["duration_s"]}s, '
        f'{r["action"]}, {r["match"]})<br><small>{e(r["evidence"])}</small>'
        + (f' · <a href="{e(r["youtube"])}">YouTube</a>' if r["youtube"] else "") + "</figcaption></figure>"
        for r in rows)
    ignored = f"<p>Ignored words: {e(', '.join(req['ignored']))}</p>" if req["ignored"] else ""
    (out_dir / "index.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>{e(query)}</title>"
        "<style>body{font:14px system-ui;margin:16px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}"
        "figure{margin:0}img{width:100%;border-radius:4px}</style>"
        f"<h1>{e(query)}</h1><p>Looking for: {e(describe_query(req))}; action: "
        f"{e(req['action'] or 'any (in view)')}. {len(rows)} results.</p>{ignored}<main>{cards}</main>")


def search_videos(query, videos, config=None, action=None, settings=None, cache=Path("output"),
                  results=Path("results/search"), bank=None, possible=False, explain=False, quick=False, log=print):
    """`yta search`: every video searched for the query; writes results/<query>/matches.csv,
    index.html and thumbs/. Returns (rows, the folder)."""
    config = config or people.load_config()
    names = [p["name"] for p in config.get("people", [])]
    req = parse(query, config, names)
    req["action"] = action = action or req["action"]
    log(f"looking for: {describe_query(req)}; action: {action or 'any (in view)'}"
        + (f"; ignored words: {', '.join(req['ignored'])}" if req["ignored"] else ""), flush=True)
    if not any(req[k] for k in PARTS) and action in (None, "visible"):
        raise ValueError("the query names nobody and no action: nothing to search for")
    videos = [Path(v) for v in videos if not media.is_format_part(v)]
    searcher = Searcher(config, settings, cache, bank=bank, possible=possible, explain=explain, videos=videos, log=log)
    out_dir = Path(results) / slug(query if not action or action in query.lower() else f"{query} {action}")
    rows = []
    for i, v in enumerate(videos, 1):
        try:
            found = searcher.search(v, req, action) if not quick or searcher.scout(v, req) else []
            for r in found:
                r["thumb"] = thumbnail(v, r, out_dir / "thumbs")
        except Exception as e:  # a bad file should not stop the search
            log(f"[{i}/{len(videos)}] {v.name}: FAILED {e}", file=sys.stderr, flush=True)
            continue
        rows += found
        log(f"[{i}/{len(videos)}] {v.name}: {len(found)} results"
            + (f" ({searcher.note})" if searcher.note and not found else ""), flush=True)
    rows.sort(key=lambda r: (r["match"] != "match", r["video"], r["start_s"]))
    write_results(rows, query, req, out_dir)
    log(f"{len(rows)} results ({sum(r['match'] == 'match' for r in rows)} matches) -> {out_dir / 'matches.csv'}, "
        f"{out_dir / 'index.html'}")
    if searcher.unsure:
        log(f"{searcher.unsure} more people might match (not seen well enough to tell); --possible lists them, "
            "--explain says why")
    return rows, out_dir
