"""Finding the sections of each video where someone does an action, and writing them out.

Frames showing the action (actions.label_people) are joined into sections; statues, posters and
other figures that never move are dropped; sections in the intro (title sequence, teaser, ads;
see intro.py) are listed apart. `yta detect` runs this on a folder of videos:

Outputs in --out (for an action other than standing, the files below except poses/, faces/,
audio/ and camera/ are in --out/<action>/ instead, with the action's name in place of "standing"):
    poses/<video>.npz     every person found in every sampled frame (17 body points each)
                          (<video>.dense.npz: the extra samples for running and jumping)
    faces/<video>.npz     face measurements for smiling, talking and silent
    audio/<video>.npz     speech and voices in the sound, for talking and silent
    camera/<video>.npz    the camera's movement between samples, for walking, sitting and --motion
    frames/<video>.csv    one row per sampled frame
    segments/<video>.csv  standing sections of that video (its presence marks the video done)
    thumbs/*.jpg          best frame of each section, with the detected skeletons drawn
    segments.csv          all standing sections in the interviews of all videos in this folder
    intro.csv             standing sections in the intros
    dropped.csv           sections dropped as still figures
    interview_starts.csv  where each video's interview starts
    review.html           thumbnails with YouTube links at each section's start time
    speech.csv, voices.csv  talking and silent: who spoke when, and each video's voices
"""
import csv
import html
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import audio, faces, media
from .actions import (
    ACTIONS,
    DENSE_ACTIONS,
    NOTE_COLUMNS,
    SOUND_ACTIONS,
    SPEECH_FIELDS,
    VOICE_FIELDS,
    frame_fields,
    label_people,
    rounded,
    segment_fields,
)
from .analysis import Settings, VideoAnalysis
from .media import hms
from .pose import ANKLE_VIS, SKELETON

COLORS = {"standing": (60, 200, 60), "sitting": (255, 150, 30), "unknown": (170, 170, 170)}  # BGR
ACTION_COLOR, OTHER_COLOR = (60, 200, 60), (255, 150, 30)  # for actions other than standing
INTERVIEW_FIELDS = ["video", "interview_start_s", "interview_start", "title_end_s", "how"]

# Statues, posters and artwork shaped like a standing person pass every rule above and come
# back every time their camera angle is shown. Unlike a person they never move: across all
# 248 test videos their pixels changed by at most 3.0 grey levels (of 255) between nearby samples,
# while real standing people, even a bystander standing still, changed by 4.4 or more.
# A section whose standing figures all stay below STILL_DIFF is dropped as a still figure.
STILL_DIFF = 3.5                       # mean absolute grey-level change inside the figure's box
STILL_PAIR_S = 3                       # compare standing samples up to this many seconds apart
STILL_CHUNK_S = 300                    # longer sections are read in pieces of this many seconds (at 144p;
                                       # pieces of larger pictures are shorter, so they fit in memory)

# Standing in the intro (title sequence, teaser, ads) is reported separately; intro.py finds
# where the interview starts. A section that starts in the intro and ends at most this many
# seconds after the interview start is counted as intro: the start can fall in a crossfade.
INTRO_GRACE_S = 3


def draw(img, people, scale=3, action="standing"):
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for p in people:
        color = COLORS.get(p["state"]) or (ACTION_COLOR if p["state"] == action else OTHER_COLOR)
        if p.get("face_box") is not None:
            x, y, bw, bh = (int(v * scale) for v in p["face_box"])
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 1, cv2.LINE_AA)
        kp, ok = p["kp"][:, :2] * scale, p["kp"][:, 2] >= ANKLE_VIS
        for a, b in SKELETON:
            if ok[a] and ok[b]:
                cv2.line(out, tuple(map(int, kp[a])), tuple(map(int, kp[b])), color, 2, cv2.LINE_AA)
        for i in np.flatnonzero(ok):
            cv2.circle(out, tuple(map(int, kp[i])), 3, color, -1, cv2.LINE_AA)
        x, y = int(p["box"][0] * scale), int(p["box"][1] * scale)
        cv2.putText(out, p["state"], (x + 2, max(y + 14, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    return out


def find_sections(poses, name, min_frames, max_gap, interview_start=0, action="standing", extra=None):
    """Classify everyone in the poses and join the frames showing the action into sections.

    Returns (frames, sections, intro): one row per sampled frame, the sections from the
    interview start on, and the sections before it (in the intro). Each has at least
    min_frames frames showing the action and keeps best_t / best_people for drawing a thumbnail.
    A section that starts in the intro and ends within INTRO_GRACE_S of the interview start
    belongs to the intro (the detected start can fall inside a crossfade); one that runs on
    longer is split at the interview start. extra: see label_people.
    """
    timeline, by_frame = label_people(poses, action, extra)
    frames, whole = join_sections(timeline, by_frame, name, min_frames, max_gap, action=action)
    if not interview_start:
        return frames, whole, []
    cut = interview_start
    for s in whole:
        if s["start_s"] < interview_start < s["end_s"] <= interview_start + INTRO_GRACE_S:
            cut = max(cut, s["end_s"])
    _, in_intro = join_sections(timeline, by_frame, name, min_frames, max_gap, until=cut, action=action)
    _, sections = join_sections(timeline, by_frame, name, min_frames, max_gap, since=cut, action=action)
    return frames, sections, in_intro


def join_sections(poses, by_frame, name, min_frames, max_gap, since=0.0, until=math.inf, action="standing"):
    """One row per sampled frame, and the sections of the action between since and until."""
    a = ACTIONS[action]
    fps = float(poses["fps"])
    frames, segments, cur = [], [], None
    for t, people in zip(poses["t"].tolist(), by_frame):
        states = [p["state"] for p in people]
        active = [p for p in people if p["state"] == action]
        row = {"t": t, "people": len(people), **{s: states.count(s) for s in a["states"]}}
        for m in a["stats"]:
            values = [p["m"][m] for p in people if p["m"].get(m) is not None]
            row[f"best_{m}"] = rounded(m, max(values)) if values else ""
        frames.append(row)
        if not active or not since <= t < until:
            continue
        score = float(max(p["m"][a["score"]] for p in active))
        if cur and t - cur["last_t"] <= (max_gap + 1) / fps:
            cur.update(end_s=t + 1 / fps, last_t=t)
            cur["n"] += 1
            cur["most"] = max(cur["most"], len(active))
        else:
            cur = {"video": name, "start_s": t, "end_s": t + 1 / fps, "last_t": t, "n": 1,
                   "most": len(active), "best": -math.inf, "boxes": []}
            segments.append(cur)
        cur["boxes"].append((t, [p["box"] for p in active]))
        for note in a.get("notes", ()):  # voices ("A+B") are listed one by one
            cur.setdefault("notes", {}).setdefault(note, set()).update(
                v for p in active for v in (str(p["m"].get(note) or "").split("+") if note != "by"
                                            else [p["m"].get("by")]) if v)
        if score > cur["best"]:  # compared with the rounded best, as the CSV shows it
            cur.update(best=rounded(a["score"], score), best_t=t, best_people=people)

    segments = [s for s in segments if s["n"] >= min_frames]
    for s in segments:
        if action != "standing":  # jumping's 0.2 s steps would otherwise show as 3.4000000000000004
            s.update(start_s=round(s["start_s"], 3), end_s=round(s["end_s"], 3))
        s.update({NOTE_COLUMNS[k]: (" " if k != "by" else ", ").join(sorted(v)) for k, v in s.get("notes", {}).items()})
        s.update({f"{action}_frames": s["n"], f"max_people_{action}": s["most"], f"best_{a['score']}": s["best"]},
                 start=hms(s["start_s"]), end=hms(s["end_s"]), duration_s=round(s["end_s"] - s["start_s"], 1))
    return frames, segments


def overlap(a, b):
    """Intersection over union of two x, y, w, h boxes."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / (a[2] * a[3] + b[2] * b[3] - ix * iy + 1e-9)


def still_diff(path, section, fps, w, h):
    """Largest pixel change of the section's figures between nearby samples.

    None when no figure is seen twice within STILL_PAIR_S seconds (then it can't be judged).
    A section longer than STILL_CHUNK_S (sitting or smiling can last most of an episode) is
    read a piece at a time, and only until a piece shows the figure moving: the largest change
    found up to then is returned.
    """
    boxes = section["boxes"]
    chunk = max(STILL_PAIR_S * 10, STILL_CHUNK_S * min(1.0, 256 * 144 / (w * h)))
    if boxes[-1][0] - boxes[0][0] <= chunk:
        return _still_diff(media.read_range(path, boxes[0][0], boxes[-1][0], fps, w, h), boxes, w, h)
    best, i = None, 0
    while i < len(boxes):
        t0 = boxes[i][0]
        part = [b for b in boxes[i:] if b[0] <= t0 + chunk + STILL_PAIR_S]
        n = sum(b[0] < t0 + chunk for b in part)  # pairs start in this piece
        d = _still_diff(media.read_range(path, t0, part[-1][0], fps, w, h), part, w, h, starts=n)
        best = d if best is None else best if d is None else max(best, d)
        if best is not None and best >= STILL_DIFF:
            break
        i += n
    return best


def _still_diff(frames, boxes, w, h, starts=None):
    best = None
    for i, (ta, boxes_a) in enumerate(boxes[:starts]):
        for tb, boxes_b in boxes[i + 1:]:
            if tb - ta > STILL_PAIR_S:
                break
            fa, fb = frames.get(round(ta, 3)), frames.get(round(tb, 3))
            if fa is None or fb is None:
                continue
            for a in boxes_a:
                for b in boxes_b:
                    if overlap(a, b) < 0.3:
                        continue
                    x0, y0 = int(max(0, min(a[0], b[0]))), int(max(0, min(a[1], b[1])))
                    x1 = int(min(w, max(a[0] + a[2], b[0] + b[2])))
                    y1 = int(min(h, max(a[1] + a[3], b[1] + b[3])))
                    if x1 - x0 >= 2 and y1 - y0 >= 2:
                        d = float(np.abs(fa[y0:y1, x0:x1].astype(np.int16) - fb[y0:y1, x0:x1]).mean())
                        best = d if best is None else max(best, d)
    return best


def drop_still_figures(path, sections, fps, w, h):
    """Split sections into (kept, dropped): dropped ones are statues, posters and the like."""
    kept, dropped = [], []
    for s in sections:
        d = still_diff(path, s, fps, w, h)
        s["still_diff"] = "" if d is None else round(d, 2)
        (dropped if d is not None and d < STILL_DIFF else kept).append(s)
    return kept, dropped


def detect(path, poses, action, min_frames, max_gap, interview_start=0, extra=None):
    """The sections of one video: (frames, sections, intro sections, sections dropped as still)."""
    frames, sections, in_intro = find_sections(poses, path.name, min_frames, max_gap, interview_start,
                                               action, extra)
    if ACTIONS[action]["still"]:
        w, h = (int(x) for x in poses["size"])
        sections, dropped = drop_still_figures(path, sections, float(poses["fps"]), w, h)
    else:
        dropped = []
        for s in sections:
            s["still_diff"] = ""
    return frames, sections, in_intro, dropped


def write_video(analysis, action, min_frames, max_gap, out_dir, bank=None, extra=None):
    """Find one video's sections and write its CSVs and thumbnails into out_dir.
    Returns (frames, sections, dropped as still, in the intro, interview start)."""
    path, poses = analysis.path, analysis.poses
    stem = path.stem
    w, h = (int(x) for x in poses["size"])
    fps = float((extra if action in DENSE_ACTIONS else poses)["fps"])  # the rate the sections were sampled at
    url = media.youtube_url(path)
    start = analysis.interview_start(bank)
    frames, segments, in_intro, dropped = detect(path, poses, action, min_frames, max_gap,
                                                 start["interview_start_s"], extra)
    for old in (out_dir / "thumbs").glob(f"{glob_escape(stem)}_*.jpg"):
        old.unlink()
    for s in segments + dropped + in_intro:
        s["thumb"] = f"{stem}_{int(s['start_s']):05d}.jpg"
        img = media.grab_frame(path, s["best_t"], fps, w, h)
        cv2.imwrite(str(out_dir / "thumbs" / s["thumb"]), draw(img, s["best_people"], action=action))
        s["youtube"] = f"{url}&t={int(s['start_s'])}s" if url else ""
    fields = segment_fields(action)
    if action in SOUND_ACTIONS:
        speech, voices = speech_timeline(path, poses, action, extra, out_dir / "thumbs", url)
        write_csv(out_dir / "speech" / f"{stem}.csv", SPEECH_FIELDS, speech)
        write_csv(out_dir / "voices" / f"{stem}.csv", VOICE_FIELDS, voices)
    write_csv(out_dir / "frames" / f"{stem}.csv", frame_fields(action), frames)
    write_csv(out_dir / "interview" / f"{stem}.csv", INTERVIEW_FIELDS, [start])
    write_csv(out_dir / "intro" / f"{stem}.csv", fields, in_intro)
    write_csv(out_dir / "dropped" / f"{stem}.csv", fields, dropped)
    write_csv(out_dir / "segments" / f"{stem}.csv", fields, segments)  # written last: marks done
    return frames, segments, dropped, in_intro, start


def speech_timeline(path, poses, action, extra, thumbs=None, url=""):
    """Who spoke when (the sound's turns), whether they were on screen, and a summary per voice.

    A turn is on screen ("yes") when, in most of its sampled seconds with a face that could be
    judged, a visible person was talking with that voice; "no" when faces were judged but none
    was; "unknown" when no face could be judged. Each voice's thumbnail is the sampled frame
    where it was matched to the largest single talking face. Returns (turn rows, voice rows).
    """
    sound = extra.get("audio")
    if sound is None:
        return [], []
    _, by_frame = label_people(poses, action, extra)
    voices = audio.frame_voices(sound)
    t = media.picture_time(poses["t"].astype(np.float64), float(poses["fps"])) \
        + faces.BURST_N * faces.BURST_STEP / 2  # the middle of each sample's burst
    rows, best = [], {}
    for s, e, k in audio.turns(sound, voices):
        v = audio.voice_name(k)
        judged = seen = 0
        for f in range(np.searchsorted(t, s), np.searchsorted(t, e)):
            seen_faces = [p for p in by_frame[f] if p["state"] in ("talking", "silent")]
            talking = [p for p in seen_faces if p["state"] == "talking" and v in str(p["m"].get("voice", "")).split("+")]
            judged += bool(seen_faces)
            seen += bool(talking)
            if len(talking) == 1 and p_size(talking[0]) > best.get(v, (0,))[0]:
                best[v] = (p_size(talking[0]), f, talking[0])
        rows.append({"video": path.name, "start_s": s, "end_s": e, "start": hms(s), "end": hms(e),
                     "duration_s": round(e - s, 1), "voice": v,
                     "on_screen": "unknown" if not judged else "yes" if seen >= judged / 2 else "no",
                     "sound_trusted": "yes" if audio.heard(sound, s, e, voices)[0] else "no",
                     "youtube": f"{url}&t={int(s)}s" if url else ""})
    summary = []
    for v in sorted({r["voice"] for r in rows}):
        mine = [r for r in rows if r["voice"] == v]
        thumb = ""
        if thumbs is not None and v in best:
            _, f, person = best[v]
            w, h = (int(x) for x in poses["size"])
            thumb = f"{path.stem}_voice_{v}.jpg"
            img = media.grab_frame(path, float(poses["t"][f]), float(poses["fps"]), w, h)
            cv2.imwrite(str(thumbs / thumb), draw(img, [person], action="talking"))
        summary.append({"video": path.name, "voice": v, "seconds": round(sum(r["duration_s"] for r in mine), 1),
                        "turns": len(mine), "thumb": thumb,
                        "sound_trusted": round(sum(r["duration_s"] for r in mine if r["sound_trusted"] == "yes")
                                               / max(sum(r["duration_s"] for r in mine), 1e-9), 2),
                        "on_screen_seconds": round(sum(r["duration_s"] for r in mine if r["on_screen"] == "yes"), 1)})
    return rows, summary


def p_size(p):
    """A person's face size (eye distance) for picking the clearest face."""
    return float(p["m"].get("eyes") or 0)


def glob_escape(s):
    return re.sub(r"([*?\[])", r"[\1]", s)


def write_csv(path, fields, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def combine(out_dir, action="standing"):
    """Write the combined CSVs and review.html for every video processed into out_dir."""
    fields, label = segment_fields(action), action.capitalize()
    found = {}
    for kind, kind_fields, out in (("segments", fields, "segments"), ("dropped", fields, "dropped"),
                                   ("intro", fields, "intro"), ("interview", INTERVIEW_FIELDS, "interview_starts")):
        found[kind] = []
        for p in sorted((out_dir / kind).glob("*.csv")):
            with open(p, newline="") as f:
                found[kind] += list(csv.DictReader(f))
        write_csv(out_dir / f"{out}.csv", kind_fields, found[kind])

    notes = [NOTE_COLUMNS[n] for n in ACTIONS[action].get("notes", ())]

    def cards(rows):
        return "\n".join(
            f'<figure><a href="thumbs/{html.escape(r["thumb"])}"><img src="thumbs/{html.escape(r["thumb"])}" loading="lazy"></a>'
            f'<figcaption>{html.escape(r["video"])}<br>{r["start"]} to {r["end"]} ({r["duration_s"]}s, '
            f'{r[f"{action}_frames"]} frames, pixel change {r["still_diff"] or "n/a"})'
            + "".join(f"<br>{n.replace('_', ' ')}: {html.escape(r[n] or 'none')}" for n in notes)
            + (f' · <a href="{html.escape(r["youtube"])}">YouTube</a>' if r["youtube"] else "")
            + "</figcaption></figure>" for r in rows)

    voices_html = ""
    if action in SOUND_ACTIONS:
        for kind, kind_fields in (("speech", SPEECH_FIELDS), ("voices", VOICE_FIELDS)):
            found[kind] = []
            for p in sorted((out_dir / kind).glob("*.csv")):
                with open(p, newline="") as f:
                    found[kind] += list(csv.DictReader(f))
            write_csv(out_dir / f"{kind}.csv", kind_fields, found[kind])
        voices_html = (
            f"<h2>Voices ({len(found['voices'])})</h2><p>Each video's voices, from the sound (A speaks most). "
            "The picture is the clearest moment the voice was matched to a talking face; without one, the voice "
            "was never seen talking (off screen, or faces too small or turned away). Who spoke when is in "
            "speech.csv.</p><main>" + "\n".join(
                (f'<figure><a href="thumbs/{html.escape(r["thumb"])}"><img src="thumbs/{html.escape(r["thumb"])}" '
                 'loading="lazy"></a>' if r["thumb"] else "<figure>")
                + f'<figcaption>{html.escape(r["video"])}: voice {r["voice"]}, {float(r["seconds"]) / 60:.1f} min in '
                f'{r["turns"]} turns, {float(r["on_screen_seconds"]) / 60:.1f} min of it seen talking'
                + ("" if float(r["sound_trusted"] or 0) >= 0.5 else " (sound too noisy to trust)") + '</figcaption></figure>'
                for r in found["voices"]) + "</main>")

    (out_dir / "review.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>{label} sections</title>"
        "<style>body{font:14px system-ui;margin:16px;background:#fff;color:#111}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}"
        "figure{margin:0}img{width:100%;border-radius:4px}figcaption{margin-top:4px}</style>"
        f"<h1>{label} sections ({len(found['segments'])})</h1><main>{cards(found['segments'])}</main>{voices_html}"
        f"<h2>Dropped as still figures ({len(found['dropped'])})</h2>"
        f"<p>Statues, posters and other figures that never moved (pixel change under {STILL_DIFF}).</p>"
        f"<main>{cards(found['dropped'])}</main>"
        f"<h2>In the intro ({len(found['intro'])})</h2>"
        f"<p>{label} before the interview starts: title sequences, teasers and ads.</p>"
        f"<main>{cards(found['intro'])}</main>")
    return found["segments"], found["dropped"], found["intro"]



def detect_videos(videos, action="standing", out=Path("output"), settings=None, min_frames=None, max_gap=None,
                  motion_filter=None, use_sound=True, reclassify=False, force=False, bank=None, models=None,
                  log=print):
    """The detector on a list of videos (`yta detect`): each video's sections and thumbnails,
    then the combined CSVs and review.html. A video whose segments/<video>.csv exists is done and
    skipped, unless reclassify (sections again, from the saved model output) or force (everything
    again). A video that fails is reported and left for the next run. Returns (kept, dropped,
    intro) sections of all videos in out."""
    settings = settings or Settings()
    rules = ACTIONS[action]
    min_frames = rules["min_frames"] if min_frames is None else min_frames
    max_gap = rules["max_gap"] if max_gap is None else max_gap
    out = Path(out)
    out_dir = out if action == "standing" else out / action
    for sub in ("frames", "segments", "dropped", "intro", "interview", "thumbs") + \
            (("speech", "voices") if action in SOUND_ACTIONS else ()):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    models = models or settings.models()
    videos, seen = [Path(v) for v in videos if not media.is_format_part(v)], {}
    for i, v in enumerate(videos, 1):
        head = f"[{i}/{len(videos)}] {v.name}:"
        if v.stem in seen:  # the saved model output is named after the file: two with one name would mix
            log(f"{head} skipped: another video is also called {v.stem} ({seen[v.stem]})", file=sys.stderr)
            continue
        seen[v.stem] = v
        done = (out_dir / "segments" / f"{v.stem}.csv").exists()
        if done and not (reclassify or force):
            log(f"{head} done already, skipping", flush=True)
            continue
        started = time.perf_counter()
        try:
            analysis = VideoAnalysis(v, settings, models, out, force=force)
            extra = analysis.inputs(action, motion_filter, use_sound)
            frames, sections, dropped, in_intro, start = write_video(analysis, action, min_frames, max_gap, out_dir,
                                                                     bank, extra)
        except Exception as e:  # keep going on a bad file; it stays unmarked and is retried next run
            log(f"{head} FAILED {e}", file=sys.stderr, flush=True)
            continue
        more = "".join(f"; {n}" for n in analysis.notes)
        log(f"{head} interview from {start['interview_start']}, {len(frames)} frames, {len(sections)} {action} sections"
            + (f" ({len(in_intro)} more in the intro)" if in_intro else "")
            + (f" ({len(dropped)} still figures dropped)" if dropped else "")
            + f", {time.perf_counter() - started:.0f}s ({analysis.pose_note}{more})", flush=True)
    kept, dropped, in_intro = combine(out_dir, action)
    log(f"{len(kept)} {action} sections in total ({len(in_intro)} more in intros, {len(dropped)} still "
        f"figures dropped) -> {out_dir / 'segments.csv'}, {out_dir / 'review.html'}")
    return kept, dropped, in_intro
