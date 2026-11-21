"""Cut every standing section out of the local videos and join them into one review video.

The sections come from results/standing_sections.csv (`yta report`), which only has standing
during the interviews (intros are left out; see intro.py).

Each clip gets a few seconds of context on either side and a caption bar with its number,
upload date, video ID, the standing time range(s) and the verdict. Sections of the same video
whose clips would overlap are merged into one clip. Every clip is re-encoded to one format
(640x360, 25 fps, H.264 + AAC, with silent audio where the source has none) so the clips join
without glitches. An index CSV next to the video says where each clip starts in it.

With --action (sitting, walking, running, jumping, smiling, talking, silent) it reads
results/<action>/<action>_sections.csv (from `yta report --action`) and writes
output/<action>/review.mp4; there the default also includes sections that are not reviewed yet,
since this video is how you review them.

    yta review                         # confirmed + unclear
    yta review --verdicts all --pad 3
    yta review --verdicts "false alarm" --out output/false_alarms.mp4
    yta review --action talking
"""
import csv
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from .media import duration, has_audio, hms

BAR_H = 44  # caption bar height in output pixels
FPS = 25


def ascii_text(s):
    """OpenCV's fonts only draw ASCII; drop accents, emoji and curly quotes."""
    s = s.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().strip()


def fit(text, width, scale):
    """Shorten text with '...' until it fits in width pixels."""
    while cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width and len(text) > 4:
        text = text[:-4] + "..."
    return text


def caption(path, width, line1, line2):
    bar = np.zeros((BAR_H, width, 3), np.uint8)
    cv2.putText(bar, fit(line1, width - 12, 0.5), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, fit(line2, width - 12, 0.42), (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (190, 190, 190), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), bar)


def plan_clips(sections, pad):
    """Group sections by video, pad them, and merge the ones whose padded clips overlap."""
    clips = []
    for s in sorted(sections, key=lambda s: (s["video"], float(s["start_s"]))):
        start, end = max(0.0, float(s["start_s"]) - pad), float(s["end_s"]) + pad
        last = clips[-1] if clips else None
        if last and last["video"] == s["video"] and start <= last["end"]:
            last["end"] = max(last["end"], end)
            last["sections"].append(s)
        else:
            clips.append({"video": s["video"], "start": start, "end": end, "sections": [s]})
    return clips


def make_review(action="standing", sections=None, videos_dir=Path("."), verdicts=None, pad=2.0, out=None,
                width=640, height=360, log=print):
    """Cut the sections out of the videos and join them into out (see above); writes an index
    CSV next to it. Returns (out, the index rows)."""
    standing = action == "standing"
    sections_path = Path(sections or (Path("results") / ("standing_sections.csv" if standing else
                                                          f"{action}/{action}_sections.csv")))
    verdicts = verdicts or ("confirmed,unclear" if standing else "confirmed,unclear,not reviewed")
    out = Path(out or Path("output") / ("" if standing else action) / "review.mp4")
    videos_dir = Path(videos_dir)
    with open(sections_path, newline="") as f:
        rows = list(csv.DictReader(f))
    wanted = None if verdicts == "all" else {v.strip() for v in verdicts.split(",")}
    rows = [s for s in rows if wanted is None or s["verdict"] in wanted]
    if not rows:
        raise ValueError(f"no sections with verdict {verdicts!r} in {sections_path}")
    clips = plan_clips(rows, pad)
    log(f"{len(rows)} sections -> {len(clips)} clips", flush=True)

    w, h = width, height
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="review-", dir=out.parent))
    made, index = [], []
    try:
        for n, c in enumerate(clips, 1):
            src = videos_dir / c["video"]
            if not src.exists():
                log(f"[{n}/{len(clips)}] {c['video']}: not found in {videos_dir}, skipped", flush=True)
                continue
            first = c["sections"][0]
            ranges = ", ".join(f"{s['start']}-{s['end']}" for s in c["sections"])
            clip_verdicts = "/".join(sorted({s["verdict"] for s in c["sections"]}))
            video_id = Path(c["video"]).stem.split("_", 1)[-1]
            cap = work / f"cap{n:04d}.png"
            caption(cap, w, f"{n}/{len(clips)}  {first['upload_date']}  {video_id}  {action} {ranges}  [{clip_verdicts}]",
                    ascii_text(first["title"]))
            clip = work / f"clip{n:04d}.mp4"
            dur = c["end"] - c["start"]
            cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{c['start']:.3f}", "-i", str(src), "-i", str(cap)]
            audio = "0:a:0"
            if not has_audio(src):
                cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
                audio = "2:a"
            vf = (f"[0:v]scale={w}:{h - BAR_H}:force_original_aspect_ratio=decrease,"
                  f"pad={w}:{h}:(ow-iw)/2:{BAR_H}+({h - BAR_H}-ih)/2,setsar=1,fps={FPS}[v];"
                  "[v][1:v]overlay=0:0[out]")
            cmd += ["-filter_complex", vf, "-map", "[out]", "-map", audio, "-t", f"{dur:.3f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "96k", str(clip)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode or not clip.exists():
                log(f"[{n}/{len(clips)}] {c['video']}: ffmpeg failed: {r.stderr.strip()[-300:]}", flush=True)
                continue
            at = sum(i["clip_seconds"] for i in index)
            index.append({"clip": n, "review_at": hms(at), "video": c["video"], "title": first["title"],
                          action: ranges, "verdict": clip_verdicts, "clip_seconds": round(duration(clip), 2),
                          "youtube": first["youtube"]})
            made.append(clip)
            log(f"[{n}/{len(clips)}] {c['video']} {ranges}", flush=True)
        if not made:
            raise RuntimeError("no clips could be made")

        listing = work / "clips.txt"
        listing.write_text("".join(f"file '{p.name}'\n" for p in made))
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                            "-c", "copy", "-movflags", "+faststart", str(out)], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"joining the clips failed: {r.stderr.strip()[-500:]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    index_path = out.with_suffix(".csv")
    with open(index_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(index[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(index)
    log(f"{len(made)} clips, {hms(duration(out))} long -> {out}\nindex -> {index_path}")
    return out, index

