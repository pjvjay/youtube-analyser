"""Cut every standing section out of the local videos and join them into one review video.

The sections come from results/standing_sections.csv, which only has standing during the
interviews (intros are left out by detect_standing.py; see intro.py).

Each clip gets a few seconds of context on either side and a caption bar with its number,
upload date, video ID, the standing time range(s) and the verdict. Sections of the same video
whose clips would overlap are merged into one clip. Every clip is re-encoded to one format
(640x360, 25 fps, H.264 + AAC, with silent audio where the source has none) so the clips join
without glitches. An index CSV next to the video says where each clip starts in it.

Usage (from the youtube-analyser folder):
    .venv/bin/python standing/review_video.py                         # confirmed + unclear
    .venv/bin/python standing/review_video.py --verdicts all --pad 3
    .venv/bin/python standing/review_video.py --verdicts "false alarm" --out standing/output/false_alarms.mp4
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
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


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                          "stream=index", "-of", "json", str(path)], capture_output=True, text=True).stdout
    return bool(json.loads(out or "{}").get("streams"))


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json",
                          str(path)], capture_output=True, text=True).stdout
    return float(json.loads(out or "{}").get("format", {}).get("duration", 0))


def hms(t):
    t = int(round(t))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}"


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", type=Path, default=ROOT / "results" / "standing_sections.csv")
    ap.add_argument("--videos-dir", type=Path, default=ROOT, help="folder holding the downloaded videos")
    ap.add_argument("--verdicts", default="confirmed,unclear",
                    help='comma-separated verdicts to include, or "all"')
    ap.add_argument("--pad", type=float, default=2.0, help="seconds of context before and after each section")
    ap.add_argument("--out", type=Path, default=ROOT / "standing" / "output" / "review.mp4")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    args = ap.parse_args()

    with open(args.sections, newline="") as f:
        sections = list(csv.DictReader(f))
    wanted = None if args.verdicts == "all" else {v.strip() for v in args.verdicts.split(",")}
    sections = [s for s in sections if wanted is None or s["verdict"] in wanted]
    if not sections:
        sys.exit(f"no sections with verdict {args.verdicts!r} in {args.sections}")
    clips = plan_clips(sections, args.pad)
    print(f"{len(sections)} sections -> {len(clips)} clips", flush=True)

    w, h = args.width, args.height
    work = Path(tempfile.mkdtemp(prefix="review-", dir=args.out.parent if args.out.parent.exists() else None))
    made, index = [], []
    try:
        for n, c in enumerate(clips, 1):
            src = args.videos_dir / c["video"]
            if not src.exists():
                print(f"[{n}/{len(clips)}] {c['video']}: not found in {args.videos_dir}, skipped", flush=True)
                continue
            first = c["sections"][0]
            ranges = ", ".join(f"{s['start']}-{s['end']}" for s in c["sections"])
            verdicts = "/".join(sorted({s["verdict"] for s in c["sections"]}))
            video_id = Path(c["video"]).stem.split("_", 1)[-1]
            cap = work / f"cap{n:04d}.png"
            caption(cap, w, f"{n}/{len(clips)}  {first['upload_date']}  {video_id}  standing {ranges}  [{verdicts}]",
                    ascii_text(first["title"]))
            out = work / f"clip{n:04d}.mp4"
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
                    "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "96k", str(out)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode or not out.exists():
                print(f"[{n}/{len(clips)}] {c['video']}: ffmpeg failed: {r.stderr.strip()[-300:]}", flush=True)
                continue
            at = sum(i["clip_seconds"] for i in index)
            index.append({"clip": n, "review_at": hms(at), "video": c["video"], "title": first["title"],
                          "standing": ranges, "verdict": verdicts, "clip_seconds": round(duration(out), 2),
                          "youtube": first["youtube"]})
            made.append(out)
            print(f"[{n}/{len(clips)}] {c['video']} {ranges}", flush=True)
        if not made:
            sys.exit("no clips could be made")

        listing = work / "clips.txt"
        listing.write_text("".join(f"file '{p.name}'\n" for p in made))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                            "-c", "copy", "-movflags", "+faststart", str(args.out)], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"joining the clips failed: {r.stderr.strip()[-500:]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    index_path = args.out.with_suffix(".csv")
    with open(index_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(index[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(index)
    print(f"{len(made)} clips, {hms(duration(args.out))} long -> {args.out}\nindex -> {index_path}")


if __name__ == "__main__":
    main()
