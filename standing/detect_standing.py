"""Find the sections of each video where someone is standing.

Samples frames with ffmpeg, runs a YOLO pose model (ONNX, CPU) on each frame, classifies
every detected person as standing, sitting or unknown from their hip/knee/ankle geometry,
and joins standing frames into time ranges.

The pose model is the slow part, so its output is saved per video (poses/<video>.npz).
Changing a threshold below and running again with --reclassify redoes only the cheap part.

Usage (from the youtube-analyser folder):
    .venv/bin/python standing/detect_standing.py 2020*.mp4 --out standing/output

Outputs in --out:
    poses/<video>.npz     every person found in every sampled frame (17 body points each)
    frames/<video>.csv    one row per sampled frame
    segments/<video>.csv  standing sections of that video (its presence marks the video done)
    thumbs/*.jpg          best frame of each section, with the detected skeletons drawn
    segments.csv          all sections of all videos processed into this folder
    review.html           thumbnails with YouTube links at each section's start time
"""
import argparse
import csv
import html
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

try:
    import openvino as ov
except ImportError:  # onnxruntime alone still works, just slower
    ov = None

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "yolo11s-pose.onnx"
NAME_RE = re.compile(r"^(\d{8})_(.{11})$")

# COCO keypoint indices used by YOLO pose models.
L_SHO, R_SHO, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 5, 6, 11, 12, 13, 14, 15, 16
LEGS = ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK))
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]
COLORS = {"standing": (60, 200, 60), "sitting": (255, 150, 30), "unknown": (170, 170, 170)}  # BGR

FRAME_FIELDS = ["t", "people", "standing", "sitting", "unknown", "best_drop", "best_knee"]
SEGMENT_FIELDS = ["video", "start_s", "end_s", "start", "end", "duration_s", "standing_frames",
                  "max_people_standing", "best_drop", "still_diff", "youtube", "thumb"]

# Thresholds for one leg, in units of the person's torso length (mid-shoulder to mid-hip).
# A standing thigh hangs below the hip by roughly a torso length; a seated thigh points
# sideways or at the camera, so the knee sits level with the hip or only a little below it.
# Someone seated square-on to the camera is the hard case: their thigh points at the lens,
# so in 2D the knee still lands below the hip. What gives them away is the bent knee
# (78-121 degrees in the seated examples checked, 160-180 standing) and a thigh that looks
# much shorter than the shin. So standing needs an ankle; without one a leg can only vote
# sitting or unknown.
STAND_DROP, SIT_DROP = 0.55, 0.35      # vertical hip-to-knee drop / torso length
STAND_TILT, SIT_TILT = 35, 55          # thigh angle away from vertical, degrees
STAND_KNEE, SIT_KNEE = 150, 125        # hip-knee-ankle angle, degrees (180 = straight leg)
MAX_SHIN_THIGH = 1.35                  # shin / thigh length; higher means a foreshortened thigh
VIS = 0.5                              # keypoint confidence needed for shoulders, hips, knees
ANKLE_VIS = 0.3                        # ankles of walking people at 144p often score 0.3-0.5

# A guest seated square-on to the camera can match every leg rule above (in one 2019 episode
# a seated guest in a patterned onesie did, for 20 sections). The giveaway is the seated person
# next to them: at the same size in frame, someone standing has their shoulders well above the
# sitter's. So a standing vote is overruled when a sitting person in the same frame has a torso
# within SAME_SCALE of theirs and shoulders less than ABOVE_SITTER torso lengths lower.
SAME_SCALE = 0.35                      # torso lengths within 35% of each other
ABOVE_SITTER = 0.35                    # standing shoulders must be this far above a sitter's

# Statues, posters and artwork shaped like a standing person pass every rule above and come
# back every time their camera angle is shown. Unlike a person they never move: across all
# 248 videos their pixels changed by at most 3.0 grey levels (of 255) between nearby samples,
# while real standing people, even an expo bystander standing still, changed by 4.4 or more.
# A section whose standing figures all stay below STILL_DIFF is dropped as a still figure.
STILL_DIFF = 3.5                       # mean absolute grey-level change inside the figure's box
STILL_PAIR_S = 3                       # compare standing samples up to this many seconds apart


def hms(t):
    t = int(round(t))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}"


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "json", str(path)], capture_output=True, text=True)
    streams = json.loads(r.stdout or "{}").get("streams") if r.returncode == 0 else None
    if not streams:
        raise ValueError(f"{Path(path).name}: no readable video stream")
    return streams[0]["width"], streams[0]["height"]


def read_frames(path, fps, w, h):
    """Yield (seconds, BGR frame) at `fps` frames per second, decoded by ffmpeg."""
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps}",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    size, k = w * h * 3, 0
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield k / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def grab_frame(path, t, fps, w, h):
    """Return the frame read_frames() produced at time t.

    Seeking straight to t can land one source frame away from the frame the fps filter
    picked, which at a camera cut is a different shot. So seek a whole number of sample
    intervals earlier, where the fps filter's grid lines up with the original one, and
    run the same filter forward to t.
    """
    k = min(round(t * fps), 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t - k / fps:.6f}", "-i", str(path),
                          "-vf", f"fps={fps}", "-frames:v", str(k + 1),
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    size = w * h * 3
    if len(raw) < size:
        return np.zeros((h, w, 3), np.uint8)
    return np.frombuffer(raw[-size:] if len(raw) % size == 0 else raw[:size], np.uint8).reshape(h, w, 3)


class PoseModel:
    """YOLO pose model on the CPU, through OpenVINO when it is installed, else onnxruntime.

    Both give the same detections (raw outputs agree to within 0.006). On this Intel Mac
    OpenVINO is about 1.4x faster, mostly from keeping several frames in flight at once.
    """

    def __init__(self, path, imgsz, threads=0, runtime="auto"):
        self.path, self.imgsz = str(path), imgsz
        self.runtime = runtime if runtime != "auto" else "openvino" if ov else "onnxruntime"
        if self.runtime == "openvino":
            if ov is None:
                raise SystemExit("openvino is not installed; use --runtime onnxruntime")
            self.core = ov.Core()
            self.config = {"PERFORMANCE_HINT": "THROUGHPUT"}
            if threads:
                self.config["INFERENCE_NUM_THREADS"] = threads
            self.compiled = {}  # one per input shape: landscape videos and vertical Shorts differ
        else:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            if threads:
                so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name

    def _compiled(self, shape):
        if shape not in self.compiled:
            m = self.core.read_model(self.path)
            m.reshape(list(shape))
            self.compiled[shape] = self.core.compile_model(m, "CPU", self.config)
        return self.compiled[shape]

    def _prep(self, img):
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)
        nw, nh = round(w * r), round(h * r)
        pw, ph = -(-nw // 32) * 32, -(-nh // 32) * 32  # letterbox to a multiple of 32
        dx, dy = (pw - nw) // 2, (ph - nh) // 2
        canvas = np.full((ph, pw, 3), 114, np.uint8)
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], np.float32) / 255
        return x, (r, dx, dy)

    @staticmethod
    def _post(raw, meta, conf, iou):
        """Return (conf (n,), box (n, 4) x y w h, kp (n, 17, 3) x y visibility) in image pixels."""
        r, dx, dy = meta
        out = raw[0].T  # (anchors, 4 box + 1 conf + 17*3)
        out = out[out[:, 4] > conf]
        if not len(out):
            return np.zeros(0, np.float32), np.zeros((0, 4), np.float32), np.zeros((0, 17, 3), np.float32)
        xywh = np.c_[out[:, 0] - out[:, 2] / 2, out[:, 1] - out[:, 3] / 2, out[:, 2], out[:, 3]]
        keep = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), out[:, 4].tolist(), conf, iou)).flatten()
        out, xywh = out[keep], xywh[keep]
        kp = out[:, 5:].reshape(-1, 17, 3).copy()
        kp[:, :, 0] = (kp[:, :, 0] - dx) / r
        kp[:, :, 1] = (kp[:, :, 1] - dy) / r
        box = np.c_[(xywh[:, 0] - dx) / r, (xywh[:, 1] - dy) / r, xywh[:, 2] / r, xywh[:, 3] / r]
        return out[:, 4].copy(), box.astype(np.float32), kp

    def __call__(self, img, conf, iou=0.45):
        """Detect people in one image (see _post for the result)."""
        x, meta = self._prep(img)
        if self.runtime == "openvino":
            raw = self._compiled(x.shape)(x)[0]
        else:
            raw = self.sess.run(None, {self.inp: x})[0]
        return self._post(raw, meta, conf, iou)

    def run(self, images, conf, iou=0.45):
        """Yield detections for each image of one video, in order."""
        if self.runtime != "openvino":
            for img in images:
                yield self(img, conf, iou)
            return
        done, nxt, count, queue = {}, 0, 0, None

        def finished(request, userdata):
            i, meta = userdata
            done[i] = self._post(request.get_output_tensor(0).data, meta, conf, iou)

        for i, img in enumerate(images):
            x, meta = self._prep(img)
            if queue is None:
                queue = ov.AsyncInferQueue(self._compiled(x.shape))
                queue.set_callback(finished)
            queue.start_async({0: x}, (i, meta))  # waits while every request is busy
            count = i + 1
            while nxt in done:
                yield done.pop(nxt)
                nxt += 1
        if queue is not None:
            queue.wait_all()
        while nxt in done:
            yield done.pop(nxt)
            nxt += 1
        if nxt != count:
            raise RuntimeError(f"pose model returned {nxt} of {count} frames")


def infer(path, model, fps, conf):
    """Run the pose model on every sampled frame. Returns the arrays saved in poses/*.npz."""
    w, h = probe(path)
    times, frame, confs, boxes, kps = [], [], [], [], []

    def images():
        for t, img in read_frames(path, fps, w, h):
            times.append(t)
            yield img

    for i, (c, b, k) in enumerate(model.run(images(), conf)):
        frame += [i] * len(c)
        confs.append(c)
        boxes.append(b)
        kps.append(k)
    cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape, np.float32)
    return {"t": np.array(times, np.float32), "frame": np.array(frame, np.int32),
            "conf": cat(confs, (0,)), "box": cat(boxes, (0, 4)), "kp": cat(kps, (0, 17, 3)),
            "size": np.array([w, h]), "fps": np.float32(fps)}


def angle(a, b, c):
    """Angle at b between b->a and b->c, in degrees."""
    u, v = a - b, c - b
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def classify(kp, img_h):
    """Return (state, measurements) for one person: standing, sitting or unknown."""
    inside = kp[:, 1] < img_h - 1  # keypoints guessed below the bottom edge are not real
    ok = (kp[:, 2] >= VIS) & inside
    ok_ank = (kp[:, 2] >= ANKLE_VIS) & inside
    pt = kp[:, :2]

    def mid(a, b):
        return (pt[a] + pt[b]) / 2 if ok[a] and ok[b] else pt[a] if ok[a] else pt[b] if ok[b] else None

    sho, hip = mid(L_SHO, R_SHO), mid(L_HIP, R_HIP)
    if sho is None or hip is None:
        return "unknown", {}
    torso = float(np.linalg.norm(hip - sho))
    if torso < 3:
        return "unknown", {}
    votes, drops, knees = [], [], []
    for h, k, a in LEGS:
        if not (ok[h] and ok[k]):
            continue
        thigh = pt[k] - pt[h]
        drop = thigh[1] / torso
        tilt = math.degrees(math.atan2(abs(thigh[0]), max(thigh[1], 1e-9)))
        knee = angle(pt[h], pt[k], pt[a]) if ok_ank[a] else None
        shin_thigh = (np.linalg.norm(pt[a] - pt[k]) / max(np.linalg.norm(thigh), 1e-9)
                      if ok_ank[a] else None)
        drops.append(drop)
        if knee is not None:
            knees.append(knee)
        if (knee is not None and knee >= STAND_KNEE and drop >= STAND_DROP
                and tilt <= STAND_TILT and shin_thigh <= MAX_SHIN_THIGH):
            votes.append("standing")
        elif drop < SIT_DROP or tilt > SIT_TILT or (knee is not None and knee < SIT_KNEE):
            votes.append("sitting")
        else:
            votes.append("unknown")
    m = {"drop": max(drops) if drops else None, "knee": max(knees) if knees else None,
         "torso": torso, "sho_y": float(sho[1])}
    if "sitting" in votes:
        return "sitting", m
    if "standing" in votes:
        return "standing", m
    return "unknown", m


def level_with(stander, sitter):
    """True when a sitter of about the same size has shoulders nearly as high as the stander's."""
    a, b = stander["torso"], sitter["torso"]
    if abs(a - b) > SAME_SCALE * max(a, b):
        return False
    return sitter["sho_y"] - stander["sho_y"] < ABOVE_SITTER * a  # image y grows downwards


def draw(img, people, scale=3):
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for p in people:
        color = COLORS[p["state"]]
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


def find_sections(poses, name, min_frames, max_gap):
    """Classify everyone in the poses and join standing frames into sections.

    Returns (frames, sections): one row per sampled frame, and every section with at least
    min_frames standing frames. Sections keep best_t / best_people for drawing a thumbnail.
    """
    h = int(poses["size"][1])
    fps = float(poses["fps"])
    by_frame = [[] for _ in poses["t"]]
    for j, f in enumerate(poses["frame"]):
        state, meas = classify(poses["kp"][j], h)
        by_frame[f].append({"kp": poses["kp"][j], "box": poses["box"][j], "state": state, "m": meas})
    for people in by_frame:
        sitters = [p["m"] for p in people if p["state"] == "sitting"]
        for p in people:
            if p["state"] == "standing" and any(level_with(p["m"], s) for s in sitters):
                p["state"] = "sitting"

    frames, segments, cur = [], [], None
    for t, people in zip(poses["t"].tolist(), by_frame):
        states = [p["state"] for p in people]
        standing = [p for p in people if p["state"] == "standing"]
        drops = [p["m"]["drop"] for p in people if p["m"].get("drop") is not None]
        knees = [p["m"]["knee"] for p in people if p["m"].get("knee") is not None]
        frames.append({"t": t, "people": len(people), "standing": len(standing),
                       "sitting": states.count("sitting"), "unknown": states.count("unknown"),
                       "best_drop": round(float(max(drops)), 2) if drops else "",
                       "best_knee": round(max(knees)) if knees else ""})
        if not standing:
            continue
        score = float(max(p["m"]["drop"] for p in standing))
        if cur and t - cur["last_t"] <= (max_gap + 1) / fps:
            cur.update(end_s=t + 1 / fps, last_t=t)
            cur["standing_frames"] += 1
            cur["max_people_standing"] = max(cur["max_people_standing"], len(standing))
        else:
            cur = {"video": name, "start_s": t, "end_s": t + 1 / fps, "last_t": t,
                   "standing_frames": 1, "max_people_standing": len(standing), "best_drop": -1,
                   "boxes": []}
            segments.append(cur)
        cur["boxes"].append((t, [p["box"] for p in standing]))
        if score > cur["best_drop"]:
            cur.update(best_drop=round(score, 2), best_t=t, best_people=people)

    segments = [s for s in segments if s["standing_frames"] >= min_frames]
    for s in segments:
        s.update(start=hms(s["start_s"]), end=hms(s["end_s"]), duration_s=round(s["end_s"] - s["start_s"], 1))
    return frames, segments


def read_range(path, t0, t1, fps, w, h):
    """Grey frames that read_frames() produces between t0 and t1 (inclusive), by time."""
    k = min(round(t0 * fps), 2)  # see grab_frame for why it starts k samples early
    n = round((t1 - t0) * fps) + 1
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0 - k / fps:.6f}", "-i", str(path),
                          "-vf", f"fps={fps}", "-frames:v", str(k + n), "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True).stdout
    frames = np.frombuffer(raw[:len(raw) // (w * h) * w * h], np.uint8).reshape(-1, h, w)[k:]
    return {round(t0 + i / fps, 3): f.astype(np.int16) for i, f in enumerate(frames)}


def overlap(a, b):
    """Intersection over union of two x, y, w, h boxes."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / (a[2] * a[3] + b[2] * b[3] - ix * iy + 1e-9)


def still_diff(path, section, fps, w, h):
    """Largest pixel change of the section's standing figures between nearby samples.

    None when no figure is seen twice within STILL_PAIR_S seconds (then it can't be judged).
    """
    frames = read_range(path, section["boxes"][0][0], section["boxes"][-1][0], fps, w, h)
    best = None
    for i, (ta, boxes_a) in enumerate(section["boxes"]):
        for tb, boxes_b in section["boxes"][i + 1:]:
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
                        d = float(np.abs(fa[y0:y1, x0:x1] - fb[y0:y1, x0:x1]).mean())
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


def analyse(path, poses, args, out_dir):
    """Find the sections in the saved poses; write frames/ and segments/ CSVs and thumbnails."""
    stem = path.stem
    w, h = (int(x) for x in poses["size"])
    fps = float(poses["fps"])
    m = NAME_RE.match(stem)
    url = f"https://www.youtube.com/watch?v={m.group(2)}" if m else ""
    frames, segments = find_sections(poses, path.name, args.min_frames, args.max_gap)
    segments, dropped = drop_still_figures(path, segments, fps, w, h)
    for old in (out_dir / "thumbs").glob(f"{glob_escape(stem)}_*.jpg"):
        old.unlink()
    for s in segments + dropped:
        s["thumb"] = f"{stem}_{int(s['start_s']):05d}.jpg"
        img = grab_frame(path, s["best_t"], fps, w, h)
        cv2.imwrite(str(out_dir / "thumbs" / s["thumb"]), draw(img, s["best_people"]))
        s["youtube"] = f"{url}&t={int(s['start_s'])}s" if url else ""
    write_csv(out_dir / "frames" / f"{stem}.csv", FRAME_FIELDS, frames)
    write_csv(out_dir / "dropped" / f"{stem}.csv", SEGMENT_FIELDS, dropped)
    write_csv(out_dir / "segments" / f"{stem}.csv", SEGMENT_FIELDS, segments)  # written last: marks done
    return len(frames), len(segments), len(dropped)


def glob_escape(s):
    return re.sub(r"([*?\[])", r"[\1]", s)


def write_csv(path, fields, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def combine(out_dir):
    """Write segments.csv, dropped.csv and review.html for every video processed into out_dir."""
    found = {}
    for kind in ("segments", "dropped"):
        found[kind] = []
        for p in sorted((out_dir / kind).glob("*.csv")):
            found[kind] += list(csv.DictReader(open(p)))
        write_csv(out_dir / f"{kind}.csv", SEGMENT_FIELDS, found[kind])

    def cards(rows):
        return "\n".join(
            f'<figure><a href="thumbs/{html.escape(r["thumb"])}"><img src="thumbs/{html.escape(r["thumb"])}" loading="lazy"></a>'
            f'<figcaption>{html.escape(r["video"])}<br>{r["start"]} to {r["end"]} ({r["duration_s"]}s, '
            f'{r["standing_frames"]} frames, pixel change {r["still_diff"] or "n/a"})'
            + (f' · <a href="{html.escape(r["youtube"])}">YouTube</a>' if r["youtube"] else "")
            + "</figcaption></figure>" for r in rows)

    (out_dir / "review.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Standing sections</title>"
        "<style>body{font:14px system-ui;margin:16px;background:#fff;color:#111}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}"
        "figure{margin:0}img{width:100%;border-radius:4px}figcaption{margin-top:4px}</style>"
        f"<h1>Standing sections ({len(found['segments'])})</h1><main>{cards(found['segments'])}</main>"
        f"<h2>Dropped as still figures ({len(found['dropped'])})</h2>"
        f"<p>Statues, posters and other figures that never moved (pixel change under {STILL_DIFF}).</p>"
        f"<main>{cards(found['dropped'])}</main>")
    return found["segments"], found["dropped"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "output")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--imgsz", type=int, default=640, help="longest side fed to the model")
    ap.add_argument("--fps", type=float, default=1.0, help="frames sampled per second of video")
    ap.add_argument("--conf", type=float, default=0.35, help="minimum person confidence")
    ap.add_argument("--min-frames", type=int, default=2, help="standing frames needed for a section")
    ap.add_argument("--max-gap", type=int, default=2, help="non-standing frames allowed inside a section")
    ap.add_argument("--runtime", choices=["auto", "openvino", "onnxruntime"], default="auto",
                    help="auto = OpenVINO if installed, else onnxruntime")
    ap.add_argument("--threads", type=int, default=0, help="inference threads (0 = automatic)")
    ap.add_argument("--reclassify", action="store_true",
                    help="redo sections for videos already done, from their saved poses")
    ap.add_argument("--force", action="store_true", help="rerun the pose model even if poses are saved")
    args = ap.parse_args()

    for sub in ("poses", "frames", "segments", "dropped", "thumbs"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    model = None
    # yt-dlp leaves unmerged format files such as X.f278.webm next to X.webm; skip those.
    videos = [v for v in args.videos if not re.search(r"\.f\d+$", v.stem)]
    for i, v in enumerate(videos, 1):
        pose_file = args.out / "poses" / f"{v.stem}.npz"
        done = (args.out / "segments" / f"{v.stem}.csv").exists()
        if done and not (args.reclassify or args.force):
            print(f"[{i}/{len(videos)}] {v.name}: done already, skipping", flush=True)
            continue
        started = time.perf_counter()
        try:
            if pose_file.exists() and not args.force:
                poses = dict(np.load(pose_file))
                how = "saved poses"
            else:
                model = model or PoseModel(args.model, args.imgsz, args.threads, args.runtime)
                poses = infer(v, model, args.fps, args.conf)
                np.savez_compressed(pose_file.with_suffix(".tmp.npz"), **poses)
                pose_file.with_suffix(".tmp.npz").replace(pose_file)
                how = f"{1000 * (time.perf_counter() - started) / max(len(poses['t']), 1):.0f} ms/frame"
            n_frames, n_segments, n_dropped = analyse(v, poses, args, args.out)
        except Exception as e:  # keep going on a bad file; it stays unmarked and is retried next run
            print(f"[{i}/{len(videos)}] {v.name}: FAILED {e}", file=sys.stderr, flush=True)
            continue
        print(f"[{i}/{len(videos)}] {v.name}: {n_frames} frames, {n_segments} standing sections"
              + (f" ({n_dropped} still figures dropped)" if n_dropped else "")
              + f", {time.perf_counter() - started:.0f}s ({how})", flush=True)
    kept, dropped = combine(args.out)
    print(f"{len(kept)} standing sections in total ({len(dropped)} still figures dropped) -> "
          f"{args.out / 'segments.csv'}, {args.out / 'review.html'}")


if __name__ == "__main__":
    main()
