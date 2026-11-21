"""Find the sections of each video where someone is standing, or doing another --action.

Samples frames with ffmpeg, runs a YOLO pose model (ONNX, CPU) on each frame, classifies
every detected person as standing, sitting or unknown from their hip/knee/ankle geometry,
and joins standing frames into time ranges.

--action picks what to look for (default standing):
    standing, sitting   the leg-geometry rules below
    jumping             the lowest foot and the hips rise and come back down; only the moments
                        with a whole body in view are sampled again, at 5 fps (actions.py)
    smiling, talking    each face, found from the pose model's eye and nose points, is measured
                        by two small face models over a short burst of frames (actions.py)
All actions share the same pose-model output, so a second action on the same videos does not
run the pose model again.

Standing before the interview starts (title sequence, teaser, ads; see intro.py) is listed
separately in intro.csv, and statues and posters are dropped into dropped.csv.

The pose model is the slow part, so its output is saved per video (poses/<video>.npz).
Changing a threshold below and running again with --reclassify redoes only the cheap part.

Usage (from the youtube-analyser folder):
    .venv/bin/python standing/detect_standing.py *.mp4 --out standing/output
    .venv/bin/python standing/detect_standing.py *.mp4 --out standing/output --action talking

Outputs in --out (for an action other than standing, the files below except poses/ and
faces/ are in --out/<action>/ instead, with the action's name in place of "standing"):
    poses/<video>.npz     every person found in every sampled frame (17 body points each)
                          (<video>.dense.npz: the extra samples for jumping)
    faces/<video>.npz     face measurements for smiling and talking
    frames/<video>.csv    one row per sampled frame
    segments/<video>.csv  standing sections of that video (its presence marks the video done)
    thumbs/*.jpg          best frame of each section, with the detected skeletons drawn
    segments.csv          all standing sections in the interviews of all videos in this folder
    intro.csv             standing sections in the intros
    dropped.csv           sections dropped as still figures
    interview_starts.csv  where each video's interview starts
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
sys.path.insert(0, str(HERE))
import intro  # noqa: E402  (standing/intro.py: where the interview starts)
import actions  # noqa: E402  (standing/actions.py: rules for jumping, smiling and talking)
DEFAULT_MODEL = HERE / "models" / "yolo11s-pose.onnx"
NAME_RE = re.compile(r"^(\d{8})_(.{11})$")

# COCO keypoint indices used by YOLO pose models.
L_SHO, R_SHO, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 5, 6, 11, 12, 13, 14, 15, 16
LEGS = ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK))
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]
COLORS = {"standing": (60, 200, 60), "sitting": (255, 150, 30), "unknown": (170, 170, 170)}  # BGR
ACTION_COLOR, OTHER_COLOR = (60, 200, 60), (255, 150, 30)  # for actions other than standing

INTERVIEW_FIELDS = ["video", "interview_start_s", "interview_start", "title_end_s", "how"]

# What each --action looks for. states: the labels counted in frames/<video>.csv, the action's
# own label first. stats: measurements reported per frame as best_<name> (largest in the frame).
# score: the measurement whose largest value picks a section's thumbnail, reported as
# best_<score>. min_frames / max_gap: the default section rules, in sampled frames.
# still: drop sections whose figures never move (statues, posters); jumping and talking are
# movement already.
ACTIONS = {
    "standing": {"states": ("standing", "sitting", "unknown"), "stats": ("drop", "knee"), "score": "drop",
                 "min_frames": 2, "max_gap": 2, "still": True},
    "sitting": {"states": ("sitting", "standing", "unknown"), "stats": ("drop", "knee"), "score": "conf",
                "min_frames": 2, "max_gap": 2, "still": True},
    # in JUMP_FPS samples: one airborne frame is a jump; jumps up to 1 s apart are one section
    "jumping": {"states": ("jumping", "on the ground", "unknown"), "stats": ("rise",), "score": "rise",
                "min_frames": 1, "max_gap": 4, "still": False},
    "smiling": {"states": ("smiling", "not smiling", "unknown"), "stats": ("happy", "eyes"), "score": "happy",
                "min_frames": 2, "max_gap": 2, "still": True},
    "talking": {"states": ("talking", "quiet", "unknown"), "stats": ("mouth_move", "eyes"), "score": "mouth_move",
                "min_frames": 2, "max_gap": 2, "still": False},
}
ROUND = {"knee": 0, "eyes": 1}  # decimals in the CSVs; everything else has 2


def frame_fields(action="standing"):
    a = ACTIONS[action]
    return ["t", "people", *a["states"], *[f"best_{m}" for m in a["stats"]]]


def segment_fields(action="standing"):
    return ["video", "start_s", "end_s", "start", "end", "duration_s", f"{action}_frames",
            f"max_people_{action}", f"best_{ACTIONS[action]['score']}", "still_diff", "youtube", "thumb"]


FRAME_FIELDS, SEGMENT_FIELDS = frame_fields(), segment_fields()  # standing's


def rounded(name, v):
    d = ROUND.get(name, 2)
    return round(v) if d == 0 else round(float(v), d)

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

# A guest seated square-on to the camera can match every leg rule above (in one test video a
# seated guest did, for 20 sections). The giveaway is the seated person
# next to them: at the same size in frame, someone standing has their shoulders well above the
# sitter's. So a standing vote is overruled when a sitting person in the same frame has a torso
# within SAME_SCALE of theirs and shoulders less than ABOVE_SITTER torso lengths lower.
SAME_SCALE = 0.35                      # torso lengths within 35% of each other
ABOVE_SITTER = 0.35                    # standing shoulders must be this far above a sitter's

# Statues, posters and artwork shaped like a standing person pass every rule above and come
# back every time their camera angle is shown. Unlike a person they never move: across all
# 248 test videos their pixels changed by at most 3.0 grey levels (of 255) between nearby samples,
# while real standing people, even a bystander standing still, changed by 4.4 or more.
# A section whose standing figures all stay below STILL_DIFF is dropped as a still figure.
STILL_DIFF = 3.5                       # mean absolute grey-level change inside the figure's box
STILL_PAIR_S = 3                       # compare standing samples up to this many seconds apart
STILL_CHUNK_S = 300                    # longer sections are read in pieces of this many seconds

# Standing in the intro (title sequence, teaser, ads) is reported separately; intro.py finds
# where the interview starts. A section that starts in the intro and ends at most this many
# seconds after the interview start is counted as intro: the start can fall in a crossfade.
INTRO_GRACE_S = 3


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
    yield from _decode(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps}",
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], fps, w, h)


def read_windows(path, windows, fps, w, h):
    """Yield (window number, seconds, BGR frame) at `fps` inside each (t0, t1) window."""
    for n, (t0, t1) in enumerate(windows):
        for t, img in _decode(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", str(path), "-t",
                               f"{t1 - t0:.3f}", "-an", "-vf", f"fps={fps}", "-f", "rawvideo",
                               "-pix_fmt", "bgr24", "-"], fps, w, h):
            yield n, round(t0 + t, 3), img


def _decode(cmd, fps, w, h):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
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


def infer(path, model, fps, conf, windows=None):
    """Run the pose model on every sampled frame. Returns the arrays saved in poses/*.npz.

    With windows (a list of (start, end) seconds), only frames inside them are sampled, and
    each frame's window number and the picture's shift from the frame before (the camera
    moving; phase correlation, with its response, which drops at a cut) are added.
    """
    w, h = probe(path)
    times, frame, confs, boxes, kps = [], [], [], [], []
    motion = {"window": [], "shift": [], "response": []}

    def images():
        if windows is None:
            for t, img in read_frames(path, fps, w, h):
                times.append(t)
                yield img
            return
        hann, prev = cv2.createHanningWindow((w, h), cv2.CV_32F), (None, None)
        for n, t, img in read_windows(path, windows, fps, w, h):
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            (dx, dy), r = cv2.phaseCorrelate(prev[1], grey, hann) if prev[0] == n else ((np.nan, np.nan), 0.0)
            prev = (n, grey)
            times.append(t)
            motion["window"].append(n)
            motion["shift"].append((dx, dy))
            motion["response"].append(r)
            yield img

    for i, (c, b, k) in enumerate(model.run(images(), conf) if model is not None else ()):
        frame += [i] * len(c)
        confs.append(c)
        boxes.append(b)
        kps.append(k)
    cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape, np.float32)
    out = {"t": np.array(times, np.float32), "frame": np.array(frame, np.int32),
           "conf": cat(confs, (0,)), "box": cat(boxes, (0, 4)), "kp": cat(kps, (0, 17, 3)),
           "size": np.array([w, h]), "fps": np.float32(fps)}
    if windows is not None:  # times like 7.4 s are kept exact (float32 would print 7.400000095)
        out.update(t=np.array(times, np.float64),
                   window=np.array(motion["window"], np.int32), response=np.array(motion["response"], np.float32),
                   shift=np.array(motion["shift"], np.float32).reshape(-1, 2),
                   windows=np.array(windows, np.float32).reshape(-1, 2))
    return out


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
    votes, why, drops, knees = [], [], [], []
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
            why.append("")
        elif drop < SIT_DROP or tilt > SIT_TILT or (knee is not None and knee < SIT_KNEE):
            votes.append("sitting")
            why.append("thigh" if drop < SIT_DROP or tilt > SIT_TILT else "knee")
        else:
            votes.append("unknown")
            why.append("")
    m = {"drop": max(drops) if drops else None, "knee": max(knees) if knees else None,
         "torso": torso, "sho_y": float(sho[1]), "votes": list(zip(votes, why))}
    if "sitting" in votes:
        return "sitting", m
    if "standing" in votes:
        return "standing", m
    return "unknown", m


def seated(m):
    """Stricter sitting, for --action sitting: no leg looks standing, and a thigh is level or
    sideways, or every leg seen is bent. The standing rules count a single bent knee as sitting,
    which also catches the back leg of someone walking (seen on a test clip)."""
    legs = m.get("votes", [])
    if any(v == "standing" for v, _ in legs):
        return False
    sitting = [w for v, w in legs if v == "sitting"]
    return "thigh" in sitting or len(legs) >= 2 and len(sitting) == len(legs)


def level_with(stander, sitter):
    """True when a sitter of about the same size has shoulders nearly as high as the stander's."""
    a, b = stander["torso"], sitter["torso"]
    if abs(a - b) > SAME_SCALE * max(a, b):
        return False
    return sitter["sho_y"] - stander["sho_y"] < ABOVE_SITTER * a  # image y grows downwards


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


def label_people(poses, action="standing", extra=None):
    """Everyone in every sampled frame, with a state for the action.

    Returns (timeline, people per frame): timeline is poses, except for jumping, where it is the
    dense samples (extra). For smiling and talking, extra is the face measurements.
    """
    if action == "jumping":
        return extra, actions.label_jumps(extra)
    h = int(poses["size"][1])
    by_frame = [[] for _ in poses["t"]]
    for j, f in enumerate(poses["frame"]):
        state, meas = classify(poses["kp"][j], h)
        by_frame[f].append({"j": j, "kp": poses["kp"][j], "box": poses["box"][j], "state": state,
                            "m": {**meas, "conf": float(poses["conf"][j])}})
    for people in by_frame:
        sitters = [p["m"] for p in people if p["state"] == "sitting"]
        for p in people:
            if p["state"] == "standing" and any(level_with(p["m"], s) for s in sitters):
                p["state"] = "sitting"
            elif action == "sitting" and p["state"] == "sitting" and not seated(p["m"]):
                p["state"] = "unknown"
    if action in ("smiling", "talking"):
        actions.label_faces(by_frame, poses, extra, action)
    return poses, by_frame


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
        if score > cur["best"]:  # compared with the rounded best, as the CSV shows it
            cur.update(best=rounded(a["score"], score), best_t=t, best_people=people)

    segments = [s for s in segments if s["n"] >= min_frames]
    for s in segments:
        if action != "standing":  # jumping's 0.2 s steps would otherwise show as 3.4000000000000004
            s.update(start_s=round(s["start_s"], 3), end_s=round(s["end_s"], 3))
        s.update({f"{action}_frames": s["n"], f"max_people_{action}": s["most"], f"best_{a['score']}": s["best"]},
                 start=hms(s["start_s"]), end=hms(s["end_s"]), duration_s=round(s["end_s"] - s["start_s"], 1))
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
    """Largest pixel change of the section's figures between nearby samples.

    None when no figure is seen twice within STILL_PAIR_S seconds (then it can't be judged).
    A section longer than STILL_CHUNK_S (sitting or smiling can last most of an episode) is
    read a piece at a time, and only until a piece shows the figure moving: the largest change
    found up to then is returned.
    """
    boxes = section["boxes"]
    if boxes[-1][0] - boxes[0][0] <= STILL_CHUNK_S:
        return _still_diff(read_range(path, boxes[0][0], boxes[-1][0], fps, w, h), boxes, w, h)
    best, i = None, 0
    while i < len(boxes):
        t0 = boxes[i][0]
        part = [b for b in boxes[i:] if b[0] <= t0 + STILL_CHUNK_S + STILL_PAIR_S]
        n = sum(b[0] < t0 + STILL_CHUNK_S for b in part)  # pairs start in this piece
        d = _still_diff(read_range(path, t0, part[-1][0], fps, w, h), part, w, h, starts=n)
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


def find_interview_start(path, bank):
    """intro.interview_start() as a CSV row; the whole video is the interview without a bank."""
    r = intro.interview_start(path, bank) if bank is not None else \
        {"interview_start_s": 0, "title_end_s": 0, "how": "intro kept"}
    return {"video": path.name, **r, "interview_start": hms(r["interview_start_s"])}


def action_input(path, poses, action, pose_model, face_models, conf, jump_fps=actions.JUMP_FPS):
    """What an action needs besides the poses: the dense samples for jumping, faces for smiling
    and talking, nothing for standing and sitting. pose_model / face_models are only used when
    needed, and may be functions that load them (a PoseModel is callable itself, so the type
    tells them apart)."""
    if action == "jumping":
        _, by_frame = label_people(poses)
        states = {p["j"]: p["state"] for people in by_frame for p in people}
        windows = actions.jump_windows(poses, [states[j] for j in range(len(poses["frame"]))])
        if not windows:  # nobody in full view: nothing to sample, so the model is not needed
            model = None
        else:
            model = pose_model if isinstance(pose_model, PoseModel) else pose_model()
        return infer(path, model, jump_fps, conf, windows)
    if action in ("smiling", "talking"):
        w, h = (int(x) for x in poses["size"])
        models = face_models if isinstance(face_models, actions.FaceModels) else face_models()
        faces = actions.analyse_faces(read_frames(path, 1 / actions.BURST_STEP, w, h), poses, models)
        faces["poses_n"] = np.int32(len(poses["frame"]))
        return faces
    return None


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


def analyse(path, poses, args, out_dir, bank=None, extra=None):
    """Find the sections in the saved poses; write the per-video CSVs and thumbnails."""
    action = getattr(args, "action", "standing")
    stem = path.stem
    w, h = (int(x) for x in poses["size"])
    fps = float((extra if action == "jumping" else poses)["fps"])  # the rate the sections were sampled at
    m = NAME_RE.match(stem)
    url = f"https://www.youtube.com/watch?v={m.group(2)}" if m else ""
    start = find_interview_start(path, bank)
    frames, segments, in_intro, dropped = detect(path, poses, action, args.min_frames, args.max_gap,
                                                 start["interview_start_s"], extra)
    for old in (out_dir / "thumbs").glob(f"{glob_escape(stem)}_*.jpg"):
        old.unlink()
    for s in segments + dropped + in_intro:
        s["thumb"] = f"{stem}_{int(s['start_s']):05d}.jpg"
        img = grab_frame(path, s["best_t"], fps, w, h)
        cv2.imwrite(str(out_dir / "thumbs" / s["thumb"]), draw(img, s["best_people"], action=action))
        s["youtube"] = f"{url}&t={int(s['start_s'])}s" if url else ""
    fields = segment_fields(action)
    write_csv(out_dir / "frames" / f"{stem}.csv", frame_fields(action), frames)
    write_csv(out_dir / "interview" / f"{stem}.csv", INTERVIEW_FIELDS, [start])
    write_csv(out_dir / "intro" / f"{stem}.csv", fields, in_intro)
    write_csv(out_dir / "dropped" / f"{stem}.csv", fields, dropped)
    write_csv(out_dir / "segments" / f"{stem}.csv", fields, segments)  # written last: marks done
    return len(frames), len(segments), len(dropped), len(in_intro), start["interview_start"]


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

    def cards(rows):
        return "\n".join(
            f'<figure><a href="thumbs/{html.escape(r["thumb"])}"><img src="thumbs/{html.escape(r["thumb"])}" loading="lazy"></a>'
            f'<figcaption>{html.escape(r["video"])}<br>{r["start"]} to {r["end"]} ({r["duration_s"]}s, '
            f'{r[f"{action}_frames"]} frames, pixel change {r["still_diff"] or "n/a"})'
            + (f' · <a href="{html.escape(r["youtube"])}">YouTube</a>' if r["youtube"] else "")
            + "</figcaption></figure>" for r in rows)

    (out_dir / "review.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>{label} sections</title>"
        "<style>body{font:14px system-ui;margin:16px;background:#fff;color:#111}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}"
        "figure{margin:0}img{width:100%;border-radius:4px}figcaption{margin-top:4px}</style>"
        f"<h1>{label} sections ({len(found['segments'])})</h1><main>{cards(found['segments'])}</main>"
        f"<h2>Dropped as still figures ({len(found['dropped'])})</h2>"
        f"<p>Statues, posters and other figures that never moved (pixel change under {STILL_DIFF}).</p>"
        f"<main>{cards(found['dropped'])}</main>"
        f"<h2>In the intro ({len(found['intro'])})</h2>"
        f"<p>{label} before the interview starts: title sequences, teasers and ads.</p>"
        f"<main>{cards(found['intro'])}</main>")
    return found["segments"], found["dropped"], found["intro"]


def save_npz(path, arrays):
    np.savez_compressed(path.with_suffix(".tmp.npz"), **arrays)
    path.with_suffix(".tmp.npz").replace(path)


def load_input(v, poses, args, models):
    """action_input() for one video, from its cache file in --out when still valid."""
    if args.action == "jumping":
        path = args.out / "poses" / f"{v.stem}.dense.npz"
        check = lambda d: float(d["fps"]) == args.jump_fps
    elif args.action in ("smiling", "talking"):
        path = args.out / "faces" / f"{v.stem}.npz"
        check = lambda d: (int(d["poses_n"]) == len(poses["frame"])
                           and float(d["burst_step"]) == np.float32(actions.BURST_STEP))
    else:
        return None, ""
    if path.exists() and not args.force:
        saved = dict(np.load(path))
        if check(saved):
            return saved, ""
    extra = action_input(v, poses, args.action, models["pose"], models["faces"], args.conf, args.jump_fps)
    save_npz(path, extra)
    if args.action == "jumping":
        seconds = sum(b - a for a, b in extra["windows"])
        return extra, f"; {seconds:.0f}s sampled again at {args.jump_fps:g} fps"
    return extra, f"; {len(extra['det'])} faces measured"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--action", choices=list(ACTIONS), default="standing", help="what to look for")
    ap.add_argument("--out", type=Path, default=HERE / "output",
                    help="output folder; actions other than standing write to a subfolder named after "
                         "the action, and all actions share its poses/")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--imgsz", type=int, default=640, help="longest side fed to the model")
    ap.add_argument("--fps", type=float, default=1.0, help="frames sampled per second of video")
    ap.add_argument("--jump-fps", type=float, default=actions.JUMP_FPS,
                    help="jumping: frames per second sampled where a whole body is in view")
    ap.add_argument("--conf", type=float, default=0.35, help="minimum person confidence")
    ap.add_argument("--min-frames", type=int, help="frames showing the action needed for a section "
                    "(default 2; 1 for jumping)")
    ap.add_argument("--max-gap", type=int, help="frames without it allowed inside a section "
                    "(default 2; 4 for jumping, whose frames are 0.2 s apart)")
    ap.add_argument("--runtime", choices=["auto", "openvino", "onnxruntime"], default="auto",
                    help="auto = OpenVINO if installed, else onnxruntime")
    ap.add_argument("--threads", type=int, default=0, help="inference threads (0 = automatic)")
    ap.add_argument("--reclassify", action="store_true",
                    help="redo sections for videos already done, from their saved poses")
    ap.add_argument("--force", action="store_true", help="rerun the pose model even if poses are saved")
    ap.add_argument("--intro-bank", type=Path, default=intro.DEFAULT_BANK,
                    help="title frames that mark the intro (see intro.py)")
    ap.add_argument("--keep-intro", action="store_true", help="treat the whole video as the interview")
    args = ap.parse_args()
    action = args.action
    if args.min_frames is None:
        args.min_frames = ACTIONS[action]["min_frames"]
    if args.max_gap is None:
        args.max_gap = ACTIONS[action]["max_gap"]

    out_dir = args.out if action == "standing" else args.out / action
    (args.out / "poses").mkdir(parents=True, exist_ok=True)
    if action in ("smiling", "talking"):
        (args.out / "faces").mkdir(parents=True, exist_ok=True)
    for sub in ("frames", "segments", "dropped", "intro", "interview", "thumbs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    bank = None if args.keep_intro else intro.load_bank(args.intro_bank)
    if bank is None and not args.keep_intro:
        print(f"no intro bank at {args.intro_bank}: the intro is not separated", file=sys.stderr)
    loaded = {}

    def model(name):  # load each model the first time it is needed
        if name not in loaded:
            loaded[name] = (PoseModel(args.model, args.imgsz, args.threads, args.runtime) if name == "pose"
                            else actions.FaceModels(args.runtime, args.threads))
        return loaded[name]

    models = {"pose": lambda: model("pose"), "faces": lambda: model("faces")}
    # yt-dlp leaves unmerged format files such as X.f278.webm next to X.webm; skip those.
    videos = [v for v in args.videos if not re.search(r"\.f\d+$", v.stem)]
    for i, v in enumerate(videos, 1):
        pose_file = args.out / "poses" / f"{v.stem}.npz"
        done = (out_dir / "segments" / f"{v.stem}.csv").exists()
        if done and not (args.reclassify or args.force):
            print(f"[{i}/{len(videos)}] {v.name}: done already, skipping", flush=True)
            continue
        started = time.perf_counter()
        try:
            if pose_file.exists() and not args.force:
                poses = dict(np.load(pose_file))
                how = "saved poses"
            else:
                poses = infer(v, model("pose"), args.fps, args.conf)
                save_npz(pose_file, poses)
                how = f"{1000 * (time.perf_counter() - started) / max(len(poses['t']), 1):.0f} ms/frame"
            extra, more = load_input(v, poses, args, models)
            n_frames, n_segments, n_dropped, n_intro, start = analyse(v, poses, args, out_dir, bank, extra)
        except Exception as e:  # keep going on a bad file; it stays unmarked and is retried next run
            print(f"[{i}/{len(videos)}] {v.name}: FAILED {e}", file=sys.stderr, flush=True)
            continue
        print(f"[{i}/{len(videos)}] {v.name}: interview from {start}, {n_frames} frames, "
              f"{n_segments} {action} sections"
              + (f" ({n_intro} more in the intro)" if n_intro else "")
              + (f" ({n_dropped} still figures dropped)" if n_dropped else "")
              + f", {time.perf_counter() - started:.0f}s ({how}{more})", flush=True)
    kept, dropped, in_intro = combine(out_dir, action)
    print(f"{len(kept)} {action} sections in total ({len(in_intro)} more in intros, {len(dropped)} still "
          f"figures dropped) -> {out_dir / 'segments.csv'}, {out_dir / 'review.html'}")


if __name__ == "__main__":
    main()
