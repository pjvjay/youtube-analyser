"""Find the sections of each video where someone is standing, or doing another --action.

Samples frames with ffmpeg, runs a YOLO pose model (ONNX, CPU) on each frame, classifies
every detected person as standing, sitting or unknown from their hip/knee/ankle geometry,
and joins standing frames into time ranges.

--action picks what to look for (default standing):
    standing, sitting   the leg-geometry rules below; --motion still (or moving) also uses how
                        far people travel between samples (sitting: still by default)
    walking             upright people whose body travels 0.6-4 torso lengths a second between
                        the 1-fps samples, with the camera's pan, tilt and zoom taken out (actions.py)
    running             4 torso lengths a second or more, in 5-fps samples taken again around
                        every moment a whole body is in view (actions.py)
    jumping             the lowest foot and the hips rise and come back down, in the same
                        5-fps samples (actions.py)
    smiling             each face, found from the pose model's eye and nose points, is measured
                        by a small expression model over a short burst of frames (actions.py)
    talking, silent     the faces' mouth movement together with the sound, where it is clear:
                        whether anyone speaks and which voice (audio.py); each face keeps its
                        own voice; also who spoke when (speech.csv)
All actions share the same pose-model output, so a second action on the same videos does not
run the pose model again.

Standing before the interview starts (title sequence, teaser, ads; see intro.py) is listed
separately in intro.csv, and statues and posters are dropped into dropped.csv.

The pose model is the slow part, so its output is saved per video (poses/<video>.npz).
Changing a threshold below and running again with --reclassify redoes only the cheap part.

Usage (from the youtube-analyser folder):
    .venv/bin/python standing/detect_standing.py *.mp4 --out standing/output
    .venv/bin/python standing/detect_standing.py *.mp4 --out standing/output --action talking

Outputs in --out (for an action other than standing, the files below except poses/, faces/,
audio/ and camera/ are in --out/<action>/ instead, with the action's name in place of "standing"):
    poses/<video>.npz     every person found in every sampled frame (17 body points each)
                          (<video>.dense.npz: the extra samples for jumping)
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
import actions  # noqa: E402  (standing/actions.py: rules for jumping, smiling, talking, walking)
import audio  # noqa: E402  (standing/audio.py: speech and voices in the sound)
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
    "talking": {"states": ("talking", "silent", "unknown"), "stats": ("mouth_move", "eyes"), "score": "mouth_move",
                "min_frames": 2, "max_gap": 2, "still": False, "notes": ("voice", "by")},
    # a visible face that is not talking: listening, or nobody speaking
    "silent": {"states": ("silent", "talking", "unknown"), "stats": ("mouth_move", "eyes"), "score": "eyes",
               "min_frames": 2, "max_gap": 1, "still": True, "notes": ("heard", "by")},  # talking ends it
    # moving across the ground upright, from how far the body travels between 1-fps samples
    "walking": {"states": ("walking", "running", "still", "unknown"), "stats": ("speed",), "score": "speed",
                "min_frames": 2, "max_gap": 1, "still": False},
    # in jumping's 5-fps samples: 0.6 s of running, with gaps up to 0.6 s
    "running": {"states": ("running", "walking", "still", "unknown"), "stats": ("speed",), "score": "speed",
                "min_frames": 3, "max_gap": 2, "still": False},
}
ROUND = {"knee": 0, "eyes": 1}  # decimals in the CSVs; everything else has 2
# notes: what the people showing the action say about it, listed per section (voices heard
# while talking or while silent, and whether it was judged from the sound and the face or the
# face alone), in these columns
NOTE_COLUMNS = {"voice": "voices", "heard": "voices_heard", "by": "judged_by"}
FACE_ACTIONS = ("smiling", "talking", "silent")
SOUND_ACTIONS = ("talking", "silent")
MOTION = {"standing": "any", "sitting": "still", "walking": "moving"}  # default --motion per action
DENSE_ACTIONS = ("jumping", "running")  # judged in the 5-fps samples around whole bodies


def frame_fields(action="standing"):
    a = ACTIONS[action]
    return ["t", "people", *a["states"], *[f"best_{m}" for m in a["stats"]]]


def segment_fields(action="standing"):
    notes = [NOTE_COLUMNS[n] for n in ACTIONS[action].get("notes", ())]
    return ["video", "start_s", "end_s", "start", "end", "duration_s", f"{action}_frames",
            f"max_people_{action}", f"best_{ACTIONS[action]['score']}", *notes, "still_diff", "youtube", "thumb"]


SPEECH_FIELDS = ["video", "start_s", "end_s", "start", "end", "duration_s", "voice", "on_screen", "sound_trusted",
                 "youtube"]
VOICE_FIELDS = ["video", "voice", "seconds", "turns", "on_screen_seconds", "sound_trusted", "thumb"]


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
    each frame's window number, the camera's movement from the frame before (shift, scale, and
    response, which drops at a cut) and each person's look are added (actions.camera_motion).
    """
    w, h = probe(path)
    times, frame, confs, boxes, kps, window = [], [], [], [], [], []

    def images():
        if windows is None:
            for t, img in read_frames(path, fps, w, h):
                times.append(t)
                yield img
            return
        for n, t, img in read_windows(path, windows, fps, w, h):
            times.append(t)
            window.append(n)
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
        out.update(t=np.array(times, np.float64), window=np.array(window, np.int32),
                   windows=np.array(windows, np.float32).reshape(-1, 2))
        # the camera's movement between the samples, measured around the people, and their looks
        out.update(actions.camera_motion(read_windows(path, windows, fps, w, h), out),
                   camera_version=np.int32(actions.CAMERA_VERSION))
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
    dense samples. extra is what action_input() made: jumping's dense samples; smiling's face
    measurements; for talking and silent {"faces", "audio"} (audio None without sound); for
    standing, sitting and walking {"camera", "motion"}, or None to ignore movement (standing's
    default, which keeps its results as they were).
    """
    h = int(poses["size"][1])
    if action == "jumping":
        return extra, actions.label_jumps(extra)
    if action == "running":
        return extra, actions.label_runs(extra, seated=lambda kp: is_seated(kp, h))
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
    if action == "smiling":
        actions.label_faces(by_frame, poses, extra, action)
    elif action in SOUND_ACTIONS:
        sound, heard = extra.get("audio"), None
        if sound is not None:
            voices = audio.frame_voices(sound)
            burst, fps = actions.BURST_N * actions.BURST_STEP, float(poses["fps"])
            heard = [audio.heard(sound, actions.picture_time(t, fps), actions.picture_time(t, fps) + burst, voices)
                     for t in poses["t"].tolist()]
        actions.label_faces(by_frame, poses, extra["faces"], action, heard, extra.get("camera"))
    elif extra is not None:
        label_movement(by_frame, poses, action, extra["camera"], extra["motion"])
    return poses, by_frame


def is_seated(kp, img_h):
    """A person the stricter sitting rules call seated (see seated())."""
    state, m = classify(kp, img_h)
    return state == "sitting" and seated(m)


def label_movement(by_frame, poses, action, camera, motion):
    """Walking: upright people whose body travels 0.6-4 torso lengths a second between samples
    (actions.label_motion) are walking; faster is running (the running action judges that in the
    5-fps samples); slower is still. Standing and sitting with --motion still (or moving): people
    whose movement says otherwise are relabelled walking or running (or still); people who could
    not be followed keep their label."""
    speed = actions.label_motion(poses, camera)
    for people in by_frame:
        for p in people:
            state = actions.motion_state(float(speed[p["j"]]))
            if state is not None:
                p["m"]["speed"] = float(speed[p["j"]])
            if action == "walking":
                upright = not (p["state"] == "sitting" and seated(p["m"]))
                p["state"] = "unknown" if state is None else state if upright else "still"
            elif p["state"] == action and state is not None and motion != "any" \
                    and (state != "still") != (motion == "moving"):
                p["state"] = state


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


def _model(models, name):
    """models[name], loading it first if it is a function that loads it."""
    m = models[name]
    return m() if callable(m) and not isinstance(m, PoseModel) else m


def dense_input(path, poses, models, conf, jump_fps=actions.JUMP_FPS):
    """Jumping: the pose model again at jump_fps around every moment a whole body is in view."""
    _, by_frame = label_people(poses)
    states = {p["j"]: p["state"] for people in by_frame for p in people}
    windows = actions.jump_windows(poses, [states[j] for j in range(len(poses["frame"]))])
    model = _model(models, "pose") if windows else None  # nobody in full view: nothing to sample
    return infer(path, model, jump_fps, conf, windows)


def face_input(path, poses, models):
    """Smiling, talking, silent: every usable face measured over its burst (actions.analyse_faces)."""
    w, h = (int(x) for x in poses["size"])
    faces = actions.analyse_faces(read_frames(path, 1 / actions.BURST_STEP, w, h), poses, _model(models, "faces"))
    faces["poses_n"] = np.int32(len(poses["frame"]))
    return faces


def sound_input(path, models):
    """Talking, silent: speech and voices in the sound (audio.analyse); has_sound False without sound."""
    wav = audio.load(path)
    if wav is None:
        return {"has_sound": np.bool_(False)}
    return {**audio.analyse(wav, _model(models, "voices")), "has_sound": np.bool_(True)}


def camera_input(path, poses):
    """Walking, and standing or sitting still: the camera's movement between the 1-fps samples."""
    w, h = (int(x) for x in poses["size"])
    steps = actions.CAMERA_STEPS
    camera = actions.camera_motion(read_frames(path, float(poses["fps"]) * steps, w, h), poses, steps)
    camera["poses_t"], camera["steps"] = np.int32(len(poses["t"])), np.int32(steps)
    camera["camera_version"] = np.int32(actions.CAMERA_VERSION)
    return camera


def action_input(path, poses, action, models, conf, jump_fps=actions.JUMP_FPS, motion=None, use_sound=True):
    """What an action needs besides the poses (see label_people), computed here without caching.

    models: {"pose", "faces", "voices"}, each a model or a function that loads it; only the ones
    the action needs are used. motion: "any", "still" or "moving" (default MOTION[action]).
    """
    if action in DENSE_ACTIONS:
        return dense_input(path, poses, models, conf, jump_fps)
    if action == "smiling":
        return face_input(path, poses, models)
    if action in SOUND_ACTIONS:
        sound = sound_input(path, models) if use_sound else {"has_sound": np.bool_(False)}
        return {"faces": face_input(path, poses, models), "audio": sound if sound["has_sound"] else None,
                "camera": camera_input(path, poses)}
    motion = motion or MOTION.get(action, "any")
    if motion == "any" and action != "walking":
        return None
    return {"camera": camera_input(path, poses), "motion": motion}


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
    fps = float((extra if action in DENSE_ACTIONS else poses)["fps"])  # the rate the sections were sampled at
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
    if action in SOUND_ACTIONS:
        speech, voices = speech_timeline(path, poses, action, extra, out_dir / "thumbs", url)
        write_csv(out_dir / "speech" / f"{stem}.csv", SPEECH_FIELDS, speech)
        write_csv(out_dir / "voices" / f"{stem}.csv", VOICE_FIELDS, voices)
    write_csv(out_dir / "frames" / f"{stem}.csv", frame_fields(action), frames)
    write_csv(out_dir / "interview" / f"{stem}.csv", INTERVIEW_FIELDS, [start])
    write_csv(out_dir / "intro" / f"{stem}.csv", fields, in_intro)
    write_csv(out_dir / "dropped" / f"{stem}.csv", fields, dropped)
    write_csv(out_dir / "segments" / f"{stem}.csv", fields, segments)  # written last: marks done
    return len(frames), len(segments), len(dropped), len(in_intro), start["interview_start"]


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
    t = actions.picture_time(poses["t"].astype(np.float64), float(poses["fps"])) \
        + actions.BURST_N * actions.BURST_STEP / 2  # the middle of each sample's burst
    rows, best = [], {}
    for s, e, k in audio.turns(sound, voices):
        v = audio.voice_name(k)
        judged = seen = 0
        for f in range(np.searchsorted(t, s), np.searchsorted(t, e)):
            faces = [p for p in by_frame[f] if p["state"] in ("talking", "silent")]
            talking = [p for p in faces if p["state"] == "talking" and v in str(p["m"].get("voice", "")).split("+")]
            judged += bool(faces)
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
            img = grab_frame(path, float(poses["t"][f]), float(poses["fps"]), w, h)
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


def save_npz(path, arrays):
    np.savez_compressed(path.with_suffix(".tmp.npz"), **arrays)
    path.with_suffix(".tmp.npz").replace(path)


def cached(path, valid, make, force=False):
    """(data, made): the arrays saved at path if valid(them), else make() them and save them."""
    if path.exists() and not force:
        saved = dict(np.load(path))
        if valid(saved):
            return saved, False
    data = make()
    save_npz(path, data)
    return data, True


def load_input(v, poses, args, models):
    """action_input() for one video, each part from its cache file in --out when still valid:
    poses/<video>.dense.npz, faces/, audio/ and camera/<video>.npz. Returns (extra, note)."""
    out, action, notes = args.out, args.action, []

    def faces():
        d, made = cached(out / "faces" / f"{v.stem}.npz",
                         lambda d: int(d["poses_n"]) == len(poses["frame"]) and "aligned" in d
                         and float(d["burst_step"]) == np.float32(actions.BURST_STEP),
                         lambda: face_input(v, poses, models), args.force)
        if made:
            notes.append(f"{len(d['det'])} faces measured")
        return d

    def camera():
        return cached(out / "camera" / f"{v.stem}.npz",
                      lambda d: int(d["poses_t"]) == len(poses["t"]) and "steps" in d
                      and int(d.get("camera_version", 0)) == actions.CAMERA_VERSION,
                      lambda: camera_input(v, poses), args.force)[0]

    if action in DENSE_ACTIONS:
        extra, made = cached(out / "poses" / f"{v.stem}.dense.npz",
                             lambda d: float(d["fps"]) == args.jump_fps
                             and int(d.get("camera_version", 0)) == actions.CAMERA_VERSION,
                             lambda: dense_input(v, poses, models, args.conf, args.jump_fps), args.force)
        if made:
            notes.append(f"{sum(b - a for a, b in extra['windows']):.0f}s sampled again at {args.jump_fps:g} fps")
    elif action == "smiling":
        extra = faces()
    elif action in SOUND_ACTIONS:
        sound = None
        if not args.no_sound:
            sound, made = cached(out / "audio" / f"{v.stem}.npz", lambda d: True, lambda: sound_input(v, models),
                                 args.force)
            if not bool(sound["has_sound"]):
                sound = None
            if made:
                notes.append("no sound" if sound is None else f"{int(sound['voices'])} voices, sound trusted "
                             f"{np.mean(sound['snr_db'] >= audio.SNR_OK_DB):.0%} of the time")
        extra = {"faces": faces(), "audio": sound, "camera": camera()}
    else:
        motion = args.motion or MOTION.get(action, "any")
        if motion == "any" and action != "walking":
            return None, ""
        extra = {"camera": camera(), "motion": motion}
    return extra, "".join(f"; {n}" for n in notes)


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
                    help="running and jumping: frames per second sampled where a whole body is in view")
    ap.add_argument("--conf", type=float, default=0.35, help="minimum person confidence")
    ap.add_argument("--min-frames", type=int, help="frames showing the action needed for a section "
                    "(default 2; 1 for jumping, 3 for running)")
    ap.add_argument("--max-gap", type=int, help="frames without it allowed inside a section "
                    "(default 2; 1 for walking and silent; 4 for jumping and 2 for running, whose frames are 0.2 s apart)")
    ap.add_argument("--motion", choices=["any", "still", "moving"],
                    help="standing and sitting: only people standing or sitting still (or moving), judged "
                         "from how far they travel between samples (default: any for standing, still for sitting)")
    ap.add_argument("--no-sound", action="store_true",
                    help="talking and silent: judge from the faces alone, ignoring the sound")
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
    shared = ["poses"] + ["faces"] * (action in FACE_ACTIONS) + ["audio"] * (action in SOUND_ACTIONS) \
        + ["camera"] * (action in ("walking",) + SOUND_ACTIONS or (args.motion or MOTION.get(action, "any")) != "any")
    for sub in shared:  # shared by all actions
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("frames", "segments", "dropped", "intro", "interview", "thumbs") + \
            (("speech", "voices") if action in SOUND_ACTIONS else ()):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    bank = None if args.keep_intro else intro.load_bank(args.intro_bank)
    if bank is None and not args.keep_intro:
        print(f"no intro bank at {args.intro_bank}: the intro is not separated", file=sys.stderr)
    loaded = {}

    def model(name):  # load each model the first time it is needed
        if name not in loaded:
            loaded[name] = (PoseModel(args.model, args.imgsz, args.threads, args.runtime) if name == "pose"
                            else actions.FaceModels(args.runtime, args.threads) if name == "faces"
                            else audio.VoiceModels(args.runtime, args.threads))
        return loaded[name]

    models = {name: (lambda name=name: model(name)) for name in ("pose", "faces", "voices")}
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
