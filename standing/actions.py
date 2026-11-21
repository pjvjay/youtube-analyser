"""Rules for the actions other than standing and sitting: jumping, smiling and talking.

detect_standing.py runs the pipeline (sampling, the pose model, sections, still figures, intro,
output) for every action. This module holds the parts that differ:

- jumping: the 1-fps poses show where a whole body is in view; only those moments are sampled
  again at JUMP_FPS, and a jump is the lowest foot and the hips rising together and coming back
  down, with camera movement taken out.
- smiling and talking: each person's face is found from the pose model's eye and nose points
  (so there is no separate face detector), cut out, and measured by two small face models:
  models/face_expression.onnx ("happy" probability) for smiling, and models/face_landmark.onnx
  (468 face points, including the lips) for talking, which is the mouth opening and closing.

Sitting needs nothing here: the standing rules in detect_standing.py already label people
"sitting".

The thresholds and the reasons for them are below.
"""
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

try:
    import openvino as ov
except ImportError:
    ov = None

HERE = Path(__file__).resolve().parent
LANDMARK_MODEL = HERE / "models" / "face_landmark.onnx"      # MediaPipe face mesh, Apache-2.0
EXPRESSION_MODEL = HERE / "models" / "face_expression.onnx"  # mini-Xception on FER2013, MIT

# COCO keypoint indices used by YOLO pose models (as in detect_standing.py).
NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHO, R_SHO, L_HIP, R_HIP, L_ANK, R_ANK = 5, 6, 11, 12, 15, 16
VIS, ANKLE_VIS = 0.5, 0.3              # the same keypoint confidences the standing rules use

# ---- Jumping ----------------------------------------------------------------------------
# At 1 frame per second a jump (under half a second in the air) is almost never sampled, and
# sampling whole videos faster would multiply the cost. So only the moments where someone's whole
# body is in view at 1 fps (shoulders, hips and an ankle, and not sitting) are sampled again,
# at JUMP_FPS, from JUMP_PAD_S before to JUMP_PAD_S after. In a sit-down show that is a small
# part of each video.
JUMP_FPS = 5
JUMP_PAD_S = 1.0
# A person is airborne in a frame when their lowest foot AND their hips are both at least
# JUMP_RISE torso lengths higher than in a frame before and a frame after (each within
# JUMP_RETURN_S), and the ground is at the same height in those two frames (within JUMP_GROUND).
# Using the lowest foot ignores steps and leg lifts (the other foot stays down); requiring the
# hips ignores a foot lifted on its own; requiring the way back down ignores stepping up onto
# something, walking away from the camera and standing up. A casual jump lifts the body by
# 0.3-0.7 torso lengths; walking moves the hips by about 0.1 and the lowest foot hardly at all,
# and keypoint jitter at 144p is about 0.05-0.1.
JUMP_RISE = 0.25
JUMP_GROUND = 0.12
JUMP_RETURN_S = 1.0
JUMP_SAME_SIZE = 0.25                  # torso length may change by this much (not a zoom or a new person)
# Camera tilts and shakes move everyone: the picture's own shift between samples (phase
# correlation, which a small moving person hardly affects) is taken out of every position.
# A response below CUT_RESPONSE is a cut: people are not followed across it.
CUT_RESPONSE = 0.2

# ---- Faces ------------------------------------------------------------------------------
# Faces are cut out around the pose model's eyes: eye distance d, turned so the eyes are level.
# The face landmark model gets a 4d square centred 0.6d below the eyes (what MediaPipe itself
# uses for these faces), the expression model a 2.6d square centred 0.5d below the eyes, which
# is about how tightly the FER2013 training faces are cropped.
# Faces turned further than FACE_YAW (nose more than 0.35 d off the middle of the eyes) are
# skipped: the mouth can no longer be measured.
FACE_YAW = 0.35
# Each sampled frame is followed by a short burst: BURST_N frames BURST_STEP seconds apart
# (so t, t+0.2, ... t+0.8 s), all cut at the same place. A burst ends at a camera cut (as for
# jumping), and a burst frame counts when the face model still finds a face there (FACE_SCORE)
# at about the same place and size (below), so another shot or a person walking out of the crop
# is not read as a moving mouth or a smile.
BURST_N, BURST_STEP = 5, 0.2
FACE_SCORE = 0.5
# Smiling: median "happy" probability over the burst. Tested on faces shrunk to 144p sizes: a
# smile scored 0.94-1.00 and talking, shouting and neutral faces at most 0.17 with at least
# 12 px between the eyes; at 9 px the two overlapped and at 6 px neutral faces scored up to 0.74.
# So smiling is only judged on faces with SMILE_MIN_EYES px or more between the eyes (close-ups
# at 144p).
SMILE_HAPPY = 0.6
SMILE_MIN_EYES = 11
# Talking: the mouth opening (inner lips apart / outer eye corners apart) changes by at least
# TALK_MOVE within the burst. Speech opens and closes the mouth several times a second: in the
# 144p talking test clip (about 20 px between the eyes) the opening moved by 0.17-0.36 within
# most bursts while he spoke, and by at most 0.05 on a still frame of his closed mouth with
# camera wobble and noise added; shrunk to 7-10 px between the eyes, 0.14-0.19 against 0.05.
# An open mouth was still measured at 6 px between the eyes (0.25 open against 0.06 closed),
# so talking is judged from TALK_MIN_EYES px.
TALK_MOVE = 0.12
TALK_MIN_EYES = 6
TALK_MIN_BURST = 3                     # burst frames with a face needed to judge
# MediaPipe face mesh indices: inner lips (upper, lower) and outer eye corners.
LIP_UP, LIP_DOWN, EYE_R_OUT, EYE_L_OUT = 13, 14, 33, 263
HAPPY = 3                              # class order: angry disgust fear happy sad surprise neutral
# In the 192-px landmark crop the pose model's eye distance d is CROP_D pixels and the middle of
# its eyes is at CROP_EYES. A burst frame counts when the face model finds the middle of the eyes
# within EYES_OFF x d of there (the burst is read separately, so fast head movements put it up to
# about 0.5 d away) and the outer eye corners SPAN_RANGE x d apart (1.45-1.56 d on the test faces).
CROP_D = 192 / 4
CROP_EYES = np.array([96, 96 - 0.6 * CROP_D])
EYES_OFF = 0.75
SPAN_RANGE = (1.0, 2.2)


# ---- Models -----------------------------------------------------------------------------

class OnnxModel:
    """A fixed-shape ONNX model on the CPU, through OpenVINO or onnxruntime (as PoseModel)."""

    def __init__(self, path, shape, runtime="openvino", threads=0):
        self.runtime = runtime
        if runtime == "openvino":
            core = ov.Core()
            m = core.read_model(str(path))
            m.reshape(list(shape))
            config = {"PERFORMANCE_HINT": "LATENCY"}
            if threads:
                config["INFERENCE_NUM_THREADS"] = threads
            self.compiled = core.compile_model(m, "CPU", config)
        else:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            if threads:
                so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name

    def __call__(self, x):
        if self.runtime == "openvino":
            r = self.compiled(x)
            return [r[o] for o in self.compiled.outputs]
        return self.sess.run(None, {self.inp: x})


class FaceModels:
    """The face landmark and expression models, on the same runtime as the pose model."""

    def __init__(self, runtime="auto", threads=0):
        self.runtime = runtime if runtime != "auto" else "openvino" if ov else "onnxruntime"
        self.landmark = OnnxModel(LANDMARK_MODEL, (1, 192, 192, 3), self.runtime, threads)
        self.expression = OnnxModel(EXPRESSION_MODEL, (1, 64, 64, 1), self.runtime, threads)

    def face(self, img, geo):
        """(468 x 2 landmarks in the 192-px crop, face score, happy probability) for one face."""
        mid, d, ang = geo
        crop = _crop(img, _below(mid, d, ang, 0.6), 4 * d, ang, 192)
        lm, score = self.landmark(np.ascontiguousarray(crop[None, :, :, ::-1], np.float32) / 255)
        grey = cv2.cvtColor(_crop(img, _below(mid, d, ang, 0.5), 2.6 * d, ang, 64), cv2.COLOR_BGR2GRAY)
        probs = self.expression(((grey.astype(np.float32) / 255 - 0.5) * 2)[None, :, :, None])[0]
        return (np.asarray(lm).reshape(468, 3)[:, :2], 1 / (1 + math.exp(-float(np.ravel(score)[0]))),
                float(np.ravel(probs)[HAPPY]))


def _below(mid, d, ang, k):
    """The point k eye-distances below the middle of the eyes, along the face's own vertical."""
    a = math.radians(ang)
    return mid + k * d * np.array([-math.sin(a), math.cos(a)])


def _crop(img, center, side, ang, out):
    """A side x side square around center, turned by ang degrees, scaled to out x out pixels."""
    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), ang, out / side)
    m[:, 2] += np.array([out / 2, out / 2]) - np.asarray(center, np.float64)
    return cv2.warpAffine(img, m, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def face_geometry(kp):
    """(middle of the eyes, eye distance d, roll in degrees) for a face turned to the camera, else None."""
    if min(kp[NOSE, 2], kp[L_EYE, 2], kp[R_EYE, 2]) < VIS:
        return None
    le, re = kp[L_EYE, :2].astype(np.float64), kp[R_EYE, :2].astype(np.float64)
    v = le - re  # the person's left eye is on the right of the picture when they face the camera
    d = float(np.linalg.norm(v))
    if v[0] <= 0 or d < 2:
        return None
    mid = (le + re) / 2
    if abs(float(np.dot(kp[NOSE, :2] - mid, v / d))) > FACE_YAW * d:
        return None
    return mid, d, math.degrees(math.atan2(v[1], v[0]))


def face_box(geo):
    """x, y, w, h of the expression crop, for drawing."""
    mid, d, ang = geo
    c = _below(mid, d, ang, 0.5)
    return np.array([c[0] - 1.3 * d, c[1] - 1.3 * d, 2.6 * d, 2.6 * d], np.float32)


def analyse_faces(frames, poses, models):
    """Measure every usable face of the sampled frames over a burst of following frames.

    frames yields (seconds, BGR image) at 1 / BURST_STEP frames per second (read_frames).
    Returns arrays for faces/<video>.npz: det (index into the pose arrays), eyes (d in pixels),
    and mouth / happy (n, BURST_N), NaN where the burst frame showed no face at that place.
    """
    rate = 1 / BURST_STEP
    dets, geos = [], []
    for j, kp in enumerate(poses["kp"]):
        g = face_geometry(kp)
        if g is not None and g[1] >= min(TALK_MIN_EYES, SMILE_MIN_EYES):
            dets.append(j)
            geos.append(g)
    mouth = np.full((len(dets), BURST_N), np.nan, np.float32)
    happy = np.full((len(dets), BURST_N), np.nan, np.float32)
    wanted = {}  # burst frame index -> [(face row, position in burst)]
    for row, j in enumerate(dets):
        t = float(poses["t"][poses["frame"][j]])
        for b in range(BURST_N):
            wanted.setdefault(round((t + b * BURST_STEP) * rate), []).append((row, b))
    span = np.full((len(dets), BURST_N), np.nan, np.float32)
    start = [round(float(poses["t"][poses["frame"][j]]) * rate) for j in dets]  # each burst's first frame
    last, prev, cut, hann = max(wanted, default=-1), None, -1, None
    for k, (_, img) in enumerate(frames):
        if k > last:
            break
        jobs = wanted.pop(k, ())
        if jobs or (k + 1) in wanted:  # a cut inside a burst ends it: after it is another shot
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if hann is None:
                hann = cv2.createHanningWindow(grey.shape[::-1], cv2.CV_32F)
            if prev is not None and prev[0] == k - 1 and cv2.phaseCorrelate(prev[1], grey, hann)[1] < CUT_RESPONSE:
                cut = k
            prev = (k, grey)
        for row, b in jobs:
            if b and cut > start[row]:
                continue
            lm, score, p = models.face(img, geos[row])
            eyes = (lm[EYE_R_OUT] + lm[EYE_L_OUT]) / 2
            s = float(np.linalg.norm(lm[EYE_L_OUT] - lm[EYE_R_OUT])) / CROP_D
            # a face where the pose model put the eyes, at the size it gave them
            if score >= FACE_SCORE and np.linalg.norm(eyes - CROP_EYES) <= EYES_OFF * CROP_D \
                    and SPAN_RANGE[0] <= s <= SPAN_RANGE[1]:
                mouth[row, b] = np.linalg.norm(lm[LIP_DOWN] - lm[LIP_UP]) / (s * CROP_D)
                happy[row, b] = p
                span[row, b] = s
    # a burst frame whose face is a different size from the rest shows someone else (a cut)
    measured = ~np.isnan(span).all(1)
    typical = np.full((len(dets), 1), np.nan, np.float32)
    typical[measured] = np.nanmedian(span[measured], axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        odd = np.abs(span / typical - 1) > 0.25
    mouth[odd], happy[odd] = np.nan, np.nan
    return {"det": np.array(dets, np.int32), "eyes": np.array([g[1] for g in geos], np.float32),
            "mouth": mouth, "happy": happy, "fps": np.float32(poses["fps"]),
            "burst_step": np.float32(BURST_STEP)}


def label_faces(by_frame, poses, faces, action):
    """Relabel everyone in by_frame as smiling / not smiling or talking / quiet from their face.

    People whose face could not be judged (out of view, turned away, too small) are "unknown".
    """
    rows = {int(j): r for r, j in enumerate(faces["det"])}
    for people in by_frame:
        for p in people:
            p["state"], p["m"] = "unknown", {"conf": p["m"].get("conf")}
            r = rows.get(p["j"])
            if r is None:
                continue
            eyes = float(faces["eyes"][r])
            p["face_box"] = face_box(face_geometry(poses["kp"][p["j"]]))
            if action == "smiling":
                h = faces["happy"][r]
                h = h[~np.isnan(h)]
                if eyes < SMILE_MIN_EYES or not len(h):
                    continue
                happy = float(np.median(h))
                p["m"].update(happy=happy, eyes=eyes)
                p["state"] = "smiling" if happy >= SMILE_HAPPY else "not smiling"
            else:
                m = faces["mouth"][r]
                m = m[~np.isnan(m)]
                if eyes < TALK_MIN_EYES or len(m) < TALK_MIN_BURST:
                    continue
                move = float(m.max() - m.min())
                p["m"].update(mouth_move=move, eyes=eyes)
                p["state"] = "talking" if move >= TALK_MOVE else "quiet"
    return by_frame


# ---- Jumping ----------------------------------------------------------------------------

def _body(kp, img_h):
    """(mid-hip x, mid-hip y, lowest ankle y, torso length) of an upright person, else None."""
    inside = kp[:, 1] < img_h - 1  # keypoints guessed below the bottom edge are not real
    ok, ok_ank = (kp[:, 2] >= VIS) & inside, (kp[:, 2] >= ANKLE_VIS) & inside
    if not (ok[L_SHO] or ok[R_SHO]) or not (ok[L_HIP] or ok[R_HIP]) or not (ok_ank[L_ANK] or ok_ank[R_ANK]):
        return None
    sho = np.mean([kp[i, :2] for i in (L_SHO, R_SHO) if ok[i]], 0)
    hip = np.mean([kp[i, :2] for i in (L_HIP, R_HIP) if ok[i]], 0)
    torso = float(np.linalg.norm(hip - sho))
    if torso < 3 or sho[1] >= hip[1]:
        return None
    foot = max(float(kp[a, 1]) for a in (L_ANK, R_ANK) if ok_ank[a])
    return float(hip[0]), float(hip[1]), foot, torso


def jump_windows(poses, states):
    """Time ranges to sample again at JUMP_FPS: around every 1-fps frame with a whole body in view.

    states[j] is the standing rules' label for pose detection j; sitting people are left out.
    """
    fps, img_h = float(poses["fps"]), int(poses["size"][1])
    times = sorted({float(poses["t"][f]) for j, f in enumerate(poses["frame"])
                    if states[j] != "sitting" and _body(poses["kp"][j], img_h) is not None})
    windows = []
    for t in times:
        t0, t1 = max(0.0, t - JUMP_PAD_S), t + 1 / fps + JUMP_PAD_S
        if windows and t0 <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], t1)
        else:
            windows.append([t0, t1])
    return [(round(a, 3), round(b, 3)) for a, b in windows]


def _tracks(dense):
    """Follow people through the dense samples. Returns tracks: lists of per-frame points.

    Positions have the camera's movement taken out; a cut or a new window starts afresh.
    """
    img_h = int(dense["size"][1])
    bodies = [[] for _ in dense["t"]]
    for j, f in enumerate(dense["frame"]):
        b = _body(dense["kp"][j], img_h)
        if b is not None:
            bodies[f].append((j, b))
    tracks, live, cam = [], [], 0.0
    for f in range(len(dense["t"])):
        if f == 0 or dense["window"][f] != dense["window"][f - 1] or dense["response"][f] < CUT_RESPONSE:
            live, cam = [], 0.0
        else:
            cam += float(dense["shift"][f, 1])  # how far the picture moved down since the last frame
        live = [tr for tr in live if f - tr[-1]["f"] <= 2]  # a track may miss one frame
        pairs = []
        for ti, tr in enumerate(live):
            last = tr[-1]
            for di, (_, (hx, hy, _, torso)) in enumerate(bodies[f]):
                dx = abs(hx - last["hx"]) / last["torso"]
                dy = abs(hy - cam - last["hy"]) / last["torso"]
                ratio = torso / last["torso"]
                if dx <= 0.6 and dy <= 1.5 and 1 / 1.3 <= ratio <= 1.3:
                    pairs.append((dx + abs(math.log(ratio)) + 0.3 * dy, ti, di))
        taken_t, taken_d = set(), set()
        for _, ti, di in sorted(pairs):
            if ti not in taken_t and di not in taken_d:
                taken_t.add(ti)
                taken_d.add(di)
                live[ti].append(_point(f, *bodies[f][di], cam))
        for di, body in enumerate(bodies[f]):
            if di not in taken_d:
                tracks.append([_point(f, *body, cam)])
                live.append(tracks[-1])
    return tracks


def _point(f, j, body, cam):
    hx, hy, foot, torso = body
    return {"f": f, "j": j, "hx": hx, "hy": hy - cam, "foot": foot - cam, "torso": torso}


def label_jumps(dense):
    """One list of people per dense frame, each "jumping", "on the ground" or "unknown", with their rise.

    rise: how far (in torso lengths) the lowest foot and the hips are both above where they
    were in a frame before and a frame after, on the same ground. Only people followed from
    before to after get one.
    """
    t = dense["t"]
    people = [[] for _ in t]
    person = {}
    for j, f in enumerate(dense["frame"]):
        person[j] = {"j": j, "kp": dense["kp"][j], "box": dense["box"][j], "state": "unknown",
                     "m": {"conf": float(dense["conf"][j])}}
        people[f].append(person[j])
    for tr in _tracks(dense):
        lo = hi = 0  # tr[lo:i] and tr[i + 1:hi] are the points within JUMP_RETURN_S of point i
        for i, p in enumerate(tr):
            tp = t[p["f"]]
            while tp - t[tr[lo]["f"]] > JUMP_RETURN_S:
                lo += 1
            hi = max(hi, i + 1)
            while hi < len(tr) and t[tr[hi]["f"]] - tp <= JUMP_RETURN_S:
                hi += 1
            best = None
            for a in tr[lo:i]:
                for b in tr[i + 1:hi]:
                    size = (a["torso"] + b["torso"]) / 2
                    if (abs(p["torso"] - size) > JUMP_SAME_SIZE * size
                            or abs(a["foot"] - b["foot"]) > JUMP_GROUND * size):
                        continue
                    rise = min((a["foot"] + b["foot"]) / 2 - p["foot"], (a["hy"] + b["hy"]) / 2 - p["hy"]) / size
                    best = rise if best is None else max(best, rise)
            if best is not None:
                person[p["j"]]["m"]["rise"] = best
                person[p["j"]]["state"] = "jumping" if best >= JUMP_RISE else "on the ground"
    return people
