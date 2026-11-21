"""Faces: smiling, and talking or silent (with the sound, see audio.py).

Each person's face is found from the pose model's eye and nose points (so there is no separate
face detector), cut out, and measured by two small face models (inference.FaceModels): the
expression model's "happy" probability for smiling, and the face landmark model's lips for
talking, which is the mouth opening and closing. Talking and silent also use the sound where it
can be trusted: whether anyone is speaking, and which voice, decides which of the visible mouths
is talking; each face followed through a shot keeps its own voice.

The thresholds and the reasons for them are below.
"""
import math

import cv2
import numpy as np

from .inference import below
from .media import picture_time
from .motion import CUT_RESPONSE
from .pose import L_EYE, NOSE, R_EYE, VIS

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
# With trusted sound (audio.py) the faces and the sound are judged together, per sampled second:
# - no speech in the burst (under SPEECH_SHARE of it): everyone visible is silent, whatever their
#   mouth does (chewing, smiling, mouthing along);
# - speech: of the visible mouths moving at least TALK_MOVE_VOICED, the one that moves most is
#   talking (two when two voices are heard, and also anyone whose mouth moves TALK_MOVE or more,
#   since two people talking at once can sound like one voice); everyone else is silent
#   (listening). The sound already says someone is talking, so a smaller movement is enough: in
#   the talking test clip speech moved the mouth by 0.07-0.36 per burst, a still face at most 0.05.
# - the sound cannot be trusted (a loud background, no sound track): the faces alone, as above.
SPEECH_SHARE = 0.2
TALK_MOVE_VOICED = 0.08
# Whose voice is whose: each visible face is followed from sample to sample within a shot (the
# middle of its eyes within FACE_FOLLOW eye distances, after the camera's own movement), and takes
# as its own voice the one heard most while it was the only one talking (at least VOICE_VOTES
# times, and VOICE_SHARE of them). Then two faces talking at once each get their own voice.
FACE_FOLLOW = 1.0
VOICE_VOTES = 2
VOICE_SHARE = 0.6
# MediaPipe face mesh indices: inner lips (upper, lower) and outer eye corners.
LIP_UP, LIP_DOWN, EYE_R_OUT, EYE_L_OUT = 13, 14, 33, 263
# In the 192-px landmark crop the pose model's eye distance d is CROP_D pixels and the middle of
# its eyes is at CROP_EYES. A burst frame counts when the face model finds the middle of the eyes
# within EYES_OFF x d of there (heads move during the 0.8 s burst) and the outer eye corners
# SPAN_RANGE x d apart (1.45-1.56 d on the test faces).
CROP_D = 192 / 4
CROP_EYES = np.array([96, 96 - 0.6 * CROP_D])
EYES_OFF = 0.75
SPAN_RANGE = (1.0, 2.2)


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
    c = below(mid, d, ang, 0.5)
    return np.array([c[0] - 1.3 * d, c[1] - 1.3 * d, 2.6 * d, 2.6 * d], np.float32)


def analyse_faces(frames, poses, models):
    """Measure every usable face of the sampled frames over a burst of following frames.

    frames yields (seconds, BGR image) at 1 / BURST_STEP frames per second (read_frames).
    Returns arrays for faces/<video>.npz: det (index into the pose arrays), eyes (d in pixels),
    and mouth / happy (n, BURST_N), NaN where the burst frame showed no face at that place.
    """
    rate, fps = 1 / BURST_STEP, float(poses["fps"])
    dets, geos = [], []
    for j, kp in enumerate(poses["kp"]):
        g = face_geometry(kp)
        if g is not None and g[1] >= min(TALK_MIN_EYES, SMILE_MIN_EYES):
            dets.append(j)
            geos.append(g)
    mouth = np.full((len(dets), BURST_N), np.nan, np.float32)
    happy = np.full((len(dets), BURST_N), np.nan, np.float32)
    wanted = {}  # burst frame index -> [(face row, position in burst)]
    start = []   # each burst's first frame: the one showing what the pose sample shows
    for row, j in enumerate(dets):
        first = round((picture_time(float(poses["t"][poses["frame"][j]]), fps) - picture_time(0, rate)) * rate)
        start.append(first)
        for b in range(BURST_N):
            wanted.setdefault(first + b, []).append((row, b))
    span = np.full((len(dets), BURST_N), np.nan, np.float32)
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
            "burst_step": np.float32(BURST_STEP), "aligned": np.int32(1)}


def label_faces(by_frame, poses, faces, action, heard=None, camera=None):
    """Relabel everyone in by_frame from their face: smiling / not smiling, or talking / silent.

    heard: for talking and silent, per sampled frame, what the sound says about its burst
    (audio.heard: trusted, share of speech, voices heard), or None to use the faces alone.
    camera: camera_motion() of the sampled frames, so faces are followed within a shot (not
    across cuts) to learn whose voice is whose; None follows them by position alone.
    People whose face could not be judged (out of view, turned away, too small) are "unknown".
    """
    rows = {int(j): r for r, j in enumerate(faces["det"])}
    judged = []
    for f, people in enumerate(by_frame):
        judged.append([])
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
                p["m"].update(mouth_move=float(m.max() - m.min()), eyes=eyes)
                judged[-1].append(p)
    if action == "smiling":
        return by_frame
    sounds = [heard[f] if heard is not None else (False, 0.0, []) for f in range(len(by_frame))]
    for people, sound in zip(judged, sounds):
        _talkers(people, sound)
    _face_tracks(judged, poses, camera)
    _name_voices(judged, sounds)
    return by_frame


def _talkers(judged, sound):
    """Decide talking / silent for the judged faces of one sampled frame (see TALK_MOVE_VOICED)."""
    trusted, share, voices = sound
    if not trusted:
        for p in judged:
            p["state"] = "talking" if p["m"]["mouth_move"] >= TALK_MOVE else "silent"
            p["m"]["by"] = "face"
        return
    if share < SPEECH_SHARE:
        for p in judged:
            p["state"], p["m"]["by"] = "silent", "sound"
        return
    ranked = sorted((p for p in judged if p["m"]["mouth_move"] >= TALK_MOVE_VOICED),
                    key=lambda p: -p["m"]["mouth_move"])
    n = max(1, len(voices))
    # one talker per voice heard; more when their mouths move as clearly as talking without sound
    # (two people talking over each other often sound like one voice to the voice grouping)
    talkers = ranked[:n] + [p for p in ranked[n:] if p["m"]["mouth_move"] >= TALK_MOVE]
    for p in judged:
        p["m"]["by"] = "sound+face"
        p["state"] = "talking" if p in talkers else "silent"


def _face_tracks(judged, poses, camera=None):
    """Give each judged face a track number: the same face in the samples before and after
    (one missed sample allowed), within a shot. Faces are matched by where the middle of the
    eyes is, after the camera's own movement, within FACE_FOLLOW eye distances."""
    img_w, img_h = (int(x) for x in poses["size"])
    centre = np.array([img_w / 2, img_h / 2])
    live, next_id = [], 0
    for f, people in enumerate(judged):
        cut = camera is not None and f > 0 and camera["response"][f] < CUT_RESPONSE
        if cut:
            live = []
        here = []
        for p in people:
            mid, d, _ = face_geometry(poses["kp"][p["j"]])
            here.append((p, mid, d))
        # move the live faces' last positions to where the camera puts them in this frame
        if camera is not None and f > 0 and not cut:
            s, shift = float(camera["scale"][f]), camera["shift"][f].astype(np.float64)
            live = [dict(x, mid=s * (x["mid"] - centre) + centre + shift, d=x["d"] * s) for x in live]
        live = [x for x in live if f - x["f"] <= 2]
        pairs = []
        for li, x in enumerate(live):
            for hi, (p, mid, d) in enumerate(here):
                gap = float(np.linalg.norm(mid - x["mid"])) / x["d"]
                if gap <= FACE_FOLLOW and 1 / 1.4 <= d / x["d"] <= 1.4:
                    pairs.append((gap, li, hi))
        taken_l, taken_h = set(), set()
        for _, li, hi in sorted(pairs):
            if li not in taken_l and hi not in taken_h:
                taken_l.add(li)
                taken_h.add(hi)
                here[hi][0]["track"] = live[li]["track"]
                live[li].update(f=f, mid=here[hi][1], d=here[hi][2])
        for hi, (p, mid, d) in enumerate(here):
            if hi not in taken_h:
                p["track"] = next_id
                live.append({"track": next_id, "f": f, "mid": mid, "d": d})
                next_id += 1


def _name_voices(judged, sounds):
    """Say which voice each talker speaks with, and which voices each silent face hears.

    A face track's own voice is the one heard most when it was the only talker and one voice was
    heard (at least VOICE_VOTES times, VOICE_SHARE of them). A talker with an own voice speaks
    with it, whatever is heard at that moment: when two people talk at once the voice grouping
    mostly hears one of them, or a blend (on the test conversation it gave one talker three
    different voices in four seconds). A talker without one takes the voice heard that no other
    talker owns, when that is a single voice, else the voices heard are listed."""
    votes = {}
    for people, (trusted, _, voices) in zip(judged, sounds):
        talkers = [p for p in people if p["state"] == "talking"]
        if trusted and len(talkers) == 1 and len(voices) == 1:
            votes.setdefault(talkers[0]["track"], []).append(int(voices[0]))
    own = {}
    for track, v in votes.items():
        counts = np.bincount(v)
        if len(v) >= VOICE_VOTES and counts.max() >= VOICE_SHARE * len(v):
            own[track] = int(counts.argmax())
    for people, (trusted, share, voices) in zip(judged, sounds):
        if not trusted or share < SPEECH_SHARE:
            continue
        voices = [int(v) for v in voices]
        talkers = [p for p in people if p["state"] == "talking"]
        owned = {own[p["track"]] for p in talkers if p["track"] in own}
        left = [v for v in voices if v not in owned]
        for p in talkers:
            mine = own.get(p["track"])
            p["m"]["voice"] = _name([mine] if mine is not None else left[:1] if len(left) == 1 else left)
        for p in people:
            if p["state"] == "silent":
                p["m"]["heard"] = _name([v for v in voices if v != own.get(p["track"])])


def _name(voices):
    return "+".join(chr(ord("A") + int(v)) for v in voices)

