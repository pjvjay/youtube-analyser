"""Who is in the videos: each person's characteristics, followed through a shot, and their voice.

Used by search.py to find a described person ("woman with purple hair") doing an action. For
every person found by the pose model in every 1-fps sample (configured in people.toml):

- hair colour: the person's pixels (models/person_mask.onnx, MediaPipe's selfie segmentation)
  around the head, minus the face (its outline from the face landmark model) and minus anything
  the colour of their cheeks, named by the nearest colour in the hair palette. Only on faces with
  at least hair_min_eyes px between the eyes: on faces shrunk to 144p sizes, dyed purple and green
  hair was named right at 9 px and 13 px, pink and blue at 13 px; natural shades are rougher (ash
  blond and grey are the same colour; dark brown and black too).
- clothes colours: the person's pixels between the shoulders and hips (top) and the hips and
  knees (bottom), named by the clothes palette.
- apparent gender and age: models/gender_age.onnx (optional, see people.toml), on faces with at
  least gender_min_eyes px between the eyes. On 17 test faces, 88-100% right with 13 px or more,
  82% at 9 px, near chance at 6 px; a woman in sunglasses came out a man at every size, and a
  woman with dyed purple hair scored as a man too. So when the person talks, the pitch of their
  voice decides instead (women 168-250 Hz, men 98-119 Hz in the test recordings).
- their voice (when they talk): pitch, and the voice's fingerprint, which can be recognised in
  other videos (and given a name, see people.toml).

A person is followed from sample to sample within a shot (tracks(): as the walking rules link
them, or by their box in close-ups), so each characteristic is decided from all the samples of
a track: noisy single samples are outvoted.
"""
import math
import re
import tomllib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from . import assets, motion
from .faces import FACE_SCORE, face_geometry
from .pose import L_HIP, L_SHO, R_HIP, R_SHO, VIS

DEFAULT_CONFIG = assets.PEOPLE_TOML
APPEARANCE_VERSION = 2

# MediaPipe face mesh: the face outline, from the top of the forehead round to the chin (index 18).
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
CHIN = 18                          # position of landmark 152 in FACE_OVAL
TRACK_IOU = 0.3                    # a person's box overlaps their box in the next sample at least this much
HAIR_LINK_MAX = 30                 # hair colours further apart than this are two people (on the test clips,
                                   # dyed purple against natural brown: 51; one person's consecutive samples:
                                   # at most 4, from a still portrait; real heads and light vary more)
MIN_PX = 8                         # pixels needed to measure a colour


# ---- configuration ---------------------------------------------------------------------------

def load_config(path=None, text=None):
    """people.toml, with a user's file (path) or its text laid over it: tables are merged,
    values replaced."""
    with open(DEFAULT_CONFIG, "rb") as f:
        config = tomllib.load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path, "rb") as f:
            _merge(config, tomllib.load(f))
    if text:
        _merge(config, tomllib.loads(text))
    for kind in ("hair", "clothes"):   # one example may be written without the list's brackets
        config["colours"][kind] = {name: _lab([hexes] if isinstance(hexes, str) else hexes)
                                   for name, hexes in config["colours"][kind].items()}
    return config


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v


def _lab(hexes):
    """sRGB '#rrggbb' strings -> OpenCV 8-bit CIELAB rows (ValueError for anything else)."""
    bad = [h for h in hexes if not re.fullmatch(r"#?[0-9a-fA-F]{6}", str(h))]
    if bad or not hexes:
        raise ValueError(f"colours are written #rrggbb, not {bad or hexes}")
    rgb = np.array([[int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)] for h in hexes], np.uint8)
    return cv2.cvtColor(rgb[None, :, ::-1], cv2.COLOR_BGR2LAB)[0].astype(np.float32)


def colour_name(lab, palette):
    """(name, distance) of the palette colour nearest to lab (OpenCV CIELAB), lightness counting
    half: the same hair or cloth in light and shade differs mostly in lightness."""
    best, name = np.inf, None
    for n, protos in palette.items():
        d = float(np.sqrt((((protos - lab) * [0.5, 1, 1]) ** 2).sum(1)).min())
        if d < best:
            best, name = d, n
    return name, best



# ---- what each detection looks like ----------------------------------------------------------

def empty(poses):
    """Appearance arrays (appearance/<video>.npz) with nothing measured yet: hair, top and bottom
    colours (CIELAB), female and age NaN; eyes (px between them, NaN: no face turned to the camera)
    from the poses; looked: whether hair and clothes have been measured; faced: which detections'
    faces have been judged for gender and age."""
    n = len(poses["frame"])
    out = {k: np.full((n, 3), np.nan, np.float32) for k in ("hair", "top", "bottom")}
    out.update({k: np.full(n, np.nan, np.float32) for k in ("female", "age")})
    geos = [face_geometry(kp) for kp in poses["kp"]]
    out["eyes"] = np.array([np.nan if g is None else g[1] for g in geos], np.float32).reshape(n)
    out.update(looked=np.int32(0), faced=np.zeros(n, bool), poses_n=np.int32(n), version=np.int32(APPEARANCE_VERSION))
    return out


def describe(frames, poses, models, config, out=None, looks=True, faces=()):
    """Measure what is asked for, adding to out (empty(poses) when None), and return it.

    looks: hair and clothes colours of every detection (once per video); faces: the detections
    whose apparent gender and age to judge (only faces with gender_min_eyes px between the eyes;
    search.py asks for a few per person). frames: (seconds, BGR) at the poses' samples
    (read_frames); only the frames needed are looked at."""
    look = config["look"]
    out = empty(poses) if out is None else {k: np.array(v) for k, v in out.items()}
    looks = looks and not int(out["looked"])
    faces = {int(j) for j in faces if not out["faced"][j]} if models.gender_age is not None else set()
    by_frame = {}
    for j, f in enumerate(poses["frame"]):
        if looks or j in faces:
            by_frame.setdefault(int(f), []).append(j)
    if not by_frame:
        return out
    last = max(by_frame)
    for f, (_, img) in enumerate(frames):
        if f > last:
            break
        if f not in by_frame:
            continue
        if looks:
            mask = models.person_mask(img)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        for j in by_frame[f]:
            kp, box = poses["kp"][j], poses["box"][j]
            geo = face_geometry(kp)
            if looks:
                out["top"][j], out["bottom"][j] = _clothes(lab, mask, kp, box)
                if geo is not None and geo[1] >= look["hair_min_eyes"]:
                    out["hair"][j] = _hair(img, lab, mask, box, geo, models)
            if j in faces:
                out["faced"][j] = True
                if geo is not None and geo[1] >= look["gender_min_eyes"]:
                    out["female"][j], out["age"][j] = models.female_age(img, geo)
    if looks:
        out["looked"] = np.int32(1)
    return out


def face_picks(appearance, track, tracks_wanted, config):
    """The detections to judge gender and age on for these tracks: each track's faces_per_person
    largest faces with at least gender_min_eyes px between the eyes (the clearest, and enough
    for a median)."""
    look = config["look"]
    eyes = appearance["eyes"]
    picks = []
    for tr in tracks_wanted:
        js = np.flatnonzero((track == tr) & (eyes >= look["gender_min_eyes"]))
        picks += js[np.argsort(-eyes[js], kind="stable")][:look["faces_per_person"]].tolist()
    return picks


def _median(lab, where):
    vals = lab[where]
    return np.median(vals, 0) if len(vals) >= MIN_PX else np.full(3, np.nan, np.float32)


def _hair(img, lab, mask, box, geo, models):
    """Median colour of the hair: person pixels around the head, not the face, not skin-coloured.
    (Worked out in a crop around the head, which gives the same pixels as the whole picture.)"""
    mid, d, _ = geo
    lm, score = models.faces.landmarks(img, geo)
    oval = lm[FACE_OVAL]
    if score < FACE_SCORE:
        return np.full(3, np.nan, np.float32)
    h, w = mask.shape
    grow = max(1, int(d / 10))
    x0, x1 = oval[:, 0].min(), oval[:, 0].max()
    fw = x1 - x0
    zx0, zx1 = x0 - 0.45 * fw, x1 + 0.45 * fw
    zy0, zy1 = max(box[1], oval[0][1] - 0.9 * fw), oval[CHIN][1]
    # the crop: the zone and the (grown) face outline, with a pixel to spare
    cx0 = int(max(0, math.floor(min(zx0, oval[:, 0].min())) - grow - 1))
    cx1 = int(min(w, math.ceil(max(zx1, oval[:, 0].max())) + grow + 2))
    cy0 = int(max(0, math.floor(min(zy0, oval[:, 1].min())) - grow - 1))
    cy1 = int(min(h, math.ceil(max(zy1, oval[:, 1].max())) + grow + 2))
    if cx1 <= cx0 or cy1 <= cy0:
        return np.full(3, np.nan, np.float32)
    lab, mask = lab[cy0:cy1, cx0:cx1], mask[cy0:cy1, cx0:cx1]
    face = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(face, [np.round(oval).astype(np.int32) - np.array([cx0, cy0], np.int32)], 1)
    face = cv2.dilate(face, np.ones((3, 3), np.uint8), iterations=grow) > 0
    yy, xx = np.mgrid[cy0:cy1, cx0:cx1]
    zone = (xx > zx0) & (xx < zx1) & (yy >= zy0) & (yy < zy1)
    cheeks = lab[face & (yy > mid[1] + 0.3 * d) & (yy < mid[1] + 1.0 * d)]
    hair = zone & ~face & mask
    if len(cheeks):  # forehead and neck skin that the outline missed
        skin = np.median(cheeks, 0)
        hair &= np.sqrt((((lab - skin) * [0.5, 1, 1]) ** 2).sum(2)) > 10
    return _median(lab, hair)


def _clothes(lab, mask, kp, box):
    """Median colours of the top (shoulders to hips) and bottom (hips to knees), inside the
    person's pixels and drawn in from the edges (arms, background); NaN where not in view."""
    ok = kp[:, 2] >= VIS
    h, w = mask.shape
    nan = np.full(3, np.nan, np.float32)
    if not (ok[L_SHO] and ok[R_SHO]):
        return nan, nan
    sho = kp[[L_SHO, R_SHO], :2]
    span = abs(sho[0, 0] - sho[1, 0])
    if span < 4:
        return nan, nan
    hips_in = ok[L_HIP] and ok[R_HIP]
    hip_y = kp[[L_HIP, R_HIP], 1].mean() if hips_in else min(h, box[1] + box[3], sho[:, 1].mean() + 1.2 * span)
    x0, x1 = sho[:, 0].min() + 0.2 * span, sho[:, 0].max() - 0.2 * span
    y0, y1 = sho[:, 1].mean() + 0.15 * (hip_y - sho[:, 1].mean()), hip_y - 0.1 * (hip_y - sho[:, 1].mean())
    region = np.zeros((h, w), bool)
    region[int(max(0, y0)):int(min(h, y1)), int(max(0, x0)):int(min(w, x1))] = True
    top = _median(lab, region & mask)
    knees = [13, 14]
    if not hips_in or not ok[knees].all():
        return top, nan
    hips = kp[[L_HIP, R_HIP], :2]
    knee_y = kp[knees, 1].mean()
    hx0, hx1 = hips[:, 0].min(), hips[:, 0].max()
    region = np.zeros((h, w), bool)
    region[int(max(0, hips[:, 1].mean() + 0.1 * (knee_y - hips[:, 1].mean()))):int(min(h, knee_y)),
           int(max(0, hx0 - 0.1 * span)):int(min(w, hx1 + 0.1 * span))] = True
    return top, _median(lab, region & mask)



# ---- following people through a shot ---------------------------------------------------------

def tracks(poses, camera=None, looks=None, hair=None):
    """A track number for every pose detection: the same person in the sample before (or the one
    before that), in the same shot. As for walking (motion.label_motion): where the camera alone
    would have put them, their mid-hip has moved at most LINK_MAX torso lengths a second and their
    torso is about the same size; people not seen down to the hips (close-ups) by their box instead,
    overlapping by TRACK_IOU or more; and in both cases their clothes alike (looks, as in
    motion.LOOK_MAX) and, where both were measured, their hair colour too (hair: CIELAB per
    detection, within HAIR_LINK_MAX): two people can sit in the same place in two shots that
    the camera measurement did not see as a cut. The nearest pairs are linked first."""
    w, h = (int(x) for x in poses["size"])
    centre = np.array([w / 2, h / 2])
    by_frame = [[] for _ in poses["t"]]
    for j, f in enumerate(poses["frame"]):
        by_frame[f].append(j)
    track = np.full(len(poses["frame"]), -1, np.int32)
    # live: (frame, detection, box x0 y0 x1 y1, body (hip x, hip y, torso) or None), moved with the camera
    live, next_id = [], 0
    last_hair = {}   # track -> its latest measured hair colour (a sample without one carries it over)
    for f, dets in enumerate(by_frame):
        if camera is not None and f > 0 and camera["response"][f] < motion.CUT_RESPONSE:
            live = []
        elif camera is not None and f > 0:
            s, d = float(camera["scale"][f]), camera["shift"][f].astype(np.float64)
            if np.isfinite(d).all():
                live = [(g, j, np.r_[s * (b[:2] - centre) + centre + d, s * (b[2:] - centre) + centre + d],
                         None if body is None else (*(s * (np.array(body[:2]) - centre) + centre + d), s * body[2]))
                        for g, j, b, body in live]
        live = [x for x in live if f - x[0] <= 2]
        bodies = {jb: motion.upper_body(poses["kp"][jb], h) for jb in dets}
        pairs = []
        for li, (g, ja, ba, body_a) in enumerate(live):
            dt = max(float(poses["t"][f] - poses["t"][g]), 1e-6)
            for di, jb in enumerate(dets):
                if looks is not None and motion.look_diff(looks[ja], looks[jb]) > motion.LOOK_MAX:
                    continue
                if hair is not None and track[ja] in last_hair and _colour_diff(last_hair[track[ja]], hair[jb]) > HAIR_LINK_MAX:
                    continue
                seen = 0.0 if looks is None else motion.look_diff(looks[ja], looks[jb])
                body_b = bodies[jb]
                if body_a is not None and body_b is not None:
                    torso = (body_a[2] + body_b[2]) / 2
                    travel = math.hypot(body_b[0] - body_a[0], body_b[1] - body_a[1]) / torso
                    size = abs(math.log(body_b[2] / body_a[2]))
                    if travel <= motion.LINK_MAX * dt and size <= math.log(1.5):
                        pairs.append((travel + size + motion.LOOK_WEIGHT * seen, li, di))
                else:
                    bb = xyxy(poses["box"][jb])
                    share = iou(ba, bb)
                    size = abs(math.log(max(bb[3] - bb[1], 1e-6) / max(ba[3] - ba[1], 1e-6)))
                    if share >= TRACK_IOU and size <= math.log(1.5):
                        pairs.append((2 * (1 - share) + size + motion.LOOK_WEIGHT * seen, li, di))
        taken_l, taken_d = set(), set()
        for _, li, di in sorted(pairs):
            if li not in taken_l and di not in taken_d:
                taken_l.add(li)
                taken_d.add(di)
                track[dets[di]] = track[live[li][1]]
        for di, j in enumerate(dets):
            if di not in taken_d:
                track[j] = next_id
                next_id += 1
            if hair is not None and not np.isnan(hair[j]).any():
                last_hair[track[j]] = hair[j]
        live = [x for li, x in enumerate(live) if li not in taken_l] + \
            [(f, j, xyxy(poses["box"][j]), bodies[j]) for j in dets]
    return track


def _colour_diff(a, b):
    """Distance of two CIELAB colours, lightness counting half (as colour_name); 0 if either is unknown."""
    if np.isnan(a).any() or np.isnan(b).any():
        return 0.0
    return float(np.sqrt((((a - b) * [0.5, 1, 1]) ** 2).sum()))


def xyxy(b):
    return np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], np.float64)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0



# ---- each person's characteristics ------------------------------------------------------------

def summarise(poses, track, appearance, config, own_voice=None, voices=None):
    """One summary per track: characteristics decided from all its samples.

    own_voice: {detection: voice index} for detections seen talking (faces.label_faces);
    voices: the video's voice characteristics ({"f0", "names"}: pitch per voice, and each voice's
    channel-wide name if known). Returns {track: dict}."""
    look, palettes = config["look"], config["colours"]
    out = {}
    for tr in np.unique(track):
        js = np.flatnonzero(track == tr)
        times = poses["t"][poses["frame"][js]]
        s = {"track": int(tr), "detections": js, "first_t": float(times.min()), "last_t": float(times.max())}
        for part, palette in (("hair", "hair"), ("top", "clothes"), ("bottom", "clothes")):
            names = [colour_name(v, palettes[palette])[0] for v in appearance[part][js] if not np.isnan(v).any()]
            s[part] = _vote(names, look["min_samples"])
        female = appearance["female"][js]
        female = female[~np.isnan(female)]
        age = appearance["age"][js]
        age = age[~np.isnan(age)]
        s["female"] = float(np.median(female)) if len(female) >= look["min_samples"] else None
        s["age"] = float(np.median(age)) if len(age) >= look["min_samples"] else None
        s["faces"] = len(female)
        mine = [own_voice[j] for j in js if own_voice and j in own_voice]
        s["voice"] = Counter(mine).most_common(1)[0][0] if mine else None
        s["voice_f0"] = float(voices["f0"][s["voice"]]) if s["voice"] is not None and voices is not None \
            and s["voice"] < len(voices["f0"]) else None
        s["voice_name"] = voices["names"].get(s["voice"]) if s["voice"] is not None and voices is not None else None
        s["gender"], s["gender_by"] = _gender(s, config)
        out[int(tr)] = s
    return out


def _vote(names, min_samples):
    """(most common name, its share, samples) or (None, 0, samples) with too few samples."""
    if len(names) < min_samples:
        return None, 0.0, len(names)
    best, n = Counter(names).most_common(1)[0]   # a tie goes to the name seen first
    return best, n / len(names), len(names)


def _gender(s, config):
    """Apparent gender from the pitch of their voice if they talked (the more reliable of the
    two at 144p, see people.toml), else from their face if it was seen well enough:
    ("female" | "male" | None, "voice" | "face" | "")."""
    look, voice = config["look"], config["voice"]
    f0 = s.get("voice_f0")
    if f0 is not None and not math.isnan(f0):
        if f0 >= voice["high_above_hz"]:
            return "female", "voice"
        if f0 <= voice["low_below_hz"]:
            return "male", "voice"
    if s["female"] is not None:
        if s["female"] >= look["female_above"]:
            return "female", "face"
        if s["female"] <= look["male_below"]:
            return "male", "face"
    return None, ""



# ---- voices across videos ---------------------------------------------------------------------

def same_voice(a_centroid, a_f0, b_centroid, b_f0, config):
    """Fingerprint similarity of two voices, or 0 when their pitch says they cannot be one person
    (one low, one high)."""
    v = config["voice"]
    sim = float(np.dot(a_centroid, b_centroid))
    if not (math.isnan(a_f0) or math.isnan(b_f0)):
        lo, hi = v["low_below_hz"], v["high_above_hz"]
        if (a_f0 <= lo and b_f0 >= hi) or (a_f0 >= hi and b_f0 <= lo):
            return 0.0
    return sim


def name_voices(centroids, f0, enrolled, config):
    """Which enrolled person (if any) each voice of a video is: {voice index: name}. enrolled:
    {name: (fingerprint, pitch)}. A voice is named when it is at least same_voice_above alike,
    and the best match."""
    out = {}
    for v in range(len(centroids)):
        scores = {n: same_voice(centroids[v], float(f0[v]), c, p, config) for n, (c, p) in enrolled.items()}
        if scores:
            best = max(scores, key=scores.get)
            if scores[best] >= config["voice"]["same_voice_above"]:
                out[v] = best
    return out

