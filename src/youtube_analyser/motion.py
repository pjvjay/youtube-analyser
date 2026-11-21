"""Movement: the camera's own pan, tilt and zoom, and people followed over time, for walking,
running, jumping, and standing or sitting still.

- walking: people are followed from one 1-fps sample to the next and judged by how far their
  body travels over the ground. The camera's own movement is measured from the background
  (people cut out) and taken out first, and people are told apart by their clothes.
- jumping and running: the 1-fps poses show where a whole body is in view; only those moments
  are sampled again at JUMP_FPS. A jump is the lowest foot and the hips rising together and
  coming back down; running is the body travelling 4 torso lengths a second or more.

The thresholds and the reasons for them are below.
"""
import math

import cv2
import numpy as np

from .pose import ANKLE_VIS, L_ANK, L_HIP, L_SHO, R_ANK, R_HIP, R_SHO, VIS

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

# ---- Moving or still --------------------------------------------------------------------
# People are followed from sample to sample and judged by how fast their body travels over the
# ground, in torso lengths a second. The camera's own movement is taken out first: its pan, tilt
# and zoom between two frames are measured from the whole picture (Fourier-Mellin phase
# correlation: zoom from the spectrum in log-polar form, then the shift; a person filling part of
# the picture hardly affects either, and on the street photo at 144p zooms of 0.9-1.2 came out
# within 0.01 and shifts within 0.1 px), and positions are put in the coordinates of the first
# frame of the shot. Nobody is followed across a cut (CUT_RESPONSE).
#   speed across the picture: how far the middle of the hips moves, in torso lengths;
#   speed towards or away from the camera: from how fast the person grows or shrinks. A person at
#   distance D with a torso of T pixels has D/torso = f/T torso lengths, f the focal length in
#   pixels, so a size change of r per second (log ratio) is (f/T) r torso lengths a second. f is
#   not known; FOCAL_H assumes a normal lens (about 37 degrees top to bottom). A longer lens makes
#   this an underestimate, so it can miss walking towards the camera, not invent it. Size changes
#   under MOVE_SCALE a second are keypoint jitter and count as none.
# Walking is MOVE_SPEED or more and under RUN_SPEED; running RUN_SPEED or more. A stroll is about
# 2 torso lengths a second (1 m/s), a brisk walk up to 3.5, a jog 4.5-6.5, a run more. Standing
# people sway and gesture by up to about 0.3, and keypoints jitter by about 0.1 at 144p.
MOVE_SPEED = 0.6
RUN_SPEED = 4.0
MOVE_SCALE = 0.15
FOCAL_H = 1.5
# Walking is judged between the 1-fps samples (it shows clearly 1 s apart, and waist-up shots
# count). Running covers 5-13 torso lengths a second, often more than the distance to the edge
# of the picture, so it is judged in jumping's 5-fps samples (where a whole body is in view),
# from the track over RUN_WINDOW_S either side: at least RUN_MIN_POINTS samples of the person.
LINK_MAX = 4.0                         # torso lengths a person may travel between 1-fps samples
DENSE_LINK_MAX = 3.0                   # and between 5-fps samples (0.2 s): a sprint
RUN_WINDOW_S = 0.5
RUN_MIN_POINTS = 3
# In the 5-fps samples the size change is measured over SIZE_WINDOW_S either side (11 samples):
# size jitter then averages out to about 0.02 a second, so from DENSE_MOVE_SCALE it counts. (A
# person 12 torso lengths away walking towards the camera grows by 0.12 a second.)
SIZE_WINDOW_S = 1.0
DENSE_MOVE_SCALE = 0.1
# In 1 s the camera can move too far to measure in one go: a camera following a runner moved the
# picture by 64% of its width, which came out as +71 px instead of -163 (phase correlation cannot
# tell a shift of more than half the picture from one the other way), and the runner as walking.
# So the camera's movement between 1-fps samples is measured in CAMERA_STEPS steps (5 frames a
# second, decoded just for this), and a cut in any step is a cut between the samples.
CAMERA_STEPS = 5
CAMERA_VERSION = 2                     # saved with the camera's movement; a change rebuilds camera/ and dense samples
MIN_BACKGROUND = 0.3                   # share of the picture outside people needed to measure the camera on it
HOLE_SOFTNESS = 6                      # px: blur of the holes cut around people (3-10 all measured a followed runner right)
SMOOTH_STEP = 3                        # px a step may differ from its neighbours' median before it is replaced
# People whose box touches the edge of the picture are only partly in view: their shoulders or
# hips are guesses, and so is their speed (a bystander cut off at the top, head out of view,
# measured torsos of 20-28 px from one second to the next, which read as walking towards the
# camera). They are not measured until they are fully in.
EDGE_PX = 1
# Someone mostly hidden by another person is not measured while they overlap (by OCCLUDED or
# more of the smaller of the two boxes): the pose model's points for the hidden one drift
# towards the other. People standing side by side overlap by 14-33% (measured on the street
# clips), and at 0.3 a jump next to someone was lost, so it is half. Detections overlapping by
# CROWDED or more are often two people blended into one (their clothes look like neither), so
# between 1-fps samples they are no sign that someone moved too far to follow (see label_motion).
OCCLUDED = 0.5
CROWDED = 0.3
# Position alone mixes people up: someone walking past a bystander hides them, and the walker was
# then matched to where the bystander had stood (0.7 instead of 1.5 torso lengths a second). So
# each person also has a look: the mean colour (CIELAB, 0-255) of 4 x 2 patches over their torso,
# where clothes are. The same person 1 s later differs by about 3-10 (noise, light), different
# clothes by 20 or more; links differing by more than LOOK_MAX are not made.
LOOK_MAX = 20
LOOK_WEIGHT = 1 / 10                   # a look difference of 10 counts like 1 torso length of travel


def frame_motion(prev, cur, hann, people=()):
    """How the camera moved from prev to cur (grey float32 frames): (dx, dy, response, scale).
    A point p in prev is at scale * (p - centre) + centre + (dx, dy) in cur.

    The zoom comes from the whole pictures' spectra (Fourier-Mellin). response is phase
    correlation's, on the whole pictures after the zoom: low, under CUT_RESPONSE, at a cut.
    The shift is measured on the background only: someone the camera follows stays put in the
    picture and can win otherwise (a followed runner's stillness came out as the camera's, every
    other frame). So the people's boxes (x, y, w, h, from either frame) are cut out of both
    pictures with soft-edged holes: hard-edged ones are still themselves and win instead, soft
    ones (like the Hann window already on the picture's edges) add nothing. With under
    MIN_BACKGROUND of the picture left (a close-up), the whole pictures are used."""
    h, w = prev.shape
    lp_a, radius = _log_polar(prev, hann)
    lp_b, _ = _log_polar(cur, hann)
    (sx, _), _ = cv2.phaseCorrelate(lp_a, lp_b)
    scale = float(np.exp(-sx * np.log(radius) / lp_a.shape[1]))
    if not 0.5 < scale < 2:
        scale = 1.0
    unzoomed = cv2.warpAffine(cur, cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 / scale), (w, h),
                              borderMode=cv2.BORDER_REFLECT)
    (dx, dy), response = cv2.phaseCorrelate(prev, unzoomed, hann)
    if len(people):
        keep = np.ones((h, w), np.float32)
        for x, y, bw, bh in people:
            keep[int(max(0, y - 0.1 * bh)):int(min(h, y + 1.1 * bh)), int(max(0, x - 0.1 * bw)):int(min(w, x + 1.1 * bw))] = 0
        if keep.mean() >= MIN_BACKGROUND:
            inside = keep > 0.5
            small = cv2.resize(keep, (w // 4, h // 4), interpolation=cv2.INTER_AREA)  # blurring a quarter-size mask
            keep = cv2.resize(cv2.GaussianBlur(small, (0, 0), HOLE_SOFTNESS / 4), (w, h)) * hann  # is 40x faster
            (dx, dy), _ = cv2.phaseCorrelate((prev - prev[inside].mean()) * keep,
                                             (unzoomed - unzoomed[inside].mean()) * keep)
    return dx * scale, dy * scale, float(response), scale



_SPECTRA = {}    # the last frame's log-polar spectrum (it is the next step's previous frame), and filters


def _log_polar(grey, hann):
    """The picture's magnitude spectrum (which a shift does not change), high-passed, in
    log-polar form, where a zoom becomes a shift along the first axis."""
    last = _SPECTRA.get("last")
    if last is not None and last[0] is grey:
        return last[1]
    h, w = grey.shape
    if ("filter", h, w) not in _SPECTRA:
        yy, xx = np.meshgrid(np.linspace(-0.5, 0.5, h), np.linspace(-0.5, 0.5, w), indexing="ij")
        k = np.cos(np.pi * xx) * np.cos(np.pi * yy)
        _SPECTRA["filter", h, w] = ((1 - k) * (2 - k)).astype(np.float32)
    spectrum = cv2.dft(grey * hann, flags=cv2.DFT_COMPLEX_OUTPUT)
    magnitude = np.fft.fftshift(cv2.magnitude(spectrum[:, :, 0], spectrum[:, :, 1]))
    radius = min(h, w) / 2
    lp = cv2.warpPolar(np.log1p(magnitude) * _SPECTRA["filter", h, w], (w, h), (w / 2, h / 2), radius,
                       cv2.WARP_POLAR_LOG | cv2.INTER_LINEAR)
    _SPECTRA["last"] = (grey, (lp, radius))
    return lp, radius


def camera_motion(frames, poses, steps=1):
    """The camera's movement from each sample to the next (frame_motion: shift, response, scale),
    measured around the people, and each pose detection's look (see LOOK_MAX).

    frames: (..., BGR) in order, `steps` per sample (read_frames at steps times the samples'
    rate), or read_windows for jumping's samples (a new window starts afresh: response 0). The
    movement between two samples is the steps between them put together; its response is the
    weakest step's, so a cut anywhere in between shows. A step out of line with the steps either
    side of it (see SMOOTH_STEP) is replaced by their median: a camera does not jump from
    panning to still and back in a fifth of a second, but a measurement can (a followed runner
    who was missed in one frame, so not cut out, came out as the camera's movement once)."""
    n = len(poses["t"])
    look = np.full((len(poses["frame"]), 24), np.nan, np.float32)
    by_frame = {}
    for j, f in enumerate(poses["frame"]):
        by_frame.setdefault(int(f), []).append(j)
    window = poses.get("window")
    first = (steps - 1) // 2       # the frame showing what sample 0 shows (see picture_time)
    step, sample_at, prev, hann = [], {}, None, None  # step[i]: frame i-1 to i (dx, dy, response, scale)
    for i, item in enumerate(frames):
        k, part = divmod(i - first, steps)
        img = item[-1]
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if hann is None:
            hann = cv2.createHanningWindow(grey.shape[::-1], cv2.CV_32F)
        new_shot = window is not None and steps == 1 and i > 0 and window[i] != window[i - 1]
        if prev is None or new_shot:
            step.append((np.nan, np.nan, 0.0, 1.0))
        else:
            near = min(n - 1, max(0, round((i - first) / steps)))
            boxes = [poses["box"][j] for j in by_frame.get(near, []) + by_frame.get(near - 1, [])]
            step.append(frame_motion(prev, grey, hann, boxes))
        prev = grey
        if part == 0 and 0 <= k < n:
            sample_at[k] = i
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            for j in by_frame.get(k, ()):
                look[j] = person_look(lab, poses["kp"][j])
    step = _smooth_steps(np.array(step, np.float64).reshape(-1, 4))
    shift = np.full((n, 2), np.nan, np.float32)
    response, scale = np.zeros(n, np.float32), np.ones(n, np.float32)
    for k in range(1, n):
        if k not in sample_at or k - 1 not in sample_at or (window is not None and window[k] != window[k - 1]):
            continue
        zoom, offset, weakest = 1.0, np.zeros(2), 1.0
        for dx, dy, r, z in step[sample_at[k - 1] + 1:sample_at[k] + 1]:
            zoom, offset, weakest = z * zoom, z * offset + np.array([dx, dy]), min(weakest, r)
        shift[k], response[k], scale[k] = offset, weakest, zoom
    return {"shift": shift, "response": response, "scale": scale, "look": look}


def _smooth_steps(step):
    """Replace steps out of line with both neighbours (within a shot) by the three's median."""
    out = step.copy()
    for i in range(1, len(step) - 1):
        trio = step[i - 1:i + 2]
        if (trio[:, 2] < CUT_RESPONSE).any() or np.isnan(trio[:, :2]).any():
            continue
        med = np.median(trio, axis=0)
        if np.abs(step[i, :2] - med[:2]).max() > max(SMOOTH_STEP, 0.3 * np.abs(med[:2]).max()):
            out[i, [0, 1, 3]] = med[[0, 1, 3]]
    return out


def person_look(lab, kp):
    """Mean CIELAB colour of 4 x 2 patches over the torso (shoulders to hips), 24 numbers; NaN if
    the shoulders and hips are not all in view."""
    unknown = np.full(24, np.nan, np.float32)
    ok = kp[:, 2] >= VIS
    if not (ok[L_SHO] and ok[R_SHO] and ok[L_HIP] and ok[R_HIP]):
        return unknown
    pts = kp[[L_SHO, R_SHO, L_HIP, R_HIP], :2]
    h, w = lab.shape[:2]
    x0, x1 = int(max(0, pts[:, 0].min())), int(min(w, pts[:, 0].max() + 1))
    y0, y1 = int(max(0, pts[:, 1].min())), int(min(h, pts[:, 1].max() + 1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return unknown
    return cv2.resize(lab[y0:y1, x0:x1], (2, 4), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def look_diff(a, b):
    """Mean colour difference of two looks: 0 when both are unknown (position decides), and
    over LOOK_MAX (no link) when only one is: a detection whose torso is not all in view is
    often two people merged into one box (one walking in front of the other), which must not
    carry on either person's track."""
    known_a, known_b = not np.isnan(a).any(), not np.isnan(b).any()
    if not known_a and not known_b:
        return 0.0
    if known_a != known_b:
        return LOOK_MAX + 1.0
    return float(np.abs(a - b).mean())


def upper_body(kp, img_h):
    """(mid-hip x, mid-hip y, torso length) of a person whose shoulders and hips are in view."""
    inside = kp[:, 1] < img_h - 1
    ok = (kp[:, 2] >= VIS) & inside
    if not (ok[L_SHO] or ok[R_SHO]) or not (ok[L_HIP] or ok[R_HIP]):
        return None
    sho = np.mean([kp[i, :2] for i in (L_SHO, R_SHO) if ok[i]], 0)
    hip = np.mean([kp[i, :2] for i in (L_HIP, R_HIP) if ok[i]], 0)
    torso = float(np.linalg.norm(hip - sho))
    if torso < 3 or sho[1] >= hip[1]:  # upright people only
        return None
    return float(hip[0]), float(hip[1]), torso


def whole_body(kp, img_h):
    """(mid-hip x, mid-hip y, lowest ankle y, torso length) of an upright person whose shoulders,
    hips and an ankle are in view, else None."""
    up = upper_body(kp, img_h)
    if up is None:
        return None
    ok_ank = (kp[:, 2] >= ANKLE_VIS) & (kp[:, 1] < img_h - 1)
    if not (ok_ank[L_ANK] or ok_ank[R_ANK]):
        return None
    foot = max(float(kp[a, 1]) for a in (L_ANK, R_ANK) if ok_ank[a])
    return up[0], up[1], foot, up[2]


def ground_speed(across, grow, torso_px, img_h, min_grow=MOVE_SCALE):
    """Torso lengths a second over the ground: across the picture, and towards or away from the
    camera from the size change (see FOCAL_H), which counts from min_grow."""
    depth = FOCAL_H * img_h / torso_px * grow if grow >= min_grow else 0.0
    return math.hypot(across, depth)


def motion_state(speed):
    """running, walking or still from a ground speed; None when it is not known."""
    if speed is None or np.isnan(speed):
        return None
    return "running" if speed >= RUN_SPEED else "walking" if speed >= MOVE_SPEED else "still"


def jump_windows(poses, states):
    """Time ranges to sample again at JUMP_FPS: around every 1-fps frame with a whole body in view.

    states[j] is the standing rules' label for pose detection j; sitting people are left out.
    """
    fps, img_h = float(poses["fps"]), int(poses["size"][1])
    times = sorted({float(poses["t"][f]) for j, f in enumerate(poses["frame"])
                    if states[j] != "sitting" and whole_body(poses["kp"][j], img_h) is not None})
    windows = []
    for t in times:
        t0, t1 = max(0.0, t - JUMP_PAD_S), t + 1 / fps + JUMP_PAD_S
        if windows and t0 <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], t1)
        else:
            windows.append([t0, t1])
    return [(round(a, 3), round(b, 3)) for a, b in windows]


def _tracks(dense, need_foot=True):
    """Follow people through the dense samples. Returns tracks: lists of per-frame points with
    positions and torso length in the coordinates of the first frame of the shot (camera pan,
    tilt and zoom taken out). A cut or a new window starts afresh; a track may miss one frame.
    need_foot: only people with an ankle in view (jumping needs the feet; running does not)."""
    w, img_h = (int(x) for x in dense["size"])
    centre = np.array([w / 2, img_h / 2])
    look = dense.get("look")
    people = [[] for _ in dense["t"]]
    hidden = occluded(dense) | at_edge(dense)
    for j, f in enumerate(dense["frame"]):
        b = whole_body(dense["kp"][j], img_h) if need_foot else upper_body(dense["kp"][j], img_h)
        if b is not None and j not in hidden:
            people[f].append((j, b))
    tracks, live = [], []
    zoom, offset = 1.0, np.zeros(2)
    for f in range(len(dense["t"])):
        if f == 0 or dense["window"][f] != dense["window"][f - 1] or dense["response"][f] < CUT_RESPONSE:
            live, zoom, offset = [], 1.0, np.zeros(2)
        else:  # compose this frame's camera movement onto the shot's
            s = float(dense["scale"][f]) if "scale" in dense else 1.0
            zoom, offset = s * zoom, s * offset + dense["shift"][f].astype(np.float64)
        live = [tr for tr in live if f - tr[-1]["f"] <= 2]
        points = []
        for j, b in people[f]:
            hip = (np.array(b[:2]) - centre - offset) / zoom + centre
            foot = ((b[2] - centre[1] - offset[1]) / zoom + centre[1]) if need_foot else None
            points.append({"f": f, "j": j, "hx": float(hip[0]), "hy": float(hip[1]), "foot": foot,
                           "torso": b[-1] / zoom})
        pairs = []
        for ti, tr in enumerate(live):
            last, gap = tr[-1], f - tr[-1]["f"]
            for di, p in enumerate(points):
                dx = abs(p["hx"] - last["hx"]) / last["torso"]
                dy = abs(p["hy"] - last["hy"]) / last["torso"]
                ratio = p["torso"] / last["torso"]
                seen = look_diff(look[last["j"]], look[p["j"]]) if look is not None else 0.0
                if dx <= DENSE_LINK_MAX * gap and dy <= 1.5 * gap and 1 / 1.3 <= ratio <= 1.3 and seen <= LOOK_MAX:
                    pairs.append((dx + abs(math.log(ratio)) + 0.3 * dy + LOOK_WEIGHT * seen, ti, di))
        taken_t, taken_d = set(), set()
        for _, ti, di in sorted(pairs):
            if ti not in taken_t and di not in taken_d:
                taken_t.add(ti)
                taken_d.add(di)
                live[ti].append(points[di])
        for di, p in enumerate(points):
            if di not in taken_d:
                tracks.append([p])
                live.append(tracks[-1])
    return tracks


def at_edge(poses):
    """Detections whose box touches the edge of the picture (see EDGE_PX)."""
    w, h = (int(x) for x in poses["size"])
    b = poses["box"]
    edge = (b[:, 0] <= EDGE_PX) | (b[:, 1] <= EDGE_PX) | (b[:, 0] + b[:, 2] >= w - EDGE_PX) | (b[:, 1] + b[:, 3] >= h - EDGE_PX)
    return set(np.flatnonzero(edge).tolist())


def occluded(poses, share=OCCLUDED):
    """Detections overlapping another in the same frame by `share` or more of the smaller box."""
    hidden, by_frame = set(), {}
    for j, f in enumerate(poses["frame"]):
        by_frame.setdefault(int(f), []).append(j)
    for js in by_frame.values():
        for i, a in enumerate(js):
            for b in js[i + 1:]:
                if covered(poses["box"][a], poses["box"][b]) >= share:
                    hidden.update((a, b))
    return hidden


def covered(a, b):
    """How much of the smaller of two x, y, w, h boxes the other covers."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return float(ix * iy / (min(a[2] * a[3], b[2] * b[3]) + 1e-9))


def _dense_people(dense):
    """One list of people per dense frame (state unknown), and each person by detection."""
    people, person = [[] for _ in dense["t"]], {}
    for j, f in enumerate(dense["frame"]):
        person[j] = {"j": j, "kp": dense["kp"][j], "box": dense["box"][j], "state": "unknown",
                     "m": {"conf": float(dense["conf"][j])}}
        people[f].append(person[j])
    return people, person


def label_jumps(dense):
    """One list of people per dense frame, each "jumping", "on the ground" or "unknown", with their rise.

    rise: how far (in torso lengths) the lowest foot and the hips are both above where they
    were in a frame before and a frame after, on the same ground. Only people followed from
    before to after get one.
    """
    t = dense["t"]
    people, person = _dense_people(dense)
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


def label_runs(dense, seated=None):
    """One list of people per dense frame, each "running", "walking", "still" or "unknown", with
    their ground speed (torso lengths a second) over RUN_WINDOW_S either side.
    seated(kp): True for a seated person (on a bicycle, say), who is not running."""
    t, img_h = dense["t"], int(dense["size"][1])
    people, person = _dense_people(dense)
    for tr in _tracks(dense, need_foot=False):
        times = np.array([t[p["f"]] for p in tr])
        for i, p in enumerate(tr):
            near = (times >= times[i] - RUN_WINDOW_S - 1e-6) & (times <= times[i] + RUN_WINDOW_S + 1e-6)
            if near.sum() < RUN_MIN_POINTS or np.ptp(times[near]) < 0.4:
                continue
            wide = np.abs(times - times[i]) <= SIZE_WINDOW_S + 1e-6
            torso = float(np.mean([q["torso"] for q, k in zip(tr, near) if k]))
            across = math.hypot(_slope(times[near], [q["hx"] for q, k in zip(tr, near) if k]),
                                _slope(times[near], [q["hy"] for q, k in zip(tr, near) if k])) / torso
            grow = abs(_slope(times[wide], np.log([q["torso"] for q, k in zip(tr, wide) if k])))
            speed = ground_speed(across, grow, torso, img_h, DENSE_MOVE_SCALE)
            who = person[p["j"]]
            who["m"]["speed"] = speed
            who["state"] = "still" if seated is not None and seated(who["kp"]) else motion_state(speed)
    return people


def _slope(t, v):
    """Least-squares change of v per second."""
    t, v = np.asarray(t, np.float64), np.asarray(v, np.float64)
    tt = t - t.mean()
    return float(np.dot(tt, v - v.mean()) / np.dot(tt, tt))


def label_motion(poses, camera):
    """Ground speed of each pose detection between the 1-fps samples (torso lengths a second,
    ground_speed): the slower of its links to the samples before and after. With a link on one
    side only, that link counts when nobody was left unmatched on the other side (the person was
    hidden there, or out of the shot); otherwise they may have moved too far to follow, and the
    one link is mostly a change of pace (a runner setting off averaged 2.9 over his first second,
    and came out walking), so the speed is unknown (NaN)."""
    w, img_h = (int(x) for x in poses["size"])
    centre = np.array([w / 2, img_h / 2])
    bodies = [[] for _ in poses["t"]]
    hidden = occluded(poses) | at_edge(poses)
    crowded = occluded(poses, CROWDED)
    for j, f in enumerate(poses["frame"]):
        b = upper_body(poses["kp"][j], img_h)
        if b is not None and j not in hidden:
            bodies[f].append((j, b))
    scale = camera["scale"] if "scale" in camera else np.ones(len(camera["response"]), np.float32)
    links, refused = {}, set()
    for f in range(len(poses["t"]) - 1):
        if not bodies[f] or not bodies[f + 1] or camera["response"][f + 1] < CUT_RESPONSE:
            continue
        dt = float(poses["t"][f + 1] - poses["t"][f])
        s, d = float(scale[f + 1]), camera["shift"][f + 1].astype(np.float64)
        pairs = []
        for a, (ja, (xa, ya, ta)) in enumerate(bodies[f]):
            expect = s * (np.array([xa, ya]) - centre) + centre + d  # where the camera alone puts a
            for b, (jb, (xb, yb, tb)) in enumerate(bodies[f + 1]):
                torso = (s * ta + tb) / 2
                travel = float(np.linalg.norm(np.array([xb, yb]) - expect)) / torso
                size = abs(math.log(tb / (s * ta)))
                seen = look_diff(camera["look"][ja], camera["look"][jb])
                if travel <= LINK_MAX and size <= math.log(1.5) and seen <= LOOK_MAX:
                    speed = ground_speed(travel / dt, size / dt, torso, img_h)
                    pairs.append((travel + size + LOOK_WEIGHT * seen, a, b, speed))
        taken_a, taken_b = set(), set()
        for _, a, b, speed in sorted(pairs):
            if a not in taken_a and b not in taken_b:
                taken_a.add(a)
                taken_b.add(b)
                for j in (bodies[f][a][0], bodies[f + 1][b][0]):
                    links.setdefault(j, []).append(speed)
        left_a = [bodies[f][a][0] for a in range(len(bodies[f])) if a not in taken_a]
        left_b = [bodies[f + 1][b][0] for b in range(len(bodies[f + 1])) if b not in taken_b]
        clear_a = [j for j in left_a if j not in crowded]   # a blend of two people is no sign of
        clear_b = [j for j in left_b if j not in crowded]   # someone moving too far to follow
        for j in left_a:
            if clear_b:
                refused.add(j)
        for j in left_b:
            if clear_a:
                refused.add(j)
    speed = np.full(len(poses["frame"]), np.nan, np.float32)
    for j, v in links.items():
        if len(v) == 2 or j not in refused:
            speed[j] = min(v)
    return speed

