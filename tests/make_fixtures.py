"""Builds the test clips from the small sources in tests/fixtures/sources/, with their ground truth.

The clips are made, not filmed: a person cut out of a street photo is placed on the empty street
and moved (walking, running, jumping), a virtual camera pans, tilts, zooms and cuts, and real
speech from four people is laid under faces taken from a real talking-head clip. So each clip's
truth is known exactly. They are 256 x 144 at 25 fps, H.264 at CRF 28 with sensor noise: as
rough as the 144p downloads.

The moving person is a still cut-out that slides: legs do not swing. Walking and running are
told apart by speed (which is what the detector uses), jumping by the body rising and landing.

Usage (from the youtube-analyser folder; the tests call it themselves):
    .venv/bin/python tests/make_fixtures.py            # build any missing clips
    .venv/bin/python tests/make_fixtures.py --force    # rebuild all of them
Clips go to tests/.cache/clips/ (not committed).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "fixtures" / "sources"
VERSION = 9                                 # bump when a clip changes: cached clips and results are rebuilt
CACHE = ROOT / ".cache" / f"v{VERSION}"
CLIPS = CACHE / "clips"
W, H, FPS = 256, 144, 25
SR = 16000

# ---- the stage: an empty street, a person, a camera --------------------------------------------
PERSON_TORSO = 148          # shoulder-to-hip length of the person cut-out, in its pixels
TORSO = 85                  # the person's torso on the street, in street pixels (27 px in the clips)
GROUND = 720                # street row the feet stand on
VIEW_H = 450                # camera view height in street pixels (width 800): 9.4 torso lengths across
VIEW_TOP = 300


def _street():
    """The empty street, four times side by side (crossfaded) so the camera can pan far."""
    s = cv2.imread(str(SOURCES / "street.jpg")).astype(np.float32)
    out, fade = s, 80
    for _ in range(3):
        ramp = np.linspace(0, 1, fade)[None, :, None]
        joint = out[:, -fade:] * (1 - ramp) + s[:, :fade] * ramp
        out = np.concatenate([out[:, :-fade], joint, s[:, fade:]], 1)
    return out.astype(np.uint8)


def _person(recolour=False):
    p = cv2.imread(str(SOURCES / "person.png"), cv2.IMREAD_UNCHANGED)
    if recolour:  # the same cut-out in a blue coat: another person to the look matching
        hsv = cv2.cvtColor(p[:, :, :3], cv2.COLOR_BGR2HSV)
        coat = (hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 120)
        hsv[:, :, 0] = np.where(coat, 110, hsv[:, :, 0])
        hsv[:, :, 1] = np.where(coat, 170, hsv[:, :, 1])
        p[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return p


class Person:
    """Where a person is at time t: feet x, feet y on the street, and torso length (street px)."""

    def __init__(self, path, recolour=False):
        self.path, self.img = path, _person(recolour)

    def at(self, t):
        x, y, torso = self.path(t)
        return float(x), float(y), float(torso)


def _paste(canvas, img, feet_x, feet_y, torso):
    k = torso / PERSON_TORSO
    h, w = max(1, round(img.shape[0] * k)), max(1, round(img.shape[1] * k))
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    x0, y0 = int(round(feet_x - w / 2)), int(round(feet_y - h))
    a0, b0 = max(0, x0), max(0, y0)
    a1, b1 = min(canvas.shape[1], x0 + w), min(canvas.shape[0], y0 + h)
    if a1 <= a0 or b1 <= b0:
        return
    part = small[b0 - y0:b1 - y0, a0 - x0:a1 - x0].astype(np.float32)
    alpha = part[:, :, 3:] / 255
    canvas[b0:b1, a0:a1] = canvas[b0:b1, a0:a1] * (1 - alpha) + part[:, :, :3] * alpha


def _other_street(street):
    """Another place: the street mirrored, in other colours (for cuts)."""
    hsv = cv2.cvtColor(street[:, ::-1], cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + 70) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def render(street, view, people, t):
    """One 256 x 144 frame: the people pasted onto the street, seen through view(t) = (left, top,
    height) in street pixels, or (left, top, height, "elsewhere") for another place (a cut)."""
    left, top, vh, *where = view(t)
    if where:
        street = _other_street(street)
    vw = vh * W / H
    x0, y0 = int(max(0, left - 10)), int(max(0, top - 10))
    x1, y1 = int(min(street.shape[1], left + vw + 10)), int(min(street.shape[0], top + vh + 10))
    canvas = street[y0:y1, x0:x1].astype(np.float32)
    for p in people:
        fx, fy, torso = p.at(t)
        _paste(canvas, p.img, fx - x0, fy - y0, torso)
    m = np.float32([[W / vw, 0, -(left - x0) * W / vw], [0, H / vh, -(top - y0) * H / vh]])
    return cv2.warpAffine(canvas, m, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)


def box(view, person, t):
    """Where the person appears in the clip at t: x0, y0, x1, y1 in clip pixels."""
    left, top, vh = view(t)[:3]
    k = H / vh
    fx, fy, torso = person.at(t)
    s = torso / PERSON_TORSO
    h, w = person.img.shape[0] * s, person.img.shape[1] * s
    return ((fx - w / 2 - left) * k, (fy - h - top) * k, (fx + w / 2 - left) * k, (fy - top) * k)


def _write(path, frames, sound=None):
    """Encode BGR frames (and float32 16 kHz sound, if any) like a small YouTube download."""
    rng = np.random.default_rng(1)
    tmp = path.with_suffix(".video.mp4")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p",
                          str(tmp)], stdin=subprocess.PIPE)
    for f in frames:
        noisy = np.clip(f.astype(np.float32) + rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8)
        p.stdin.write(noisy.tobytes())
    p.stdin.close()
    p.wait()
    if sound is None:
        tmp.replace(path)
        return
    raw = path.with_suffix(".f32")
    raw.write_bytes(sound.astype(np.float32).tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-f", "f32le", "-ar", str(SR), "-ac", "1",
                    "-i", str(raw), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "64k",
                    "-shortest", str(path)], check=True)
    tmp.unlink()
    raw.unlink()


# ---- movement clips -----------------------------------------------------------------------

def still_view(left=400):
    return lambda t: (left, VIEW_TOP, VIEW_H)


def hop(t, t0, height, air=0.45):
    """Height of a jump at t (parabola), in torso lengths."""
    u = (t - t0) / air
    return 4 * height * u * (1 - u) if 0 <= u <= 1 else 0.0


def walk(t, legs):
    """Distance covered at t (torso lengths) over legs [(t0, t1, speed)]; between legs, standing."""
    return sum(v * (min(t, b) - a) for a, b, v in legs if t > a)


def _movement():
    """name -> (seconds, view, {person name: Person}, {person name: truth(t)})."""
    clips = {}
    v = still_view()

    # walking and standing, with a still camera
    legs = [(1, 4, 1.5), (7, 10, -1.5)]      # 3.8 torso lengths there and back, in view all along
    man = Person(lambda t, legs=legs: (470 + TORSO * walk(t, legs), GROUND, TORSO))
    clips["walk_stop_walk"] = (12, v, {"man": man}, {"man": lambda t: "walking" if 1 < t < 4 or 7 < t < 10 else "still"})

    man = Person(lambda t: (800, GROUND, TORSO))
    clips["stand_still"] = (10, v, {"man": man}, {"man": lambda t: "still"})

    # the camera pans past someone standing: still
    man = Person(lambda t: (1000, GROUND, TORSO))
    clips["pan_past_standing"] = (10, lambda t: (300 + 1.5 * TORSO * t, VIEW_TOP, VIEW_H), {"man": man},
                                  {"man": lambda t: "still"})

    # the camera follows someone walking: walking (he stays put in the picture)
    man = Person(lambda t: (700 + 1.5 * TORSO * max(0.0, t - 1), GROUND, TORSO))
    clips["follow_walker"] = (10, lambda t: (300 + 1.5 * TORSO * max(0.0, t - 1), VIEW_TOP, VIEW_H), {"man": man},
                              {"man": lambda t: "walking" if t > 1 else "still"})

    # the camera zooms in on someone standing: still (not walking towards the camera)
    def zoom(t):
        vh = VIEW_H - 150 * min(1.0, max(0.0, (t - 2) / 6))
        return 800 - vh * W / H / 2, GROUND + 30 - vh, vh
    man = Person(lambda t: (800, GROUND, TORSO))
    clips["zoom_on_standing"] = (10, zoom, {"man": man}, {"man": lambda t: "still"})

    # someone walks towards the camera: walking (size grows as distance shrinks, normal lens)
    focal = 1.5 * H                       # clip pixels, as actions.FOCAL_H assumes
    t_out = 24.0                          # his torso in the clip at the start (9 torso lengths away)
    d0 = focal / t_out                    # his distance, in torso lengths
    def approach(t):
        d = d0 - 1.5 * min(2.4, max(0.0, t - 2))   # 1.5 torso lengths a second for 2.4 s
        g = d0 / d
        torso_street = t_out * VIEW_H / H * g
        return 800, VIEW_TOP + (40 + (95 - 40) * g) * VIEW_H / H, torso_street
    man = Person(approach)
    clips["towards_camera"] = (8, v, {"man": man}, {"man": lambda t: "walking" if 2 < t < 4.4 else "still"})

    # running: across a still camera, and followed by the camera
    man = Person(lambda t: (300 + 6 * TORSO * max(0.0, t - 3), GROUND, TORSO))
    clips["run_across"] = (8, v, {"man": man}, {"man": lambda t: "running" if 3 < t < 4.8 else "gone" if t >= 4.8 else "still"})
    man = Person(lambda t: (700 + 6 * TORSO * min(4.0, max(0.0, t - 2)), GROUND, TORSO))
    clips["follow_runner"] = (9, lambda t: (300 + 6 * TORSO * min(4.0, max(0.0, t - 2)), VIEW_TOP, VIEW_H), {"man": man},
                              {"man": lambda t: "running" if 2 < t < 6 else "still"})
    # a brisk walk is still walking
    legs = [(1, 3.5, 3.0)]
    man = Person(lambda t, legs=legs: (470 + TORSO * walk(t, legs), GROUND, TORSO))
    clips["brisk_walk"] = (8, v, {"man": man}, {"man": lambda t: "walking" if 1 < t < 3.5 else "still"})

    # two people cross, and someone walks past a bystander (hiding them): no mix-ups
    a = Person(lambda t: (470 + 1.5 * TORSO * min(4.0, max(0.0, t - 1)), GROUND, TORSO))
    b = Person(lambda t: (1130 - 1.5 * TORSO * min(4.0, max(0.0, t - 1)), GROUND, TORSO), recolour=True)
    clips["two_cross"] = (8, v, {"beige": a, "blue": b}, {"beige": lambda t: "walking" if 1 < t < 5 else "still",
                                                          "blue": lambda t: "walking" if 1 < t < 5 else "still"})
    stand = Person(lambda t: (800, GROUND - 10, TORSO), recolour=True)
    walker = Person(lambda t: (470 + 1.5 * TORSO * min(4.0, max(0.0, t - 1)), GROUND + 10, TORSO))
    clips["walk_past_bystander"] = (8, v, {"bystander": stand, "walker": walker},
                                    {"bystander": lambda t: "still", "walker": lambda t: "walking" if 1 < t < 5 else "still"})

    # a cut: from someone standing to another place with someone standing elsewhere in the picture
    man = Person(lambda t: (800 if t < 5 else 1050, GROUND, TORSO))
    clips["cut_on_standing"] = (10, lambda t: (400, VIEW_TOP, VIEW_H) if t < 5 else (400, VIEW_TOP, VIEW_H, "elsewhere"),
                                {"man": man}, {"man": lambda t: "still"})

    # jumping
    def jumper(times, height, x=800, recolour=False, forward=0.0):
        return Person(lambda t: (x + TORSO * forward * min(4.0, max(0.0, t - 3)),
                                 GROUND - TORSO * sum(hop(t, j, height) for j in times), TORSO), recolour)
    man = jumper([3, 6.5, 10], 0.45)
    clips["jump_three"] = (12, v, {"man": man}, {"man": lambda t: "jumping" if any(j + 0.1 < t < j + 0.35 for j in (3, 6.5, 10)) else "ground"})
    man = jumper([3, 6.5], 0.12)
    clips["tiny_hops"] = (9, v, {"man": man}, {"man": lambda t: "ground"})
    man = Person(lambda t: (800, GROUND, TORSO))
    tilt = lambda t: (400, VIEW_TOP + TORSO * sum(hop(t, j, 0.45) for j in (3, 6.5)), VIEW_H)
    clips["camera_bob"] = (9, tilt, {"man": man}, {"man": lambda t: "ground"})
    man = Person(lambda t: (800, GROUND - TORSO * 0.45 * min(1.0, max(0.0, (t - 4) / 0.4)), TORSO))
    clips["step_up"] = (9, v, {"man": man}, {"man": lambda t: "ground"})
    a, b = jumper([3, 8], 0.45, x=650), jumper([5.5, 8], 0.45, x=950, recolour=True)
    clips["two_jumpers"] = (10, v, {"beige": a, "blue": b},
                            {"beige": lambda t: "jumping" if any(j + 0.1 < t < j + 0.35 for j in (3, 8)) else "ground",
                             "blue": lambda t: "jumping" if any(j + 0.1 < t < j + 0.35 for j in (5.5, 8)) else "ground"})
    # jumping right beside someone standing (their boxes overlap by about a third)
    a, b = jumper([3, 7], 0.45, x=760), Person(lambda t: (835, GROUND + 5, TORSO), recolour=True)
    clips["jump_beside_someone"] = (10, v, {"jumper": a, "neighbour": b},
                                    {"jumper": lambda t: "jumping" if any(j + 0.1 < t < j + 0.35 for j in (3, 7)) else "ground",
                                     "neighbour": lambda t: "ground"})
    man = jumper([5], 0.45, x=500, forward=1.5)     # walking 3-7 s, jumping at 5 s
    clips["jump_while_walking"] = (9, v, {"man": man}, {"man": lambda t: "jumping" if 5.1 < t < 5.35 else "ground"})
    return clips


MOVEMENT = _movement()


# ---- talking clips ---------------------------------------------------------------------------

TALK = {
    # three faces side by side (left, middle, right); who talks, with which speaker's voice
    "three_talkers": [
        (0, 12, {"left": "a"}),
        (12, 24, {"middle": "b"}),
        (24, 36, {"right": "c"}),
        (36, 42, {}),                                   # nobody speaks
        (42, 47, {"left": "a", "right": "c"}),          # two at once
        (47, 53, {"offscreen": "d"}),                   # a voice, but no visible mouth moves
        (53, 59, {"middle": None}),                     # a mouth moves, but no speech (chewing)
        (59, 66, {"middle": "b"}),
        (66, 72, {"left": "a"}),
    ],
}
SEATS = {"left": 0, "middle": 1, "right": 2}


def _talker_frames():
    """The talking clip's frames: ping-pong over its most active 2 s (mouth moving), and a still
    frame of it with the mouth closed."""
    cap = cv2.VideoCapture(str(SOURCES / "talker.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    active = frames[55:120]
    return active + active[::-1], frames[95]


def _speech():
    """Each speaker's speech (float32, 16 kHz), pauses already removed."""
    out = {}
    for k in "abcd":
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(SOURCES / f"speech_{k}.ogg"), "-ac", "1", "-ar",
                              str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
        out[k] = np.frombuffer(raw, np.float32).copy()
    return out


def _sound(script, seconds, noise_db=None, start=None):
    """The sound for a script: each speaker's speech where they talk (taking their recording in
    order from start[voice] seconds, looping), room tone throughout, and optionally crowd babble
    noise_db below the voices."""
    rng = np.random.default_rng(2)
    speech = _speech()
    pos = {k: int((start or {}).get(k, 0) * SR) for k in speech}
    out = rng.normal(0, 0.0005, int(seconds * SR)).astype(np.float32)
    for t0, t1, who in script:
        for voice in who.values():
            if voice is None:
                continue
            n = int((t1 - t0) * SR)
            src = speech[voice]
            idx = (pos[voice] + np.arange(n)) % len(src)
            pos[voice] = (pos[voice] + n) % len(src)
            out[int(t0 * SR):int(t0 * SR) + n] += src[idx]
    if noise_db is not None:
        babble = np.zeros_like(out)
        for k, src in speech.items():  # everyone at once, from a different place in their recording
            for shift in (0.37, 0.71):
                babble += np.resize(np.roll(src, int(shift * len(src))), len(out))
        babble *= np.sqrt(0.05 ** 2 / 10 ** (noise_db / 10) / np.mean(babble ** 2))
        out += babble
    return out


def _three_faces(script, seconds):
    talk, still = _talker_frames()
    crop = lambda f: cv2.resize(f[0:120, 28:148], (84, 84), interpolation=cv2.INTER_AREA)
    frames = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        who = next(w for a, b, w in script if a <= t < b) if t < script[-1][1] else {}
        img = np.full((H, W, 3), 45, np.uint8)
        for seat, k in SEATS.items():
            tile = crop(talk[(i + 17 * k) % len(talk)]) if seat in who else crop(still)
            dx, dy = np.sin(t * 1.3 + k), np.cos(t * 0.9 + k)  # each person sways a little
            tile = cv2.warpAffine(tile, np.float32([[1, 0, dx], [0, 1, dy]]), (84, 84), borderMode=cv2.BORDER_REFLECT)
            img[30:114, 2 + 85 * k:86 + 85 * k] = tile
        frames.append(img)
    return frames


def _cutaways(seconds, shot=6):
    """Full-frame close-ups of two different-looking people, cutting between them every shot s:
    X (as filmed) talks in his shots, Y (mirrored, another colour and background) in his."""
    talk, _ = _talker_frames()
    frames = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        f = talk[i % len(talk)]
        if int(t // shot) % 2:
            f = cv2.flip(f, 1)
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + 60) % 180
            f = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img = np.full((H, W, 3), 30 if int(t // shot) % 2 else 80, np.uint8)
        img[:, 40:216] = f
        frames.append(img)
    return frames


CUTAWAYS = [(k * 6, k * 6 + 6, {"X" if k % 2 == 0 else "Y": "a" if k % 2 == 0 else "b"}) for k in range(6)]


# ---- people with characteristics: a woman (hair natural, purple or blue) and a man, talking --------

# Each clip: seconds, the people in their seats ({seat: (who, hair)}), who talks when with which
# voice, and where each voice's recording starts (so two clips do not share the same sentences).
# Voice c (FEE083) is "Jane": enrolled from duo_purple, and heard again in solo_blue.
CHARACTERS = {
    "duo_purple": (36, {"left": ("woman", "purple"), "right": ("man", None)},
                   [(0, 8, {"left": "c"}), (8, 16, {"right": "b"}), (16, 20, {}), (20, 28, {"left": "c"}),
                    (28, 36, {"right": "b"})], {"c": 0.0, "b": 0.0}),
    "duo_natural": (36, {"left": ("woman", "natural"), "right": ("man", None)},
                    [(0, 8, {"left": "a"}), (8, 16, {"right": "d"}), (16, 20, {}), (20, 28, {"left": "a"}),
                     (28, 36, {"right": "d"})], {"a": 0.0, "d": 0.0}),
    "solo_blue": (24, {"centre": ("woman", "blue")},
                  [(0, 8, {"centre": "c"}), (8, 12, {}), (12, 20, {"centre": "c"}), (20, 24, {})], {"c": 17.0}),
}
SEAT_X = {"left": 0, "right": 128, "centre": 64}
TILE = 128


def _woman():
    """The astronaut portrait with her hair mask and mouth (found with the project's own models),
    and functions to dye her hair and open her mouth."""
    sys.path.insert(0, str(ROOT.parent / "standing"))
    import actions as A
    import detect_standing as ds
    import people as P
    img = cv2.imread(str(SOURCES / "astronaut.jpg"))
    pose = ds.PoseModel(ds.DEFAULT_MODEL, 640, runtime="onnxruntime")
    looks = P.AppearanceModels("onnxruntime", gender_age=False)
    conf, boxes, kps = pose(img, 0.35)
    kp, box = kps[int(np.argmax(conf))], boxes[int(np.argmax(conf))]
    geo = A.face_geometry(kp)
    oval, _ = looks.face_outline(img, geo)
    h, w = img.shape[:2]
    face = np.zeros((h, w), np.uint8)
    cv2.fillPoly(face, [np.round(oval).astype(np.int32)], 1)
    face = cv2.dilate(face, np.ones((5, 5), np.uint8), iterations=2) > 0
    fx0, fx1 = oval[:, 0].min(), oval[:, 0].max()
    fw = fx1 - fx0
    yy, xx = np.mgrid[0:h, 0:w]
    zone = (xx > fx0 - 0.6 * fw) & (xx < fx1 + 0.6 * fw) & (yy >= box[1] - 5) & (yy < oval[P.CHIN][1] - 0.1 * fw)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hairish = (lab[:, :, 2] - 128 > 8) & (lab[:, :, 0] > 40) & (lab[:, :, 0] < 200)   # her warm light-brown hair
    hair = zone & ~face & looks.person_mask(img) & hairish
    hair = cv2.morphologyEx(hair.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    # the mouth, from the face landmarks back in the picture
    mid, d, ang = geo
    centre = A._below(mid, d, ang, 0.6)
    m = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), ang, 192 / (4 * d))
    m[:, 2] += np.array([96, 96]) - centre
    lm, _, _ = looks.faces.face(img, geo)
    back = cv2.invertAffineTransform(m)
    lm = lm @ back[:, :2].T + back[:, 2]
    top, lip, chin = lm[13][1], lm[14][1], lm[152][1]                         # inner lips, chin
    mx, half = (lm[61][0] + lm[291][0]) / 2, abs(lm[291][0] - lm[61][0]) / 2

    def dye(hue, chroma=45, light=0.75):
        lab2 = lab.copy()
        soft = cv2.GaussianBlur(hair.astype(np.float32), (0, 0), 1.5)
        lab2[:, :, 0] *= 1 - soft * (1 - light)
        lab2[:, :, 1] = lab2[:, :, 1] * (1 - soft) + (128 + chroma * np.cos(np.radians(hue))) * soft
        lab2[:, :, 2] = lab2[:, :, 2] * (1 - soft) + (128 + chroma * np.sin(np.radians(hue))) * soft
        return cv2.cvtColor(np.clip(lab2, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    wx = np.exp(-((xx - mx) / (1.6 * half)) ** 2)
    below = np.clip((yy - lip) / 2.0, 0, 1)                                   # nothing moves above the lower lip
    fade = np.clip(1 - (yy - chin) / (0.6 * (chin - lip)), 0, 1)              # still again under the chin

    # her smile shows her teeth, which the face model reads as a slightly open mouth: at rest the
    # teeth are covered with lip colour (lips closed), and talking opens a dark mouth under them
    lips = np.zeros((h, w), np.float32)
    cv2.ellipse(lips, (int(mx), int((top + lip) / 2)), (int(0.9 * half), max(1, int((lip - top) / 2) + 1)), 0, 0, 360, 1, -1)
    lips = cv2.GaussianBlur(lips, (0, 0), 1.0)[:, :, None]
    upper = (abs(xx - mx) < 0.5 * half) & (yy > lm[0][1]) & (yy < top - 1)
    lip_colour = np.median(img[upper], axis=0)

    def talking(src, opening):
        """Her mouth opened by opening (0-1): the lower lip and chin move down, showing the dark
        inside of her mouth."""
        out = (src * (1 - lips) + lip_colour * lips).astype(np.uint8)
        drop = 1.2 * opening * (chin - lip)
        if drop < 1:
            return out
        dy = drop * wx * below * fade
        out = cv2.remap(out, xx.astype(np.float32), (yy - dy).astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
        inside = np.zeros((h, w), np.float32)
        cv2.ellipse(inside, (int(mx), int((top + lip + drop) / 2)), (int(0.75 * half), int((lip + drop - top) / 2)),
                    0, 0, 360, 1, -1)
        inside = cv2.GaussianBlur(inside, (0, 0), 1.5)[:, :, None]
        return (out * (1 - inside) + np.array([35, 30, 70]) * inside).astype(np.uint8)
    variants = {"natural": img, "purple": dye(320), "blue": dye(290)}
    return variants, talking


def _mouth(script, seconds, seat, start):
    """How far open a woman's mouth is (0-1) at each frame: the loudness of her own voice, so her
    syllables and pauses are seen as well as heard."""
    own = [(a, b, {k: v for k, v in who.items() if k == seat}) for a, b, who in script]
    wav = _sound(own, seconds, start=start)
    hop = SR // FPS
    rms = np.sqrt(np.convolve(wav ** 2, np.ones(int(0.06 * SR)) / int(0.06 * SR), "same"))[::hop][:int(seconds * FPS)]
    # each syllable opens it wide and it closes between them: loudness against the loudest in the
    # surrounding 0.6 s
    peak = np.array([rms[max(0, i - 7):i + 8].max() for i in range(len(rms))])
    ratio = np.where(peak > 0.01, rms / np.maximum(peak, 1e-6), 0.0)
    return np.clip((ratio - 0.3) / 0.4, 0, 1)


def _characters(name):
    seconds, seats, script, start = CHARACTERS[name]
    talk, still = _talker_frames()
    mouths = {seat: _mouth(script, seconds, seat, start) for seat, (who, _) in seats.items() if who == "woman"}
    women = _woman() if any(who == "woman" for who, _ in seats.values()) else None
    man_tile = lambda f: cv2.resize(f[0:144, 16:160], (TILE, TILE), interpolation=cv2.INTER_AREA)
    frames = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        who = next((w for a, b, w in script if a <= t < b), {})
        img = np.full((H, W, 3), 45, np.uint8)
        for k, (seat, (person, hair)) in enumerate(seats.items()):
            if person == "man":
                tile = man_tile(talk[(i + 11) % len(talk)] if seat in who else still)
            else:
                variants, talking = women
                opening = mouths[seat][i] if seat in who else 0.0
                tile = cv2.resize(talking(variants[hair], opening)[20:280, 50:310], (TILE, TILE), interpolation=cv2.INTER_AREA)
            dx, dy = np.sin(t * 1.1 + k), np.cos(t * 0.7 + k)   # a little sway
            tile = cv2.warpAffine(tile, np.float32([[1, 0, dx], [0, 1, dy]]), (TILE, TILE), borderMode=cv2.BORDER_REFLECT)
            x0 = SEAT_X[seat]
            img[8:8 + TILE, x0:x0 + TILE] = tile
        frames.append(img)
    return frames


# ---- building --------------------------------------------------------------------------------

def path(name):
    return CLIPS / f"{name}.mp4"


def build(name, force=False):
    """Make one clip (if missing or force) and return its path."""
    out = path(name)
    if out.exists() and not force:
        return out
    CLIPS.mkdir(parents=True, exist_ok=True)
    if name in MOVEMENT:
        seconds, view, people, _ = MOVEMENT[name]
        street = _street()
        frames = (render(street, view, list(people.values()), i / FPS) for i in range(int(seconds * FPS)))
        _write(out, frames)
    elif name.startswith("three_talkers"):
        script = TALK["three_talkers"]
        seconds = script[-1][1]
        frames = _three_faces(script, seconds)
        sound = None if name.endswith("no_sound") else _sound(script, seconds, 5 if name.endswith("noisy") else None)
        _write(out, frames, sound)
    elif name == "cutaways":
        seconds = CUTAWAYS[-1][1]
        _write(out, _cutaways(seconds), _sound(CUTAWAYS, seconds))
    elif name in CHARACTERS:
        seconds, _, script, start = CHARACTERS[name]
        _write(out, _characters(name), _sound(script, seconds, start=start))
    elif name.startswith("voices_"):  # sound only, with a still picture
        script, seconds, noise = VOICE_CLIPS[name]
        _write(out, [np.full((H, W, 3), 60, np.uint8)] * int(seconds * FPS), _sound(script, seconds, noise))
    else:
        raise KeyError(name)
    return out


def _turns(voices, seconds, rng):
    """Alternating turns of 2-6 s among the voices, in a shuffled order."""
    out, t, last = [], 0.0, None
    while t < seconds:
        v = rng.choice([x for x in voices if x != last] or voices)
        d = min(float(rng.uniform(2, 6)), seconds - t)
        out.append((t, t + d, {"v": v}))
        t, last = t + d, v
    return out


def _shift(turns, dt):
    return [(a + dt, b + dt, w) for a, b, w in turns]


_rng = np.random.default_rng(7)
VOICE_CLIPS = {  # name -> (script, seconds, babble dB below the voices or None)
    "voices_one": (_turns(["b"], 40, _rng), 40, None),
    "voices_two": (_turns(["a", "b"], 60, _rng), 60, None),
    "voices_three": (_turns(["a", "b", "c"], 90, _rng), 90, None),
    "voices_three_noisy": (_turns(["a", "b", "c"], 90, _rng), 90, 5),
    "voices_silence": ([], 30, None),
    # two voices, and a third who only says a few words (3 s, at 40-43 s)
    "voices_brief_third": (_turns(["a", "b"], 40, _rng) + [(40, 43, {"v": "d"})]
                           + _shift(_turns(["a", "b"], 20, _rng), 43), 63, None),
}

ALL = list(MOVEMENT) + ["three_talkers", "three_talkers_noisy", "three_talkers_no_sound", "cutaways"] + list(VOICE_CLIPS) \
    + list(CHARACTERS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="clips to build (default: all)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for name in args.names or ALL:
        print(build(name, args.force), flush=True)


if __name__ == "__main__":
    sys.exit(main())
