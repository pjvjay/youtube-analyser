"""Shared by the tests: build a clip, run the detector on it like a user would, and look at the result.

run(action, names) runs standing/detect_standing.py on the clips (once; the detector's own caches
and "done" files make later runs instant), into tests/.cache/v<N>/out. people(name, action) then
labels everyone in every sample from the same cached inputs, so a test can check each person,
not just the sections.

DETECTOR_OPTIONS (environment) runs the end-to-end tests with other pose model settings, into a
cache folder of their own, to see what a faster setting costs:
    DETECTOR_OPTIONS="--imgsz 320" .venv/bin/python -m unittest tests.test_movement tests.test_talking tests.test_search
"""
import argparse
import csv
import functools
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "standing"))
sys.path.insert(0, str(HERE))

import actions  # noqa: E402
import audio  # noqa: E402,F401  (for the tests: from tests.helpers import audio)
import detect_standing as ds  # noqa: E402
import make_fixtures as mf  # noqa: E402
import people as characteristics  # noqa: E402  (standing/people.py; "people" is the helper below)
import search  # noqa: E402

OPTIONS = shlex.split(os.environ.get("DETECTOR_OPTIONS", ""))
OUT = mf.CACHE / ("out-" + re.sub(r"[^a-z0-9.]+", "-", " ".join(OPTIONS).lower()).strip("-") if OPTIONS else "out")
__all__ = ["actions", "audio", "ds", "mf", "characteristics", "search", "REPO", "OUT", "OPTIONS", "search_options", "run", "people", "sections", "csv_rows", "out_dir",
           "track_person", "overlap", "in_view", "steady"]


def run(action, names, *options):
    """Run the detector CLI for an action on these clips (building them first). Clips already
    done for this action are skipped by the detector itself."""
    clips = [str(mf.build(n)) for n in names]
    r = subprocess.run([sys.executable, str(REPO / "standing" / "detect_standing.py"), *clips, "--out", str(OUT),
                        "--keep-intro", "--action", action, *OPTIONS, *options], capture_output=True, text=True)
    if r.returncode or "FAILED" in r.stderr:
        raise AssertionError(f"detect_standing.py failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    return r


def search_options(**over):
    """search.options() for the tests' cache, with the pose model settings of DETECTOR_OPTIONS."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", type=Path, default=ds.DEFAULT_MODEL)
    ap.add_argument("--fps", type=float, default=1.0)
    known, _ = ap.parse_known_args(OPTIONS)
    if not known.model.is_absolute():
        known.model = REPO / known.model
    return search.options(OUT, imgsz=known.imgsz, model=known.model, fps=known.fps, keep_intro=True, **over)


def out_dir(action):
    return OUT if action == "standing" else OUT / action


def sections(name, action, kind="segments"):
    """The sections the detector wrote for one clip (segments, intro or dropped)."""
    with open(out_dir(action) / kind / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


def csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@functools.lru_cache(maxsize=None)
def people(name, action, motion=None, no_sound=False):
    """(timeline, people per sampled frame) for a clip, labelled for the action from the inputs
    the detector cached (so this runs the detector first if needed)."""
    options = (["--motion", motion] if motion else []) + (["--no-sound"] if no_sound else [])
    run(action, [name], *options)
    clip = mf.path(name)
    poses = dict(np.load(OUT / "poses" / f"{clip.stem}.npz"))
    args = argparse.Namespace(out=OUT, action=action, force=False, no_sound=no_sound, motion=motion,
                              jump_fps=actions.JUMP_FPS, conf=0.35)
    extra, _ = ds.load_input(clip, poses, args, models=None)
    timeline, by_frame = ds.label_people(poses, action, extra)
    return timeline, by_frame, poses, extra


def overlap(a, b):
    """Intersection over union of two x0, y0, x1, y1 boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def in_view(bx, margin=2):
    return bx[0] >= margin and bx[2] <= mf.W - margin and bx[1] >= -margin and bx[3] <= mf.H + margin


def track_person(name, action, who):
    """For each sample: (picture time, the truth for that person then, the detector's state for
    them or None if they were not found, their measurements, with "hidden": True when they
    overlapped someone or touched the edge, so were not measured on purpose). Only samples with
    the person fully in view. People are matched to detections one to one (best overlap first), so someone
    walking past a bystander is not taken for the bystander."""
    seconds, view, persons, truth = mf.MOVEMENT[name]
    timeline, by_frame, _, _ = people(name, action)
    fps = float(timeline["fps"])
    hidden = actions._occluded(timeline) | actions._at_edge(timeline)  # not measured on purpose
    out = []
    for t, frame in zip(timeline["t"].tolist(), by_frame):
        when = actions.picture_time(t, fps)
        boxes = {k: mf.box(view, p, when) for k, p in persons.items()}
        pairs = sorted(((overlap(bx, _xyxy(p["box"])), k, i) for k, bx in boxes.items() for i, p in enumerate(frame)),
                       reverse=True)
        # a detection over two people (one walking in front of the other) is nobody's
        ambiguous = {i for i, p in enumerate(frame) if sum(overlap(bx, _xyxy(p["box"])) >= 0.3 for bx in boxes.values()) > 1}
        mine, used_k, used_i = None, set(), set()
        for score, k, i in pairs:
            if score < 0.3:
                break
            if k in used_k or i in used_i or i in ambiguous:
                continue
            used_k.add(k)
            used_i.add(i)
            if k == who:
                mine = frame[i]
        if not in_view(boxes[who]):
            continue
        m = dict(mine["m"], hidden=mine["j"] in hidden) if mine else {}
        out.append((when, truth[who](when), mine["state"] if mine else None, m))
    return out


def _xyxy(b):
    return (float(b[0]), float(b[1]), float(b[0] + b[2]), float(b[1] + b[3]))


def steady(truth_fn, t, margin):
    """True when the truth is the same from t - margin to t + margin (away from a change)."""
    return all(truth_fn(t + d) == truth_fn(t) for d in np.linspace(-margin, margin, 9))
