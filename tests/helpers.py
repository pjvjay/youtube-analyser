"""Shared by the tests: build a clip, run the commands on it like a user would (in this process,
so the models load once and coverage sees everything), and look at the result.

run(action, names) runs `yta detect` on the clips into tests/.cache/v<N>/out (the detector's own
caches and "done" files make later runs instant). people(name, action) then labels everyone in
every sample from the same saved model output, so a test can check each person, not just the
sections.

DETECTOR_OPTIONS (environment) runs the end-to-end tests with other pose model settings, into a
cache folder of their own, to see what a faster setting costs:
    DETECTOR_OPTIONS="--imgsz 320" pytest tests/test_movement.py tests/test_talking.py tests/test_search.py
"""
import argparse
import csv
import functools
import os
import re
import shlex
from pathlib import Path

import numpy as np

from youtube_analyser import actions, analysis, audio, cli, faces, media, motion, query, search
from youtube_analyser import people as characteristics
from youtube_analyser.analysis import Settings, VideoAnalysis

from . import make_fixtures as mf

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OPTIONS = shlex.split(os.environ.get("DETECTOR_OPTIONS", ""))
OUT = mf.CACHE / ("out-" + re.sub(r"[^a-z0-9.]+", "-", " ".join(OPTIONS).lower()).strip("-") if OPTIONS else "out")
__all__ = ["actions", "analysis", "audio", "faces", "media", "motion", "characteristics", "query", "search", "mf",
           "REPO", "OUT", "OPTIONS", "SETTINGS", "Captured", "yta", "run", "people", "sections", "csv_rows", "out_dir",
           "searcher", "csv_rows_text", "track_person", "overlap", "in_view", "steady"]


def _settings():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default=Settings().pose_model)
    ap.add_argument("--fps", type=float, default=1.0)
    known, _ = ap.parse_known_args(OPTIONS)
    return Settings(pose_model=known.model, imgsz=known.imgsz, fps=known.fps)


SETTINGS = _settings()


class Captured:
    """What a command printed (stdout / stderr) and its exit code."""

    def __init__(self):
        self.out, self.err, self.code = [], [], None

    def __call__(self, *args, file=None, flush=None, sep=" ", end="\n"):
        (self.err if file is not None else self.out).append(sep.join(str(a) for a in args) + end)

    @property
    def stdout(self):
        return "".join(self.out)

    @property
    def stderr(self):
        return "".join(self.err)


def yta(*argv):
    """Run a `yta` command in this process; returns a Captured."""
    c = Captured()
    c.code = cli.main([str(a) for a in argv], log=c)
    return c


def run(action, names, *options):
    """`yta detect` for an action on these clips (building them first). Clips already done for
    this action are skipped by the detector itself."""
    clips = [mf.build(n) for n in names]
    r = yta("detect", *clips, "--out", OUT, "--keep-intro", "--action", action, *OPTIONS, *options)
    if r.code or "FAILED" in r.stderr:
        raise AssertionError(f"yta detect failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    return r


def out_dir(action):
    return OUT if action == "standing" else OUT / action


def sections(name, action, kind="segments"):
    """The sections the detector wrote for one clip (segments, intro or dropped)."""
    return csv_rows(out_dir(action) / kind / f"{name}.csv")


def csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def csv_rows_text(text):
    return list(csv.DictReader((text or "").splitlines()))


@functools.cache
def people(name, action, motion_filter=None, no_sound=False):
    """(timeline, people per sampled frame, poses, inputs) for a clip, labelled for the action
    from the model output the detector saved (so this runs the detector first if needed)."""
    run(action, [name], *(["--motion", motion_filter] if motion_filter else []), *(["--no-sound"] if no_sound else []))
    va = VideoAnalysis(mf.path(name), SETTINGS, cache=OUT)
    extra = va.inputs(action, motion_filter, not no_sound)
    timeline, by_frame = actions.label_people(va.poses, action, extra)
    return timeline, by_frame, va.poses, extra


def searcher(config=None, **kw):
    """A Searcher on the tests' model output folder."""
    return search.Searcher(config or characteristics.load_config(), SETTINGS, OUT, **kw)


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
    hidden = motion.occluded(timeline) | motion.at_edge(timeline)  # not measured on purpose
    out = []
    for t, frame in zip(timeline["t"].tolist(), by_frame):
        when = media.picture_time(t, fps)
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
