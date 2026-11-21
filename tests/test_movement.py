"""Walking, running and jumping, end to end: the detector runs on made clips whose truth is known.

The clips (see make_fixtures.py) cover: walking, stopping and walking back; standing still; a
camera panning past someone standing; a camera following a walker; a camera zooming in on
someone standing; walking towards the camera; running across a still camera and with the camera
following; a brisk walk (not a run); two people crossing; someone walking past a bystander who is
hidden for a moment; a cut; three jumps; tiny hops; the camera bobbing up and down; stepping up
onto something; two people jumping (once together); a jump right beside someone standing; a jump
while walking.

For every person in every sample where they are fully in view and their truth is steady (the
same for a second either side, since speeds are measured over time), the detector's verdict
must be the truth or "unknown" (it may decline, never be wrong), and each person must get a
verdict often enough. Sections must be where the truth says, and nowhere else.

    .venv/bin/python -m unittest tests.test_movement -v      (first run about 5 minutes)
"""
import unittest

from tests import helpers as h
from tests.helpers import mf

CLIPS = list(mf.MOVEMENT)
JUMP_CLIPS = ["jump_three", "tiny_hops", "camera_bob", "step_up", "two_jumpers", "jump_beside_someone",
              "jump_while_walking"]
WALK_CLIPS = [c for c in CLIPS if c not in JUMP_CLIPS] + ["jump_while_walking"]


def setUpModule():
    for action in ("walking", "running", "jumping"):
        h.run(action, CLIPS)


def truth_interval(name, who, value, step=0.05):
    """(start, end) pairs of picture time when a person's truth is value."""
    seconds, _, _, truth = mf.MOVEMENT[name]
    out, start = [], None
    t = 0.0
    while t <= seconds:
        on = truth[who](t) == value
        if on and start is None:
            start = t
        if not on and start is not None:
            out.append((start, t))
            start = None
        t += step
    if start is not None:
        out.append((start, seconds))
    return out


def overlaps(section, interval, slack):
    return float(section["start_s"]) - slack < interval[1] and interval[0] < float(section["end_s"]) + slack


class Walking(unittest.TestCase):
    """--action walking: people judged between 1-fps samples."""

    def test_every_verdict_is_right_or_unknown(self):
        for name in WALK_CLIPS:
            for who, truth in mf.MOVEMENT[name][3].items():
                for t, expect, got, m in h.track_person(name, "walking", who):
                    if expect == "gone" or not h.steady(truth, t, 1.0) or got in (None, "unknown"):
                        continue
                    want = {"jumping": "walking", "ground": "still"}.get(expect, expect)
                    if name == "jump_while_walking":
                        want = "walking" if 3 < t < 7 else "still"
                    self.assertEqual(got, want, f"{name} {who} at {t:.2f}s (speed {m.get('speed')})")

    def test_people_get_verdicts(self):
        for name in WALK_CLIPS:
            for who, truth in mf.MOVEMENT[name][3].items():
                rows = h.track_person(name, "walking", who)
                for state in ("walking", "still"):
                    steady = [got for t, e, got, m in rows
                              if e == state and h.steady(truth, t, 1.0) and not m.get("hidden")]
                    if len(steady) >= 2:
                        self.assertGreaterEqual(sum(g == state for g in steady), len(steady) / 2,
                                                f"{name} {who}: {state} in {steady}")

    def test_runners_are_never_walking(self):
        for name in ("run_across", "follow_runner"):
            for t, expect, got, m in h.track_person(name, "walking", "man"):
                if expect == "running" and h.steady(mf.MOVEMENT[name][3]["man"], t, 1.0):
                    self.assertIn(got, ("running", "unknown", None), f"{name} at {t:.2f}s")

    def test_walking_sections(self):
        for name in WALK_CLIPS:
            sections = h.sections(name, "walking")
            walks = [iv for who in mf.MOVEMENT[name][3] for iv in truth_interval(name, who, "walking")]
            if name == "jump_while_walking":
                walks = [(3, 7)]
            if name == "towards_camera":   # too short at 1 fps for a 2-sample section (see README)
                self.assertFalse([s for s in sections if not any(overlaps(s, iv, 1.0) for iv in walks)])
                continue
            for iv in walks:
                if iv[1] - iv[0] >= 2.5:
                    self.assertTrue(any(overlaps(s, iv, 1.0) for s in sections), f"{name}: no section for {iv}")
            for s in sections:
                self.assertTrue(any(overlaps(s, iv, 1.0) for iv in walks), f"{name}: section {s['start']}-{s['end']}")

    def test_standing_still_leaves_out_the_walker(self):
        _, by_any, _, _ = h.people("walk_stop_walk", "standing")
        _, by_still, _, _ = h.people("walk_stop_walk", "standing", motion="still")
        walking = [f for f, people in enumerate(by_still) if any(p["state"] == "walking" for p in people)]
        self.assertTrue(walking)
        for f in walking:
            self.assertLess(sum(p["state"] == "standing" for p in by_still[f]),
                            sum(p["state"] == "standing" for p in by_any[f]))


class Running(unittest.TestCase):
    """--action running: people judged in the 5-fps samples around whole bodies."""

    def test_running_only_when_running(self):
        for name in CLIPS:
            for who, truth in mf.MOVEMENT[name][3].items():
                for t, expect, got, m in h.track_person(name, "running", who):
                    if expect == "gone" or not h.steady(truth, t, 1.0) or got in (None, "unknown"):
                        continue
                    self.assertEqual(got == "running", expect == "running",
                                     f"{name} {who} at {t:.2f}s: {got}, truth {expect} (speed {m.get('speed')})")

    def test_runners_are_found(self):
        for name in ("run_across", "follow_runner"):
            rows = [got for t, e, got, _ in h.track_person(name, "running", "man") if e == "running"]
            self.assertGreaterEqual(rows.count("running"), len(rows) / 2, f"{name}: {rows}")

    def test_running_sections(self):
        for name in CLIPS:
            sections = h.sections(name, "running")
            runs = truth_interval(name, "man", "running") if "man" in mf.MOVEMENT[name][2] else []
            if runs:
                self.assertEqual(len(sections), 1, f"{name}: {sections}")
                self.assertTrue(overlaps(sections[0], runs[0], 0.5))
                self.assertGreaterEqual(float(sections[0]["best_speed"]), 4.0)
            else:
                self.assertEqual(sections, [], name)


class Jumping(unittest.TestCase):
    """--action jumping: a body that rises and lands, in the 5-fps samples."""

    EXPECTED = {"jump_three": [3, 6.5, 10], "two_jumpers": [3, 5.5, 8], "jump_beside_someone": [3, 7],
                "jump_while_walking": [5]}

    def test_each_jump_once_and_only_jumps(self):
        for name in CLIPS:
            sections = h.sections(name, "jumping")
            jumps = self.EXPECTED.get(name, [])
            self.assertEqual(len(sections), len(jumps), f"{name}: {[(s['start_s'], s['end_s']) for s in sections]}")
            for s, j in zip(sections, jumps):
                self.assertLessEqual(abs(float(s["start_s"]) - j), 0.5, f"{name}: jump at {j}, section {s['start_s']}")
                self.assertGreaterEqual(float(s["best_rise"]), 0.25)

    def test_both_jumpers_are_found_jumping(self):
        for who, times in (("beige", [3, 8]), ("blue", [5.5, 8])):
            rows = h.track_person("two_jumpers", "jumping", who)
            for j in times:
                self.assertTrue(any(got == "jumping" for t, e, got, _ in rows if j <= t <= j + 0.45), f"{who} at {j}")
            for t, e, got, _ in rows:
                if got == "jumping":
                    self.assertTrue(any(j <= t <= j + 0.45 for j in times), f"{who} jumping at {t:.2f}s")

    def test_a_jump_beside_someone(self):
        rows = h.track_person("jump_beside_someone", "jumping", "jumper")
        for j in (3, 7):
            self.assertTrue(any(got == "jumping" for t, e, got, _ in rows if j <= t <= j + 0.45), f"jump at {j}")
        for t, e, got, _ in h.track_person("jump_beside_someone", "jumping", "neighbour"):
            self.assertNotEqual(got, "jumping", f"the neighbour at {t:.2f}s")

    def test_runners_and_walkers_are_not_jumping(self):
        for name in ("run_across", "follow_runner", "walk_stop_walk", "brisk_walk", "two_cross"):
            self.assertEqual(h.sections(name, "jumping"), [], name)


if __name__ == "__main__":
    unittest.main()
