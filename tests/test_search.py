"""Searching for a described person doing an action, end to end (standing/search.py, people.py).

The character clips (see make_fixtures.CHARACTERS) put a woman (the astronaut portrait, her hair
dyed purple or blue, or left natural) and a man (the carphone talker) side by side, taking turns
to talk with real voices:

    duo_purple   left: woman, purple hair, voice c ("Jane")   right: man, voice b
    duo_natural  left: woman, natural hair, voice a            right: man, voice d
                 both: 0-8 s left talks, 8-16 right, 16-20 nobody, 20-28 left, 28-36 right
    solo_blue    centre: woman, blue hair, voice c again (another part of her recording)
                 0-8 s talks, 8-12 silent, 12-20 talks, 20-24 silent

Jane is enrolled from duo_purple 0-8 s, so finding her in solo_blue is recognising her voice in
another video. The street clips two_cross and walk_past_bystander have a man in a beige coat and
one in a blue coat.

    .venv/bin/python -m unittest tests.test_search -v      (first run about 2 minutes)
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import helpers as h
from tests.helpers import characteristics as people, mf, search

CHARACTER_CLIPS = list(mf.CHARACTERS)
STREET_CLIPS = ["two_cross", "walk_past_bystander"]
JANE = '[[people]]\nname = "Jane"\nvoice_from = { video = "duo_purple.mp4", start = 0.0, end = 8.0 }\n' \
       'describe = "woman with purple hair"\n'
state = {}


def setUpModule():
    state["tmp"] = tempfile.TemporaryDirectory()
    config = Path(state["tmp"].name) / "people.toml"
    config.write_text(JANE)
    state["config_file"] = config
    state["config"] = people.load_config(config)
    videos = [mf.build(n) for n in CHARACTER_CLIPS + STREET_CLIPS]
    state["searcher"] = search.Searcher(search.options(h.OUT, keep_intro=True, videos=videos), state["config"])


def tearDownModule():
    state["tmp"].cleanup()


def find(query, clips=CHARACTER_CLIPS):
    """The search's rows for a query, each with the seat of the person found (left, right, centre)."""
    req = search.parse(query, state["config"], ["Jane"])
    out = []
    for name in clips:
        v = mf.path(name)
        for r in state["searcher"].search(v, req, req["action"]):
            if name in mf.CHARACTERS:
                box = np.load(h.OUT / "poses" / f"{v.stem}.npz")["box"][r["best_j"]]
                centre = box[0] + box[2] / 2
                r["seat"] = min(mf.SEAT_X, key=lambda k: abs(centre - (mf.SEAT_X[k] + mf.TILE / 2)))
            r["clip"] = name
            out.append(r)
    return out


def phases(name, seat):
    """The script's times when seat talks."""
    return [(a, b) for a, b, who in mf.CHARACTERS[name][2] if seat in who]


def overlap(a0, a1, spans):
    return sum(max(0.0, min(a1, b) - max(a0, a)) for a, b in spans)


class Talkers(unittest.TestCase):
    def check_talking(self, rows, expect):
        """rows cover at least 75% of each expected (clip, seat)'s talking time, and no row is
        more than 1.5 s outside it (a burst reaches 0.8 s past its sample)."""
        for clip, seat in expect:
            spans = phases(clip, seat)
            mine = [r for r in rows if (r["clip"], r["seat"]) == (clip, seat)]
            found = sum(overlap(r["start_s"], r["end_s"], spans) for r in mine)
            self.assertGreaterEqual(found, 0.75 * sum(b - a for a, b in spans), (clip, seat, mine))
        for r in rows:
            self.assertIn((r["clip"], r["seat"]), expect, r)
            outside = (r["end_s"] - r["start_s"]) - overlap(r["start_s"], r["end_s"], phases(r["clip"], r["seat"]))
            self.assertLessEqual(outside, 1.5, r)

    def test_woman_with_purple_hair_talking(self):
        rows = find("woman with purple hair talking")
        self.check_talking(rows, {("duo_purple", "left")})
        self.assertTrue(all(r["match"] == "match" for r in rows))
        self.assertIn("hair purple", rows[0]["evidence"])

    def test_men_and_women_talking(self):
        self.check_talking(find("man talking"), {("duo_purple", "right"), ("duo_natural", "right")})
        self.check_talking(find("woman talking"), {("duo_purple", "left"), ("duo_natural", "left"),
                                                   ("solo_blue", "centre")})

    def test_a_deep_voice(self):
        self.check_talking(find("someone with a deep voice talking"), {("duo_purple", "right"), ("duo_natural", "right")})

    def test_jane_by_her_voice_in_another_video(self):
        rows = find("Jane talking")
        self.check_talking(rows, {("duo_purple", "left"), ("solo_blue", "centre")})
        self.assertTrue(all("voice of Jane" in r["evidence"] for r in rows))


class Looks(unittest.TestCase):
    def test_in_view(self):
        rows = find("woman with purple hair")
        self.assertEqual([(r["clip"], r["seat"]) for r in rows], [("duo_purple", "left")])
        self.assertGreaterEqual(rows[0]["end_s"] - rows[0]["start_s"], 30)
        rows = find("woman with blue hair")
        self.assertEqual([(r["clip"], r["seat"]) for r in rows], [("solo_blue", "centre")])
        rows = find("woman with brown hair")
        self.assertEqual([(r["clip"], r["seat"]) for r in rows], [("duo_natural", "left")])

    def test_nobody(self):
        for q in ("man with purple hair", "someone with green hair", "woman with purple hair in a white shirt"):
            self.assertEqual(find(q), [], q)

    def test_coats_on_the_street(self):
        for colour in ("blue", "beige"):
            rows = find(f"person in a {colour} coat", ["two_cross"])
            self.assertEqual(len({r["person"] for r in rows}), 1, rows)
            seconds, view, persons, _ = mf.MOVEMENT["two_cross"]
            for r in rows:   # the boxed person is the one in that coat
                poses = np.load(h.OUT / "poses" / "two_cross.npz")
                b = poses["box"][r["best_j"]]
                when = h.actions.picture_time(r["best_t"], 1.0)
                self.assertGreater(h.overlap(mf.box(view, persons[colour], when), (b[0], b[1], b[0] + b[2], b[1] + b[3])),
                                   0.3, (colour, r))

    def test_walking_in_a_blue_coat(self):
        rows = find("man in a blue coat walking", STREET_CLIPS)
        # only in two_cross (in the other clip he is the bystander); the man's face is too small to
        # tell he is a man, so without --possible he is not listed ...
        self.assertEqual(rows, [])
        rows = find("someone in a blue coat walking", STREET_CLIPS)
        self.assertEqual({r["clip"] for r in rows}, {"two_cross"})
        for r in rows:   # he walks from 1 to 5 s
            self.assertGreaterEqual(r["start_s"], 0)
            self.assertLessEqual(r["end_s"], 6.5)
        self.assertGreaterEqual(sum(r["end_s"] - r["start_s"] for r in rows), 3)


class CommandLine(unittest.TestCase):
    def test_the_command_line_and_its_files(self):
        with tempfile.TemporaryDirectory() as out:
            clips = [str(mf.path(n)) for n in CHARACTER_CLIPS]
            r = subprocess.run([sys.executable, str(h.REPO / "standing" / "search.py"), "Jane talking", *clips,
                                "--out", str(h.OUT), "--results", out, "--keep-intro", "--config",
                                str(state["config_file"])], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("looking for: name Jane; action: talking", r.stdout)
            folder = Path(out) / "jane-talking"
            rows = h.csv_rows(folder / "matches.csv")
            want = find("Jane talking")
            self.assertEqual(sorted((x["video"], float(x["start_s"]), float(x["end_s"])) for x in rows),
                             sorted((x["video"], x["start_s"], x["end_s"]) for x in want))
            self.assertTrue(all((folder / "thumbs" / x["thumb"]).exists() for x in rows))
            self.assertIn("Jane talking", (folder / "index.html").read_text())

            r = subprocess.run([sys.executable, str(h.REPO / "standing" / "search.py"), "woman with purple hair dancing",
                                clips[0], "--out", str(h.OUT), "--results", out, "--keep-intro", "--explain"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("ignored words: dancing", r.stdout)
            self.assertIn("duo_purple.mp4 person", r.stdout)          # --explain: everyone seen

    def test_what_it_adds_is_saved(self):
        saved = h.OUT / "appearance" / "duo_purple.npz"
        find("woman with purple hair", ["duo_purple"])
        before = saved.stat().st_mtime
        find("man in a black top", ["duo_purple"])
        self.assertEqual(saved.stat().st_mtime, before)


if __name__ == "__main__":
    unittest.main()
