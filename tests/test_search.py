"""Searching for a described person doing an action, end to end (`yta search`: search.py, people.py, query.py).

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

    pytest tests/test_search.py      (first run about 2 minutes)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import helpers as h
from tests.helpers import characteristics as people
from tests.helpers import mf, search
from tests.helpers import query as queries

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
    state["searcher"] = h.searcher(state["config"], videos=videos)


def tearDownModule():
    state["tmp"].cleanup()


def find(query, clips=CHARACTER_CLIPS):
    """The search's rows for a query, each with the seat of the person found (left, right, centre)."""
    req = queries.parse(query, state["config"], ["Jane"])
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
                when = h.media.picture_time(r["best_t"], 1.0)
                self.assertGreater(h.overlap(mf.box(view, persons[colour], when), (b[0], b[1], b[0] + b[2], b[1] + b[3])),
                                   0.3, (colour, r))

    def test_a_described_person_jumping(self):
        """Jumps are judged in the 5-fps samples: each is put down to the person it belongs to."""
        h.run("jumping", ["jump_three"])
        want = [(float(r["start_s"]), float(r["end_s"])) for r in h.sections("jump_three", "jumping")]
        rows = find("someone in a beige coat jumping", ["jump_three"])
        self.assertEqual([(r["start_s"], r["end_s"]) for r in rows], want)
        self.assertEqual(len({r["person"] for r in rows}), 1)
        self.assertEqual(find("someone in a blue coat jumping", ["jump_three"]), [])

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
            r = h.yta("search", "Jane talking", *clips, "--out", h.OUT, "--results", out, "--keep-intro", "--config",
                      state["config_file"], *h.OPTIONS)
            self.assertEqual(r.code, 0, r.stderr)
            self.assertIn("looking for: name Jane; action: talking", r.stdout)
            folder = Path(out) / "jane-talking"
            rows = h.csv_rows(folder / "matches.csv")
            want = find("Jane talking")
            self.assertEqual(sorted((x["video"], float(x["start_s"]), float(x["end_s"])) for x in rows),
                             sorted((x["video"], x["start_s"], x["end_s"]) for x in want))
            self.assertTrue(all((folder / "thumbs" / x["thumb"]).exists() for x in rows))
            self.assertIn("Jane talking", (folder / "index.html").read_text())

            r = h.yta("search", "woman with purple hair dancing", clips[0], "--out", h.OUT, "--results", out,
                      "--keep-intro", "--explain", *h.OPTIONS)
            self.assertEqual(r.code, 0, r.stderr)
            self.assertIn("ignored words: dancing", r.stdout)
            self.assertIn("duo_purple.mp4 person", r.stdout)          # --explain: everyone seen

    def test_what_it_adds_is_saved(self):
        """A search saves what it measures (appearance/), and measures only what a query needs."""
        find("woman with purple hair talking", ["duo_purple"])   # everything else is saved already
        with tempfile.TemporaryDirectory() as tmp:
            for part in ("poses", "faces", "audio", "camera"):   # a cache with no appearance yet
                (Path(tmp) / part).mkdir()
                (Path(tmp) / part / "duo_purple.npz").write_bytes((h.OUT / part / "duo_purple.npz").read_bytes())
            s = search.Searcher(state["config"], h.SETTINGS, Path(tmp))
            saved = Path(tmp) / "appearance" / "duo_purple.npz"
            req = lambda q: queries.parse(q, state["config"])
            s.search(mf.path("duo_purple"), req("woman with purple hair"))
            before = saved.stat().st_mtime_ns
            s.search(mf.path("duo_purple"), req("woman with purple hair talking"))   # nothing new to measure
            self.assertEqual(saved.stat().st_mtime_ns, before)
            # her gender came from her voice, so the face model has not run; asking for an age runs
            # it, on the clearest faces of the man only (the one with black hair), who looks about 38
            self.assertEqual(int(np.load(saved)["faced"].sum()), 0)
            self.assertEqual(s.search(mf.path("duo_purple"), req("old man with black hair")), [])
            self.assertEqual(int(np.load(saved)["faced"].sum()), state["config"]["look"]["faces_per_person"])


class Cheapest(unittest.TestCase):
    def test_nobody_with_the_colour_means_no_more_work(self):
        with tempfile.TemporaryDirectory() as out:
            s = search.Searcher(state["config"], h.SETTINGS, Path(out))
            req = queries.parse("woman with green hair talking", state["config"])
            self.assertEqual(s.search(mf.path("duo_purple"), req, "talking"), [])
            self.assertEqual(s.note, "nobody looks right: no green hair seen")
            self.assertEqual(sorted(p.name for p in Path(out).iterdir()), ["appearance", "poses"])   # no faces, sound, camera

    def test_the_quick_look(self):
        s, req = state["searcher"], queries.parse("woman with purple hair talking", state["config"])
        self.assertTrue(s.scout(mf.path("duo_purple"), req))
        self.assertFalse(s.scout(mf.path("duo_natural"), req))
        self.assertIn("no purple hair seen", s.note)
        self.assertTrue(s.scout(mf.path("duo_natural"), queries.parse("man talking", state["config"])))  # no colour: nothing to look for

    def test_anyone_doing_it_is_the_detectors_sections(self):
        h.run("jumping", ["jump_three"])
        want = [(float(r["start_s"]), float(r["end_s"])) for r in h.sections("jump_three", "jumping")]
        rows = find("people jumping", ["jump_three"])
        self.assertEqual([(r["start_s"], r["end_s"]) for r in rows], want)
        self.assertTrue(want)
        self.assertTrue(all(r["best_boxes"] for r in rows))


if __name__ == "__main__":
    unittest.main()
