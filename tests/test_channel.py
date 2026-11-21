"""Searching a whole channel, end to end (standing/channel.py), with a stand-in for yt-dlp.

The "channel" (fake_yt_dlp.py) has five videos, newest first: duo_purple, duo_natural, solo_blue
(see test_search.py), two_cross (two men crossing a street) and jump_three (a man jumping three
times). "Downloading" copies the clip, without its sound when the format asks for video only.
YouTube itself is not reached: what is tested is everything around it (listing, formats, dates,
resuming, failures, the report and its links).

    .venv/bin/python -m unittest tests.test_channel -v      (about 3 minutes)
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import helpers as h
from tests.helpers import mf

CHANNEL = [("duoPurple01", "Guest with purple hair | EP 3", "20240301", "duo_purple"),
           ("duoNatural1", "Another guest | EP 2", "20240215", "duo_natural"),
           ("soloBlue001", "A monologue", "20240201", "solo_blue"),
           ("twoCross001", "Street scene", "20240115", "two_cross"),
           ("jumpThree01", "Jumping about | EP 1", "20240101", "jump_three")]
URL = "https://www.youtube.com/@FakeShow"
state = {}


def setUpModule():
    state["tmp"] = tempfile.TemporaryDirectory()
    tmp = Path(state["tmp"].name)
    videos = [{"id": i, "title": t, "upload_date": d, "clip": str(mf.build(c))} for i, t, d, c in CHANNEL]
    (tmp / "channel.json").write_text(json.dumps({"videos": videos}))
    state["env"] = dict(os.environ, FAKE_CHANNEL=str(tmp / "channel.json"), FAKE_LOG=str(tmp / "calls.log"))


def tearDownModule():
    state["tmp"].cleanup()


def channel(query, work, *options, source=URL, fail=""):
    """Run channel.py; returns (stdout, the search folder, the yt-dlp calls made)."""
    log = Path(state["env"]["FAKE_LOG"])
    log.unlink(missing_ok=True)
    cmd = [sys.executable, str(h.REPO / "standing" / "channel.py"), source, query, "--work", str(work),
           "--yt-dlp", f"{sys.executable} {h.REPO / 'tests' / 'fake_yt_dlp.py'}", *options]
    if "--speed" not in options and "--imgsz" in " ".join(h.OPTIONS):
        cmd += ["--speed", "fastest"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(state["env"], FAKE_FAIL=fail))
    if r.returncode:
        raise AssertionError(f"channel.py failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    folder = Path(work) / "searches" / h.search.slug(query + (" fastest" if "fastest" in cmd else ""))
    calls = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    return r.stdout, folder, calls


def matches(folder):
    with open(folder / "matches.csv", newline="") as f:
        return list(csv.DictReader(f))


def downloads(calls):
    return [c for c in calls if "--flat-playlist" not in c]


class PersonSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(state["tmp"].name) / "person"
        cls.out, cls.folder, cls.calls = channel("woman with purple hair talking", cls.work)

    def test_the_report_links_to_each_moment(self):
        rows = matches(self.folder)
        self.assertEqual({r["video_id"] for r in rows}, {"duoPurple01"})
        self.assertEqual([r["youtube"] for r in rows], ["https://www.youtube.com/watch?v=duoPurple01&t=0s",
                                                        "https://www.youtube.com/watch?v=duoPurple01&t=20s"])
        self.assertEqual({r["title"] for r in rows}, {"Guest with purple hair | EP 3"})
        md = (self.folder / "report.md").read_text()
        self.assertIn("## 2024-03-01 Guest with purple hair | EP 3", md)
        self.assertIn("(https://www.youtube.com/watch?v=duoPurple01&t=20s)", md)
        page = (self.folder / "report.html").read_text()
        self.assertIn("watch?v=duoPurple01&amp;t=0s", page)
        self.assertTrue(all((self.folder / "thumbs" / r["thumb"]).exists() for r in rows))
        self.assertIn("4 videos with nothing found", page)

    def test_the_others_end_at_the_quick_look(self):
        for vid in ("duoNatural1", "soloBlue001", "twoCross001", "jumpThree01"):
            done = json.loads((self.folder / "done" / f"{vid}.json").read_text())
            self.assertEqual(done["rows"], [])
            self.assertIn("quick look", done["note"])
            self.assertIn("no purple hair seen", done["note"])

    def test_lowest_resolution_with_sound_and_the_inventory(self):
        got = downloads(self.calls)
        self.assertEqual(len(got), 5)
        for c in got:
            self.assertIn("worstvideo[height>=144]+worstaudio", c[c.index("-f") + 1])
        self.assertEqual(self.calls[0][-1], URL + "/videos")     # the channel's uploads
        with open(self.work / "videos.csv", newline="") as f:
            inv = {r["video_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(inv["duoPurple01"]["upload_date"], "2024-03-01")
        self.assertEqual(inv["jumpThree01"]["title"], "Jumping about | EP 1")

    def test_a_second_run_carries_on_and_a_new_query_reuses_the_videos(self):
        out, _, calls = channel("woman with purple hair talking", self.work)
        self.assertIn("5 already searched; 0 to go", out)
        self.assertEqual(downloads(calls), [])
        out, folder, calls = channel("woman with blue hair", self.work)
        self.assertEqual(downloads(calls), [])                   # already downloaded, with sound
        self.assertEqual({r["video_id"] for r in matches(folder)}, {"soloBlue001"})


class ActionSearch(unittest.TestCase):
    def test_anyone_jumping_video_only_then_talking_with_sound(self):
        work = Path(state["tmp"].name) / "action"
        out, folder, calls = channel("people jumping", work)
        got = downloads(calls)
        self.assertTrue(got and all("worstaudio" not in c[c.index("-f") + 1] for c in got))   # no sound needed
        video = next((work / "videos").glob("*_jumpThree01.mp4"))
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                                str(video)], capture_output=True, text=True).stdout.split()
        self.assertEqual(probe, ["video"])
        rows = matches(folder)
        self.assertEqual({r["video_id"] for r in rows}, {"jumpThree01"})
        h.run("jumping", ["jump_three"])
        want = [(float(r["start_s"]), float(r["end_s"])) for r in h.sections("jump_three", "jumping")]
        self.assertEqual([(float(r["start_s"]), float(r["end_s"])) for r in rows], want)
        # talking needs the sound: the silent copies are downloaded again, with it
        out, folder, calls = channel("man talking", work)
        self.assertEqual(len(downloads(calls)), 5)
        self.assertEqual({r["video_id"] for r in matches(folder)}, {"duoPurple01", "duoNatural1"})


class Options(unittest.TestCase):
    def test_newest_dates_failures_deleting_and_a_folder(self):
        work = Path(state["tmp"].name) / "options"
        out, folder, calls = channel("people jumping", work, "--max-videos", "2")
        self.assertEqual(len(downloads(calls)), 2)
        self.assertEqual(sorted(p.stem for p in (folder / "done").iterdir()), ["duoNatural1", "duoPurple01"])
        # dates, a failing download, and deleting the videos after searching them
        work = Path(state["tmp"].name) / "options2"
        out, folder, calls = channel("someone in a blue coat", work, "--since", "20240115", "--delete-videos",
                                     fail="soloBlue001")
        self.assertIn("outside the dates asked for", out)
        self.assertIn("soloBlue001", (folder / "report.html").read_text())        # listed under problems
        self.assertEqual(sorted(p.stem for p in (folder / "done").iterdir()), ["duoNatural1", "duoPurple01", "twoCross001"])
        self.assertEqual(list((work / "videos").iterdir()), [])
        rows = matches(folder)
        self.assertEqual({r["video_id"] for r in rows}, {"twoCross001"})
        self.assertTrue(all((folder / "thumbs" / r["thumb"]).exists() for r in rows))
        # videos already on disk
        mine = Path(state["tmp"].name) / "mine"
        mine.mkdir()
        for name in ("jump_three", "two_cross"):
            (mine / f"{name}.mp4").symlink_to(mf.path(name))
        out, folder, calls = channel("people jumping", Path(state["tmp"].name) / "folder", source=str(mine))
        self.assertEqual(calls, [])
        self.assertEqual({r["video"] for r in matches(folder)}, {"jump_three.mp4"})


class Intros(unittest.TestCase):
    def test_the_shared_title_sequence_is_found_and_its_results_set_apart(self):
        """Five 4-minute "episodes" open with the same title sequence (a man jumping three times);
        after it each has a different moving pattern and nobody, except the first, which has two
        men jumping near its end. The bank of title frames is built from the episodes themselves."""
        eps = Path(state["tmp"].name) / "episodes"
        eps.mkdir()
        for k in range(1, 6):
            parts = [["-i", str(mf.build("jump_three"))],
                     ["-f", "lavfi", "-t", "200" if k == 1 else "240", "-i",
                      f"life=s={mf.W}x{mf.H}:r={mf.FPS}:seed={k}:mold=10,format=yuv420p"]]
            if k == 1:
                parts += [["-i", str(mf.build("two_jumpers"))],
                          ["-f", "lavfi", "-t", "40", "-i", f"life=s={mf.W}x{mf.H}:r={mf.FPS}:seed=99,format=yuv420p"]]
            inputs = [x for p in parts for x in p]
            chain = "".join(f"[{i}:v]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[v]"
            subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", chain, "-map", "[v]", "-c:v",
                            "libx264", "-crf", "30", "-preset", "veryfast", "-g", "50", str(eps / f"2024010{k}_episode000{k}.mp4")],
                           check=True)
        out, folder, calls = channel("people jumping", Path(state["tmp"].name) / "episodes-work", "--speed", "fastest",
                                     source=str(eps))
        self.assertIn("intro bank built from 5 videos", out)
        rows = matches(folder)
        main = [r for r in rows if r["in_intro"] == "False"]
        self.assertEqual({r["video_id"] for r in main}, {"episode0001"})
        self.assertTrue(all(float(r["start_s"]) > 200 for r in main))            # the two men near the end
        intro_rows = [r for r in rows if r["in_intro"] == "True"]
        self.assertEqual(len({r["video_id"] for r in intro_rows}), 5)             # the title's jumps, in every episode
        self.assertTrue(all(float(r["start_s"]) < 12 for r in intro_rows))
        md = (folder / "report.md").read_text()
        self.assertLess(md.index("episode0001&t=2"), md.index("# In the intros"))


if __name__ == "__main__":
    unittest.main()
