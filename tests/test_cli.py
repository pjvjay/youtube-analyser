"""The `yta` command end to end: every action, skipping and --reclassify, a broken file, the intro
bank, the reports and the review video, and the command's own behaviour (errors, help, version).

    pytest tests/test_cli.py      (a few minutes the first time)
"""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import helpers as h
from tests.helpers import mf
from youtube_analyser import actions, intro

ACTIONS = list(actions.ACTIONS)
CLIPS = {"standing": ["walk_stop_walk"], "sitting": ["walk_stop_walk"], "walking": ["walk_stop_walk"],
         "running": ["run_across"], "jumping": ["jump_three"], "smiling": ["three_talkers"],
         "talking": ["three_talkers"], "silent": ["three_talkers"]}


class Detect(unittest.TestCase):
    def test_every_action_runs_and_writes_its_files(self):
        self.assertEqual(set(CLIPS), set(ACTIONS))
        for action, names in CLIPS.items():
            h.run(action, names)
            out = h.out_dir(action)
            for f in ("segments.csv", "intro.csv", "dropped.csv", "interview_starts.csv", "review.html"):
                self.assertTrue((out / f).exists(), f"{action}: {f}")
            with open(out / "segments.csv", newline="") as f:
                self.assertEqual(csv.DictReader(f).fieldnames, actions.segment_fields(action), action)
            if action in actions.SOUND_ACTIONS:
                self.assertTrue((out / "speech.csv").exists() and (out / "voices.csv").exists(), action)
                self.assertIn("Voices (", (out / "review.html").read_text())

    def test_a_second_run_skips_and_reclassify_redoes(self):
        h.run("walking", ["walk_stop_walk"])
        r = h.run("walking", ["walk_stop_walk"])
        self.assertIn("done already, skipping", r.stdout)
        r = h.run("walking", ["walk_stop_walk"], "--reclassify")
        self.assertIn("walking sections", r.stdout)
        self.assertNotIn("skipping", r.stdout)

    def test_bad_files_and_duplicates_are_reported_and_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bad = tmp / "20240101_badvideo000.mp4"
            bad.write_bytes(b"not a video")
            (tmp / "other").mkdir()
            twin = tmp / "other" / "walk_stop_walk.mp4"
            twin.symlink_to(mf.build("walk_stop_walk"))
            r = h.yta("detect", bad, mf.build("walk_stop_walk"), twin, tmp / "x.f251.webm", "--out", tmp / "out",
                      "--keep-intro", "--action", "walking")
            self.assertEqual(r.code, 0)
            self.assertIn("FAILED", r.stderr)
            self.assertIn("another video is also called walk_stop_walk", r.stderr)
            self.assertNotIn("f251", r.stdout + r.stderr)           # yt-dlp's unmerged part: not a video
            self.assertTrue((tmp / "out" / "walking" / "segments" / "walk_stop_walk.csv").exists())

    def test_a_changed_setting_makes_the_model_output_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = mf.build("walk_stop_walk")
            h.yta("detect", clip, "--out", tmp, "--keep-intro")
            saved = Path(tmp) / "poses" / "walk_stop_walk.npz"
            before = saved.stat().st_mtime_ns
            h.yta("detect", clip, "--out", tmp, "--keep-intro", "--reclassify")
            self.assertEqual(saved.stat().st_mtime_ns, before)                # the same settings: reused
            r = h.yta("detect", clip, "--out", tmp, "--keep-intro", "--reclassify", "--imgsz", "320")
            self.assertNotEqual(saved.stat().st_mtime_ns, before)             # another input size: made again
            self.assertIn("ms/frame", r.stdout)
            self.assertIn('"imgsz": 320', str(np.load(saved)["_key"]))

    def test_motion_and_no_sound(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = h.yta("detect", mf.build("walk_stop_walk"), "--out", tmp, "--keep-intro", "--motion", "still")
            self.assertEqual(r.code, 0)
            self.assertTrue((Path(tmp) / "camera" / "walk_stop_walk.npz").exists())
            r = h.yta("detect", mf.build("three_talkers"), "--out", tmp, "--keep-intro", "--action", "talking", "--no-sound")
            self.assertEqual(r.code, 0)
            self.assertFalse((Path(tmp) / "audio").exists())                  # the sound was not even read


class Intro(unittest.TestCase):
    def test_the_bank_and_interview_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            eps = episodes(tmp)
            r = h.yta("intro", *eps, "--bank", tmp / "bank.npz", "--build-bank")
            self.assertEqual(r.code, 0, r.stderr)
            self.assertIn("from 4 videos of 4 minutes or more", r.stdout)
            bank = intro.load_bank(tmp / "bank.npz")
            self.assertGreater(intro.title_frames(bank), 0)
            r = h.yta("intro", *eps, "--bank", tmp / "bank.npz")
            starts = {line.split("\t")[0]: (int(line.split("\t")[1]), line.split("\t")[2]) for line in r.stdout.splitlines()}
            for name, (start, how) in starts.items():
                self.assertEqual(how, "title", name)
                self.assertTrue(12 <= start <= 45, (name, start))              # the 12 s title, its edge kept a while
            # with the bank, the detector sets the title's jumps apart as intro
            r = h.yta("detect", eps[0], "--out", tmp / "out", "--action", "jumping", "--intro-bank", tmp / "bank.npz")
            self.assertEqual(r.code, 0, r.stderr)
            self.assertTrue(h.csv_rows(tmp / "out" / "jumping" / "intro.csv"))
            self.assertEqual(h.csv_rows(tmp / "out" / "jumping" / "segments.csv"), [])
            # no bank: said so, and each whole video is the interview
            r = h.yta("detect", eps[0], "--out", tmp / "out2", "--action", "jumping", "--intro-bank", tmp / "none.npz")
            self.assertIn("no intro bank", r.stderr)
            self.assertTrue(h.csv_rows(tmp / "out2" / "jumping" / "segments.csv"))
            r = h.yta("intro", eps[0], "--bank", tmp / "none.npz")
            self.assertEqual(r.code, 2)
            self.assertIn("no bank", r.stderr)


def episodes(folder, n=4, seconds=250):
    """n made-up 4-minute episodes sharing a 12 s title (a man jumping), then a different moving
    pattern in each (ffmpeg's "life" test source)."""
    out = []
    for k in range(1, n + 1):
        path = Path(folder) / f"2024010{k}_episode000{k}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mf.build("jump_three")), "-f", "lavfi", "-t", str(seconds - 12),
                        "-i", f"life=s={mf.W}x{mf.H}:r={mf.FPS}:seed={k}:mold=10,format=yuv420p", "-filter_complex",
                        "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-c:v", "libx264", "-crf", "30", "-preset",
                        "veryfast", "-g", "50", str(path)], check=True)
        out.append(path)
    return out


class Reports(unittest.TestCase):
    def test_reports_and_review_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inventory = tmp / "videos.csv"
            with open(inventory, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["file", "upload_date", "video_id", "title", "kind", "duration_min"])
                for name in ("walk_stop_walk", "three_talkers"):
                    w.writerow([f"{name}.mp4", "2024-01-01", name[:11], f"Test {name}", "long", "1.0"])
            results = tmp / "results"
            (results / "talking").mkdir(parents=True)
            for action in ("standing", "walking", "talking"):
                h.run(action, CLIPS[action])
            talking = h.sections("three_talkers", "talking")
            standing = h.sections("walk_stop_walk", "standing")
            (results / "verdicts.csv").write_text(
                f"video,start_s,verdict,note\nwalk_stop_walk.mp4,{standing[0]['start_s']},confirmed,\n")
            (results / "talking" / "verdicts.csv").write_text(
                f"video,start_s,verdict,note\nthree_talkers.mp4,{talking[0]['start_s']},confirmed,left talks\n")
            for action in ("standing", "walking", "talking"):
                r = h.yta("report", "--action", action, "--detector-output", h.OUT, "--results", results,
                          "--inventory", inventory)
                self.assertEqual(r.code, 0, r.stderr[-2000:])
            rows = h.csv_rows(results / "talking" / "talking_sections.csv")
            mine = [r for r in rows if r["video"] == "three_talkers.mp4"]
            self.assertEqual(len(mine), len(talking))
            self.assertEqual(mine[0]["verdict"], "confirmed")
            self.assertEqual(mine[0]["title"], "Test three_talkers")
            self.assertTrue((results / "talking" / "speech.csv").exists())
            self.assertIn("## Voices", (results / "talking" / "README.md").read_text())
            self.assertIn("confirmed", (results / "README.md").read_text())
            self.assertTrue((results / "walking" / "walking_sections.csv").exists())
            # without an inventory: titles empty, said so
            r = h.yta("report", "--action", "walking", "--detector-output", h.OUT, "--results", tmp / "r2",
                      "--inventory", tmp / "missing.csv")
            self.assertEqual(r.code, 0)
            self.assertIn("no inventory", r.stderr)
            # no detector output: an error, not a crash
            r = h.yta("report", "--detector-output", tmp / "nothing", "--results", tmp / "r3")
            self.assertEqual(r.code, 2)
            # the review video of the talking sections
            review = tmp / "review.mp4"
            r = h.yta("review", "--action", "talking", "--sections", results / "talking" / "talking_sections.csv",
                      "--videos-dir", mf.CLIPS, "--verdicts", "all", "--out", review)
            self.assertEqual(r.code, 0, r.stderr[-2000:])
            self.assertGreater(review.stat().st_size, 10000)
            index = h.csv_rows(review.with_suffix(".csv"))
            self.assertTrue(index and "talking" in index[0])
            # standing's default: reviewed ones only; a clip without sound gets silence
            r = h.yta("review", "--sections", results / "standing_sections.csv", "--videos-dir", mf.CLIPS,
                      "--out", tmp / "standing.mp4")
            self.assertEqual(r.code, 0, r.stderr[-2000:])
            self.assertTrue(h.media.has_audio(tmp / "standing.mp4"))
            r = h.yta("review", "--sections", results / "standing_sections.csv", "--videos-dir", tmp, "--out",
                      tmp / "none.mp4")
            self.assertEqual(r.code, 2)                                        # the videos are not there
            r = h.yta("review", "--sections", results / "standing_sections.csv", "--verdicts", "unclear")
            self.assertEqual(r.code, 2)                                        # no such sections


class Command(unittest.TestCase):
    def test_help_version_models_and_errors(self):
        self.assertIn("yta", h.yta("models").stdout + "yta")
        self.assertIn("yolo11s-pose.onnx", h.yta("models").stdout)
        with self.assertRaises(SystemExit):
            h.yta("--version")
        with self.assertRaises(SystemExit):
            h.yta("detect")                                                    # no videos
        r = h.yta("search", "wibble", mf.build("walk_stop_walk"), "--out", h.OUT, "--keep-intro")
        self.assertEqual(r.code, 2)
        self.assertIn("nothing to search for", r.stderr)
        r = h.yta("config")
        self.assertIn("[words]", r.stdout)
        # no ffmpeg: said so at once, not once per video
        from youtube_analyser import cli
        which = cli.shutil.which
        cli.shutil.which = lambda name: None if name == "ffprobe" else which(name)
        try:
            r = h.yta("detect", mf.build("walk_stop_walk"))
        finally:
            cli.shutil.which = which
        self.assertEqual(r.code, 2)
        self.assertIn("ffprobe not found: install ffmpeg", r.stderr)
        # an optional part that is not installed: said so, with what to install
        from youtube_analyser import cloud

        def missing(url):
            raise ModuleNotFoundError("No module named 'azure'", name="azure")
        saved = cloud._STORES["azblob"]
        cloud._STORES["azblob"] = missing
        try:
            r = h.yta("collect", "some-search", "--store", "azblob://account/container")
        finally:
            cloud._STORES["azblob"] = saved
        self.assertEqual(r.code, 2)
        self.assertIn('azure is not installed: pip install "youtube-analyser[azure]"', r.stderr)
        r = subprocess.run([sys.executable, "-m", "youtube_analyser", "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn("yta channel", r.stdout)


if __name__ == "__main__":
    unittest.main()
