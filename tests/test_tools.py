"""The command line, the reports, the review video and the HTTP API, end to end.

    .venv/bin/python -m unittest tests.test_tools -v      (a few minutes the first time)
"""
import csv
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tests import helpers as h
from tests.helpers import REPO, ds, mf, search

ACTIONS = list(ds.ACTIONS)
CLIPS = {"standing": ["walk_stop_walk"], "sitting": ["walk_stop_walk"], "walking": ["walk_stop_walk"],
         "running": ["run_across"], "jumping": ["jump_three"], "smiling": ["three_talkers"],
         "talking": ["three_talkers"], "silent": ["three_talkers"]}


class CommandLine(unittest.TestCase):
    def test_every_action_runs_and_writes_its_files(self):
        self.assertEqual(set(CLIPS), set(ACTIONS))
        for action, names in CLIPS.items():
            h.run(action, names)
            out = h.out_dir(action)
            for f in ("segments.csv", "intro.csv", "dropped.csv", "interview_starts.csv", "review.html"):
                self.assertTrue((out / f).exists(), f"{action}: {f}")
            with open(out / "segments.csv", newline="") as f:
                self.assertEqual(csv.DictReader(f).fieldnames, ds.segment_fields(action), action)
            if action in ds.SOUND_ACTIONS:
                self.assertTrue((out / "speech.csv").exists() and (out / "voices.csv").exists(), action)
                self.assertIn("Voices (", (out / "review.html").read_text())

    def test_a_second_run_skips_and_reclassify_redoes(self):
        r = h.run("walking", ["walk_stop_walk"])
        self.assertIn("done already, skipping", r.stdout)
        r = h.run("walking", ["walk_stop_walk"], "--reclassify")
        self.assertIn("walking sections", r.stdout)
        self.assertNotIn("skipping", r.stdout)

    def test_a_bad_file_is_reported_and_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "20240101_badvideo000.mp4"
            bad.write_bytes(b"not a video")
            r = subprocess.run([sys.executable, str(REPO / "standing" / "detect_standing.py"), str(bad),
                                str(mf.build("walk_stop_walk")), "--out", str(Path(tmp) / "out"), "--keep-intro",
                                "--action", "walking"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0)
            self.assertIn("FAILED", r.stderr)
            self.assertTrue((Path(tmp) / "out" / "walking" / "segments" / "walk_stop_walk.csv").exists())


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
            talking = h.sections("three_talkers", "talking")
            (results / "verdicts.csv").write_text("video,start_s,verdict,note\n")
            (results / "talking" / "verdicts.csv").write_text(
                f"video,start_s,verdict,note\nthree_talkers.mp4,{talking[0]['start_s']},confirmed,left talks\n")
            for action in ("standing", "walking", "talking"):
                h.run(action, CLIPS[action])
                r = subprocess.run([sys.executable, str(REPO / "standing" / "make_report.py"), "--action", action,
                                    "--detector-output", str(h.OUT), "--results", str(results),
                                    "--inventory", str(inventory)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            rows = h.csv_rows(results / "talking" / "talking_sections.csv")
            mine = [r for r in rows if r["video"] == "three_talkers.mp4"]
            self.assertEqual(len(mine), len(talking))
            self.assertEqual(mine[0]["verdict"], "confirmed")
            self.assertEqual(mine[0]["title"], "Test three_talkers")
            self.assertTrue((results / "talking" / "speech.csv").exists())
            self.assertIn("## Voices", (results / "talking" / "README.md").read_text())
            self.assertTrue((results / "standing_sections.csv").exists())
            self.assertTrue((results / "walking" / "walking_sections.csv").exists())
            # the review video of the talking sections
            review = tmp / "review.mp4"
            r = subprocess.run([sys.executable, str(REPO / "standing" / "review_video.py"), "--action", "talking",
                                "--sections", str(results / "talking" / "talking_sections.csv"), "--videos-dir",
                                str(mf.CLIPS), "--verdicts", "all", "--out", str(review)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertGreater(review.stat().st_size, 10000)
            index = h.csv_rows(review.with_suffix(".csv"))
            self.assertTrue(index and "talking" in index[0])


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Service(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.port = _free_port()
        # Jane's voice, enrolled from a recording of her (0-8 s of duo_purple), for who=Jane
        jane = Path(cls.tmp.name) / "jane.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(mf.build("duo_purple")), "-t", "8", "-vn", str(jane)], check=True)
        config = Path(cls.tmp.name) / "people.toml"
        config.write_text(f'[[people]]\nname = "Jane"\nvoice_file = "{jane}"\n')
        env = dict(os.environ, API_KEY="test-key", WORK_DIR=str(Path(cls.tmp.name) / "work"), PEOPLE_CONFIG=str(config))
        cls.server = subprocess.Popen([sys.executable, "-m", "uvicorn", "service.app:app", "--port", str(cls.port)],
                                      cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                cls.call("GET", "/healthz")
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait()
        cls.tmp.cleanup()

    @classmethod
    def call(cls, method, path, body=None, key="test-key"):
        req = urllib.request.Request(f"http://127.0.0.1:{cls.port}{path}", data=body, method=method,
                                     headers={"x-api-key": key} if key else {})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
            return data if r.headers.get_content_type() == "text/csv" else json.loads(data)

    def job(self, name, action, who=None):
        query = f"action={action}" if who is None else urllib.parse.urlencode({"who": who})
        job = self.call("POST", f"/jobs?name={name}.mp4&{query}", mf.build(name).read_bytes())
        for _ in range(600):
            status = self.call("GET", f"/jobs/{job['job_id']}")
            if status["status"] in ("done", "failed"):
                return job["job_id"], status
            time.sleep(1)
        self.fail("job did not finish")

    def test_health_lists_the_actions(self):
        self.assertEqual(self.call("GET", "/healthz")["actions"], ACTIONS)

    def test_a_wrong_key_or_action_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("POST", "/jobs?name=x.mp4", b"x", key="wrong")
        self.assertEqual(e.exception.code, 401)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("POST", "/jobs?name=x.mp4&action=dancing", b"x")
        self.assertEqual(e.exception.code, 400)

    def test_same_sections_as_the_command_line(self):
        for name, action in (("walk_stop_walk", "walking"), ("three_talkers", "talking"), ("jump_three", "jumping")):
            job_id, status = self.job(name, action)
            self.assertEqual(status["status"], "done", status)
            self.assertTrue(status["video_deleted"])
            got = list(csv.DictReader(self.call("GET", f"/jobs/{job_id}/sections.csv").decode().splitlines()))
            h.run(action, [name])
            want = h.sections(name, action)
            self.assertEqual([(r["start_s"], r["end_s"]) for r in got], [(r["start_s"], r["end_s"]) for r in want], name)
            if action == "talking":
                speech = self.call("GET", f"/jobs/{job_id}/speech.csv").decode()
                self.assertGreater(len(speech.splitlines()), 3)
                self.assertEqual(status["voices"], 3)
        self.assertEqual(list((Path(self.tmp.name) / "work").glob("*")), [])   # every upload deleted

    def test_finding_a_person(self):
        job_id, status = self.job("duo_purple", None, who="woman with purple hair talking")
        self.assertEqual(status["status"], "done", status)
        self.assertEqual((status["action"], status["looking_for"]), ("talking", "gender female, hair purple"))
        got = list(csv.DictReader(self.call("GET", f"/jobs/{job_id}/matches.csv").decode().splitlines()))
        searcher = search.Searcher(h.search_options(), search.people.load_config())
        req = search.parse("woman with purple hair talking", searcher.config)
        want = searcher.search(mf.path("duo_purple"), req, "talking")
        self.assertEqual([(float(r["start_s"]), float(r["end_s"])) for r in got], [(r["start_s"], r["end_s"]) for r in want])
        self.assertEqual(len(got), 2)
        # Jane, enrolled from a recording (PEOPLE_CONFIG), found by her voice in another video
        job_id, status = self.job("solo_blue", None, who="Jane talking")
        self.assertEqual(status["status"], "done", status)
        got = list(csv.DictReader(self.call("GET", f"/jobs/{job_id}/matches.csv").decode().splitlines()))
        self.assertEqual(len(got), 2)
        self.assertIn("voice of Jane", got[0]["evidence"])
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("POST", "/jobs?name=x.mp4&who=wibble", b"x")
        self.assertEqual(e.exception.code, 400)
        self.assertEqual(list((Path(self.tmp.name) / "work").glob("*")), [])


if __name__ == "__main__":
    unittest.main()
