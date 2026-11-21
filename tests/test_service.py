"""The HTTP API (service.py) with its own worker, in this process: the same sections as the
command line, who=, enrolment from a recording, errors, deleting, expiry, and what happens to
uploads (always deleted once analysed; deleted at startup when the queue did not survive).

    pytest tests/test_service.py
"""
import csv
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from tests import helpers as h
from tests.helpers import mf
from youtube_analyser import actions, cloud, jobs, service

KEY = {"x-api-key": "test-key"}


class Service(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        # Jane's voice, enrolled from a recording of her (0-8 s of duo_purple), for who=Jane
        jane = tmp / "jane.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(mf.build("duo_purple")), "-t", "8", "-vn", str(jane)], check=True)
        (tmp / "people.toml").write_text(f'[[people]]\nname = "Jane"\nvoice_file = "{jane}"\n')
        cls.config = service.Config({"API_KEY": "test-key", "WORK_DIR": str(tmp / "work"), "PEOPLE_CONFIG": str(tmp / "people.toml"),
                                     "IMGSZ": str(h.SETTINGS.imgsz), "MODEL": h.SETTINGS.pose_model})
        cls.client = TestClient(service.create_app(cls.config))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.tmp.cleanup()

    def job(self, name, action=None, who=None, **params):
        q = {"name": f"{name}.mp4", **({"action": action} if action else {}), **({"who": who} if who else {}), **params}
        r = self.client.post("/jobs", params=q, content=mf.build(name).read_bytes(), headers=KEY)
        self.assertEqual(r.status_code, 202, r.text)
        job_id = r.json()["job_id"]
        for _ in range(1200):
            status = self.client.get(f"/jobs/{job_id}", headers=KEY).json()
            if status["status"] in ("done", "failed") and status["finished"]:
                return job_id, status
            time.sleep(0.25)
        self.fail("job did not finish")

    def csv(self, path):
        r = self.client.get(path, headers=KEY)
        self.assertEqual(r.status_code, 200, r.text)
        return list(csv.DictReader(r.text.splitlines()))

    def test_health_lists_the_actions(self):
        r = self.client.get("/healthz").json()
        self.assertEqual(r["actions"], list(actions.ACTIONS))
        self.assertEqual(r["people_enrolled"], ["Jane"])
        self.assertEqual(r["workers"], 1)

    def test_a_wrong_key_action_or_body_is_refused(self):
        self.assertEqual(self.client.post("/jobs?name=x.mp4", content=b"x", headers={"x-api-key": "wrong"}).status_code, 401)
        self.assertEqual(self.client.post("/jobs?name=x.mp4&action=dancing", content=b"x", headers=KEY).status_code, 400)
        self.assertEqual(self.client.post("/jobs?name=x.mp4", content=b"", headers=KEY).status_code, 400)
        r = self.client.post("/jobs?name=x.mp4&who=wibble", content=b"x", headers=KEY)
        self.assertEqual(r.status_code, 400)
        self.assertIn("ignored words wibble", r.text)
        self.assertEqual(self.client.get("/jobs/nosuchjob", headers=KEY).status_code, 404)
        self.assertEqual(self.client.get("/jobs/../../etc", headers=KEY).status_code, 404)

    def test_same_sections_as_the_command_line(self):
        for name, action in (("walk_stop_walk", "walking"), ("three_talkers", "talking"), ("jump_three", "jumping")):
            job_id, status = self.job(name, action)
            self.assertEqual(status["status"], "done", status)
            self.assertTrue(status["video_deleted"])
            got = self.csv(f"/jobs/{job_id}/sections.csv")
            h.run(action, [name])
            want = h.sections(name, action)
            self.assertEqual([(r["start_s"], r["end_s"]) for r in got], [(r["start_s"], r["end_s"]) for r in want], name)
            if action == "talking":
                self.assertGreater(len(self.csv(f"/jobs/{job_id}/speech.csv")), 2)
                self.assertEqual(status["voices"], 3)
            else:
                self.assertEqual(self.client.get(f"/jobs/{job_id}/speech.csv", headers=KEY).status_code, 404)
            self.assertEqual(self.client.get(f"/jobs/{job_id}/matches.csv", headers=KEY).status_code, 404)
        self.assertEqual(list((self.config.work_dir / "store").glob("uploads/*")), [])   # every upload deleted

    def test_finding_a_person(self):
        job_id, status = self.job("duo_purple", who="woman with purple hair talking")
        self.assertEqual(status["status"], "done", status)
        self.assertEqual((status["action"], status["looking_for"]), ("talking", "gender female, hair purple"))
        got = self.csv(f"/jobs/{job_id}/matches.csv")
        want = h.searcher().search(mf.path("duo_purple"), h.query.parse("woman with purple hair talking",
                                                                         h.characteristics.load_config()), "talking")
        self.assertEqual([(float(r["start_s"]), float(r["end_s"])) for r in got], [(r["start_s"], r["end_s"]) for r in want])
        self.assertEqual(len(got), 2)
        self.assertEqual(self.client.get(f"/jobs/{job_id}/sections.csv", headers=KEY).status_code, 404)
        # Jane, enrolled from a recording (PEOPLE_CONFIG), found by her voice in another video
        job_id, status = self.job("solo_blue", who="Jane talking")
        got = self.csv(f"/jobs/{job_id}/matches.csv")
        self.assertEqual(len(got), 2)
        self.assertIn("voice of Jane", got[0]["evidence"])

    def test_a_broken_video_fails_and_is_deleted(self):
        r = self.client.post("/jobs", params={"name": "broken.mp4"}, content=b"not a video at all", headers=KEY)
        job_id = r.json()["job_id"]
        for _ in range(200):
            status = self.client.get(f"/jobs/{job_id}", headers=KEY).json()
            if status["finished"]:
                break
            time.sleep(0.1)
        self.assertEqual(status["status"], "failed")
        self.assertIn("no readable video stream", status["error"])
        self.assertTrue(status["video_deleted"])
        r = self.client.get(f"/jobs/{job_id}/sections.csv", headers=KEY)
        self.assertEqual(r.status_code, 409)
        self.assertIn("failed", r.text)
        self.assertEqual(self.client.delete(f"/jobs/{job_id}", headers=KEY).status_code, 204)
        self.assertEqual(self.client.get(f"/jobs/{job_id}", headers=KEY).status_code, 404)
        self.assertEqual(self.client.delete(f"/jobs/{job_id}", headers=KEY).status_code, 404)


class Lifecycle(unittest.TestCase):
    def test_too_big_expiry_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = service.Config({"WORK_DIR": tmp, "MAX_UPLOAD_MB": "0", "YTA_WORKERS": "0", "RESULT_TTL_S": "60"})
            store = cloud.FileStore(Path(tmp) / "store")
            # an upload left by a service that stopped: its queue (in memory) is gone, so it is deleted
            store.put("uploads/oldjob", b"video")
            js = jobs.JobStore(store)
            old = js.new("old.mp4", "standing")
            store.put(f"uploads/{old['job_id']}", b"video")
            with TestClient(service.create_app(cfg, store=store)) as client:
                self.assertEqual(store.list("uploads/"), [])
                self.assertEqual(js.get(old["job_id"])["status"], "failed")
                self.assertEqual(client.post("/jobs?name=x.mp4", content=b"x" * 10).status_code, 413)
                self.assertEqual(list(Path(tmp).glob("*.upload")), [])     # the part-upload is gone too
                self.assertEqual(client.get("/healthz").json()["workers"], 0)
            self.assertEqual(js.expire(60, now=time.time() + 61), 1)      # finished jobs are forgotten
            self.assertIsNone(js.get(old["job_id"]))


if __name__ == "__main__":
    unittest.main()
