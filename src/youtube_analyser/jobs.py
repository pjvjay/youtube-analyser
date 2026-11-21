"""Jobs and workers: the same processing on one machine and in the cloud.

A job is a video to analyse (uploaded to the HTTP API) or one video of a channel search. The API
(service.py) and `yta channel --queue` put work in a cloud.Queue and files in a cloud.Store;
Workers take the work, run the same pipeline as the command line, and put the results in the
Store. On one machine the store is a folder and the queue lives in the API's process (its
workers are threads); on Azure they are Blob Storage and Queue Storage, and the workers run as a
Container Apps job (`yta worker --until-empty`) that scales with the queue.

Store layout:
    uploads/<job>              an uploaded video, until it has been analysed (then deleted)
    jobs/<job>/status.json     the job's status (queued, processing, done, failed) and summary
    jobs/<job>/sections.csv    the sections found (or matches.csv for a who= job, speech.csv for talking)
    searches/<search>/...      a channel search (see channel.py)
"""
import contextlib
import csv
import io
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from . import audio, people
from .actions import ACTIONS, NOTE_COLUMNS, SOUND_ACTIONS, SPEECH_FIELDS
from .analysis import Settings, VideoAnalysis
from .channel import ChannelTasks
from .query import PARTS, describe_query, parse
from .search import Searcher

SECTION_FIELDS = ["start_s", "end_s", "start", "end", "duration_s"]   # then <action>_frames, max_people_<action>, notes
SPEECH_JOB_FIELDS = [f for f in SPEECH_FIELDS if f not in ("video", "youtube")]
MATCH_JOB_FIELDS = ["person", "start_s", "end_s", "start", "end", "duration_s", "action", "match", "evidence"]
STATUS_FIELDS = ["job_id", "name", "action", "who", "looking_for", "status", "size_mb", "created", "started",
                 "finished", "frames", "interview_start_s", "sections", "intro_sections", "still_figures_dropped",
                 "voices", "sound_trusted", "matches", "error", "video_deleted"]
MAX_TRIES = 3            # a channel video whose handling fails this many times is given up (and reported)


def section_fields(action):
    notes = [NOTE_COLUMNS[n] for n in ACTIONS[action].get("notes", ())]
    return [*SECTION_FIELDS, f"{action}_frames", f"max_people_{action}", *notes]


def to_csv(fields, rows):
    out = io.StringIO()
    w = csv.DictWriter(out, fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


class JobStore:
    """Job records in a Store (jobs/<id>/status.json) and their result files."""

    def __init__(self, store):
        self.store = store
        self._lock = threading.Lock()

    def new(self, name, action, who=None, looking_for=None, possible=False, size=0):
        job_id = uuid.uuid4().hex
        job = dict.fromkeys(STATUS_FIELDS)
        job.update(job_id=job_id, name=name[:200], action=action, who=who, looking_for=looking_for, status="queued",
                   size_mb=round(size / 1e6, 1), created=time.time(), video_deleted=False, possible=possible)
        self.store.put_json(f"jobs/{job_id}/status.json", job)
        return job

    def get(self, job_id):
        """The job's record, or None."""
        try:
            return self.store.get_json(f"jobs/{job_id}/status.json")
        except KeyError:
            return None

    def update(self, job_id, **fields):
        """Change a job's record; nothing happens (False) if the job has been deleted meanwhile."""
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return False
            job.update(fields)
            self.store.put_json(f"jobs/{job_id}/status.json", job)
            return True

    def put_result(self, job_id, name, text):
        self.store.put(f"jobs/{job_id}/{name}", text.encode())

    def result(self, job_id, name):
        """A result file's text, or None."""
        try:
            return self.store.get(f"jobs/{job_id}/{name}").decode()
        except KeyError:
            return None

    def delete(self, job_id):
        for key in self.store.list(f"jobs/{job_id}/"):
            self.store.delete(key)
        self.store.delete(f"uploads/{job_id}")

    def ids(self):
        return sorted({k.split("/")[1] for k in self.store.list("jobs/") if k.endswith("/status.json")})

    def expire(self, ttl_s, now=None):
        """Forget jobs finished more than ttl_s ago. Returns how many."""
        now = now or time.time()
        gone = 0
        for job_id in self.ids():
            job = self.get(job_id)
            if job and job.get("finished") and now - job["finished"] > ttl_s:
                self.delete(job_id)
                gone += 1
        return gone


class Worker:
    """Takes work from a queue and does it: "analyse" (an uploaded video: the detector's sections
    for an action, or with who= the times a described person does it) and "channel-video" (one
    video of a channel search). One video at a time: the models already use every core."""

    def __init__(self, store, task_queue, settings=None, config=None, bank=None, log=print, visibility_s=3600,
                 yt=None):
        self.store, self.queue = store, task_queue
        self.settings = settings or Settings()
        self.config = config or people.load_config()
        self.models = self.settings.models(self.config["models"]["gender_age"])
        self.bank, self.log, self.visibility_s = bank, log, visibility_s
        self.jobs = JobStore(store)
        self.channel = ChannelTasks(store, self.settings.runtime, self.settings.threads, yt=yt, log=log)
        self.enrolled = None

    def run_once(self):
        """Handle the next message. Returns False when the queue was empty."""
        m = self.queue.receive(self.visibility_s)
        if m is None:
            return False
        kind = m.body.get("kind")
        try:
            with self._kept_hidden(m):
                if kind == "analyse":
                    self.analyse_upload(m.body["job"])
                elif kind == "channel-video":
                    self.channel.handle(m.body)
                else:
                    raise ValueError(f"unknown kind of work {kind!r}")
        except Exception as e:  # a failing channel video is tried again, then reported
            self.log(f"{kind} failed (try {m.dequeue_count}): {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            if kind == "channel-video" and m.dequeue_count < MAX_TRIES:
                self._give_back(m, self.queue.retry)
                return True
            if kind == "channel-video":
                self.store.put_json(f"searches/{m.body['search']}/failed/{m.body['entry']['id']}.json",
                                    {"entry": m.body["entry"], "error": f"{type(e).__name__}: {e}"})
        self._give_back(m, self.queue.done)
        return True

    @contextlib.contextmanager
    def _kept_hidden(self, m):
        """While the work goes on, keep the message hidden from other workers (a long video can
        take longer than the visibility the message was taken with)."""
        stop = threading.Event()

        def beat():
            while not stop.wait(self.visibility_s / 3):
                try:
                    self.queue.extend(m, self.visibility_s)
                except Exception as e:  # pragma: no cover - the queue's service hiccuped; the next beat tries again
                    self.log(f"could not keep the message hidden: {e}", file=sys.stderr, flush=True)
        beater = threading.Thread(target=beat, daemon=True)
        beater.start()
        try:
            yield
        finally:
            stop.set()
            beater.join()

    def _give_back(self, m, how):
        """done() or retry(); if the queue no longer knows the message (its visibility ran out and
        another worker took it), say so rather than stopping the worker."""
        try:
            how(m)
        except Exception as e:
            self.log(f"the queue no longer had the message ({type(e).__name__}: {e})", file=sys.stderr, flush=True)

    def run(self, until_empty=False, poll_s=2.0, stop=None):
        """Handle messages until stopped (stop: a threading.Event), or until the queue is empty."""
        while stop is None or not stop.is_set():
            if not self.run_once():
                if until_empty:
                    return
                if stop is not None:
                    stop.wait(poll_s)
                else:
                    time.sleep(poll_s)

    # ---- an uploaded video ---------------------------------------------------------------------

    def analyse_upload(self, job_id):
        """Analyse an uploaded video; the video is deleted from the store whatever happens."""
        key = f"uploads/{job_id}"
        job = self.jobs.get(job_id)
        if job is None:  # deleted before it was started
            self.store.delete(key)
            return
        if job["status"] in ("done", "failed"):   # handled already (a message delivered twice)
            return
        if not self.store.exists(key):             # a worker stopped while analysing it, after taking the video
            self.jobs.update(job_id, status="failed", finished=time.time(), video_deleted=True,
                             error="the video was lost when a worker stopped while analysing it: upload it again")
            return
        self.jobs.update(job_id, status="processing", started=time.time())
        try:
            with tempfile.TemporaryDirectory() as tmp:
                video = Path(tmp) / ("video" + (Path(job["name"]).suffix[:8] if Path(job["name"]).suffix else ".video"))
                self.store.get_file(key, video)
                self.store.delete(key)
                summary, results = (self._find_person(video, job) if job.get("who") else self._detect(video, job["action"]))
            if self.jobs.update(job_id, status="done", **summary):
                for name, text in results.items():
                    self.jobs.put_result(job_id, name, text)
        except Exception as e:  # reported on the job; the worker carries on
            self.jobs.update(job_id, status="failed", error=f"{type(e).__name__}: {e}")
        finally:
            self.store.delete(key)   # the video never outlives its processing
            self.jobs.update(job_id, finished=time.time(), video_deleted=not self.store.exists(key))

    def _detect(self, video, action):
        rules = ACTIONS[action]
        va = VideoAnalysis(video, self.settings, self.models)
        if not len(va.poses["t"]):
            raise ValueError("no video frames could be decoded")
        from .detector import detect, speech_timeline
        start = va.interview_start(self.bank)["interview_start_s"]
        extra = va.inputs(action)
        _, sections, in_intro, still = detect(video, va.poses, action, rules["min_frames"], rules["max_gap"], start, extra)
        summary = {"frames": len(va.poses["t"]), "interview_start_s": start, "sections": len(sections),
                   "intro_sections": len(in_intro), "still_figures_dropped": len(still)}
        results = {"sections.csv": to_csv(section_fields(action), sections)}
        if action in SOUND_ACTIONS:
            results["speech.csv"] = to_csv(SPEECH_JOB_FIELDS, speech_timeline(video, va.poses, action, extra)[0])
            a = extra["audio"]
            summary.update(voices=int(a["voices"]) if a is not None else 0,
                           sound_trusted=round(float((a["snr_db"] >= audio.SNR_OK_DB).mean()), 2)
                           if a is not None and len(a["snr_db"]) else 0.0)
        return summary, results

    def _find_person(self, video, job):
        req = request_for(job["who"], self.config, job["action"])
        searcher = Searcher(self.config, self.settings, None, self.models, bank=self.bank, possible=job.get("possible"),
                            log=self.log)
        if self.enrolled is None:
            self.enrolled = searcher.enrol()        # the same people for every job
        searcher.enrolled = self.enrolled
        rows = searcher.search(video, req, req["action"])
        va = searcher.analysis(video)
        if not len(va.poses["t"]):
            raise ValueError("no video frames could be decoded")
        return ({"matches": len(rows), "frames": len(va.poses["t"]),
                 "interview_start_s": va.interview_start(self.bank)["interview_start_s"]},
                {"matches.csv": to_csv(MATCH_JOB_FIELDS, rows)})


def request_for(who, config, action=None):
    """A who= description as a search request (ValueError if it describes nobody); action:
    given separately (else the description's, else any time they are in view)."""
    req = parse(who, config, [p["name"] for p in config.get("people", [])])
    if action not in (None, *ACTIONS, "visible"):
        raise ValueError(f"action must be one of: {', '.join(ACTIONS)}, visible")
    req["action"] = action or req["action"] or "visible"
    if not any(req[k] for k in PARTS):
        raise ValueError("who= describes nobody: " + (f"ignored words {', '.join(req['ignored'])}; " if req["ignored"] else "")
                         + "see people.toml for the words it knows")
    return req


def looking_for(req):
    return describe_query(req) + (f" (ignored: {', '.join(req['ignored'])})" if req["ignored"] else "")
