"""HTTP API: upload a video, get back the sections where someone does an action (or, with who=,
the times a described person does it) as CSV (`yta serve`; needs the [service] extra).

    POST   /jobs?name=<file name>&action=<action>&who=<description>   body: the video -> 202 {"job_id", ...}
    GET    /jobs/{job_id}                 status: queued | processing | done | failed, and a summary
    GET    /jobs/{job_id}/sections.csv    start_s, end_s, start, end, duration_s, <action>_frames,
                                          max_people_<action> (+ voices, judged_by for talking ...)
    GET    /jobs/{job_id}/speech.csv      talking and silent: who spoke when, and whether on screen
    GET    /jobs/{job_id}/matches.csv     with who=: person, start_s, end_s, start, end, duration_s,
                                          action, match, evidence
    DELETE /jobs/{job_id}
    GET    /healthz

Where uploads, results and the queue of work live is set by the environment (see jobs.py and
cloud/): on one machine (the default) a folder and a queue inside this process, with its own
worker; on Azure, Blob Storage and Queue Storage, with `yta worker` running as a Container Apps
job. An uploaded video is deleted as soon as it has been analysed, whether that succeeded or
failed; results are deleted RESULT_TTL_S after the job finished, or with DELETE.

Environment: API_KEY (callers must send it in x-api-key), YTA_STORE (default: a folder in
WORK_DIR), YTA_QUEUE (default memory://), YTA_WORKERS (worker threads in this process; default
1 with the memory queue, else 0), MAX_UPLOAD_MB (2048), RESULT_TTL_S (3600), MODEL, IMGSZ (640),
FPS (1), JUMP_FPS (5), INTRO_BANK (a bank file; default none: whole videos), PEOPLE_CONFIG (your
people.toml for who=; people to find by name are enrolled from voice_file recordings).
"""
import os
import secrets
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from . import cloud, intro, people
from .actions import ACTIONS, SOUND_ACTIONS
from .analysis import Settings
from .inference import runtime_name
from .jobs import STATUS_FIELDS, JobStore, Worker, looking_for, request_for


class Config:
    """The service's settings, from the environment (or given)."""

    def __init__(self, env=None):
        env = os.environ if env is None else env
        work = Path(env.get("WORK_DIR", Path(tempfile.gettempdir()) / "yta-service"))
        self.work_dir = work
        self.store_url = env.get("YTA_STORE") or str(work / "store")
        self.queue_url = env.get("YTA_QUEUE") or "memory://"
        default_workers = 1 if self.queue_url.startswith("memory://") else 0
        self.workers = int(env.get("YTA_WORKERS", default_workers))
        self.api_key = env.get("API_KEY", "")
        self.max_upload = int(env.get("MAX_UPLOAD_MB", "2048")) * 1024 * 1024
        self.ttl_s = int(env.get("RESULT_TTL_S", "3600"))
        self.settings = Settings(pose_model=env.get("MODEL") or Settings().pose_model, imgsz=int(env.get("IMGSZ", "640")),
                                 fps=float(env.get("FPS", "1")), jump_fps=float(env.get("JUMP_FPS", "5")))
        self.intro_bank = env.get("INTRO_BANK") or None
        self.people_config = env.get("PEOPLE_CONFIG") or None


def create_app(config=None, store=None, task_queue=None):
    """The FastAPI app. config: a Config (default: from the environment); store / task_queue:
    a cloud.Store and cloud.Queue to use instead of the configured ones."""
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import PlainTextResponse

    cfg = config or Config()
    state = {}

    def expire():
        while not state["stop"].wait(60):
            try:
                state["jobs"].expire(cfg.ttl_s)
            except Exception as e:  # pragma: no cover - a storage hiccup; try again next minute
                print(f"expiring results: {e}", file=sys.stderr)

    @asynccontextmanager
    async def lifespan(app):
        state["store"] = store or cloud.open_store(cfg.store_url)
        state["queue"] = task_queue or cloud.open_queue(cfg.queue_url)
        state["jobs"] = JobStore(state["store"])
        state["people"] = people.load_config(cfg.people_config)
        state["bank"] = intro.load_bank(cfg.intro_bank) if cfg.intro_bank else None
        state["stop"] = threading.Event()
        if isinstance(state["queue"], cloud.MemoryQueue):
            # a queue in memory is lost on a restart: uploads left from before cannot be processed
            for key in state["store"].list("uploads/"):
                state["store"].delete(key)
                state["jobs"].update(key.split("/", 1)[1], status="failed", error="the service restarted",
                                     finished=time.time(), video_deleted=True)
        threads = [threading.Thread(target=expire, daemon=True)]
        for _ in range(cfg.workers):
            worker = Worker(state["store"], state["queue"], cfg.settings, state["people"], state["bank"])
            threads.append(threading.Thread(target=worker.run, kwargs={"stop": state["stop"], "poll_s": 0.5}, daemon=True))
        for t in threads:
            t.start()
        yield
        state["stop"].set()

    app = FastAPI(title="youtube-analyser", lifespan=lifespan)

    def check_key(given):
        if cfg.api_key and not secrets.compare_digest(given or "", cfg.api_key):
            raise HTTPException(401, "missing or wrong x-api-key")

    def get_job(job_id):
        job = state["jobs"].get(job_id) if job_id.isalnum() else None
        if not job:
            raise HTTPException(404, "no such job (results are kept for RESULT_TTL_S after finishing)")
        return job

    def result(job, name):
        if job["status"] != "done":
            raise HTTPException(409, f"job is {job['status']}" + (f": {job['error']}" if job["error"] else ""))
        return PlainTextResponse(state["jobs"].result(job["job_id"], name) or "", media_type="text/csv")

    @app.post("/jobs", status_code=202)
    async def create_job(request: Request, name: str = "video", action: str | None = None, who: str = "",
                         possible: bool = False, x_api_key: str | None = Header(default=None)):
        check_key(x_api_key)
        req = None
        if who.strip():
            try:
                req = request_for(who, state["people"], action)
            except ValueError as e:
                raise HTTPException(400, str(e)) from None
            action = req["action"]
        elif (action or "standing") not in ACTIONS:
            raise HTTPException(400, f"action must be one of: {', '.join(ACTIONS)}")
        action = action or "standing"
        cfg.work_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cfg.work_dir, suffix=".upload")
        size = 0
        try:
            with os.fdopen(fd, "wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > cfg.max_upload:
                        raise HTTPException(413, f"video is larger than {cfg.max_upload >> 20} MB")
                    f.write(chunk)
            if not size:
                raise HTTPException(400, "send the video as the request body")
            job = state["jobs"].new(name, action, who=who.strip() or None, looking_for=looking_for(req) if req else None,
                                    possible=possible, size=size)
            try:
                state["store"].put_file(f"uploads/{job['job_id']}", tmp)
            except Exception:
                state["jobs"].delete(job["job_id"])
                raise
        finally:  # includes the client disconnecting mid-upload
            Path(tmp).unlink(missing_ok=True)
        state["queue"].send({"kind": "analyse", "job": job["job_id"]})
        return {"job_id": job["job_id"], "status_url": f"/jobs/{job['job_id']}",
                "result_url": f"/jobs/{job['job_id']}/{'matches' if req else 'sections'}.csv"}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str, x_api_key: str | None = Header(default=None)):
        check_key(x_api_key)
        job = get_job(job_id)
        return {k: job.get(k) for k in STATUS_FIELDS}

    @app.get("/jobs/{job_id}/sections.csv")
    def job_sections(job_id: str, x_api_key: str | None = Header(default=None)):
        check_key(x_api_key)
        job = get_job(job_id)
        if job["who"]:
            raise HTTPException(404, "this job looked for a person (who=): its result is matches.csv")
        return result(job, "sections.csv")

    @app.get("/jobs/{job_id}/speech.csv")
    def job_speech(job_id: str, x_api_key: str | None = Header(default=None)):
        check_key(x_api_key)
        job = get_job(job_id)
        if job["action"] not in SOUND_ACTIONS or job["who"]:
            raise HTTPException(404, "speech.csv is made for action=talking and action=silent (without who=)")
        return result(job, "speech.csv")

    @app.get("/jobs/{job_id}/matches.csv")
    def job_matches(job_id: str, x_api_key: str | None = Header(default=None)):
        check_key(x_api_key)
        job = get_job(job_id)
        if not job["who"]:
            raise HTTPException(404, "matches.csv is made for jobs with who=")
        return result(job, "matches.csv")

    @app.delete("/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, x_api_key: str | None = Header(default=None)):
        check_key(x_api_key)
        get_job(job_id)
        state["jobs"].delete(job_id)

    @app.get("/healthz")
    def healthz():
        s = cfg.settings
        return {"ok": True, "queued": state["queue"].count(), "runtime": runtime_name(s.runtime),
                "model": Path(s.pose_model).name, "imgsz": s.imgsz, "fps": s.fps, "jump_fps": s.jump_fps,
                "actions": list(ACTIONS), "intro_bank": state["bank"] is not None, "workers": cfg.workers,
                "people_enrolled": [p["name"] for p in state["people"].get("people", [])]}

    return app


def app():
    """For `uvicorn --factory youtube_analyser.service:app`."""
    return create_app()
