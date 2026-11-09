"""HTTP API for the standing detector: upload a video, get the standing sections back as CSV.

The uploaded video only exists on local disk while it waits and is processed. It is deleted
as soon as it has been analysed, whether the job succeeded or failed, and any videos
left over from a crash are deleted at startup. Results are a few hundred bytes of CSV held in
memory until DELETE or until RESULT_TTL_S after the job finished.

    POST   /jobs?name=<file name>    request body = the video bytes   -> 202 {"job_id", ...}
    GET    /jobs/{job_id}            status: queued | processing | done | failed
    GET    /jobs/{job_id}/sections.csv   standing sections from the interview start on
    DELETE /jobs/{job_id}
    GET    /healthz

Standing in the intro (title sequence, teaser, ads) is left out of the CSV; the status says
where the interview starts and how many intro sections there were. Finding the start needs the
bank of title frames in standing/models/intro_bank.npz, built locally with
standing/intro.py --build-bank; without it each whole video is treated as the interview.

Settings (environment): API_KEY (when set, callers must send it in the x-api-key header),
WORK_DIR, MAX_UPLOAD_MB (default 2048), RESULT_TTL_S (default 3600), MODEL, IMGSZ (default
640), FPS (default 1).

Run locally from the youtube-analyser folder:
    .venv/bin/uvicorn service.app:app --port 8000
    curl -X POST --data-binary @video.mp4 "http://localhost:8000/jobs?name=video.mp4"
"""
import csv
import io
import os
import queue
import secrets
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "standing"))
import detect_standing as ds  # noqa: E402

API_KEY = os.environ.get("API_KEY", "")
WORK_DIR = Path(os.environ.get("WORK_DIR", Path(tempfile.gettempdir()) / "standing-jobs"))
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_MB", "2048")) * 1024 * 1024
RESULT_TTL_S = int(os.environ.get("RESULT_TTL_S", "3600"))
MODEL = os.environ.get("MODEL", str(ds.DEFAULT_MODEL))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
FPS = float(os.environ.get("FPS", "1"))
CONF, MIN_FRAMES, MAX_GAP = 0.35, 2, 2  # the command line's defaults
CSV_FIELDS = ["start_s", "end_s", "start", "end", "duration_s", "standing_frames", "max_people_standing"]
STATUS_FIELDS = ["job_id", "name", "status", "size_mb", "created", "started", "finished",
                 "frames", "interview_start_s", "sections", "intro_sections", "still_figures_dropped",
                 "error", "video_deleted"]

jobs = {}  # job_id -> job dict; only touched under `lock`
lock = threading.Lock()
work = queue.Queue()
state = {}


def run_jobs():
    """Process uploads one at a time; the pose model already uses every core."""
    while True:
        job_id, path = work.get()
        with lock:
            job = jobs.get(job_id)
            if job:
                job.update(status="processing", started=time.time())
        try:
            if job:
                poses = ds.infer(path, state["model"], FPS, CONF)
                if not len(poses["t"]):
                    raise ValueError("no video frames could be decoded")
                start = ds.intro.interview_start(path, state["bank"])["interview_start_s"]
                _, sections, in_intro = ds.find_sections(poses, job["name"], MIN_FRAMES, MAX_GAP, start)
                width, height = (int(x) for x in poses["size"])
                sections, still = ds.drop_still_figures(path, sections, FPS, width, height)
                out = io.StringIO()
                w = csv.DictWriter(out, CSV_FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(sections)
                with lock:
                    job.update(status="done", csv=out.getvalue(), frames=len(poses["t"]),
                               interview_start_s=start, sections=len(sections),
                               intro_sections=len(in_intro), still_figures_dropped=len(still))
        except Exception as e:  # report it on the job; the service keeps running
            with lock:
                job.update(status="failed", error=f"{type(e).__name__}: {e}")
        finally:
            path.unlink(missing_ok=True)  # the video never outlives its processing
            with lock:
                if job:
                    job.update(finished=time.time(), video_deleted=not path.exists())


def expire_results():
    while True:
        time.sleep(60)
        now = time.time()
        with lock:
            for job_id in [j for j, job in jobs.items()
                           if job["finished"] and now - job["finished"] > RESULT_TTL_S]:
                del jobs[job_id]


@asynccontextmanager
async def lifespan(app):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    for leftover in WORK_DIR.glob("*.video"):  # from a crash or restart mid-job
        leftover.unlink(missing_ok=True)
    state["model"] = ds.PoseModel(MODEL, IMGSZ)
    state["bank"] = ds.intro.load_bank()
    if state["bank"] is None:
        print("no intro bank: the intro is not separated", file=sys.stderr)
    threading.Thread(target=run_jobs, daemon=True).start()
    threading.Thread(target=expire_results, daemon=True).start()
    yield


app = FastAPI(title="Standing detector", lifespan=lifespan)


def check_key(given):
    if API_KEY and not secrets.compare_digest(given or "", API_KEY):
        raise HTTPException(401, "missing or wrong x-api-key")


def get_job(job_id):
    with lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "no such job (results are kept for RESULT_TTL_S after finishing)")
        return dict(job)


@app.post("/jobs", status_code=202)
async def create_job(request: Request, name: str = "video", x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    job_id = uuid.uuid4().hex
    path = WORK_DIR / f"{job_id}.video"
    size = 0
    try:
        with open(path, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, f"video is larger than {MAX_UPLOAD >> 20} MB")
                f.write(chunk)
        if not size:
            raise HTTPException(400, "send the video as the request body")
    except BaseException:  # includes the client disconnecting mid-upload
        path.unlink(missing_ok=True)
        raise
    with lock:
        jobs[job_id] = {"job_id": job_id, "name": name[:200], "status": "queued",
                        "size_mb": round(size / 1e6, 1), "created": time.time(), "started": None,
                        "finished": None, "frames": None, "interview_start_s": None, "sections": None,
                        "intro_sections": None, "still_figures_dropped": None, "error": None,
                        "video_deleted": False, "csv": None}
    work.put((job_id, path))
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}", "result_url": f"/jobs/{job_id}/sections.csv"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str, x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    job = get_job(job_id)
    return {k: job[k] for k in STATUS_FIELDS}


@app.get("/jobs/{job_id}/sections.csv")
def job_result(job_id: str, x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    job = get_job(job_id)
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}" + (f": {job['error']}" if job["error"] else ""))
    return PlainTextResponse(job["csv"], media_type="text/csv")


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str, x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    with lock:
        if jobs.pop(job_id, None) is None:
            raise HTTPException(404, "no such job")


@app.get("/healthz")
def healthz():
    return {"ok": True, "queued": work.qsize(), "runtime": state["model"].runtime,
            "model": Path(MODEL).name, "imgsz": IMGSZ, "fps": FPS, "intro_bank": state["bank"] is not None}
