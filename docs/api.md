# The HTTP API (`yta serve`)

Upload a video, get back CSV: the sections where someone does an action, or with `who=` the times
a described person does it. Install with the `service` extra (`pip install
"youtube-analyser[service]"`), then:

```bash
API_KEY=change-me yta serve --port 8000
curl http://localhost:8000/healthz
curl -H "x-api-key: change-me" -X POST --data-binary @episode.mp4 "http://localhost:8000/jobs?name=episode.mp4&action=talking"
curl -H "x-api-key: change-me" http://localhost:8000/jobs/<job_id>                   # queued | processing | done | failed
curl -H "x-api-key: change-me" http://localhost:8000/jobs/<job_id>/sections.csv
curl -H "x-api-key: change-me" -X POST --data-binary @episode.mp4 "http://localhost:8000/jobs?name=episode.mp4" \
     --url-query "who=woman with purple hair talking"                                # then GET .../matches.csv
```

| Call | Does |
|---|---|
| `POST /jobs?name=<file>&action=<action>` with the video as the body | Stores it, queues it, returns `{"job_id", "status_url", "result_url"}` (202). `action` is `standing` (default), `sitting`, `walking`, `running`, `jumping`, `smiling`, `talking` or `silent` |
| `POST /jobs?...&who=<description>[&possible=true]` | Finds that person instead (as `yta search`): the action comes from the description or `action=` (also `visible`), default whenever they are in view. A description the service does not understand is refused (400) with the ignored words |
| `GET /jobs/{job_id}` | `status`, and when done a summary: `frames`, `interview_start_s`, `sections`, `intro_sections`, `still_figures_dropped`, `voices`, `sound_trusted`, `matches`, `error`, `video_deleted` |
| `GET /jobs/{job_id}/sections.csv` | `start_s, end_s, start, end, duration_s, <action>_frames, max_people_<action>` (+ `voices, judged_by` for talking, `voices_heard, judged_by` for silent), for the interview only |
| `GET /jobs/{job_id}/speech.csv` | Talking and silent: who spoke when (voices A, B... from the sound), whether they were on screen, and whether the sound was clear enough to trust |
| `GET /jobs/{job_id}/matches.csv` | With `who=`: `person, start_s, end_s, start, end, duration_s, action, match, evidence` |
| `DELETE /jobs/{job_id}` | Forgets the job and its results (and the video, if it is still waiting) |
| `GET /healthz` | `ok`, the queue's length, the runtime and model, the actions, whether an intro bank is loaded, the worker threads and the people enrolled by name |

Errors: 401 (missing or wrong `x-api-key` when `API_KEY` is set), 400 (unknown action, an empty
body, a description of nobody), 413 (larger than `MAX_UPLOAD_MB`), 404 (no such job, or a result
this job does not make), 409 (the job is not done; the message says why if it failed).

**Privacy.** The uploaded video is deleted as soon as it has been analysed, whether that
succeeded or failed (the status says `video_deleted`); the part of an upload that was refused is
deleted at once; results are deleted `RESULT_TTL_S` (1 hour) after the job finished, or with
`DELETE`. With the default in-memory queue, a restart loses the queue, so videos left waiting are
deleted at startup and their jobs marked failed.

**Settings** (environment):

| Variable | Default | |
|---|---|---|
| `API_KEY` | none | callers must send it in `x-api-key` |
| `YTA_STORE` | a folder in `WORK_DIR` | where uploads and results are kept ([cloud.md](cloud.md)) |
| `YTA_QUEUE` | `memory://` | where work waits |
| `YTA_WORKERS` | 1 with the memory queue, else 0 | worker threads in this process (0: workers run elsewhere, `yta worker`) |
| `WORK_DIR` | the system's temporary folder | uploads are received here before being stored |
| `MAX_UPLOAD_MB` | 2048 | |
| `RESULT_TTL_S` | 3600 | |
| `MODEL`, `IMGSZ`, `FPS`, `JUMP_FPS` | `yolo11s-pose.onnx`, 640, 1, 5 | as `yta detect`'s options |
| `INTRO_BANK` | none | a bank file (`yta intro --build-bank`): sections in intros are left out |
| `PEOPLE_CONFIG` | none | your people.toml for `who=`; people found by name are enrolled from `voice_file` recordings |

As a container: `docker build -t youtube-analyser .` and `docker run -p 8000:8000 -e
API_KEY=change-me youtube-analyser`. For Azure see [cloud.md](cloud.md).
