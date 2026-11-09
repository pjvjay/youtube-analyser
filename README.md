# youtube-analyser

Finds every time range in which someone is standing during the interviews in a folder of
downloaded episodes of a sit-down interview show (each episode's intro is found automatically
and left out), and offers the same detector as an upload API that deletes each video as soon as
it has been analysed.

**Results** are written locally to `results/` and are not committed, because they name the
videos. [`samples/`](samples/README.md) shows every results file with a few made-up rows. On the
test set (248 videos, 210 hours) the detector reported 92 standing sections during the
interviews, and a manual review found 77 confirmed, 6 unclear and 9 false alarms; 27 sections
in intros (nearly all title sequences) were left out automatically.

## What is where

| Path | What |
|---|---|
| `standing/detect_standing.py` | The detector: ffmpeg samples frames, a YOLO pose model finds body points, leg-geometry rules decide standing / sitting / unknown, still figures are dropped. [README](standing/README.md) |
| `standing/intro.py` | Finds where each episode's interview starts, after the teaser, title sequence and ads, by matching title footage shared across episodes |
| `standing/make_report.py` | Turns a detector run plus `results/verdicts.csv` into the results files |
| `standing/review_video.py` | Joins every standing section into one captioned video to watch (`standing/output/review.mp4`) |
| `standing/models/` | The pose models (ONNX, exported from Ultralytics YOLO11; AGPL-3.0) |
| `service/app.py` | HTTP API: upload a video, get a CSV of standing sections in the interview back; the video is deleted after processing |
| `Dockerfile` | The API as one container (ffmpeg, OpenVINO, model) for local use or Azure Container Apps |
| `docs/azure.md` | Running it on Azure so you pay only for compute, with costs |
| `samples/` | The format of the local-only files, with made-up names |

**Kept locally, not in this repository** (see `.gitignore`): the videos, the yt-dlp
`cookies.txt` and download archive, the video inventory (`analysis/`), the results and manual
review (`results/`), the detector's output (`standing/output/`), and the bank of title frames
(`standing/models/intro_bank.npz`), which is made from the videos.

## Quick start

```bash
/usr/local/bin/python3.12 -m venv .venv
.venv/bin/pip install -r service/requirements.txt
.venv/bin/pip uninstall -y openvino-telemetry
```

The last line removes OpenVINO's telemetry package, which otherwise sends a usage event to
Google Analytics on every import.

Build the bank of title frames from your episodes (once, and again after a new season), find
standing sections, then build the report:

```bash
.venv/bin/python standing/intro.py --build-bank *.mp4 *.mkv *.webm
.venv/bin/python standing/detect_standing.py *.mp4 *.mkv *.webm --out standing/output
.venv/bin/python standing/make_report.py
```

Without a bank the detector still runs, but treats each whole video as the interview.

Run the API and send it a video:

```bash
API_KEY=change-me .venv/bin/uvicorn service.app:app --port 8000
curl -H "x-api-key: change-me" -X POST --data-binary @episode.mp4 "http://localhost:8000/jobs?name=episode.mp4"
curl -H "x-api-key: change-me" http://localhost:8000/jobs/<job_id>/sections.csv
```

Or as a container: `docker build -t standing-api .` and
`docker run -p 8000:8000 -e API_KEY=change-me standing-api`. The image includes the bank if you
have built it locally.

## Speed

At 1 frame per second, input size 640, on a 4-core Intel Mac with OpenVINO: about 51 ms per
frame, so 100 hours of video take about 5 hours on this machine. In a 2-vCPU container a
63-minute episode took 8.9 minutes. See [`docs/azure.md`](docs/azure.md) for cloud costs.
