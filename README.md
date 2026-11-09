# youtube-analyser

Finds every time range in which someone is standing in the downloaded episodes of the YouTube
channel *Private Talk With Alexis Texas*, and offers the same detector as an upload API that
deletes each video as soon as it has been analysed.

**Results:** [`results/README.md`](results/README.md) has the analysis of all 248 videos
(210 hours). [`results/standing_sections.csv`](results/standing_sections.csv) has every standing
section with its title, start and end, a YouTube link that opens at the section, and a verdict
from a manual review: 105 confirmed, 6 unclear, 9 false alarms.

## What is where

| Path | What |
|---|---|
| `standing/detect_standing.py` | The detector: ffmpeg samples frames, a YOLO pose model finds body points, leg-geometry rules decide standing / sitting / unknown, still figures are dropped. [README](standing/README.md) |
| `standing/make_report.py` | Turns a detector run plus `results/verdicts.csv` into the results files |
| `standing/models/` | The pose models (ONNX, exported from Ultralytics YOLO11; AGPL-3.0) |
| `standing/output/` | The full run: `segments.csv`, `dropped.csv` and per-video CSVs (pose data, frames and thumbnails are not committed) |
| `results/` | Analysis, the timestamp list, and the manual verdicts |
| `service/app.py` | HTTP API: upload a video, get a CSV of standing sections back; the video is deleted after processing |
| `Dockerfile` | The API as one container (ffmpeg, OpenVINO, model) for local use or Azure Container Apps |
| `docs/azure.md` | Running it on Azure so you pay only for compute, with costs |
| `analysis/` | Inventory of the downloaded videos (titles, formats, what is missing) |

The videos themselves (6.8 GB) and the yt-dlp `cookies.txt` are not in this repository.

## Quick start

```bash
/usr/local/bin/python3.12 -m venv .venv
.venv/bin/pip install -r service/requirements.txt
.venv/bin/pip uninstall -y openvino-telemetry
```

The last line removes OpenVINO's telemetry package, which otherwise sends a usage event to
Google Analytics on every import.

Find standing sections in some videos, then build the report:

```bash
.venv/bin/python standing/detect_standing.py *.mp4 *.mkv *.webm --out standing/output
.venv/bin/python standing/make_report.py
```

Run the API and send it a video:

```bash
API_KEY=change-me .venv/bin/uvicorn service.app:app --port 8000
curl -H "x-api-key: change-me" -X POST --data-binary @episode.mp4 "http://localhost:8000/jobs?name=episode.mp4"
curl -H "x-api-key: change-me" http://localhost:8000/jobs/<job_id>/sections.csv
```

Or as a container: `docker build -t standing-api .` and
`docker run -p 8000:8000 -e API_KEY=change-me standing-api`.

## Speed

At 1 frame per second, input size 640, on a 4-core Intel Mac with OpenVINO: about 51 ms per
frame, so 100 hours of video take about 5 hours on this machine. In a 2-vCPU container a
63-minute episode took 8.9 minutes. See [`docs/azure.md`](docs/azure.md) for cloud costs.
