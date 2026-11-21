# youtube-analyser

Finds every time range in which someone is standing during the interviews in a folder of
downloaded episodes of a sit-down interview show (each episode's intro is found automatically
and left out), and offers the same detector as an upload API that deletes each video as soon as
it has been analysed.

The same pipeline finds other actions too, chosen with `--action`: **sitting**, **walking**,
**running**, **jumping**, **smiling**, **talking** and **silent** (standing is the default). They reuse the pose
model's output, so after one run the others cost little or nothing extra. Talking and silent
combine the faces with the sound, where it is clear enough, and also list who spoke when by voice
(see [below](#other-actions)).

**Results** are written locally to `results/` and are not committed, because they name the
videos. [`samples/`](samples/README.md) shows every results file with a few made-up rows. On the
test set (248 videos, 210 hours) the detector reported 92 standing sections during the
interviews, and a manual review found 77 confirmed, 6 unclear and 9 false alarms; 27 sections
in intros (nearly all title sequences) were left out automatically.

## What is where

| Path | What |
|---|---|
| `standing/detect_standing.py` | The detector: ffmpeg samples frames, a YOLO pose model finds body points, leg-geometry rules decide standing / sitting / unknown, still figures are dropped. `--action` picks what to look for. [README](standing/README.md) |
| `standing/actions.py` | The rules for walking, running and jumping (people followed over time, with the camera's pan, tilt and zoom taken out) and smiling / talking / silent (faces found from the pose model's eyes, measured by two small face models, each face keeping its own voice) |
| `standing/audio.py` | The sound: when anyone is speaking, whether the sound is clear enough to trust, and which voice (A, B, ...) |
| `standing/intro.py` | Finds where each episode's interview starts, after the teaser, title sequence and ads, by matching title footage shared across episodes |
| `standing/make_report.py` | Turns a detector run plus `results/verdicts.csv` into the results files |
| `standing/review_video.py` | Joins every standing section into one captioned video to watch (`standing/output/review.mp4`) |
| `standing/models/` | The pose models (ONNX, exported from Ultralytics YOLO11; AGPL-3.0) and, for the other actions, MediaPipe's face landmark model (Apache-2.0), a FER2013 expression model (MIT), Silero's speech detector (MIT) and Resemblyzer's voice encoder (Apache-2.0); [details](standing/README.md#models-used-by-the-actions) |
| `service/app.py` | HTTP API: upload a video, get a CSV of standing (or `action=...`) sections in the interview back; the video is deleted after processing |
| `Dockerfile` | The API as one container (ffmpeg, OpenVINO, model) for local use or Azure Container Apps |
| `docs/running.md` | **How to run everything, step by step**: setup, downloading, each action, reviewing, reports, the API, the tests, troubleshooting |
| `docs/azure.md` | Running it on Azure so you pay only for compute, with costs |
| `tests/` | 98 tests: the rules on made-up measurements, standing against the original code, and every action end to end on made clips with known truth. [README](tests/README.md) |
| `samples/` | The format of the local-only files, with made-up names |

**Kept locally, not in this repository** (see `.gitignore`): the videos, the yt-dlp
`cookies.txt` and download archive, the video inventory (`analysis/`), the results and manual
review (`results/`), the detector's output (`standing/output/`), and the bank of title frames
(`standing/models/intro_bank.npz`), which is made from the videos.

## Quick start

The full procedure, with every option, is in [`docs/running.md`](docs/running.md).

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

### Other actions

```bash
.venv/bin/python standing/detect_standing.py *.mp4 *.mkv *.webm --out standing/output --action talking
.venv/bin/python standing/make_report.py --action talking
.venv/bin/python standing/review_video.py --action talking
```

`--action` is `standing` (default), `sitting`, `walking`, `running`, `jumping`, `smiling`, `talking`
or `silent`. Results go to `standing/output/<action>/` and `results/<action>/`. Every action shares
`standing/output/poses/`, so the pose model runs once per video whichever action goes first:

| Action | Extra cost after the pose model | Notes |
|---|---|---|
| sitting | a pass over the picture at 5 frames a second for the camera's movement (about 2 minutes per hour of video) | Only when hips and knees are in view, so not in close-ups; people moving across the picture are not sitting |
| walking | the same pass (shared) | Judged by how fast the body travels between samples, with the camera's pan, tilt and zoom taken out. `--action standing --motion still` gives standing still only |
| running | the pose model again at 5 fps, only around moments where a whole body is in view | 4 torso lengths a second or more (a brisk walk is up to about 3.5) |
| jumping | the same samples (shared with running) | Jumps of under about 15 cm are often missed |
| smiling | one face pass, about 20% of the pose model's time when a face is in view | Only in close-ups (≥ 11 px between the eyes at 144p) |
| talking, silent | the same face pass, plus about 35 s per hour of sound | Where the sound is clear (speech ≥ 20 dB above the background), the sound says whether anyone speaks and which voice, and the faces say whose mouth it is; in noise, the faces alone. Also writes `speech.csv` (who spoke when, on screen or not) and `voices.csv` |

These are tested end to end on made clips with known truth (up to three people talking, walking,
running, jumping, and their edge cases; see [tests/README.md](tests/README.md)), but not yet on the
episodes: review the first results by eye before relying on them, as was done for standing.

### Tests

```bash
.venv/bin/python -m unittest discover -s tests -t . -v     # all 98, about 10 minutes the first time
.venv/bin/python -m unittest tests.test_rules              # the rules alone, seconds
```

Run the API and send it a video:

```bash
API_KEY=change-me .venv/bin/uvicorn service.app:app --port 8000
curl -H "x-api-key: change-me" -X POST --data-binary @episode.mp4 "http://localhost:8000/jobs?name=episode.mp4"
curl -H "x-api-key: change-me" http://localhost:8000/jobs/<job_id>/sections.csv
curl -H "x-api-key: change-me" -X POST --data-binary @episode.mp4 "http://localhost:8000/jobs?name=episode.mp4&action=talking"
```

Or as a container: `docker build -t standing-api .` and
`docker run -p 8000:8000 -e API_KEY=change-me standing-api`. The image includes the bank if you
have built it locally.

## Speed

At 1 frame per second, input size 640, on a 4-core Intel Mac with OpenVINO: about 51 ms per
frame, so 100 hours of video take about 5 hours on this machine. In a 2-vCPU container a
63-minute episode took 8.9 minutes. See [`docs/azure.md`](docs/azure.md) for cloud costs.

The other actions were timed on a slower 4-core Linux machine (its times vary by up to 2× from
run to run), on an 11-minute test video with a face in view 60% of the time and a whole body in
view 36% of the time (far more than in these episodes), after the pose model had run (95 s): the
camera pass for sitting took 20-35 s (walking then reused it in under a second), running 110-180 s
for its 230 s of extra 5-fps samples (jumping then reused them in 4 s), smiling 15-50 s for the
face pass, and talking and silent a few seconds each after it (this video had no sound; sound
adds about 35 s per hour). Running and jumping's cost grows with the time whole bodies are in view,
which in a sit-down show is small.
