# youtube-analyser

A Python package (`youtube_analyser`, command `yta`) that finds people doing things in YouTube
videos, cheaply, at the lowest resolution (144p):

- **Actions** (`yta detect`): every time range in which someone is **standing**, **sitting**,
  **walking**, **running**, **jumping**, **smiling**, **talking** or **silent**, in a folder of
  downloaded episodes of a sit-down interview show. Each episode's intro is found automatically
  and set apart; talking and silent combine the faces with the sound, and also list who spoke when.
- **A described person** (`yta search`): "woman with purple hair talking", "man in a blue coat
  walking", "someone with a deep voice", or someone enrolled by name (found by their voice).
- **A whole channel in one command** (`yta channel`): from the channel's address to a report with
  a link to each moment found. It downloads the videos at 144p while it analyses the ones already
  downloaded, computes only what the query needs, and rules most videos out in seconds with a
  quick look at their keyframes.
- **An upload API** (`yta serve`) and **workers** (`yta worker`) that run the same pipeline on one
  machine or in the cloud (Azure Blob and Queue Storage, or any provider plugged in), deleting
  each video as soon as it has been analysed.

On the original test set (248 videos, 210 hours) the standing detector reported 92 standing
sections during the interviews; a manual review found 77 confirmed, 6 unclear and 9 false
alarms, and 27 sections in intros (nearly all title sequences) were set apart automatically.
The other actions and the search are tested end to end on made clips with known truth, not yet
on real episodes: review their first results by eye before relying on them.

## Install

Python 3.11 or later, and ffmpeg with ffprobe (version 5 or later) on the PATH. The models come
with the package.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"                 # or pick extras: openvino, service, azure, youtube
pip uninstall -y openvino-telemetry     # OpenVINO otherwise sends a usage event on every import
```

`pip install -e .` alone is the command line on onnxruntime. The extras: `openvino` (about 1.4-1.8x
faster on Intel CPUs), `service` (the HTTP API), `azure` (Blob and Queue Storage), `youtube`
(yt-dlp, for `yta channel`; keep it up to date with `pip install -U yt-dlp`), and `dev` (the
tests). `pip wheel .` builds a wheel with the models inside (about 70 MB).

## Quick start

```bash
yta detect *.mp4 --action talking                       # -> output/talking/segments.csv, review.html
yta search "woman with purple hair talking" *.mp4        # -> results/search/<query>/matches.csv, index.html
yta channel "https://www.youtube.com/@SomeShow" "people jumping"
yta channel "https://www.youtube.com/@SomeShow" "woman with purple hair talking" --max-videos 50
                                                         # -> channels/<channel>/searches/<query>/report.html
API_KEY=change-me yta serve                              # the HTTP API on port 8000
```

From Python:

```python
from pathlib import Path
from youtube_analyser import Searcher, load_people, parse_query

config = load_people()
searcher = Searcher(config, cache="output")              # model output is saved and reused
for r in searcher.search(Path("20240105_sampleVid01.mp4"), parse_query("man talking", config)):
    print(r["start"], r["end"], r["youtube"], r["evidence"])
```

## Documentation

| | |
|---|---|
| [docs/running.md](docs/running.md) | **Step by step**: setup, downloading, each action, reviewing, reports, searching, the API, the tests, troubleshooting |
| [docs/actions.md](docs/actions.md) | How each action is found, what it misses, what was measured, the files written, the models used |
| [docs/search.md](docs/search.md) | Searching for a described person: the words a query can use, how people are matched, what to expect at 144p, enrolling people by name |
| [docs/channel.md](docs/channel.md) | Searching a whole channel: what it does, speed (measured), options, more machines |
| [docs/api.md](docs/api.md) | The HTTP API: calls, errors, settings, privacy |
| [docs/cloud.md](docs/cloud.md) | The cloud framework (swappable stores and queues, workers), Azure set-ups and costs, adding a provider |
| [docs/development.md](docs/development.md) | How the code is organised, the tests and coverage, building |
| [tests/README.md](tests/README.md) | What each test checks, the made test clips and where their media come from |
| [samples/README.md](samples/README.md) | The format of every results file, with made-up rows |

## What is where

| Path | What |
|---|---|
| `src/youtube_analyser/` | The package ([development.md](docs/development.md) has a map of its modules) |
| `src/youtube_analyser/assets/` | The models (ONNX) and the default `people.toml` (the search's words, colours and thresholds) |
| `tests/` | The tests (pytest), the clip maker and the test media |
| `docs/`, `samples/` | Documentation, and made-up examples of every results file |
| `Dockerfile` | One image for the API and the workers (ffmpeg, OpenVINO, the models) |

**Kept locally, not in this repository** (see `.gitignore`): the videos, the yt-dlp
`cookies.txt` and download archive, the video inventory (`analysis/`), the results and manual
review (`results/`), the detector's output (`output/`), the channel searches (`channels/`), and
the bank of title frames (`intro_bank.npz`), which is made from the videos.

## Speed

At 1 frame per second, input size 640, on a 4-core Intel Mac with OpenVINO: about 51 ms per frame
for the pose model, so 100 hours of video take about 5 hours for standing. In a 2-vCPU container
a 63-minute episode took 8.9 minutes. The other actions cost little on top (the pose model's
output is shared; see [actions.md](docs/actions.md)), and a channel search for someone who is in
few videos ran at 115-230x real time after the quick look ([channel.md](docs/channel.md#speed)).

## Licences

The code's licence is for its owner to choose; note what the bundled models require:

| Model | Licence |
|---|---|
| `yolo11s-pose.onnx`, `yolo11n-pose.onnx` (Ultralytics YOLO11 pose) | **AGPL-3.0**: distributing the package or an image with them (or serving them to others over a network) brings AGPL obligations |
| `face_landmark.onnx`, `person_mask.onnx` (MediaPipe) | Apache-2.0 |
| `face_expression.onnx` (mini-Xception, the `fer` package) | MIT; trained on FER2013, collected for a research challenge |
| `voice_activity.onnx` (Silero VAD) | MIT |
| `voice_embedding.onnx` (Resemblyzer) | Apache-2.0 |
| `gender_age.onnx` (yu4u/age-gender-estimation) | MIT code, **trained on IMDB-WIKI, released for academic research only**; switch it off in people.toml (`[models] gender_age = false`) if that does not suit your use |

Download only videos you are allowed to.
