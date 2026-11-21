# Development

## The package

```
src/youtube_analyser/
  cli.py          the `yta` command (detect, search, channel, collect, intro, report, review, serve, worker, models, config)
  media.py        ffmpeg and ffprobe: frames at a rate, windows, keyframes, single pictures, the sound, file names and links
  inference.py    the networks on OpenVINO or onnxruntime: PoseModel, FaceModels, VoiceModels, AppearanceModels, Models
  pose.py         standing / sitting from body points (the original rules, unchanged)
  motion.py       the camera's pan, tilt and zoom; walking, running, jumping; following people between samples
  faces.py        faces from the pose model's eyes; smiling; talking and silent with the sound; each face's own voice
  audio.py        speech, how clear the sound is, voices, pitch
  actions.py      the actions' table, and who does one in each sampled frame (label_people)
  analysis.py     Settings and VideoAnalysis: everything computed about one video, made once and saved
  detector.py     sections (joining frames), still figures, intros, the files of `yta detect`
  intro.py        where each episode's interview starts (the bank of shared title frames)
  people.py       hair and clothes colours, apparent gender and age, people followed through a shot, voices by name
  query.py        reading a search query and checking a person against it
  search.py       Searcher: a described person doing an action, cheapest step first
  youtube.py      listing a channel and downloading with yt-dlp
  channel.py      a whole channel: downloads overlapping the search, the report; the same on cloud workers
  cloud/          Store and Queue (base.py), a folder and local queues (local.py), Azure (azure.py), the URL registry
  jobs.py         JobStore and Worker: the same work for the API, the workers and channel searches
  service.py      the HTTP API (FastAPI)
  report.py       the results files of `yta report`        review.py   the review video of `yta review`
  assets/         the models (ONNX) and the default people.toml
```

## The pipeline

Every command goes through the same few steps for a video:

1. **`VideoAnalysis`** (analysis.py) gives the model output a step asks for: `poses` (the pose
   model at 1 fps), `camera`, `dense` (5 fps around whole bodies), `faces`, `sound` and
   `appearance`. Each part is made only when asked for, from the parts it depends on, and saved
   in the cache folder with a record of what made it (the pose model, its input size, the
   sampling, the video file's size, the step's version): a change of any of these makes it again,
   so no output made with one setting is used with another. One `Models` set per process
   (`Models.shared`) loads each network once, safely shared between threads.
2. **`actions.label_people`** labels everyone in every sampled frame for an action, from the
   poses and what that action needs (`VideoAnalysis.inputs`).
3. **`detector.detect`** joins the frames into sections (and drops still figures, splits off the
   intro); **`Searcher.search`** instead follows people through shots, checks them against the
   query, and joins the frames of those who match.

`yta detect`, `yta search`, `yta channel`, the API's jobs and the cloud workers are thin layers
over these three, so what they find is the same (the tests check the API and the cloud workers
against the command line).

## Tests

```bash
pip install -e ".[dev]"
pytest                                    # everything: 10-30 minutes the first time (clips are made and analysed)
pytest tests/test_rules.py tests/test_units.py     # seconds
pytest --cov --cov-report=term-missing    # with coverage
```

The end-to-end tests make their clips with `tests/make_fixtures.py` into `tests/.cache/` and keep
the model output there, so later runs take a few minutes. The Azure tests need the Azurite
emulator (`npm install -g azurite`; `tests/conftest.py` starts it) or a real account in
`AZURE_STORAGE_CONNECTION_STRING`; without either they are skipped. `DETECTOR_OPTIONS="--imgsz
320" pytest ...` runs the end-to-end tests with other pose model settings (see
[tests/README.md](../tests/README.md)).

Coverage of the package at 2.0.0: **97%** of its lines (191 tests, `pytest --cov` from a clean
cache, with Azurite); keep it above 90%. What is not covered is mostly error handling that needs
a broken runtime, storage or ffmpeg to reach, the code paths without OpenVINO or on Windows, and
`yta serve`'s call into uvicorn (the app itself is tested in-process).

## Changing things

- **A threshold**: they are at the top of pose.py, motion.py, faces.py and audio.py, each with
  the measurements behind it. Run the tests, then `yta detect ... --reclassify`.
- **The search's words, colours and thresholds**: `assets/people.toml` (users override it with
  `--config`). `tests/test_rules.py` checks the parser on many phrasings.
- **A new action**: add it to `actions.ACTIONS` (its states, measurements and section rules) and
  its labelling to `actions.label_people`, using `VideoAnalysis` for any model output it needs (a
  new part there is a `cached_property` with its own saved file and key). The detector, search,
  channel, API, reports and review video then take it as they take the others.
- **A new storage or queue provider**: see [cloud.md](cloud.md#adding-another-provider).
- **The models**: `assets/models/`, or a folder of your own in `YTA_MODELS`. The pose model was
  exported from Ultralytics' `yolo11s-pose` weights with `yolo export format=onnx dynamic=True
  simplify=True`; how each other model was converted is in [actions.md](actions.md#models-used-by-the-actions).

## Building

`pip wheel .` (or `python -m build`) makes the wheel, with the models (about 70 MB; the smaller
`yolo11n-pose.onnx` stays in the repository for `--model` and is not packaged). The `Dockerfile`
builds the wheel and installs it with all extras: one image for the API and the workers.
