# Changelog

## 2.0.0: the `youtube_analyser` package

The scripts in `standing/` and `service/` became one installable package with one command, `yta`.
Results are the same: standing is checked file by file against the original detector, and every
other action against the scripts it replaces.

### Moved and renamed

| Before | Now |
|---|---|
| `python standing/detect_standing.py ...` | `yta detect ...` |
| `python standing/search.py "<query>" ...` | `yta search "<query>" ...` |
| `python standing/channel.py <channel> "<query>"` | `yta channel <channel> "<query>"` |
| `python standing/intro.py [--build-bank] ...` | `yta intro [--build-bank] ...` (the bank is `./intro_bank.npz` by default, no longer inside the models folder) |
| `python standing/make_report.py` | `yta report` |
| `python standing/review_video.py` | `yta review` |
| `uvicorn service.app:app` | `yta serve` (settings from the environment, as before, plus `YTA_STORE`, `YTA_QUEUE`, `YTA_WORKERS`, `INTRO_BANK`) |
| `--out standing/output` (the default) | `--out output` (the default, in the current folder) |
| `standing/models/`, `standing/people.toml` | `src/youtube_analyser/assets/` (inside the package; `YTA_MODELS` points at another folder; `yta config` prints people.toml) |
| `requirements.txt` files | `pip install -e ".[all]"` (extras: `openvino`, `service`, `azure`, `youtube`, `dev`) |
| `unittest` | `pytest` (the tests run the commands in-process; `pytest --cov` measures coverage) |

Model output saved by the previous version in `standing/output/` can be reused: point `--out` at
it (files made with the default settings are accepted).

### New

- **Cloud framework** (`youtube_analyser.cloud`, `jobs.py`): a `Store` and a `Queue` chosen by
  URL (a folder, memory, Azure Blob and Queue Storage, or a plugin), and one `Worker` that runs
  the same pipeline for API uploads and channel searches. The API can now run with its workers
  elsewhere (`yta worker`), which is the Azure set-up with Blob Storage, a queue and a Container
  Apps job (`docs/cloud.md`).
- **Channel searches on workers**: `yta channel --store ... --queue ...` and `yta collect`.
- `VideoAnalysis`: one place that computes and saves every video's model output, shared by all
  commands, the API and the workers.
- `yta models`, `yta config`, `yta worker`, `yta collect`; `--yt-dlp`, cookies and yt-dlp options for
  workers too.

### Fixed

- Saved model output was reused whatever settings had made it (another `--imgsz`, `--model` or
  `--conf`, or another video under the same name): each saved file now records what made it and
  is made again when that changes. Two videos with the same name in different folders are no
  longer mixed up (the second is skipped with a warning).
- Phone videos stored sideways (with a rotation tag) were read as scrambled pictures: they are
  now read upright.
- A video whose saved model output was cut short (a crash while saving) failed every time; the
  output is now made again.
- Building the intro bank crashed when the episodes shared no title frames; a channel without a
  shared intro now simply has no bank.
- Colours and voices tied in a person's votes were decided by Python's hash seed (so could change
  from run to run); the first seen now wins.
- The face and pose models can be used from several threads at once (the API's worker threads).
- `yta report` no longer fails without `results/verdicts.csv` or without the inventory (titles are
  left empty); an inventory with an unknown duration no longer crashes it.
- Channel searches: `--since`/`--until` now also apply to a folder of videos; dates in the channel
  listing are used before downloading; live and upcoming streams are left out; titles with tabs
  or newlines are read correctly from yt-dlp; a failed video is retried on the next run (and on
  workers, three times before it is reported); thumbnails of 5-fps moments use the sampling rate
  actually used (`--jump-fps`).
- The review video no longer mixes up the verdicts asked for with each clip's verdicts.
- Folder stores tidy empty folders safely when two processes delete at once; Azure downloads
  create the folder they are written to.
