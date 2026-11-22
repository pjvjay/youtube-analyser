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
- `--since` / `--until` were ignored for videos already downloaded by an earlier search (whose
  channel listing gave no dates), and a search run again with dates still reported the videos
  its earlier run had searched outside them. A copy already downloaded is now checked against the
  dates (its name holds its upload date), a date known from an earlier download rules a video out
  before anything is done with it, and reports (also `yta collect`'s) only list videos in the
  dates and the listing asked for. Dates are checked when given (`YYYYMMDD` or `YYYY-MM-DD`).
- File names with a colon (`Interview: part 1.mp4`) or starting with a dash could not be read
  (ffmpeg took them for a protocol or an option); they now can.
- A video file with cover art or a thumbnail stored as a video stream could be read from the
  picture's size while the video was decoded (scrambled frames): the video stream is now chosen
  explicitly, and every read uses it.
- The quick look held all of a video's keyframes in memory (gigabytes for long high-definition
  videos), and the still-figure check of long sections in large videos too: both now read as they
  go, or in shorter pieces.
- Workers kept a message hidden for a fixed hour; a video taking longer was handed to a second
  worker, and the first then stopped on an out-of-date receipt. Workers now keep the message
  hidden while they work, carry on if the queue has forgotten it, skip work already done, and fail
  a job whose video was lost with a clear message.
- A single video's address given to `yta channel` was downloaded from its media-file address;
  live and upcoming streams in a listing are left out.
- Long channels: the report was rebuilt from every result file after every video (slow for
  thousands of videos); it now reads only new results and is rewritten at most every 20 s.
- `yta channel` takes a single downloaded video file too; `yta collect` lists failed videos as
  problems and does not count videos found to be outside the dates.
- Settings are checked when given (a confidence of 1.5, a negative number of videos, an odd review
  video size, `--shard 3/2`) instead of failing later; `yta intro --build-bank` says when there
  are too few long videos to find intros.
- An edited `videos.csv` (columns of your own, empty rows, values without a column) and a
  `verdicts.csv` saved by a spreadsheet (byte-order mark, stray spaces, a missing column) are read
  properly or refused with a clear message; a palette colour can be written without brackets.
- Only real YouTube IDs (11 letters, digits, `-` or `_`) in file names make links.
- The review video no longer mixes up the verdicts asked for with each clip's verdicts.
- Folder stores tidy empty folders safely when two processes delete at once; Azure downloads
  create the folder they are written to.
- `--since` together with `--until` found nothing on a channel with five or more videos newer
  than `--until` (and no dates in its listing): the newer videos were counted towards "5 in a
  row older than `--since`", so the run stopped before it reached the dates asked for and
  reported "0 of N videos searched". Only videos older than `--since` count now.
- A video that yt-dlp leaves out for its date was taken for a failed download: used with
  `--print`, yt-dlp says nothing about such a video. yt-dlp is now asked to print each video's
  upload date before its own date check, which also tells a video too new from one too old.
