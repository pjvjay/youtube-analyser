# Running everything, step by step

This is the full procedure: set up, get the videos, run every action, review the results, run
the API, and run the tests. Every command is run from the `youtube-analyser` folder.

## 1. Set up (once)

You need Python 3.12, ffmpeg (with ffprobe) on the PATH, and about 1 GB of disk for the Python
packages. Nothing else: all models are in `standing/models/`.

```bash
/usr/local/bin/python3.12 -m venv .venv
.venv/bin/pip install -r service/requirements.txt      # also installs standing/requirements.txt
.venv/bin/pip uninstall -y openvino-telemetry             # OpenVINO otherwise reports each import to Google Analytics
ffmpeg -version | head -1                                 # check ffmpeg is there
```

Check the install with the fast tests (a few seconds; they need no videos):

```bash
.venv/bin/python -m unittest tests.test_rules tests.test_standing
```

If OpenVINO will not install on your machine, everything also runs on onnxruntime (slower on
Intel CPUs): add `--runtime onnxruntime` to the commands below.

## 2. Get the videos

The scripts expect yt-dlp's naming, `YYYYMMDD_<11-character id>.<ext>`, so they can build
YouTube links; other names work but get no links. The lowest resolution keeps the cost down
(everything was built and tested for 144p), but keep the sound for talking and silent:

```bash
yt-dlp -f "worstvideo[height>=144]+worstaudio/worst" -o "%(upload_date)s_%(id)s.%(ext)s" \
       --download-archive archive.txt --cookies cookies.txt "<channel URL>"
```

yt-dlp can leave unmerged `X.f278.webm` files next to `X.webm`; the detector skips them.

Then list the videos in `analysis/videos.csv` (used by the reports for titles and durations;
see [`samples/analysis/videos.csv`](../samples/analysis/videos.csv)):

```
file,upload_date,video_id,title,kind,duration_min
20240105_sampleVid01.mp4,2024-01-05,sampleVid01,Guest A | EP 1,long,62.8
```

## 3. Find where each interview starts (once, and after a new title design)

```bash
.venv/bin/python standing/intro.py --build-bank *.mp4 *.mkv *.webm
.venv/bin/python standing/intro.py 20240105_sampleVid01.mp4       # optional: check one video's start
```

Without the bank every action still runs, but treats each whole video as the interview (you
will see `no intro bank ... the intro is not separated`). `--keep-intro` does that on purpose.

## 4. Run the actions

Run standing first: it runs the pose model on every video (the slow part) and saves the result
in `standing/output/poses/`. Every other action reuses it.

```bash
.venv/bin/python standing/detect_standing.py *.mp4 *.mkv *.webm --out standing/output
.venv/bin/python standing/detect_standing.py *.mp4 *.mkv *.webm --out standing/output --action sitting
.venv/bin/python standing/detect_standing.py *.mp4 *.mkv *.webm --out standing/output --action walking
.venv/bin/python standing/detect_standing.py *.mp4 *.mkv *.webm --out standing/output --action running
.venv/bin/python standing/detect_standing.py *.mp4 *.mkv *.webm --out standing/output --action jumping
.venv/bin/python standing/detect_standing.py *.mp4 *.mkv *.webm --out standing/output --action smiling
.venv/bin/python standing/detect_standing.py *.mp4 *.mkv *.webm --out standing/output --action talking
.venv/bin/python standing/detect_standing.py *.mp4 *.mkv *.webm --out standing/output --action silent
```

Each prints one line per video, for example
`[3/248] 20240105_sampleVid01.mp4: interview from 0:00:38, 3768 frames, 12 talking sections, 41s (saved poses; 2 voices, sound trusted 100% of the time; 1854 faces measured)`,
and a total at the end. A video that fails (a broken file) prints `FAILED` and the run carries
on; it is tried again next time.

What each action costs after the pose model, and what it saves for the others:

| Action | Extra work | Saved in `standing/output/` and shared |
|---|---|---|
| standing | none | `poses/` |
| sitting, walking, `--motion` | one pass over the picture at 5 frames a second (the camera's movement: about 2 minutes per hour of video on a 4-core machine, a fifth of the pose model's time) | `camera/` |
| running, jumping | the pose model again at 5 fps, only around moments where a whole body is in view | `poses/<video>.dense.npz` |
| smiling | the face pass (about 20% of the pose model's time when faces are in view) | `faces/` |
| talking, silent | the face pass, the camera pass, and the sound (about 35 s per hour) | `faces/`, `camera/`, `audio/` |

So after standing, `--action silent` after `--action talking` takes seconds per video.

Results for an action other than standing are in `standing/output/<action>/`:
`segments.csv` (the sections, from the interview start on), `intro.csv`, `dropped.csv`
(still figures), `interview_starts.csv`, `review.html` (thumbnails with YouTube links), and
per-video copies in `segments/`, `frames/` (every sampled frame) and `thumbs/`. Talking and
silent also write `speech.csv` (who spoke when, by voice, and whether they were on screen) and
`voices.csv` (each video's voices, with a thumbnail of each voice's face).

Useful options (all actions):

| Option | What it does |
|---|---|
| `--keep-intro` | Treat each whole video as the interview |
| `--reclassify` | Redo the sections of videos already done, from the saved model output (after changing a threshold): seconds per video |
| `--force` | Rerun everything, the pose model included |
| `--runtime onnxruntime` | Use onnxruntime instead of OpenVINO |
| `--threads N` | Limit the model to N threads |
| `--motion still` / `moving` | Standing or sitting: only people standing still (or walking); standing's default keeps its results exactly as before, sitting's default is `still` |
| `--no-sound` | Talking and silent from the faces alone |
| `--min-frames`, `--max-gap` | How many frames make a section and how many may be missing inside one |
| `--fps`, `--imgsz`, `--model`, `--jump-fps`, `--conf` | Sampling and model settings (see `standing/README.md`) |

A video counts as done for an action once `standing/output/<action>/segments/<video>.csv`
exists, so an interrupted run picks up where it stopped. The saved model outputs are checked
and rebuilt automatically when they no longer fit (for example after `--fps` or `--jump-fps`
changes). One thing to know about times: a 1-per-second sample labelled `t` shows the picture
about half a second later (ffmpeg keeps the last frame of each second); sections use the
labels, as standing always has.

## 4b. Find a particular person doing an action

```bash
.venv/bin/python standing/search.py "woman with purple hair talking" *.mp4 *.mkv *.webm
```

It prints what it understood first, then one line per video and a total, for example:

```
looking for: gender female, hair purple; action: talking
[1/248] 20240105_sampleVid01.mp4: 2 results
...
14 results (14 matches) -> results/search/woman-with-purple-hair-talking/matches.csv, .../index.html
3 more people might match (not seen well enough to tell); --possible lists them, --explain says why
```

Check the `looking for:` line: any word it did not understand is listed as `ignored words`
(say, "beard" or "glasses", which it cannot see). The words it knows are in
`standing/people.toml`. The search reuses `standing/output/` (run it with the same `--out` as the
detector, the default) and computes and saves whatever is missing, so the first search over new
videos runs the pose model (and, for voice, gender or a name, the face and sound passes that
`--action talking` uses); later searches take seconds per video.

Then open `results/search/<query>/index.html`: one thumbnail per result with the person boxed,
the evidence for each part of the query (`hair purple (100% of 36) [yes]`) and a YouTube link.

Options: `--action` (instead of an action in the query; `visible` for any time they are in view),
`--possible` (also list people who might match: some part not seen, such as hair on a small face),
`--explain` (print what was seen of every person), `--config my_people.toml`, `--keep-intro`,
`--out`, `--results`.

**To find someone by name**, enrol their voice: copy `standing/people.toml` to, say,
`my_people.toml`, keep only what you change, and add

```toml
[[people]]
name = "Jane"
voice_from = { video = "20240105_sampleVid01.mp4", start = 120.0, end = 150.0 }   # only she talks here
describe = "woman with purple hair"                                                # optional
```

then `search.py "Jane talking" *.mp4 --config my_people.toml`. Pick 20-30 s where only she
speaks. A voice match is "likely", not certain (see `standing/README.md`), so check the results.

**If it finds the wrong colours**, run with `--explain` to see what was measured (`top grey (80%
of 10)`), and add an example of the colour from your videos to the palette in your copy of
`people.toml` (colours are `#rrggbb`; a new name becomes a word you can search for).

## 5. Review and report

1. Open `standing/output/<action>/review.html` in a browser: one thumbnail per section, with
   the people drawn (green: doing the action), times and a YouTube link at the start.
2. Or watch them: `.venv/bin/python standing/review_video.py --action talking` makes
   `standing/output/talking/review.mp4` (every section with 2 s either side and a caption).
   For standing it uses reviewed sections only; add `--verdicts all` for everything.
3. Write down verdicts, one row per section you checked, in `results/verdicts.csv` (standing) or
   `results/<action>/verdicts.csv`:
   ```
   video,start_s,verdict,note
   20240105_sampleVid01.mp4,950.0,confirmed,
   20240112_sampleVid02.mp4,80.0,false alarm,guest seated square-on
   ```
   Verdicts: `confirmed`, `unclear`, `false alarm`, or `after the interview`.
4. Make the report: `.venv/bin/python standing/make_report.py --action talking` writes
   `results/talking/` (`talking_sections.csv` with titles and verdicts, `outside_interview.csv`,
   `interview_starts.csv`, `speech.csv`, `voices.csv` and a `README.md` summary). Without
   `--action` it makes the standing report exactly as before. `--detector-output`, `--results`
   and `--inventory` point it at other folders.

Everything in `standing/output/` and `results/` names the videos, so it stays local (see
`.gitignore`).

## 6. The API

```bash
API_KEY=change-me .venv/bin/uvicorn service.app:app --port 8000
curl http://localhost:8000/healthz                                               # lists the actions
curl -H "x-api-key: change-me" -X POST --data-binary @episode.mp4 "http://localhost:8000/jobs?name=episode.mp4&action=talking"
curl -H "x-api-key: change-me" http://localhost:8000/jobs/<job_id>                 # queued | processing | done | failed
curl -H "x-api-key: change-me" http://localhost:8000/jobs/<job_id>/sections.csv
curl -H "x-api-key: change-me" http://localhost:8000/jobs/<job_id>/speech.csv      # talking and silent
curl -H "x-api-key: change-me" -X POST --data-binary @episode.mp4 "http://localhost:8000/jobs?name=episode.mp4" \
     --url-query "who=woman with purple hair talking"
curl -H "x-api-key: change-me" http://localhost:8000/jobs/<job_id>/matches.csv     # with who=
curl -H "x-api-key: change-me" -X DELETE http://localhost:8000/jobs/<job_id>
```

`action` is any of the actions above (default standing). With `who=` (a description, as for
`search.py`) the job finds that person instead and its result is `matches.csv`; the action comes
from the description or `action=` (default: whenever they are in view), `possible=true` adds
possible matches, and people to find by name are enrolled in the file `PEOPLE_CONFIG` points to
(use `voice_file` recordings there; the server has no other videos). The status shows what was
understood (`looking_for`). The uploaded video, and everything computed from it, is deleted as
soon as it has been analysed. As a container:

```bash
docker build -t standing-api .
docker run -p 8000:8000 -e API_KEY=change-me standing-api
```

See [`docs/azure.md`](azure.md) for running it on Azure.

## 7. Run the tests

```bash
.venv/bin/python -m unittest discover -s tests -t . -v       # everything (see below for time)
.venv/bin/python -m unittest tests.test_rules                # the rules alone: seconds
.venv/bin/python -m unittest tests.test_talking -v           # one group
.venv/bin/python -m unittest tests.test_movement.Running.test_runners_are_found   # one test
```

The first full run builds the test clips (`tests/make_fixtures.py`) and runs the detector on
them, 5-10 minutes on a 4-core machine; later runs reuse both and take a few minutes. Delete
`tests/.cache/` to start from nothing. [`tests/README.md`](../tests/README.md) lists what each test
checks and where the test media come from.

## 8. When something goes wrong

| You see | It means / do |
|---|---|
| `no intro bank at ...` | Step 3 was not done (or `--intro-bank` points elsewhere): whole videos are treated as the interview |
| `openvino is not installed; use --runtime onnxruntime` | Install `openvino`, or use onnxruntime |
| `FAILED <error>` for a video | That file could not be read or analysed; the others carry on. Check it plays (`ffprobe video.mp4`) |
| `no readable video stream` | The file has no video (an audio-only download): download it again with video |
| Talking says `no sound` | The file has no sound track: talking and silent fall back to the faces |
| Talking says `sound trusted 0% of the time` | The background is too loud (crowd, music): talking falls back to the faces, voices are unreliable |
| Nothing found for walking, running or jumping | They need people's shoulders and hips (running and jumping: their whole body) in view: close-ups are never judged |
| A change of thresholds has no effect | Run again with `--reclassify` (done videos are otherwise skipped) |
| A search finds nobody | Run it with `--explain`: each person's hair, clothes, gender and voice as seen. Faces too small for hair (under 10 px between the eyes) or gender (12 px) are "not seen": add `--possible`. Gender by voice needs them seen talking |
| A search lists ignored words | Those words are not in `standing/people.toml`; rephrase, or add them there (`[words]`) |
| Tests fail after editing the clips | Bump `VERSION` in `tests/make_fixtures.py` so the clips and results are rebuilt |

## 9. A full run, in order

1. Download (step 2) and list the videos in `analysis/videos.csv`.
2. `intro.py --build-bank` (step 3).
3. `detect_standing.py` for standing, then each other action you need (step 4).
4. Look at each `review.html`; for the actions you will rely on, watch the review video and
   write verdicts (step 5). The new actions have not been checked on real episodes yet, so do
   this before trusting their counts.
5. `make_report.py` (with `--action`) for each action.
6. `search.py` for the people you are looking for (step 4b); check the thumbnails.
7. After changing a threshold in `standing/actions.py` or `standing/audio.py`, run the tests
   (step 7) and then `detect_standing.py ... --reclassify` for that action. Changes to
   `people.toml` take effect on the next search.
