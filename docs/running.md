# Running everything, step by step

This is the full procedure: set up, get the videos, run every action, review the results, run
the API, and run the tests. Every command is run from the `youtube-analyser` folder.

## 1. Set up (once)

You need Python 3.11 or later and ffmpeg (with ffprobe, version 5.1 or later) on the PATH. The
models come with the package.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"                 # the package, with OpenVINO, the API, Azure and yt-dlp
pip uninstall -y openvino-telemetry     # OpenVINO otherwise reports each import to Google Analytics
ffmpeg -version | head -1               # check ffmpeg is there
yta --help
```

`pip install -e .` alone is enough for the command line on onnxruntime (slower on Intel CPUs:
add `--runtime onnxruntime` if OpenVINO will not install); the extras are `openvino`, `service`
(the HTTP API), `azure` (Blob and Queue Storage) and `youtube` (yt-dlp), and `dev` adds the
tests. Check the install with the fast tests (seconds; they need no videos):

```bash
pip install -e ".[dev]"
pytest tests/test_rules.py tests/test_units.py
```

## 2. Get the videos

The commands expect yt-dlp's naming, `YYYYMMDD_<11-character id>.<ext>`, so they can build
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

## 2b. Or: search a whole channel in one command

If what you want is "every moment in this channel where someone jumps" or "where the woman with
purple hair talks", `yta channel` does steps 2 to 5 for you: it lists the channel, downloads the
videos at 144p (without the sound when the query does not need it) while it analyses the ones
already downloaded, builds the intro bank from the first episodes, searches each video, and
writes a report with a link to each moment:

```bash
pip install -U yt-dlp                    # once; again whenever YouTube downloads start failing
yta channel "https://www.youtube.com/@SomeShow" "people jumping"
yta channel "https://www.youtube.com/@SomeShow" "woman with purple hair talking"
```

It prints what it understood, then a line per video with how long it took and an estimate of
the time left, for example:

```
looking for: gender female, hair purple; action: talking
212 videos listed; work folder channels/someshow
intro bank built from 8 videos: 31 title frames shared by the episodes
[1/212] 2024-05-02 20240502_sampleVid07.mp4 (58.3 min): 0 results (quick look (612 keyframes): no purple hair seen); downloaded in 9s, searched in 31s
[2/212] 2024-04-25 20240425_sampleVid06.mp4 (61.0 min): 3 results; downloaded in 8s, searched in 14.2 min; 8x real time, about 20 h left
...
```

(The estimate is rough: videos where the person might be take far longer than the rest.) Open
`channels/<channel>/searches/<query>/report.html` at any time: it is rewritten as videos are
searched (at most every 20 s).
Stop with Ctrl-C whenever you like; running the same command again carries on. Useful options:

| Option | What it does |
|---|---|
| `--max-videos 20` | Only the newest 20 videos (try a query on a few first) |
| `--since 20240101`, `--until 2024-12-31` | Only videos uploaded in these dates (also those already downloaded by an earlier search) |
| `--speed fastest` | About 2x faster again: the pose model at a smaller input (finds fewer small, distant people) |
| `--speed thorough` | No quick look: every second of every video (for someone who may be on screen only briefly) |
| `--delete-videos` | Delete each video once searched, to save disk |
| `--cookies cookies.txt` | If YouTube asks yt-dlp to sign in |
| `--shard 1/2` and `--shard 2/2` | Split the channel between two machines (or two runs) sharing the work folder |
| `--possible`, `--explain`, `--config` | As for `yta search` (step 4b) |

A second query on the same channel reuses the downloaded videos and everything computed, so it
is much quicker. To look through the results for one action in the usual way as well (review
video, verdicts, report), run steps 3 to 5 on `channels/<channel>/videos/` with
`--out channels/<channel>/output`.

## 3. Find where each interview starts (once, and after a new title design)

```bash
yta intro --build-bank *.mp4 *.mkv *.webm
yta intro 20240105_sampleVid01.mp4       # optional: check one video's start
```

The bank is written to `intro_bank.npz` in the current folder (`--bank` puts it elsewhere; the
other commands read it from there, or from `--intro-bank`). Without the bank every action still
runs, but treats each whole video as the interview (you will see `no intro bank ... the intro is
not separated`). `--keep-intro` does that on purpose. `yta channel` builds its own bank from the
channel's first long episodes.

## 4. Run the actions

Run standing first: it runs the pose model on every video (the slow part) and saves the result
in `output/poses/`. Every other action reuses it.

```bash
yta detect *.mp4 *.mkv *.webm --out output
yta detect *.mp4 *.mkv *.webm --out output --action sitting
yta detect *.mp4 *.mkv *.webm --out output --action walking
yta detect *.mp4 *.mkv *.webm --out output --action running
yta detect *.mp4 *.mkv *.webm --out output --action jumping
yta detect *.mp4 *.mkv *.webm --out output --action smiling
yta detect *.mp4 *.mkv *.webm --out output --action talking
yta detect *.mp4 *.mkv *.webm --out output --action silent
```

Each prints one line per video, for example
`[3/248] 20240105_sampleVid01.mp4: interview from 0:00:38, 3768 frames, 12 talking sections, 41s (saved poses; 2 voices, sound trusted 100% of the time; 1854 faces measured)`,
and a total at the end. A video that fails (a broken file) prints `FAILED` and the run carries
on; it is tried again next time.

What each action costs after the pose model, and what it saves for the others:

| Action | Extra work | Saved in `output/` and shared |
|---|---|---|
| standing | none | `poses/` |
| sitting, walking, `--motion` | one pass over the picture at 5 frames a second (the camera's movement: about 2 minutes per hour of video on a 4-core machine, a fifth of the pose model's time) | `camera/` |
| running, jumping | the pose model again at 5 fps, only around moments where a whole body is in view | `poses/<video>.dense.npz` |
| smiling | the face pass (about 20% of the pose model's time when faces are in view) | `faces/` |
| talking, silent | the face pass, the camera pass, and the sound (about 35 s per hour) | `faces/`, `camera/`, `audio/` |

So after standing, `--action silent` after `--action talking` takes seconds per video.

Results for an action other than standing are in `output/<action>/`:
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
| `--fps`, `--imgsz`, `--model`, `--jump-fps`, `--conf` | Sampling and model settings (see [actions.md](actions.md#running-it)) |

A video counts as done for an action once `output/<action>/segments/<video>.csv`
exists, so an interrupted run picks up where it stopped. The saved model outputs are checked
and rebuilt automatically when they no longer fit (for example after `--fps` or `--jump-fps`
changes). One thing to know about times: a 1-per-second sample labelled `t` shows the picture
about half a second later (ffmpeg keeps the last frame of each second); sections use the
labels, as standing always has.

## 4b. Find a particular person doing an action

```bash
yta search "woman with purple hair talking" *.mp4 *.mkv *.webm
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
`people.toml` (`yta config` prints it). The search reuses `output/` (run it with the same `--out` as the
detector, the default) and computes and saves whatever is missing, so the first search over new
videos runs the pose model (and, for voice, gender or a name, the face and sound passes that
`--action talking` uses); later searches take seconds per video.

Then open `results/search/<query>/index.html`: one thumbnail per result with the person boxed,
the evidence for each part of the query (`hair purple (100% of 36) [yes]`) and a YouTube link.

Options: `--action` (instead of an action in the query; `visible` for any time they are in view),
`--possible` (also list people who might match: some part not seen, such as hair on a small face),
`--explain` (print what was seen of every person), `--quick` (with a colour in the query, a look
at each video's keyframes first, skipping videos where nobody has it; see step 2b),
`--config my_people.toml`, `--keep-intro`, `--out`, `--results`.

**To find someone by name**, enrol their voice: write `yta config > my_people.toml`, keep only
what you change, and add

```toml
[[people]]
name = "Jane"
voice_from = { video = "20240105_sampleVid01.mp4", start = 120.0, end = 150.0 }   # only she talks here
describe = "woman with purple hair"                                                # optional
```

then `yta search "Jane talking" *.mp4 --config my_people.toml`. Pick 20-30 s where only she
speaks. A voice match is "likely", not certain (see [search.md](search.md)), so check the results.

**If it finds the wrong colours**, run with `--explain` to see what was measured (`top grey (80%
of 10)`), and add an example of the colour from your videos to the palette in your copy of
`people.toml` (colours are `#rrggbb`; a new name becomes a word you can search for).

## 5. Review and report

1. Open `output/<action>/review.html` in a browser: one thumbnail per section, with
   the people drawn (green: doing the action), times and a YouTube link at the start.
2. Or watch them: `yta review --action talking` makes
   `output/talking/review.mp4` (every section with 2 s either side and a caption).
   For standing it uses reviewed sections only; add `--verdicts all` for everything.
3. Write down verdicts, one row per section you checked, in `results/verdicts.csv` (standing) or
   `results/<action>/verdicts.csv`:
   ```
   video,start_s,verdict,note
   20240105_sampleVid01.mp4,950.0,confirmed,
   20240112_sampleVid02.mp4,80.0,false alarm,guest seated square-on
   ```
   Verdicts: `confirmed`, `unclear`, `false alarm`, or `after the interview`.
4. Make the report: `yta report --action talking` writes
   `results/talking/` (`talking_sections.csv` with titles and verdicts, `outside_interview.csv`,
   `interview_starts.csv`, `speech.csv`, `voices.csv` and a `README.md` summary). Without
   `--action` it makes the standing report exactly as before. `--detector-output`, `--results`
   and `--inventory` point it at other folders.

Everything in `output/` and `results/` names the videos, so it stays local (see
`.gitignore`).

## 6. The API

```bash
API_KEY=change-me yta serve --port 8000
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
`yta search`) the job finds that person instead and its result is `matches.csv`; the action comes
from the description or `action=` (default: whenever they are in view), `possible=true` adds
possible matches, and people to find by name are enrolled in the file `PEOPLE_CONFIG` points to
(use `voice_file` recordings there; the server has no other videos). The status shows what was
understood (`looking_for`). The uploaded video, and everything computed from it, is deleted as
soon as it has been analysed. As a container:

```bash
docker build -t youtube-analyser .
docker run -p 8000:8000 -e API_KEY=change-me youtube-analyser
```

See [api.md](api.md) for every call and setting, and [cloud.md](cloud.md) for running it on Azure
(with the work spread over workers).

## 7. Run the tests

```bash
pytest                                              # everything (see below for time); pytest --cov for coverage
pytest tests/test_rules.py tests/test_units.py      # the rules and small pieces alone: seconds
pytest tests/test_talking.py                        # one group
pytest "tests/test_movement.py::Running::test_runners_are_found"   # one test
```

The first full run builds the test clips (`tests/make_fixtures.py`) and runs the detector on
them, 10-25 minutes on a 4-core machine; later runs reuse both and take a few minutes. Delete
`tests/.cache/` to start from nothing. [`tests/README.md`](../tests/README.md) lists what each test
checks and where the test media come from, and [development.md](development.md) how the code is
organised.

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
| `yta channel`: `download failed: ... Sign in to confirm you're not a bot` | YouTube wants a signed-in session: `--cookies cookies.txt` (exported from your browser) or `--cookies-from-browser firefox`; `--yt-dlp-arg "--sleep-requests 1"` slows the requests down |
| `yta channel`: every download fails | Update yt-dlp (`pip install -U yt-dlp`) and follow what its error message says (YouTube changes often) |
| A search lists ignored words | Those words are not in `people.toml`; rephrase, or add them to your copy (`[words]`, and `--config`) |
| Tests fail after editing the clips | Bump `VERSION` in `tests/make_fixtures.py` so the clips and results are rebuilt |

## 9. A full run, in order

1. Download (step 2) and list the videos in `analysis/videos.csv`.
2. `yta intro --build-bank` (step 3).
3. `yta detect` for standing, then each other action you need (step 4).
4. Look at each `review.html`; for the actions you will rely on, watch the review video and
   write verdicts (step 5). The new actions have not been checked on real episodes yet, so do
   this before trusting their counts.
5. `yta report` (with `--action`) for each action.
6. `yta search` for the people you are looking for (step 4b), or `yta channel` for a whole
   channel (step 2b); check the thumbnails.
7. After changing a threshold (in `src/youtube_analyser/`), run the tests (step 7) and then
   `yta detect ... --reclassify` for that action. Changes to
   `people.toml` take effect on the next search.
