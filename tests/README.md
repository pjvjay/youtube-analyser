# Tests

```bash
pytest                                              # all of them
pytest tests/test_rules.py tests/test_units.py      # the rules and small pieces: seconds, no video
pytest --cov --cov-report=term-missing              # with coverage of the package
```

The first full run builds the test clips and runs the models on them (10-30 minutes on a
4-core machine); later runs reuse both from `tests/.cache/` and take a few minutes. Delete that
folder to start again. The commands are run in the test process (`helpers.yta`), so the models
load once and coverage sees everything. The Azure tests start the Azurite storage emulator when
it is installed (`npm install -g azurite`, or `AZURITE=/path/to/azurite`), or use the account in
`AZURE_STORAGE_CONNECTION_STRING`; without either they are skipped.

| File | What it checks | Needs |
|---|---|---|
| `test_rules.py` | Every decision rule on made-up measurements: walking, running and jumping from body points and camera movements; the camera measurement itself; talking and silent from mouth movement and what the sound heard; whose voice is whose; speech detection, how clear the sound is, and how voices are grouped; for the search, naming colours, reading queries, the match / possible / no verdicts, gender from voice before face, voting over a person's samples, following people through a shot (and not between two hair colours), pitch, and naming voices | nothing (seconds) |
| `test_units.py` | File names and links; videos that are not videos, sideways phone videos (read upright), keyframes; the intro bank with no shared titles; votes decided the same whatever Python's hash seed; yt-dlp's listings and downloads through a stand-in (dates, live streams, titles with tabs, cookies, a silent copy downloaded again for its sound, failures); the channel report | ffmpeg (seconds) |
| `test_standing.py` | Standing is exactly as before: the original code (git commit `7bf602d`) and the current one agree on 240 made-up pose sequences (a golden file), and the original script and `yta detect` write the same files on two clips, byte for byte (the saved model output holds the same arrays plus a record of what made it) | git history for the second |
| `test_movement.py` | Walking, running and jumping end to end on 19 made clips | the models |
| `test_talking.py` | Up to three people talking, with and without usable sound, and voices across cuts | the models |
| `test_sound.py` | Speech and voices on real speech: one, two and three voices, crowd noise, silence, a brief extra voice, no sound | the models |
| `test_search.py` | Searching for people by hair colour, clothes, apparent gender, voice and an enrolled name, with and without an action, on made clips of a woman (hair dyed purple or blue, or natural) and a man taking turns to talk, and on the street clips; an action alone gives the detector's sections; a video where nobody has the colour asked for stops after the colours (no faces, sound or camera pass); the quick look at keyframes; the face model run only on the few faces needed; the command line and its files | the models |
| `test_channel.py` | A whole channel through `yta channel`, with `fake_yt_dlp.py` standing in for yt-dlp (five videos): the report's YouTube links and timestamps, the quick look ending the search in videos without the colour, 144p formats with sound only when needed (and downloading again with sound when a later query needs it), the inventory, carrying on after a stop, reusing videos for a new query, `--max-videos`, `--since`, a failed download, `--delete-videos`, a folder of videos; and five 4-minute episodes sharing a title sequence, whose results there are set apart as intro | the models, ffmpeg's `life` test source |
| `test_cli.py` | Every action through `yta detect`; skipping and `--reclassify`; a broken file, a duplicate name and yt-dlp's part files; model output made again when a setting changes; `--motion` and `--no-sound`; the intro bank built and used; the reports (with and without an inventory) and the review videos; errors, help and `python -m youtube_analyser` | the models |
| `test_service.py` | The HTTP API in-process with its own worker: the same sections as the command line, `who=` (and Jane enrolled from a recording), a broken upload, errors, deleting, uploads too big, a restart with uploads left waiting, results expiring | the models, FastAPI |
| `test_cloud.py` | Every Store (a folder, Azure Blob) and every Queue (in memory, a folder, Azure Queue) against the same checks; URLs and plugins; Azure's sign-in without a connection string; the same Worker analysing an upload through each; a channel search spread over workers giving the same moments as one machine, with a failing video retried then reported, enrolled voices and the intro bank travelling with it, and the command line starting and collecting it | the models, Azurite for the Azure cases |
| `test_edges.py` | Videos with nobody in them; portrait videos; model output saved by an older version (used) or for a replaced video (made again); models shared between threads; enrolment problems; searches with no sound and no face model; a channel folder with dates, shards and a changed query | the models |

## What is covered

| Behaviour | Clips (make_fixtures.py) | Tests |
|---|---|---|
| Walking, stopping, walking back | `walk_stop_walk` | walking verdicts and sections |
| Standing still | `stand_still`, `cut_on_standing` | never walking, running or jumping |
| Camera panning past someone standing | `pan_past_standing` | still, not walking |
| Camera following a walker (he stays put in the picture) | `follow_walker` | walking |
| Camera zooming in on someone standing | `zoom_on_standing` | still, not walking towards the camera |
| Walking towards the camera | `towards_camera` | walking or unknown, never wrong |
| Running across a still camera, and followed by the camera | `run_across`, `follow_runner` | running (once), never walking |
| A brisk walk | `brisk_walk` | walking, not running |
| Two people crossing; walking past a bystander who is hidden for a moment | `two_cross`, `walk_past_bystander` | both right, no swaps |
| A cut to another place | `cut_on_standing`, `cutaways` | nobody followed across it |
| Jumps; two jumpers, once together; a jump right beside someone; a jump while walking | `jump_three`, `two_jumpers`, `jump_beside_someone`, `jump_while_walking` | each jump once, at the right time; the neighbour not jumping |
| Tiny hops, the camera bobbing, stepping up | `tiny_hops`, `camera_bob`, `step_up` | not jumping |
| Three people taking turns, two at once, nobody, a voice off screen, a mouth moving without speech | `three_talkers` | talking and silent per person per second, each person's voice, sections, `speech.csv` |
| Crowd babble; no sound track | `three_talkers_noisy`, `three_talkers_no_sound` | the sound is not trusted, faces decide |
| Two people in close-ups, cutting between them | `cutaways` | each keeps his own voice |
| One, two, three voices; noise; silence; a voice of 3 s | `voices_*` | voice counts, right voice for 95%+ of speech |
| Standing unchanged | made-up poses, `walk_stop_walk`, `two_jumpers` | golden file, byte-identical output |
| A woman with purple, blue or natural hair and a man, taking turns to talk | `duo_purple`, `duo_natural`, `solo_blue` | "woman with purple hair (talking)", "man / woman talking", "deep voice", "woman with blue / brown hair": the right person, at the right times (75%+ of their talking, never more than 1.5 s outside it); nobody for "man with purple hair", "green hair", "purple hair in a white shirt" |
| One person's voice in two videos | `duo_purple` (enrolled, 0-8 s), `solo_blue` | "Jane talking" finds her in both, not the other woman |
| Coats on the street | `two_cross`, `walk_past_bystander` | the person boxed is the one in the blue (beige) coat; "walking in a blue coat" only while he walks; a man too small to see as a man is not listed without `--possible` |

For the movement clips each person's truth is known at every moment, and a verdict is checked
at the picture time of each sample (see `actions.picture_time`), where the truth is the same
for a second either side. A verdict may be "unknown" (the detector declines: someone hidden by
another person, cut off by the edge of the picture, or moving too fast to follow between 1-fps
samples) but never wrong, and each person must get a verdict in at least half of the samples
where one was possible. For talking, a person who is not talking must never be called talking (none was, in 137
face-seconds with sound and 132 without); talkers must be found in at least 75% of their seconds
with the sound (83% were) and 60% without it (70% were), because the source clip's speaker pauses
between words. On the movement clips, 848 steady person-samples are checked: 2% came out unknown
and none wrong.

## Faster settings

`DETECTOR_OPTIONS` runs the end-to-end tests with other pose model settings (into a cache folder
of their own):

```bash
DETECTOR_OPTIONS="--imgsz 320" pytest tests/test_movement.py tests/test_talking.py tests/test_search.py tests/test_channel.py
```

With `--imgsz 320` (what `yta channel --speed fastest` uses) all of these passed. With
`--imgsz 480`, and with the smaller model (`--model yolo11n-pose.onnx`), two
talking tests fell just short of their thresholds (talkers found in 14 of 20 seconds where 15
were needed); the smaller model at 320 also called a man standing still walking while the
camera panned. So 320 with the usual model is the faster setting offered, with the caveat that
these clips have no small, distant people, which a smaller input finds less often.

## What the clips cannot tell you

They are made, not filmed: the moving person is a still cut-out that slides (legs do not swing),
and the three talkers are the same man. The woman in the character clips is a still portrait whose
jaw is moved with the loudness of her voice and whose hair is recoloured; the face model scored
her 0.53-0.67 (0.6 decides), so her gender there comes from her voice. Real episodes will have harder
cases (hats, lighting, hair half out of shot, similar voices). They check the logic, the camera measurement and the
models on 144p pictures end to end, but not how often real people in real episodes are missed.
That needs the review in `docs/running.md` step 5.

## The test media (`fixtures/sources/`, 1.7 MB)

| File | What | From, licence |
|---|---|---|
| `street.jpg` | An empty street (the person removed by inpainting) | Ultralytics' `bus.jpg` sample image (in the `ultralytics` wheel), AGPL-3.0 like the pose models |
| `person.png` | The man in the beige coat from the same photo, cut out (GrabCut) | as above |
| `astronaut.jpg` | A portrait of a woman (astronaut Eileen Collins), 400 × 400 | NASA photo, public domain; shipped as `skimage/data/astronaut.png` in the `scikit-image` 0.25.2 wheel |
| `talker.mp4` | A man talking in a car, 176 x 144, 4 s (the "carphone" test sequence) | shipped in the `scikit-video` 1.1.11 wheel (BSD-3-Clause) |
| `speech_a.ogg` ... `speech_d.ogg` | Four people's speech (speakers FEE078, MEO069, FEE083, MEE009), pauses removed, Opus 48 kbps | AMI Meeting Corpus excerpts shipped in the `pyannote.audio` 4.0.7 source package (AMI: CC BY 4.0) |
| `../standing_golden.json` | Fingerprints of the original standing code's output | made by `python -m tests.test_standing --make-golden` |

`make_fixtures.py` builds every clip from these, deterministically (the woman's mouth is
opened with the package's own face models). `fake_yt_dlp.py` stands in for yt-dlp: it lists a
channel described in a JSON file and "downloads" by copying clips, so the channel tests need no
internet. When you change a clip,
bump its `VERSION` so cached clips and results are rebuilt.
