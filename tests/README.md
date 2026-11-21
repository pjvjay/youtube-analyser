# Tests

```bash
.venv/bin/python -m unittest discover -s tests -t . -v     # all of them
.venv/bin/python -m unittest tests.test_rules -v           # the rules only: seconds, no video
```

The first full run builds the test clips and runs the detector on them (5-10 minutes on a
4-core machine); later runs reuse both from `tests/.cache/` and take a few minutes. Delete that
folder to start again. Only the standard library's `unittest` is needed (`pytest tests` works
too, if you have it).

| File | What it checks | Needs |
|---|---|---|
| `test_rules.py` | Every decision rule on made-up measurements: walking, running and jumping from body points and camera movements; the camera measurement itself; talking and silent from mouth movement and what the sound heard; whose voice is whose; speech detection, how clear the sound is, and how voices are grouped; for the search, naming colours, reading queries, the match / possible / no verdicts, gender from voice before face, voting over a person's samples, following people through a shot, pitch, and naming voices | nothing (seconds) |
| `test_standing.py` | Standing is exactly as before: the original code (git commit `7bf602d`) and the current one agree on 240 made-up pose sequences (a golden file) and write byte-identical files on two clips | git history for the second |
| `test_movement.py` | Walking, running and jumping end to end on 19 made clips | the models |
| `test_talking.py` | Up to three people talking, with and without usable sound, and voices across cuts | the models |
| `test_sound.py` | Speech and voices on real speech: one, two and three voices, crowd noise, silence, a brief extra voice, no sound | the models |
| `test_search.py` | Searching for people by hair colour, clothes, apparent gender, voice and an enrolled name, with and without an action, on made clips of a woman (hair dyed purple or blue, or natural) and a man taking turns to talk, and on the street clips; the command line and its files | the models |
| `test_tools.py` | Every action through the command line; skipping and `--reclassify`; a broken file; the reports and the review video; the HTTP API against the command line, including `who=` with a person enrolled from a recording | the models, uvicorn |

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

`make_fixtures.py` builds every clip from these, deterministically. When you change a clip,
bump its `VERSION` so cached clips and results are rebuilt.
