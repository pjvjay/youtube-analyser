# Standing detector (and other actions)

Finds the sections of each video where someone is standing during the interview, and writes
them as time ranges with a thumbnail and a YouTube link for each. Anyone in the frame counts,
including bystanders in the background, not only the host and guest. Standing in the intro
(teaser, title sequence, sponsor ad) is listed separately.

The same pipeline also finds **sitting, jumping, smiling and talking**, chosen with
`--action` (see [Other actions](#other-actions---action) below). Standing stays the default,
and its results are unchanged.

## How it works

1. ffmpeg samples frames from the video (1 per second by default).
2. A YOLO pose model (`models/yolo11s-pose.onnx`) finds each person and 17 body points
   (shoulders, hips, knees, ankles and so on). It runs on the CPU through OpenVINO when that
   is installed, otherwise through onnxruntime. Both give the same detections; OpenVINO is
   about 1.4–1.8× faster on Intel CPUs.
3. Each person is labelled from their leg geometry, measured in torso lengths
   (mid-shoulder to mid-hip):
   - **standing**: an ankle is visible, the leg is nearly straight (≥ 150°), the knee hangs
     well below the hip (≥ 0.55 torso lengths), the thigh is within 35° of vertical, and the
     shin is not much longer than the thigh (≤ 1.35×, which rules out a thigh pointing at the camera).
   - **sitting**: the knee is level with the hip or close to it (< 0.35), the thigh is more than
     55° from vertical, or the knee is clearly bent (< 125°).
   - **unknown**: hips or knees are out of view, as in close-ups, which are most of these videos.
4. A guest seated square-on to the camera can still pass every standing test. So a "standing"
   person is overruled when a seated person of about the same size in the same frame has
   shoulders nearly as high as theirs (someone standing is clearly taller).
5. Frames with at least one standing person are joined into sections. A section needs 2 standing
   frames and may contain up to 2 frames in a row that are not standing (a cut to a close-up, say).
6. Statues, posters and artwork shaped like a standing person are dropped: within a section the
   figure's pixels are compared between samples up to 3 s apart, and a figure that never changes
   by more than 3.5 grey levels is not a person. On all 248 videos this dropped 23 sections, all
   of them still figures, while real people standing still changed by 4.4 or more.
7. Sections in the intro are listed separately (`intro.py`, below). A section that starts in the
   intro and ends within 3 s of the interview start counts as intro, since the start can fall
   in a crossfade; one that runs on longer is split there.

## Other actions (`--action`)

```bash
.venv/bin/python standing/detect_standing.py *.mp4 *.mkv *.webm --out standing/output --action talking
```

`--action` is `standing` (default), `sitting`, `jumping`, `smiling` or `talking`. Everything
above is shared: frames are sampled at 1 per second, the pose model runs once per video and its
output in `poses/` is reused by every action, frames showing the action are joined into
sections (at least 2 frames, gaps of up to 2), sections in the intro are listed separately,
and the output files are the same. What differs is how a person is judged in a frame:

| Action | How a person is judged | Extra work per video | Judged only when |
|---|---|---|---|
| standing | Leg geometry (above) | none | an ankle is in view |
| sitting | The same leg geometry, stricter (below) | none | hips and knees are in view |
| jumping | Lowest foot and hips rise and come back down | pose model at 5 fps, only around moments with a whole body in view | shoulders, hips and an ankle are in view |
| smiling | Expression model's "happy" probability | 2 small face models on a burst of 5 frames per face per second | ≥ 11 px between the eyes |
| talking | Face landmark model: the lips opening and closing | the same face pass as smiling (shared, cached) | ≥ 6 px between the eyes |

Results for an action other than standing go to a subfolder named after it (`standing/output/talking/segments.csv`,
`review.html`, `thumbs/` and so on). Column names carry the action's name (`talking_frames`,
`max_people_talking`), plus the measurement that picked each section's thumbnail (`best_rise`,
`best_happy`, `best_mouth_move`, `best_conf` for sitting). The face measurements are saved in
`faces/<video>.npz` and jumping's extra samples in `poses/<video>.dense.npz`, so `--reclassify` after
changing a threshold is quick for every action. `standing/actions.py` has the thresholds for
jumping, smiling and talking, with the reasons for each.

### Sitting

The standing rules already label people sitting, but for standing that label only had to rule
standing out, so it is generous. On a test clip, the back leg of a man walking (knee bent below
125°) and a cyclist straddling his bike with one foot down both counted as sitting. For
`--action sitting` a person also needs no leg that looks standing, and either a thigh that is level
or sideways, or every visible leg bent (a guest seated square-on to the camera). A standing person
overruled by a seated person level with them (rule 4) counts as sitting. Close-ups, which are most
of these videos, show no legs and are never reported. Still figures are dropped as for standing;
sections longer than 5 minutes are checked a piece at a time.

### Jumping

A jump is in the air for under half a second, so at 1 frame per second it is almost never
sampled, and sampling whole videos 5 times as often would cost 5 times as much. So the 1-fps
poses pick the moments where someone's whole body is in view (shoulders, hips and an ankle, not
sitting), and only those moments, ±1 s, are sampled again at 5 fps (`--jump-fps`). In a sit-down
interview show that is a small part of each video; the output line says how many seconds it was.

People are followed from frame to frame. A person is airborne when their lowest foot **and** their
hips are both at least 0.25 torso lengths higher than in a frame before and a frame after (each
within 1 s), on the same ground. The lowest foot ignores steps and leg lifts, the hips ignore a
foot lifted on its own, and needing the way back down ignores stepping up onto something, walking
away and standing up. Camera tilts and shakes are taken out first: the whole picture's shift
between samples (phase correlation) is subtracted, and people are not followed across cuts.
Sections are 0.2 s frames (jumps up to 1 s apart form one section), and one airborne frame is
enough. There is no still-figure check (a statue does not jump).

Tested on synthetic 144p clips made from a photo of people in a street, with one man moved as
a whole: 3 of 3 jumps of 0.45 torso lengths (about 25 cm) were found (measured rise 0.41–0.42), and
nothing in the same clip with the camera tilting instead, the man stepping up onto something, or
nobody moving. Small hops of 0.3 torso lengths were found 1 time in 3: at 5 fps the highest
sampled moment often falls below 0.25. Real jumps have not been tested.

### Smiling and talking

The pose model already finds each person's eyes and nose, so no face detector is needed. Each face
turned towards the camera (nose within 0.35 eye-distances of the middle of the eyes) is cut out,
turned so the eyes are level, and measured on the sampled frame and the 4 frames after it
(0.2 s apart, so t to t+0.8 s), all cut at the same place. A burst stops at a camera cut, and a
burst frame only counts when the face model finds a face of the same size at about the same place.

- **Smiling**: `models/face_expression.onnx` (below) gives the probability that the face is
  happy; a person is smiling when the median over the burst is 0.6 or more. On faces shrunk to
  144p sizes, a smile scored 0.94–1.00 and talking, shouting and neutral faces 0.17 or less with
  12 px or more between the eyes; at 9 px the two overlapped, and at 6 px a neutral face scored
  up to 0.74. So only faces with 11 px or more between the eyes are judged, which at 144p means
  close-ups. Only a few faces were available for this (one smiling, from a photo, and six that
  were not), so treat 0.6 as a starting point.
- **Talking**: `models/face_landmark.onnx` (below) gives 468 points on the face, including the
  inner lips. The mouth opening is the gap between the lips divided by the distance between the outer
  eye corners; a person is talking when it changes by 0.12 or more within the burst. In a 144p
  test clip of a man talking (the "carphone" test sequence, about 20 px between the eyes) it
  changed by 0.17–0.36 in most bursts while he spoke, and by 0.05 or less on a still frame of his
  closed mouth with camera wobble and noise added. Shrunk to 10 px and 7 px between the eyes,
  speech still changed it by 0.14–0.19 in most bursts and the still face by 0.05 or less. Faces
  from 6 px between the eyes are judged.

The two actions share one face pass (`faces/<video>.npz`): whichever runs first measures both.
It costs about 4 ms per face per burst frame on OpenVINO (8 ms on onnxruntime), about 20% on top
of the pose model for a video that shows one face most of the time.

**Known gaps:** talking is mouth movement, so laughing, chewing and singing count too, and a
voice off camera does not. Someone talking with little jaw movement can be missed (the first 2 s
of the test clip). Smiling is only judged in close-ups. Both miss faces turned away from the
camera. Neither has been checked on real episodes yet: review the first results by eye (the
review video below includes unreviewed sections for these actions) before relying on the
counts, as was done for standing.

### Models used by the actions

| File | What | Source, licence |
|---|---|---|
| `models/face_landmark.onnx` (2.4 MB) | 468 face points from a 192×192 crop, plus a face score | MediaPipe's `face_landmark.tflite` (in the `mediapipe` 0.10.14 wheel), Apache-2.0. Converted with `python -m tf2onnx.convert --tflite face_landmark.tflite --output face_landmark.onnx --opset 13` (tf2onnx 1.17, TensorFlow 2.21); the outputs match the TFLite model to within 0.0001 |
| `models/face_expression.onnx` (0.2 MB) | 7 expressions (angry, disgust, fear, happy, sad, surprise, neutral) from a 64×64 grey face | The mini-Xception `emotion_model.hdf5` shipped in the `fer` 25.10.3 wheel (MIT), trained on the FER2013 faces. Converted with `tf2onnx.convert.from_keras(model, input_signature=(tf.TensorSpec((None, 64, 64, 1), tf.float32, name="face"),), opset=13)`; outputs match Keras to within 4e-7. FER2013 was collected from web images for a research challenge, so check that its terms suit your use |

Both run through OpenVINO or onnxruntime, like the pose model (`--runtime`), and give the same
measurements on either (to within 2e-6).

## Where the interview starts (`intro.py`)

Episodes open with a cold-open teaser, the show's title sequence, a sponsor ad or a logo card,
in some combination, and the interview starts after the last of these. Title sequences, ads and
logo cards are the same across a season, so:

1. The first 4 minutes of every episode are reduced to one 32×18 grey thumbnail per second.
2. A thumbnail that matches a short moment (≤ 15 s) in 3 or more other episodes is title
   material. (Interview footage repeated in a re-run or a compilation episode is a static shot
   that matches minutes of footage, so it is not counted.) These are saved in
   `models/intro_bank.npz` (about 1,150 thumbnails, 0.5 MB, for the test set), so a new video
   can be checked alone. The bank is made from your videos, so it is built locally and not
   committed.
3. The interview starts after the last run of title frames that begins in the first 150 s;
   title frames up to 15 s apart are joined, which bridges a sponsor read between title and ad.
   Some intros go on to show episode-specific clips inside a fixed frame (a TV set, say), so
   the intro is extended while the edge of the picture stays as it was in the title frames.
4. Some episodes (a whole season, in the test set) have a title unique to them. There the
   interview starts at the first 10 s in which 7 frames show a camera angle seen again later
   in the episode.

Videos under 4 minutes (Shorts) have no intro. Checked by eye on the 210 test episodes, with the
frames either side of each detected start. The end of the interview is not detected.

Build the bank before the first run, and again after adding episodes with a new title design:

```bash
.venv/bin/python standing/intro.py --build-bank *.mp4 *.mkv *.webm
```

The thresholds and the reasons for them are at the top of `detect_standing.py`.

**Known gaps:** people whose feet are out of frame, or who stand behind a table, are not
detected, because without an ankle the rules cannot tell standing from sitting facing the
camera. Standing that shows for only one sampled frame (a quick cut in an intro) is not
reported. A guest sitting alone, square-on to the camera, can still be reported as standing.

## Running it

From the `youtube-analyser` folder:

```bash
.venv/bin/python standing/detect_standing.py 20240105_sampleVid01.mp4 --out standing/output
```

Every video in the folder (yt-dlp's unmerged `.f278.webm` leftover is skipped automatically):

```bash
.venv/bin/python standing/detect_standing.py *.mp4 *.mkv *.webm --out standing/output
```

A video whose `segments/<video>.csv` exists is skipped, so an interrupted run picks up where it
stopped. The pose model's output is saved in `poses/<video>.npz`, so after changing a threshold,
`--reclassify` redoes the sections in seconds without rerunning the model. `--force` redoes everything.

Options: `--action` (above), `--imgsz` (default 640; 480 is about 1.7× faster but misses small
background people), `--fps`, `--model` (`yolo11n-pose.onnx` is about 2.5× faster and less
accurate), `--runtime` (`auto`, `openvino` or `onnxruntime`), `--min-frames` and `--max-gap`
(defaults 2 and 2; 1 and 4 for jumping, whose frames are 0.2 s apart), `--jump-fps`, `--conf`,
`--threads`, `--keep-intro` (treat the whole video as the interview), `--intro-bank`.

Every action shares `<out>/poses/`, so running a second action on the same videos does not run
the pose model again: standing at 1 fps is the expensive part, and sitting on top of it takes
seconds.

## Output (in `--out`; for other actions, in `--out/<action>/`, except `poses/` and `faces/`)

| Path | Contents |
|---|---|
| `segments.csv` | Every standing section in the interviews: video, start, end, duration, frames, people standing, pixel change, YouTube link |
| `intro.csv` | Standing sections in the intros, for checking |
| `dropped.csv` | Sections dropped as still figures (statues, posters), for checking |
| `interview_starts.csv` | Where each video's interview starts, and how that was found |
| `review.html` | All three, as thumbnails with skeletons drawn (green standing, orange sitting, grey unknown) |
| `segments/`, `intro/`, `dropped/`, `interview/<video>.csv` | The same, per video |
| `frames/<video>.csv` | Every sampled frame: people found, standing / sitting / unknown counts, leg measurements |
| `poses/<video>.npz` | Every person found in every sampled frame, with their 17 body points (shared by all actions; `<video>.dense.npz` has jumping's extra samples) |
| `faces/<video>.npz` | Smiling and talking: each face's size, mouth opening and happy probability over its burst |
| `thumbs/` | The best frame of each section, 3× enlarged |

`make_report.py` turns a run into the `results/` files, adding episode titles and the verdicts
from the manual review in `results/verdicts.csv`: `standing_sections.csv` (interviews only),
`outside_interview.csv` (intros, plus sections the review marked "after the interview"),
`interview_starts.csv` and `README.md`. These name the videos, so they stay local; see
[`../samples/`](../samples/README.md) for their format.

`make_report.py --action talking` does the same for another action, from
`standing/output/talking/` and the review in `results/talking/verdicts.csv` (optional: without it
every section is "not reviewed"), into `results/talking/` (`talking_sections.csv` and so on).

## Review video

`review_video.py` cuts every section out of the local videos and joins them into one video to
watch, with 2 seconds of context either side and a caption bar (clip number, upload date, video
ID, standing time range, verdict, title). It also writes an index CSV saying where each clip
starts in the review video.

```bash
.venv/bin/python standing/review_video.py                      # confirmed + unclear -> standing/output/review.mp4
.venv/bin/python standing/review_video.py --verdicts "false alarm" --out standing/output/false_alarms.mp4
.venv/bin/python standing/review_video.py --verdicts all --pad 3
```

It reads `results/standing_sections.csv`, so only standing during the interviews is included.
For the current results that is 83 sections in 74 clips, 16 minutes, made in about a minute.

`--action talking` (or another action) reads `results/talking/talking_sections.csv` and writes
`standing/output/talking/review.mp4`; for these actions the default also includes sections that
are not reviewed yet, so it is the way to review them.

## Setup

```bash
/usr/local/bin/python3.12 -m venv .venv
.venv/bin/pip install -r standing/requirements.txt
.venv/bin/pip uninstall -y openvino-telemetry
```

The last step matters: `openvino-telemetry` comes with OpenVINO and sends a usage event to
Google Analytics every time `openvino` is imported (unless you have opted out). OpenVINO works
normally without it.

The models were exported from Ultralytics' `yolo11n-pose` and `yolo11s-pose` weights with
`yolo export format=onnx dynamic=True simplify=True` (Ultralytics 8.4.198, PyTorch 2.2.2, the
last PyTorch release for Intel Macs). The Ultralytics models and weights are licensed AGPL-3.0.
