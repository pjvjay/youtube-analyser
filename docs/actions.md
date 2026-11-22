# The actions: how each is found (`yta detect`)

`yta detect` finds the sections of each video where someone is standing during the interview,
and writes them as time ranges with a thumbnail and a YouTube link for each. Anyone in the frame
counts, including bystanders in the background, not only the host and guest. Standing in the
intro (teaser, title sequence, sponsor ad) is listed separately.

The same pipeline also finds **sitting, walking, running, jumping, smiling, talking and silent**,
chosen with `--action` (see [Other actions](#other-actions---action) below). Talking and silent use
the sound as well as the faces, and also list who spoke when. Standing stays the default, and
its results are exactly those of the original standing detector (tested file by file). `yta
search` finds a described person doing any of these ([search.md](search.md)), and `yta channel`
a whole YouTube channel ([channel.md](channel.md)).

In the code (`src/youtube_analyser/`): `pose.py` has the standing and sitting rules, `motion.py`
walking, running and jumping, `faces.py` and `audio.py` smiling, talking and silent, `actions.py`
puts them together per action, `analysis.py` computes and saves the model output each needs,
and `detector.py` joins frames into sections and writes the files below.

## How it works

1. ffmpeg samples frames from the video (1 per second by default).
2. A YOLO pose model (`src/youtube_analyser/assets/models/yolo11s-pose.onnx`) finds each person and 17 body points
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
7. Sections in the intro are listed separately (`yta intro`, below). A section that starts in the
   intro and ends within 3 s of the interview start counts as intro, since the start can fall
   in a crossfade; one that runs on longer is split there.

## Other actions (`--action`)

```bash
yta detect *.mp4 *.mkv *.webm --out output --action talking
```

`--action` is `standing` (default), `sitting`, `walking`, `running`, `jumping`, `smiling`,
`talking` or `silent`. Everything above is shared: frames are sampled at 1 per second, the pose model runs
once per video and its output in `poses/` is reused by every action, frames showing the action
are joined into sections (at least 2 frames, gaps of up to 2), sections in the intro are listed
separately, and the output files are the same. What differs is how a person is judged in a frame:

| Action | How a person is judged | Extra work per video | Judged only when |
|---|---|---|---|
| standing | Leg geometry (above); `--motion still` also leaves out people walking | none (`--motion`: the camera pass below) | an ankle is in view |
| sitting | The same leg geometry, stricter, and not moving (below) | a pass over the picture at 5 frames a second for the camera's movement (about 2 minutes per hour of video) | hips and knees are in view |
| walking | The body travels 0.6-4 torso lengths a second between 1-fps samples, camera pan, tilt and zoom taken out | the same pass | shoulders and hips in view, in two samples in a row |
| running | The body travels ≥ 4 torso lengths a second, in 5-fps samples | pose model at 5 fps, only around moments with a whole body in view | a whole body in view |
| jumping | Lowest foot and hips rise and come back down, in the same 5-fps samples | the same (shared with running) | shoulders, hips and an ankle in view |
| smiling | Expression model's "happy" probability | 2 small face models on a burst of 5 frames per face per second | ≥ 11 px between the eyes |
| talking | The sound says someone speaks and this mouth is the one moving, with this face's own voice; without clear sound, the mouth alone | the face pass (shared with smiling), the camera pass and one pass over the sound | ≥ 6 px between the eyes |
| silent | A face that is not talking: listening, or nobody speaking | the same | ≥ 6 px between the eyes |

Results for an action other than standing go to a subfolder named after it (`output/talking/segments.csv`,
`review.html`, `thumbs/` and so on). Column names carry the action's name (`talking_frames`,
`max_people_talking`), plus the measurement that picked each section's thumbnail (`best_rise`,
`best_happy`, `best_mouth_move`, `best_speed`, `best_conf` for sitting, `best_eyes` for silent).
Talking sections also list the voices heard (`voices`) and `judged_by` (`sound+face`, `sound`
or `face`); silent sections the voices heard while the person was silent (`voices_heard`).
What each action needs besides the poses is saved once per video and shared: `faces/`, `audio/`,
`camera/<video>.npz` and `poses/<video>.dense.npz`, so `--reclassify` after changing a threshold
is quick for every action. `motion.py`, `faces.py` and `audio.py` (in `src/youtube_analyser/`) have the thresholds,
with the reasons for each.

### Sitting

The standing rules already label people sitting, but for standing that label only had to rule
standing out, so it is generous. On a test clip, the back leg of a man walking (knee bent below
125°) and a cyclist straddling his bike with one foot down both counted as sitting. For
`--action sitting` a person also needs no leg that looks standing, and either a thigh that is level
or sideways, or every visible leg bent (a guest seated square-on to the camera). A standing person
overruled by a seated person level with them (rule 4) counts as sitting. Someone whose body is
moving across the picture (below) is not sitting either (`--motion any` turns this off). Close-ups,
which are most of these videos, show no legs and are never reported. Still figures are dropped as
for standing; sections longer than 5 minutes are checked a piece at a time.

### Walking, and standing still

Leg angles alone cannot tell walking from standing: a walker's legs pass through every angle a
stander's do, and the back leg of a stride is bent like a sitter's. What tells them apart is
movement, so people are followed from each 1-fps sample to the next and judged by how fast their
body (the middle of their hips) travels over the ground:

1. **The camera's own movement is taken out first**, so a camera panning past someone standing
   does not make them walk, and a camera following a walker (who stays put in the picture while
   the background slides past) does. It is measured 5 times a second (in 1 s a camera following a
   runner moved the picture by 64% of its width, too far to measure in one go), on the
   background only: the people found by the pose model are cut out with soft-edged holes (someone
   the camera follows is still in the picture and otherwise wins; hard-edged holes are still
   themselves and win instead). Zoom is measured too (Fourier-Mellin: on the street photo,
   zooms of 0.9-1.2 within 0.01), so zooming in on someone is not them walking towards you. A
   step out of line with its neighbours is a measuring error and is smoothed out; nobody is
   followed across a cut.
2. **People are matched** between samples by position, size and the colours of their clothes
   (4 × 2 patches over the torso). Position alone mixed people up: a man walking past a bystander
   hid her, and he was matched to where she had stood.
3. **Speed over the ground** is the travel across the picture plus, from how fast someone grows
   or shrinks, towards or away from the camera (this needs the lens, assumed normal: a long lens
   makes it an underestimate, so it can miss walking towards the camera, never invent it). 0.6-4
   torso lengths a second is walking (a stroll is about 2, a brisk walk up to 3.5); faster is
   running, which the running action judges (below).
4. A person's speed is the slower of their links to the samples before and after: a speed seen on
   one side only is mostly a change of pace (a runner's first second averaged 2.9, and came out
   walking). One link counts when the other side simply had nobody left (they were hidden, or the
   shot ended). People whose box touches the edge of the picture, or is half covered by someone
   else's, are not measured there: their points are guesses (a bystander cut off at the top
   measured torsos of 20-28 px from one second to the next). People standing side by side overlap
   by 14-33% and are measured (at a third, a jump next to someone was lost).

`--action walking` reports upright people who walk (someone seated, on a bicycle say, is not
walking). `--action standing --motion still` reports only people standing still, and `--motion
moving` only those moving; without `--motion` standing is exactly as before, so its reviewed results
stay valid. People who could not be followed keep their standing or sitting label.

**Tested** (`tests/test_movement.py`) on 144p clips made by moving a person cut out of a street
photo (see `tests/README.md`): walking, stopping and walking back; a camera panning past, following,
zooming in and cutting; walking towards the camera; two people crossing; someone walking past a
bystander who is hidden for a moment; a brisk walk; runners. Every verdict at a steady moment was
right or "unknown" (none wrong), and the walking sections matched the truth to within a sample.
**Known gaps:** at 1 fps, walking straight at the camera is only seen once the person is close
(the size must change by 15% a second to rise above jitter, which is from about 10 torso lengths
away), so those walks are often too short for a section; a walk of under 2 s may give no
section; people walking past each other are "unknown" while they overlap; real footage has not
been tested.

### Running

Running covers 5-13 torso lengths a second, often more than the distance to the edge of the
picture in one second, so it cannot be followed between 1-fps samples. It is judged in the 5-fps
samples that jumping also uses (taken again around every moment a whole body is in view,
`--jump-fps`, shared in `poses/<video>.dense.npz`): people are followed frame to frame (by
position, size and clothes, with the camera's movement taken out as for walking), and their
speed over the ground is the slope of their track over half a second either side. 4 torso lengths
a second or more is running; someone seated (on a bicycle, say) is not. A section needs 0.6 s of
running. There is no still-figure check.

**Tested** on a runner crossing a still camera (in view for 1.8 s) and one followed by the camera
(measured 5.7 torso lengths a second, truth 6): each was found once, and nobody in any other clip
was running (a brisk walk at 3 came out at 2.8). **Known gaps:** a runner whose whole body is never
in view at a 1-fps sample is not sampled again; real running has not been tested.

### Jumping

A jump is in the air for under half a second, so at 1 frame per second it is almost never
sampled, and sampling whole videos 5 times as often would cost 5 times as much. So the 1-fps
poses pick the moments where someone's whole body is in view (shoulders, hips and an ankle, not
sitting), and only those moments, ±1 s, are sampled again at 5 fps (`--jump-fps`). In a sit-down
interview show that is a small part of each video; the output line says how many seconds it was.

People are followed from frame to frame (as for running). A person is airborne when their lowest
foot **and** their hips are both at least 0.25 torso lengths higher than in a frame before and a
frame after (each within 1 s), on the same ground. The lowest foot ignores steps and leg lifts, the
hips ignore a foot lifted on its own, and needing the way back down ignores stepping up onto
something, walking away and standing up. The camera's pan, tilt and zoom are taken out first.
Sections are 0.2 s frames (jumps up to 1 s apart form one section), and one airborne frame is
enough. There is no still-figure check (a statue does not jump).

**Tested** on jumps of 0.45 torso lengths (about 25 cm): three in a row, two people jumping
(once at the same time) and a jump while walking were each found once, within 0.5 s of the
truth; tiny hops (0.12), the camera bobbing up and down, and stepping up onto something were not
jumps, and nobody walking or running jumped. **Known gaps:** hops of about 0.3 torso lengths are
found only sometimes (at 5 fps the highest moment sampled can fall under 0.25); real jumps have not
been tested.

### Smiling, talking and silent

The pose model already finds each person's eyes and nose, so no face detector is needed. Each face
turned towards the camera (nose within 0.35 eye-distances of the middle of the eyes) is cut out,
turned so the eyes are level, and measured on the sampled frame and the 4 frames after it
(0.2 s apart, so t to t+0.8 s), all cut at the same place. A burst stops at a camera cut, and a
burst frame only counts when the face model finds a face of the same size at about the same place.

- **Smiling**: `src/youtube_analyser/assets/models/face_expression.onnx` (below) gives the probability that the face is
  happy; a person is smiling when the median over the burst is 0.6 or more. On faces shrunk to
  144p sizes, a smile scored 0.94–1.00 and talking, shouting and neutral faces 0.17 or less with
  12 px or more between the eyes; at 9 px the two overlapped, and at 6 px a neutral face scored
  up to 0.74. So only faces with 11 px or more between the eyes are judged, which at 144p means
  close-ups. Only a few faces were available for this (one smiling, from a photo, and six that
  were not), so treat 0.6 as a starting point.
- **Talking**: `src/youtube_analyser/assets/models/face_landmark.onnx` (below) gives 468 points on the face, including the
  inner lips. The mouth opening is the gap between the lips divided by the distance between the outer
  eye corners; a person is talking when it changes by 0.12 or more within the burst. In a 144p
  test clip of a man talking (the "carphone" test sequence, about 20 px between the eyes) it
  changed by 0.17–0.36 in most bursts while he spoke, and by 0.05 or less on a still frame of his
  closed mouth with camera wobble and noise added. Shrunk to 10 px and 7 px between the eyes,
  speech still changed it by 0.14–0.19 in most bursts and the still face by 0.05 or less. Faces
  from 6 px between the eyes are judged.

With clear sound, talking and silent are decided together with it (next section); without it,
as above.

The actions share one face pass (`faces/<video>.npz`): whichever runs first measures both.
It costs about 4 ms per face per burst frame on OpenVINO (8 ms on onnxruntime), about 20% on top
of the pose model for a video that shows one face most of the time.

**Known gaps:** without clear sound, talking is mouth movement, so laughing, chewing and singing
count too. Someone talking with little jaw movement can be missed (the first 2 s of the test
clip). Smiling is only judged in close-ups. All miss faces turned away from the camera. None has
been checked on real episodes yet: review the first results by eye (the review video below
includes unreviewed sections for these actions) before relying on the counts, as was done for
standing.

### Talking and silent with the sound: who is speaking, and which voice

The faces say whose mouth moves; the sound says whether anyone is speaking at all, and which
voice. `audio.py` reads each video's sound once (16 kHz mono, from ffmpeg):

1. **Speech**: `src/youtube_analyser/assets/models/voice_activity.onnx` (Silero VAD) gives the probability of speech every
   32 ms. On a clean 2-speaker recording it found 98% of the speech with 1% false alarms.
2. **Is the sound clear enough to trust?** Per minute, the typical level of speech is compared with
   the quietest 5% of the sound (pauses, breaths, room tone). Speech must stand at least 20 dB
   clear. Close-miked speech measured 24-41 dB, including 9 s of talking without a pause; the same
   conversations with crowd babble 10 dB below the voices measured 10-15 dB, and there the speech
   detector also called 84% of the pauses speech, because a crowd is speech. A cartoon's music and
   squeaks measured 18 dB (the detector had called 36% of it speech). So a loud mall, music beds or
   street noise switch the sound off and talking goes back to the faces alone.
3. **Voices**: every 0.4 s, the 1.6 s of sound around it is turned into a voice fingerprint by
   `src/youtube_analyser/assets/models/voice_embedding.onnx` (Resemblyzer's speaker encoder). The fingerprints are grouped
   into voices (spectral clustering; the number of voices is where the eigenvalues jump), and
   voices that are not clearly different are merged, until every pair is at most 0.89 alike and
   fingerprints are (in the median) at least 0.10 closer to their own voice than to any other. A voice also needs
   5 s of speech. So two voices that cannot be told apart become one voice, rather than being split
   at random. Voices are named A, B, C... per video, A speaking most.
4. **Together**, for each sampled second and its 0.8 s burst: with clear sound and no speech,
   every visible face is silent, whatever its mouth does (chewing, smiling, mouthing along). With
   speech, the visible face whose mouth moves most (at least 0.08, a smaller movement than without
   sound, since the sound already says someone is talking) is talking; two when two voices are
   heard, and anyone else whose mouth moves as clearly as talking without sound (0.12), since two
   people talking at once often sound like one voice. Everyone else visible is silent: listening.
5. **Whose voice is whose**: each visible face is followed through its shot (not across cuts),
   and its own voice is the one heard most when it was the only one talking. A talker then speaks
   with their own voice, even when two talk at once and the voice grouping hears only one (on the
   test conversation it had given one talker three voices in four seconds). Silent faces list the
   voices they hear (`voices_heard`).

The sound is lined up with the pictures: a 1-per-second sample labelled `t` shows the picture at
about `t` + 0.5 s (ffmpeg keeps the last frame of each second; measured 0.48 s on the test clips),
so each sample's face burst and sound start there. Section times keep the labels, as standing's
always have.

**Tested** (`tests/test_talking.py`, `tests/test_sound.py`): a 72 s conversation of three
faces with real speech from three people under them, each talking in turn, two at once, then
nobody, a fourth voice off screen, and a mouth moving without speech. Nobody was ever called
talking who was not; the three were found talking in most of their seconds (the source clip's
speaker pauses between words); each kept one voice, three different ones, also while two talked at
once; the off-screen voice was marked not on screen, and the chewing was silent. With crowd noise
or no sound track, the faces alone decided. Close-ups cutting between two people: each kept his
voice across the cuts. Sound alone: one, two and three voices found, with 95%+ of speech under the
right voice. Earlier, on made-up conversations built from the real speakers in the pyannote.audio test
recordings (7 people with enough clean speech, alternating turns, same loudness): over 21
two-person conversations, clean and with crowd noise 20 dB down, 2.0% of speech was put under the
wrong voice (median; mean 4.7%), and no voice was split in two. An hour-long two-person
conversation was analysed in 34 s, found exactly 2 voices, and put 97.6% of the speech under the
right one. On real meeting recordings (30-60 s, 2-4 people, short overlapping turns) it was 13.9%:
short turns and overlaps are harder than an interview. Combined with the talking-face clip:
with a voice heard while his mouth moved and another while it did not, he was talking with the
first voice and silent, listening to the second (off screen); with his mouth moving but no speech,
silent; with loud babble or no sound track, judged by his face alone; with two faces side by side
and one voice, only the moving one was talking.

**Outputs** (for talking and silent): `speech.csv`, one row per speech turn (start, end, voice,
whether that voice was `on_screen`, found talking in most of the turn's seconds with a face that
could be judged, and whether the sound was trusted there); `voices.csv`, per video and voice:
minutes, turns, minutes seen talking, how much of it had clear sound, and a thumbnail of the
clearest moment the voice was matched to a talking face. `review.html` shows the voices with their
thumbnails, so you can see who A and B are. `--no-sound` judges from the faces alone.

**Known gaps:** voices are per video, so the host is A in one episode and B in the next (to find
one person's voice across episodes, enrol them by name and use `yta search`, below). In the speech timeline, overlapping speech gets
one voice. A voice with less than about 6 s of speech (5 s of fingerprints) is merged into the
nearest, so a bystander's few words are put under someone else's voice. A face's own voice needs
two moments of it talking alone. A listener who laughs as clearly as talking counts as talking.
The sound and picture are assumed to be in sync (within about half a second).

### Models used by the actions

| File | What | Source, licence |
|---|---|---|
| `src/youtube_analyser/assets/models/face_landmark.onnx` (2.4 MB) | 468 face points from a 192×192 crop, plus a face score | MediaPipe's `face_landmark.tflite` (in the `mediapipe` 0.10.14 wheel), Apache-2.0. Converted with `python -m tf2onnx.convert --tflite face_landmark.tflite --output face_landmark.onnx --opset 13` (tf2onnx 1.17, TensorFlow 2.21); the outputs match the TFLite model to within 0.0001 |
| `src/youtube_analyser/assets/models/voice_activity.onnx` (1.2 MB) | Probability of speech every 32 ms of 16 kHz sound | Silero VAD's `silero_vad_16k_sequence.onnx`, shipped in the `silero-vad` 6.2.3 wheel, MIT. Used as is; gives the same probabilities as Silero's own step-by-step model |
| `src/youtube_analyser/assets/models/voice_embedding.onnx` (5.7 MB) | A 256-number voice fingerprint of 1.6 s of speech | Resemblyzer 0.1.4's `pretrained.pt` (GE2E speaker encoder, Apache-2.0), exported with `torch.onnx.export` with the mel filterbank (librosa's, 40 bands) built in, so it takes the power spectrum (160 frames × 201) that `audio.py` computes with numpy. Fingerprints match Resemblyzer's own to a cosine of 1.00000 |
| `src/youtube_analyser/assets/models/face_expression.onnx` (0.2 MB) | 7 expressions (angry, disgust, fear, happy, sad, surprise, neutral) from a 64×64 grey face | The mini-Xception `emotion_model.hdf5` shipped in the `fer` 25.10.3 wheel (MIT), trained on the FER2013 faces. Converted with `tf2onnx.convert.from_keras(model, input_signature=(tf.TensorSpec((None, 64, 64, 1), tf.float32, name="face"),), opset=13)`; outputs match Keras to within 4e-7. FER2013 was collected from web images for a research challenge, so check that its terms suit your use |

| `src/youtube_analyser/assets/models/person_mask.onnx` (0.4 MB) | Which pixels are people, from a 256×144 picture (for hair and clothes colours) | MediaPipe's selfie segmentation, landscape (`selfie_segmentation_landscape.tflite` in the `mediapipe` wheel), Apache-2.0. Converted with tf2onnx (opset 14), its custom transposed-convolution layer rewritten as a standard `ConvTranspose` plus bias; the masks match MediaPipe's own |
| `src/youtube_analyser/assets/models/gender_age.onnx` (49 MB) | How likely a 64×64 face looks female, and its apparent age (0-100) | The WideResNet from yu4u/age-gender-estimation (MIT code), weights `weights.28-3.73.hdf5` as shipped in the `py-agender` 0.0.9 wheel, rebuilt in Keras and converted with tf2onnx (opset 13), weights stored as float16 (outputs within 7e-5 of float32). **Trained on the IMDB-WIKI faces, which are released for academic research only**: check that suits your use, or switch it off in `people.toml` (`[models] gender_age = false`; apparent gender then comes from voices only, and age is not judged) |

All run through OpenVINO or onnxruntime, like the pose model (`--runtime`), and give the same
results on either (faces to within 2e-6, voices identically). No new Python packages are needed.

## Where the interview starts (`yta intro`)

Episodes open with a cold-open teaser, the show's title sequence, a sponsor ad or a logo card,
in some combination, and the interview starts after the last of these. Title sequences, ads and
logo cards are the same across a season, so:

1. The first 4 minutes of every episode are reduced to one 32×18 grey thumbnail per second.
2. A thumbnail that matches a short moment (≤ 15 s) in 3 or more other episodes is title
   material. (Interview footage repeated in a re-run or a compilation episode is a static shot
   that matches minutes of footage, so it is not counted.) These are saved in
   `intro_bank.npz` (about 1,150 thumbnails, 0.5 MB, for the test set), so a new video
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
yta intro --build-bank *.mp4 *.mkv *.webm
```

The thresholds and the reasons for them are at the top of the detector (`yta detect`).

**Known gaps:** people whose feet are out of frame, or who stand behind a table, are not
detected, because without an ankle the rules cannot tell standing from sitting facing the
camera. Standing that shows for only one sampled frame (a quick cut in an intro) is not
reported. A guest sitting alone, square-on to the camera, can still be reported as standing.

## Running it

```bash
yta detect 20240105_sampleVid01.mp4 --out output
```

Every video in the folder (yt-dlp's unmerged `.f278.webm` leftover is skipped automatically):

```bash
yta detect *.mp4 *.mkv *.webm --out output
```

A video whose `segments/<video>.csv` exists is skipped, so an interrupted run picks up where it
stopped. The pose model's output is saved in `poses/<video>.npz`, so after changing a threshold,
`--reclassify` redoes the sections in seconds without rerunning the model. `--force` redoes everything.
Each saved file records what made it (the pose model, its input size, the sampling rate and
confidence, the video file's size and the versions of the steps), and is made again by itself
when any of that changes: running with `--imgsz 320` after a run at 640 does not mix the two.
Two videos with the same file name in different folders would share saved output, so the second
is skipped with a warning.

Options: `--action` (above), `--imgsz` (default 640; 480 is about 1.7× faster but misses small
background people), `--fps`, `--model` (`yolo11n-pose.onnx` is about 2.5× faster and less
accurate), `--runtime` (`auto`, `openvino` or `onnxruntime`), `--min-frames` and `--max-gap`
(defaults 2 and 2; 1 and 4 for jumping, whose frames are 0.2 s apart), `--jump-fps`, `--conf`,
`--threads`, `--keep-intro` (treat the whole video as the interview), `--intro-bank`, `--motion`
(`any`, `still` or `moving`, for standing and sitting) and `--no-sound` (talking and silent from
the faces alone).

Every action shares `<out>/poses/`, so running a second action on the same videos does not run
the pose model again: standing at 1 fps is the expensive part, and sitting on top of it takes
seconds.

## Output (in `--out`; for other actions, in `--out/<action>/`, except the shared `poses/`, `faces/`, `audio/` and `camera/`)

| Path | Contents |
|---|---|
| `segments.csv` | Every standing section in the interviews: video, start, end, duration, frames, people standing, pixel change, YouTube link |
| `intro.csv` | Standing sections in the intros, for checking |
| `dropped.csv` | Sections dropped as still figures (statues, posters), for checking |
| `interview_starts.csv` | Where each video's interview starts, and how that was found |
| `review.html` | All three, as thumbnails with skeletons drawn (green standing, orange sitting, grey unknown) |
| `segments/`, `intro/`, `dropped/`, `interview/<video>.csv` | The same, per video |
| `frames/<video>.csv` | Every sampled frame: people found, standing / sitting / unknown counts, leg measurements |
| `poses/<video>.npz` | Every person found in every sampled frame, with their 17 body points (shared by all actions; `<video>.dense.npz` has running and jumping's extra samples, with the camera's movement and people's clothes) |
| `faces/<video>.npz` | Smiling, talking, silent: each face's size, mouth opening and happy probability over its burst |
| `audio/<video>.npz` | Talking, silent: speech probability every 32 ms, sound clarity per minute, and the voice of each 0.4 s of speech |
| `camera/<video>.npz` | Walking, sitting, talking, silent, `--motion`: the camera's pan, tilt and zoom between 1-fps samples (measured 5 times a second), and the colours of each person's clothes |
| `speech.csv`, `voices.csv`, `speech/`, `voices/` | Talking, silent: who spoke when and each video's voices (above) |
| `appearance/<video>.npz` | `yta search`: hair and clothes colours, apparent gender and age of every person in every sample |
| `thumbs/` | The best frame of each section, 3× enlarged |

`yta report` turns a run into the `results/` files, adding episode titles and the verdicts
from the manual review in `results/verdicts.csv`: `standing_sections.csv` (interviews only),
`outside_interview.csv` (intros, plus sections the review marked "after the interview"),
`interview_starts.csv` and `README.md`. These name the videos, so they stay local; see
[`../samples/`](../samples/README.md) for their format.

`yta report --action talking` does the same for another action, from
`output/talking/` and the review in `results/talking/verdicts.csv` (optional: without it
every section is "not reviewed"), into `results/talking/` (`talking_sections.csv` and so on).

## Review video

`yta review` cuts every section out of the local videos and joins them into one video to
watch, with 2 seconds of context either side and a caption bar (clip number, upload date, video
ID, standing time range, verdict, title). It also writes an index CSV saying where each clip
starts in the review video.

```bash
yta review                      # confirmed + unclear -> output/review.mp4
yta review --verdicts "false alarm" --out output/false_alarms.mp4
yta review --verdicts all --pad 3
```

It reads `results/standing_sections.csv`, so only standing during the interviews is included.
For the current results that is 83 sections in 74 clips, 16 minutes, made in about a minute.

`--action talking` (or another action) reads `results/talking/talking_sections.csv` and writes
`output/talking/review.mp4`; for these actions the default also includes sections that
are not reviewed yet, so it is the way to review them.

