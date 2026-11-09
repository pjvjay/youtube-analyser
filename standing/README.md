# Standing detector

Finds the sections of each video where someone is standing, and writes them as time ranges
with a thumbnail and a YouTube link for each. Anyone in the frame counts, including
bystanders in the background, not only the host and guest.

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

The thresholds and the reasons for them are at the top of `detect_standing.py`.

**Known gaps:** people whose feet are out of frame, or who stand behind a table, are not
detected, because without an ankle the rules cannot tell standing from sitting facing the
camera. Standing that shows for only one sampled frame (a quick cut in an intro) is not
reported. A guest sitting alone, square-on to the camera, can still be reported as standing.

## Running it

From the `youtube-analyser` folder:

```bash
.venv/bin/python standing/detect_standing.py 20230811_g0EuMGqQpec.mp4 --out standing/output
```

Every video in the folder (yt-dlp's unmerged `.f278.webm` leftover is skipped automatically):

```bash
.venv/bin/python standing/detect_standing.py *.mp4 *.mkv *.webm --out standing/output
```

A video whose `segments/<video>.csv` exists is skipped, so an interrupted run picks up where it
stopped. The pose model's output is saved in `poses/<video>.npz`, so after changing a threshold,
`--reclassify` redoes the sections in seconds without rerunning the model. `--force` redoes everything.

Options: `--imgsz` (default 640; 480 is about 1.7× faster but misses small background people),
`--fps`, `--model` (`yolo11n-pose.onnx` is about 2.5× faster and less accurate), `--runtime`
(`auto`, `openvino` or `onnxruntime`), `--min-frames`, `--max-gap`, `--conf`, `--threads`.

## Output (in `--out`)

| Path | Contents |
|---|---|
| `segments.csv` | Every standing section: video, start, end, duration, frames, people standing, pixel change, YouTube link |
| `dropped.csv` | Sections dropped as still figures (statues, posters), for checking |
| `review.html` | Both, as thumbnails with skeletons drawn (green standing, orange sitting, grey unknown) |
| `segments/<video>.csv`, `dropped/<video>.csv` | The same, per video |
| `frames/<video>.csv` | Every sampled frame: people found, standing / sitting / unknown counts, leg measurements |
| `poses/<video>.npz` | Every person found in every sampled frame, with their 17 body points |
| `thumbs/` | The best frame of each section, 3× enlarged |

`make_report.py` turns a run into `results/standing_sections.csv` and `results/README.md`, adding
episode titles and the verdicts from the manual review in `results/verdicts.csv`.

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
