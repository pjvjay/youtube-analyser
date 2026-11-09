# Running the standing detector on Azure

Goal: send a video to an API, get back a CSV of the time ranges where someone is standing, and
pay only for the compute that analyses it. Videos are stored only while they are processed and
are deleted as soon as the CSV exists.

## What is already built

`service/app.py` is that API, and `Dockerfile` packages it with ffmpeg and the pose model.
It runs locally today and runs unchanged as an Azure Container App:

| Call | Does |
|---|---|
| `POST /jobs?name=<file>` with the video as the body | Saves it to the container's local disk, queues it, returns `job_id` |
| `GET /jobs/{job_id}` | `queued`, `processing`, `done` or `failed`, plus `interview_start_s`, `intro_sections` and `video_deleted` |
| `GET /jobs/{job_id}/sections.csv` | `start_s, end_s, start, end, duration_s, standing_frames, max_people_standing`, for the interview only (the intro is left out) |
| `DELETE /jobs/{job_id}` | Forgets the job and its result |

The video is deleted when processing ends, whether it succeeded or failed, and leftovers are
deleted at startup. Results are a few hundred bytes held in memory for `RESULT_TTL_S` (1 hour).
An `x-api-key` header is required when `API_KEY` is set.

Tested: run locally and in the Docker image on three real episodes, the API returned exactly
the same sections as the command-line tool, and the work folder was empty after each job.

## Option A: one Container App (no storage account at all)

```
client ──POST video──▶ Container App (this image, scale 0..N) ──▶ CSV
                       video on the replica's ephemeral disk, deleted after processing
```

- Nothing is stored outside the container, so there is no storage bill. Ephemeral disk is
  included in the replica: about 4 GiB at 1 vCPU and 8 GiB above 1 vCPU, according to Microsoft
  support's guidance (check the current limits page). A 1-hour video is ~30 MB at 144p and
  ~100–200 MB at 360p.
- It scales to zero, so you pay nothing between uploads.
- The catch: Container Apps scales replicas on HTTP traffic. Jobs run after the upload request
  returns, so a replica with no incoming requests can be scaled in mid-job (default cooldown
  300 s). Polling `GET /jobs/{id}` every few seconds keeps traffic flowing, but this is not
  guaranteed. Results also live in one replica's memory, so run it with `maxReplicas 1`, or
  have clients keep calling the same replica.
- Good for: a pilot, or one person uploading a few videos at a time.

## Option B (recommended for regular use): Blob staging + Container Apps job

```
client ──POST /jobs──▶ API (Container App, scale to zero) ──▶ returns a write-only upload URL
client ──PUT video──▶ Blob container "uploads"  (temporary)
        BlobCreated ──▶ Storage queue ──▶ Container Apps job (event-driven, 0..N executions)
                                           reads video ─▶ pose model ─▶ results/<job_id>.csv
                                           deletes the video blob
client ──GET /jobs/{id}/sections.csv──▶ API reads results/<job_id>.csv
```

- **Uploads go straight to Blob** with a short-lived write-only SAS URL. That avoids ingress
  timeouts and body-size limits (Microsoft does not publish a body-size limit for Container
  Apps ingress, and default ingress drops idle requests after about 4 minutes).
- **A job runs to completion.** Container Apps jobs are not scaled in by HTTP traffic. Each
  queue message starts one execution, and executions scale to zero when the queue is empty.
  Set the job's replica timeout above your longest video's processing time; at 2 vCPU a
  100-minute video takes about 15 minutes.
- **Storage costs round to zero:** a 100 MB video held for 10 minutes is a tiny fraction of a
  cent in hot LRS storage, and results are bytes. Three settings keep it that way:
  - **Turn off blob soft delete and versioning on this account.** Otherwise "deleted" videos
    are kept and billed for the retention period.
  - **Add a lifecycle rule** that deletes anything in `uploads/` older than 1 day, as a
    safety net for uploads that were never processed, and `results/` older than 7 days.
  - **Keep the account private** and let the job reach it through a managed identity, not
    account keys.

Code changes needed for B (not written yet): a `worker.py` entry point for the job that takes
the queue message, downloads the blob to local disk, runs the same
`detect_standing.infer()` / `find_sections()`, uploads the CSV, and deletes the blob in a
`finally`. The API also needs a call that issues the SAS URL. It reuses everything else.

## What it costs

Azure Container Apps consumption prices, East US, from the Azure Retail Prices API (Nov 2026):
**$0.000024 per vCPU-second, $0.000003 per GiB-second, $0.40 per million requests.** Every
subscription gets **180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests free
each month**.

Measured speed: this container, limited to 2 CPUs, processed a 63-minute episode in 8.9
minutes, which is about 0.28 vCPU-seconds per sampled frame. On the Mac itself with all 8
threads it was 51 ms per frame, about 0.4 thread-seconds. Azure CPUs differ, so the table uses
the more expensive 0.4 and a pilot run will tell. The default input size is 640: at 480 it
is about twice as fast but misses small background people.

| 100 hours of video, 1 frame per second (360,000 frames), input 640 | Estimate |
|---|---|
| vCPU-seconds | 100,000–144,000 |
| GiB-seconds (2 GiB per vCPU) | 200,000–288,000 |
| Price before the free grant | ~$3.00–4.30 |
| Price if it's the only Container Apps use that month | $0 (within the free grant) |

Wall-clock time scales with replicas: at 2 vCPU one replica does about 7 hours of video per
hour, so 4 parallel jobs do 100 hours in about 3.5 hours. The same price list also has an environment management charge of $0.10 an
hour. As far as I can tell it applies to workload-profiles (dedicated) environments, so create
a **Consumption-only** environment and check the first bill.

## Enhancement: confirm each section with a vision model

The pose rules are cheap and catch most standing people. On the 248 test videos, 77 of the 92
sections reported during the interviews were confirmed by eye. Most of the rest were a guest sitting alone,
square-on to the camera, who looks straight-legged in 2D. A second, very cheap check fixes
exactly that kind of mistake:

1. For each section, take its best frame (the detector already picks one for the thumbnail).
2. Send that one image to a small vision model on Azure OpenAI (for example `gpt-4.1-mini`):
   *"Is any real person in this frame standing? Answer yes or no."*
3. Keep sections answered "yes".

This runs once per section, not per frame. At hundreds of sections per 100 hours and a few
hundred image tokens each, it costs cents (about $0.40 per million input tokens for
`gpt-4.1-mini`, per third-party price trackers; check the Azure OpenAI pricing page). Without
the pose model it would cost far more: 360,000 frames per 100 hours instead of a few hundred.

The pose model stays in charge of recall and the vision model handles precision. That is the
cheapest arrangement because the expensive model only sees the rare frames.

## Deployment outline (Option A; nothing has been created in your Azure account)

```bash
az group create -n standing-rg -l eastus
az acr create -n <registry> -g standing-rg --sku Basic        # or use GitHub Container Registry
az acr build -r <registry> -t standing-api:1 .
az containerapp env create -n standing-env -g standing-rg -l eastus   # Consumption only
az containerapp create -n standing-api -g standing-rg --environment standing-env \
  --image <registry>.azurecr.io/standing-api:1 --registry-server <registry>.azurecr.io \
  --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1 \
  --ingress external --target-port 8000 \
  --secrets api-key=<choose-one> --env-vars API_KEY=secretref:api-key IMGSZ=640
```

Note: Azure Container Registry Basic has its own monthly fee. To keep the bill to compute
only, push the image to GitHub Container Registry instead and point `--image` there. Public
images there are free, and a private image counts against your GitHub Packages storage
allowance. The image is 1.68 GB (Python, ffmpeg, OpenVINO, OpenCV), so check your plan.
Before making it public, note that the bundled Ultralytics model is AGPL-3.0.
