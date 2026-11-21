# Running in the cloud: one framework, swappable providers

The command line, the HTTP API and the cloud workers all run the same pipeline
(`VideoAnalysis` → the detector or the search). What changes between a laptop and Azure is only
**where files are kept** and **where the work waits**, and those are two small interfaces in
`youtube_analyser.cloud`:

| Interface | Does | On one machine | On Azure |
|---|---|---|---|
| `Store` | files by key: `put`, `get`, `put_file`, `get_file`, `put_stream`, `exists`, `delete`, `list` | `FileStore` (a folder): `/path` or `file:///path`; `MemoryStore` (inside one process): `memory://` | `AzureBlobStore` (a blob container): `azblob://<account>/<container>` |
| `Queue` | work to do, each message handled once: `send`, `receive` (hidden from other workers for a while), `done`, `retry`, `count` | `MemoryQueue` (inside one process): `memory://`; `FileQueue` (processes sharing a folder): `file:///path` | `AzureQueue` (a storage queue): `azqueue://<account>/<queue>` |

Everything is chosen by URL (`cloud.open_store(url)`, `cloud.open_queue(url)`), so the same
program runs anywhere:

```
                 ┌──────────────┐  uploads/<job>   ┌───────────┐
client ─POST───▶ │ API          │ ───────────────▶ │  Store    │ ◀── results: jobs/<job>/...
                 │ (yta serve)  │ ─ message ─┐     └───────────┘        searches/<id>/...
                 └──────────────┘            ▼           ▲
yta channel --store --queue ─ messages ─▶ ┌───────┐      │
                                          │ Queue │ ─▶ Worker(s) (yta worker, or threads in the API)
                                          └───────┘     same pipeline as the command line
```

- **`jobs.Worker`** takes messages and does the work: `"analyse"` (an uploaded video: the
  detector's sections for an action, or with `who=` the times a described person does it) and
  `"channel-video"` (one video of a channel search: download, quick look, search, thumbnails).
  The uploaded video is deleted from the store whatever happens; a channel video that fails is
  tried again (3 times) and then listed as a problem in the report.
- **`jobs.JobStore`** keeps each job's status and results in the store (`jobs/<job>/status.json`,
  `sections.csv`, `speech.csv`, `matches.csv`), so the API and the workers need not share a
  machine. Results are deleted `RESULT_TTL_S` after the job finished, or with `DELETE`.
- **`service.create_app(config, store, queue)`** is the API; with the in-memory queue it runs its
  own worker threads (`YTA_WORKERS`, default 1), so a single container is a complete service.

The tests run every store and queue against the same checks (`tests/test_cloud.py`): the folder
versions directly, and Azure's against the Azurite emulator (with the real Azure SDK), and the
same worker handles the same upload and channel search through each.

## Configurations

| | Store | Queue | Workers | For |
|---|---|---|---|---|
| **Laptop / single container** (default) | a folder (`WORK_DIR/store`) | `memory://` | threads in the API | trying it out; one person uploading now and then |
| **One machine, several processes** | a folder | `file:///path` | `yta worker` processes | a big machine |
| **Azure** | `azblob://account/container` | `azqueue://account/queue` | a Container Apps job running `yta worker --until-empty`, scaled by the queue's length | regular use; channel searches spread over machines |

Settings (environment of `yta serve`): `YTA_STORE`, `YTA_QUEUE`, `YTA_WORKERS`, `API_KEY`,
`MAX_UPLOAD_MB`, `RESULT_TTL_S`, `MODEL`, `IMGSZ`, `FPS`, `JUMP_FPS`, `INTRO_BANK`,
`PEOPLE_CONFIG`, `WORK_DIR` (see [api.md](api.md)). `yta worker` takes `--store` and `--queue`,
the pose model options, `--config`, `--intro-bank`, and yt-dlp's options (`--cookies`, ...) for
channel videos.

Azure sign-in: `AZURE_STORAGE_CONNECTION_STRING` when it is set (account keys, or Azurite),
otherwise the machine's identity through azure-identity's `DefaultAzureCredential` (a container
app's managed identity). Give that identity **Storage Blob Data Contributor** on the account and
**Storage Queue Data Message Processor** and **Sender** on the queue. Install with the `azure`
extra: `pip install "youtube-analyser[azure]"`.

## Adding another provider

A provider is a class with the methods above (a Store needs only `put`, `get`, `delete` and
`list`; `MemoryStore` in `cloud/local.py` is the smallest example) and a factory taking the URL.
Register it in code, or ship it as a package with an entry point (no change here needed):

```python
from youtube_analyser import cloud

class S3Store(cloud.Store):
    def __init__(self, url):              # s3://bucket/prefix
        import boto3
        ...
    def put(self, key, data): ...
    def get(self, key): ...               # KeyError when missing
    def delete(self, key): ...
    def list(self, prefix=""): ...

cloud.register_store("s3", S3Store)
# or, in the plugin's pyproject.toml:
# [project.entry-points."youtube_analyser.stores"]
# s3 = "my_plugin:S3Store"
```

Then `yta serve` with `YTA_STORE=s3://bucket/yta`, and `yta worker --store s3://bucket/yta ...`,
work unchanged. The shared checks in `tests/test_cloud.py` (`test_a_store_keeps_files_by_key`,
`test_a_queue_hands_out_each_message_once`) are what a new provider must pass.

## Azure, option A: one Container App

```
client ──POST video──▶ Container App (yta serve, scale 0..1) ──▶ CSV
                       video and results on the replica's disk, work in its memory
```

- Nothing outside the container: no storage account. Ephemeral disk is included in the replica
  (about 4 GiB at 1 vCPU, 8 GiB above; a 1-hour video is about 30 MB at 144p).
- It scales to zero, so you pay nothing between uploads.
- The catch: Container Apps scales replicas on HTTP traffic, and jobs run after the upload
  request returns, so a replica with no incoming requests can be scaled in mid-job (default
  cooldown 300 s). Polling `GET /jobs/{id}` every few seconds keeps traffic flowing, but this is
  not guaranteed; and the queue and results live in one replica, so run it with `maxReplicas 1`.
  A restart loses the queue: uploads left waiting are then deleted and their jobs marked failed.

```bash
az group create -n yta-rg -l eastus
az acr create -n <registry> -g yta-rg --sku Basic        # or use GitHub Container Registry
az acr build -r <registry> -t youtube-analyser:2 .
az containerapp env create -n yta-env -g yta-rg -l eastus   # Consumption only
az containerapp create -n yta-api -g yta-rg --environment yta-env \
  --image <registry>.azurecr.io/youtube-analyser:2 --registry-server <registry>.azurecr.io \
  --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1 \
  --ingress external --target-port 8000 \
  --secrets api-key=<choose-one> --env-vars API_KEY=secretref:api-key
```

## Azure, option B (recommended for regular use): Blob Storage, a queue and a worker job

```
client ──POST video──▶ API (Container App, scale to zero) ──▶ uploads/<job> in Blob, message on the queue
                       Container Apps job (event-driven on the queue, 0..N executions):
                           yta worker --until-empty  ─▶ jobs/<job>/sections.csv, deletes the video
client ──GET /jobs/{id}/sections.csv──▶ API reads it from Blob
```

This is the same image with different settings: the API runs without workers, the job runs
`yta worker`. A job execution runs to completion (jobs are not scaled in by HTTP traffic), each
one handles messages until the queue is empty, and executions scale to zero when it is.

```bash
az storage account create -n <account> -g yta-rg --sku Standard_LRS --allow-blob-public-access false
az storage account blob-service-properties update -n <account> -g yta-rg \
  --enable-delete-retention false --enable-versioning false         # deleted videos are really gone
az containerapp create -n yta-api ... (as above) --system-assigned \
  --env-vars API_KEY=secretref:api-key YTA_STORE=azblob://<account>/yta YTA_QUEUE=azqueue://<account>/yta-work YTA_WORKERS=0
az containerapp job create -n yta-worker -g yta-rg --environment yta-env \
  --image <registry>.azurecr.io/youtube-analyser:2 --registry-server <registry>.azurecr.io \
  --trigger-type Event --min-executions 0 --max-executions 10 --replica-timeout 7200 \
  --cpu 2 --memory 4Gi --mi-system-assigned \
  --command yta --args "worker --store azblob://<account>/yta --queue azqueue://<account>/yta-work --until-empty" \
  --scale-rule-name queue --scale-rule-type azure-queue \
  --scale-rule-metadata accountName=<account> queueName=yta-work queueLength=1 \
  --scale-rule-identity system
# then give both identities the storage roles listed above
```

Keep storage costs at zero: no soft delete or versioning (above), and a lifecycle rule that
deletes anything under `uploads/` older than 1 day (a safety net for uploads never processed)
and `jobs/` older than 7 days. Keep the account private; the apps reach it with their managed
identities, not keys. Uploads go through the API (streamed to Blob); for very large files a
direct upload to Blob with a short-lived write-only SAS URL would avoid the API's request limits
(not built: 144p videos are small).

This has not been deployed from here (the development machine has no Azure access), so treat
the commands as an outline and check them against `az containerapp job create --help`; the
store and queue settings are the ones the tests check against the Azurite emulator.

## Azure, option C: a whole channel on workers

`yta channel ... --store azblob://<account>/yta --queue azqueue://<account>/yta-work` puts one
message per video on the same queue (see [channel.md](channel.md#more-machines)); the worker job
above handles them too (give it `--cookies` if YouTube asks it to sign in), and `yta collect`
writes the report. YouTube limits how fast one address downloads, so many workers downloading at
once may need to go slower (`--yt-dlp-arg "--sleep-requests 1"` on the workers).

## What it costs

Azure Container Apps consumption prices, East US, from the Azure Retail Prices API (Nov 2026):
**$0.000024 per vCPU-second, $0.000003 per GiB-second, $0.40 per million requests.** Every
subscription gets **180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests free each
month**. Storage for videos held minutes and results of a few KB rounds to zero.

Measured speed: the container, limited to 2 CPUs, processed a 63-minute episode in 8.9 minutes
(standing), about 0.28 vCPU-seconds per sampled frame; the table uses a more expensive 0.4
(Azure CPUs differ; a pilot run will tell).

| 100 hours of video, 1 frame per second (360,000 frames), input 640 | Estimate |
|---|---|
| vCPU-seconds | 100,000–144,000 |
| GiB-seconds (2 GiB per vCPU) | 200,000–288,000 |
| Price before the free grant | ~$3.00–4.30 |
| Price if it's the only Container Apps use that month | $0 (within the free grant) |

Other actions add to this: sitting and walking a pass over the picture at 5 frames a second (about
2 minutes per hour of video), smiling, talking and silent a face pass (about 20% more when a face
is in view most of the time), talking and silent the sound (about 35 s per hour), running and
jumping the pose model again at 5 fps around whole bodies (a small part of a sit-down show). A
`who=` job adds the colours when the description has any (and stops there when nobody has them),
and for gender, voice or a name the talking inputs.

Channel searches (from the speeds in [channel.md](channel.md#speed)): someone who appears in few
videos went through video at 100x real time or more on 4 vCPUs, so 100 hours take about 1 hour of
4 vCPUs (about $0.45 before the free grant); a search that runs every step on every video
("people talking") went at about 4x, so 100 hours take about 100 vCPU-hours (about $11 before the
free grant, or 25 workers for an hour).

The price list also has an environment management charge of $0.10 an hour, which as far as I can
tell applies to workload-profiles (dedicated) environments: create a **Consumption-only**
environment and check the first bill. Azure Container Registry Basic has its own monthly fee; a
GitHub Container Registry image avoids it (check your plan: the image is about 1.7 GB). Before
making an image public, note that the bundled Ultralytics pose model is AGPL-3.0.

## Enhancement: confirm each section with a vision model

The pose rules are cheap and catch most standing people. On the 248 test videos, 77 of the 92
sections reported during the interviews were confirmed by eye; most of the rest were a guest
sitting alone, square-on to the camera, who looks straight-legged in 2D. A second, very cheap
check fixes exactly that kind of mistake: send each section's best frame (the thumbnail) to a
small vision model on Azure OpenAI ("Is any real person in this frame standing? yes or no") and
keep the yes. This runs once per section, not per frame: cents per 100 hours of video.
