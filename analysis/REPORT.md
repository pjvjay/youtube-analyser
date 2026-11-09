# youtube-analyser: what is on disk (checked 2026-11-02)

## What this folder is

There is no analyser code here, and none anywhere else under `~/Documents`, `~/Desktop` or `~/Downloads`.
The folder is a yt-dlp download of the YouTube channel **Private Talk With Alexis Texas**
(confirmed through YouTube's public oEmbed endpoint), plus yt-dlp's bookkeeping files:

| File | What it is |
|---|---|
| `archive.txt` | yt-dlp `--download-archive`: 248 IDs, one per downloaded video |
| `long_ids.txt` | 218 IDs of the channel's long videos (written 16:36, after the first batch) |
| `cookies.txt` | Browser cookies exported by yt-dlp (1 MB). **This is a logged-in session; never commit or share this folder with it inside.** |
| `YYYYMMDD_<id>.<ext>` | The videos: upload date + YouTube ID, no title |

Tooling used: yt-dlp 2026.08.19 in `~/.venvs/ytdlp` (with the `bgutil-ytdlp-pot-provider` plugin),
and `~/.config/yt-dlp/config` containing `--remote-components ejs:github` (the line appears twice).

## Inventory

- 249 files = 248 unique videos (one duplicate, below). 6.83 GB, 211.2 hours.
- 210 long videos (16 to 99 min, median 61 min), 39 Shorts (vertical, ≤ 3 min).
- Uploads from 2019-11-15 to 2026-09-07.
- Every file has a readable video stream and decodes cleanly to the end (ffprobe + a tail decode of every file).
- Every ID in `archive.txt` is on disk, and every file on disk is in `archive.txt`.

| Resolution | Files |
|---|---|
| 256x144 (long videos) | 205 |
| 144x256 (Shorts) | 39 |
| 640x360 | 5 |

## Two download batches on 2026-09-09

| Batch | Files | Upload dates | Audio |
|---|---|---|---|
| 16:10 to 16:36 | 67 (28 long, 39 Shorts) | 2025-02-21 to 2026-09-07 | 65 have audio |
| 18:08 to 18:56 | 182 (all long) | 2019-11-15 to 2026-06-26 | **5 have audio** |

The yt-dlp config was written at 18:05, just before the second batch.

## Problems a test run will hit

1. **178 long videos have no audio track** (177 from the second batch, plus `20250221_Au3rpg7iLTY.webm`).
   That is 182 of the 211 hours. Anything based on speech (transcripts, topics, quotes) can only use
   the 31 long videos and 39 Shorts that have audio. Long videos with audio only go back to 2024-02-15.
   `needs_audio_redownload.txt` lists the 178 URLs.
2. **Duplicate:** `20250221_Au3rpg7iLTY.f278.webm` and `20250221_Au3rpg7iLTY.webm` are byte-identical
   video-only files (format 278). The audio was never merged in. The `.f278.webm` can be deleted.
3. **9 long videos in `long_ids.txt` were never downloaded** and are not in `archive.txt`.
   All 9 are still public. See `missing_long_videos.csv`.
4. **144p only.** Frames are good enough for scene changes, people on screen and large on-screen text
   (sponsor reads are readable), not for small text or fine detail.
5. **No titles or metadata in the files.** `videos.csv` adds titles from oEmbed for all 249 files.
   Long titles are mostly `GUEST | S0x EP n`. 56 files on disk are "After Dark" episodes, and 5 of the
   9 missing videos are too. Episode numbers run up to 215. 30 numbers are absent from both the
   downloads and the missing list, but 19 long videos have no EP number in their title, so this
   does not prove those episodes are missing. The full channel list was not checked against `long_ids.txt`.
6. **No transcription tool is installed** (no whisper / whisper.cpp / mlx-whisper). ffmpeg and ffprobe are.

## Files in this folder

| File | Contents |
|---|---|
| `videos.csv` | One row per file: date, ID, URL, title, episode number, kind, duration, size, codecs, has_audio, test-run flag |
| `test_run.csv` | The 4-file test-run set below |
| `missing_long_videos.csv` | The 9 long videos never downloaded |
| `needs_audio_redownload.txt` | URLs of the 178 long videos without audio |

## Test-run set

One file for each case the analyser has to handle:

| File | Case |
|---|---|
| `20260905_-Q7NEKi7Oow.mkv` | Short (21 s) with audio |
| `20260703_6lr5TYxp7Xk.webm` | Long (26 min) with audio, 144p AV1/Opus |
| `20240215_9SXcGG8lLsQ.mp4` | Long (28 min) with audio, 360p H.264/AAC |
| `20241111_QTTiwfl7Lbc.mp4` | Long (16 min), video only, no audio |

Smoke test on these four: ffmpeg extracted 60 s of 16 kHz mono WAV from the three with audio
(mean volume −17 to −25 dB, so not silent) and frames from all four, each in about a second.

## Getting the missing audio

Audio-only downloads are small. Run from this folder. Leave out `--download-archive archive.txt`:
these IDs are already in it, so yt-dlp would skip all of them.

```bash
~/.venvs/ytdlp/bin/yt-dlp -f ba --cookies cookies.txt -a analysis/needs_audio_redownload.txt -o "audio/%(upload_date)s_%(id)s.%(ext)s"
```

The 9 missing long videos (full download, recorded in the archive):

```bash
cut -d, -f2 analysis/missing_long_videos.csv | tail -n +2 > /tmp/missing_urls.txt && ~/.venvs/ytdlp/bin/yt-dlp --cookies cookies.txt --download-archive archive.txt -a /tmp/missing_urls.txt -o "%(upload_date)s_%(id)s.%(ext)s"
```
