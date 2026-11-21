# Searching a whole channel (`yta channel`)

One command from a channel's address to a report of every moment found, with a link to each on
YouTube:

```bash
yta channel "https://www.youtube.com/@SomeShow" "people jumping"
yta channel "https://www.youtube.com/@SomeShow" "woman with purple hair talking"
yta channel "https://www.youtube.com/@SomeShow" "man in a blue coat walking" --max-videos 50
yta channel ./my_videos "someone smiling"             # videos you already have
```

The query is `yta search`'s: an action alone finds **everyone doing it** (the detector's sections);
a description, with or without an action, finds **that person doing it**. It prints what it
understood before it starts (`looking for: gender female, hair purple; action: talking`).

**What it does**

1. Lists the channel's videos with yt-dlp (its Videos tab, newest first; `--shorts` adds the
   Shorts; live and upcoming streams are left out). A playlist or a single video's address works
   too.
2. Downloads them at the lowest resolution, 144p (the format the original 248 videos were
   downloaded in), **without the sound when the query does not need it** (talking, silent, a
   voice, gender and names do), 3 at a time (`--downloads`) and ahead of the analysis (up to 6
   waiting, `--ahead`), so downloading and analysing overlap. Each is named `YYYYMMDD_<id>` and
   listed in `videos.csv` (upload date, title, duration) for the links and the report.
3. Finds each episode's intro (title sequence, teaser) as `yta intro` does, from the title footage
   the first 8 long videos share (`intro_bank.npz`): results there are kept, but listed apart at
   the end of the report. A channel whose episodes share no title has no bank, and nothing is
   set apart. `--intros keep` treats every video as a whole.
4. Searches each video (`yta search`, cheapest first). **With a colour in the query, a quick look
   comes first**: the pose model and the colours on the video's keyframes only (decoding only
   keyframes takes 2 s for a 27-minute video, against 10 s for every frame), at most one every
   4 s. A video where nobody has the colour is done there; otherwise it is searched in full.
5. Rewrites the report after every video, so it can be opened while the run goes on.

**Output**, in `channels/<channel>/searches/<query>/`: `report.html` (thumbnails with the person
boxed, grouped by video, newest first; each links to its moment on YouTube), `report.md` (the
same as a list of links), `matches.csv` (upload date, title, video ID, start, end, action,
match, whether it is in the intro, the evidence, the link) and `thumbs/`. The work folder
(`--work`, default `channels/<channel>/`) also keeps the videos, `videos.csv` and the model
output (`output/`), so an interrupted run carries on where it stopped (videos already searched
are skipped), and a second query on the same channel reuses the videos and everything already
computed. `--delete-videos` deletes each video once searched (a later query that needs something
new downloads it again). Everything in `channels/` names the videos, so it is not committed.

## Speed

`--speed`:

| Setting | What it does | |
|---|---|---|
| `quick` (default) | The quick look first, for queries with a colour | Can miss someone who is on screen for under about 10 s |
| `thorough` | Every second of every video | The tested settings, throughout |
| `fastest` | `quick`, and the pose model at input size 320 instead of 640 (about 4x faster) | Passed the same end-to-end tests (movement, talking, search) as 640; finds fewer small, distant people. Its model output is kept apart (`output-320/`) |

Measured on a 27-minute test video (two or three people in close-ups nearly all the time, so
faces and voices are everywhere), on the 4-core sandbox used for development, whose speed varied
by up to 2x from run to run:

| Query | Speed | Time | Real time |
|---|---|---|---|
| "woman with green hair talking" (nobody has green hair) | quick | 14 s (33 s for the first video of a run, which also loads the models) | 115x |
| | fastest | 7 s (21 s for the first) | 230x |
| | thorough | 162 s | 10x |
| "woman with purple hair talking" (she is there: 16 results) | quick | 320-410 s (two runs) | 4-5x |
| | fastest | 220-270 s (two runs) | 6-7x |
| | thorough | 412 s | 4x |
| "people jumping" (nobody jumps; no whole bodies in view) | quick | 104 s | 16x |
| "people talking" (53 results) | quick | 377 s | 4x |

Before these changes a person search ran every step on every video, with the gender model on
every face: about 390 s for this video (measured when the same machine ran the pose model in 91 s
rather than the 115-125 s of these runs). So a search for someone who is in few of the videos,
which is what a search usually is, is now about 30x faster (`quick`) to 60x (`fastest`); in the
videos where they might be, it is 1-1.5x faster, and the face and sound passes for talking (about
a third of the time) and the pose model are most of it. Downloading at 144p (about 30 MB an hour
of video, less without the sound) happens alongside the analysis, and should rarely hold it up.

## More machines

One run uses every core of its machine. To go faster, spread the videos over more machines:

- **`--shard K/N`** searches every Nth video, starting with the Kth, so N runs share a channel:
  on one machine with many cores (each run with `--threads`), or on several machines sharing the
  work folder (a network drive); each run's report shows every run's results.
- **Cloud workers:** with `--store` and `--queue`, `yta channel` does not search itself. It lists
  the channel, fingerprints the voices of people enrolled by name, builds the intro bank from the
  first long videos, and puts one message per video on the queue; any number of `yta worker`
  processes (a Container Apps job on Azure, scaling with the queue) download and search the
  videos and write their results to the store, and `yta collect` writes the report:

  ```bash
  yta channel "https://www.youtube.com/@SomeShow" "people jumping" \
      --store azblob://myaccount/yta --queue azqueue://myaccount/yta-work
  yta worker --store azblob://myaccount/yta --queue azqueue://myaccount/yta-work --until-empty   # on each worker
  yta collect <search id> --store azblob://myaccount/yta --out report/
  ```

  The same works on one machine with a folder and a folder queue (`--store ./store --queue
  file://./queue`, and several `yta worker` processes), which is how it is tested. See
  [cloud.md](cloud.md).

## Options

`--max-videos N` (the newest N), `--since` / `--until YYYYMMDD` (upload dates: videos outside
them are skipped before downloading when the channel's listing gives dates, else by yt-dlp before
it downloads; a channel is listed newest first, so a run stops after 5 in a row older than
`--since`; on a folder, the dates in the file names),
`--shorts`, `--action` (instead of one in the query), `--config` (your people.toml),
`--possible`, `--explain`, `--cookies cookies.txt` or `--cookies-from-browser firefox` (if
YouTube asks you to sign in), `--yt-dlp-arg "--sleep-requests 1"` (any yt-dlp option, e.g. to go
slower if YouTube limits you), `--yt-dlp` (the yt-dlp command), `--runtime`, `--threads`.

## yt-dlp

yt-dlp is not pinned with the other packages, because YouTube changes often and only a recent
yt-dlp keeps up: install or update it with `pip install -U yt-dlp` (or use one on your PATH). If it reports that YouTube needs a JavaScript runtime or a sign-in, follow its
message (its wiki covers both). This tool was tested with a stand-in for yt-dlp
(`tests/fake_yt_dlp.py`), not against YouTube itself, which the development machine could not
reach; the download format is the one used for the original videos. Download only what you are
allowed to.

