# Sample files (made-up names)

The real inventory, results and manual review name the videos, so they are kept locally and
are not in this repository (see `.gitignore`). These samples show the format of each file with
a few rows each. All video names, IDs, titles and notes are made up; the times and counts come
from real rows.

| Sample | Real file (local only) | Written by |
|---|---|---|
| `analysis/videos.csv` | `analysis/videos.csv` | You: one row per downloaded video. `make_report.py` reads `file`, `upload_date`, `title` and `duration_min` |
| `results/verdicts.csv` | `results/verdicts.csv` | You: the manual review, one row per section (`video` + `start_s`). Verdicts: `confirmed`, `unclear`, `false alarm`, or `after the interview` for standing in material played after the interview |
| `results/standing_sections.csv` | `results/standing_sections.csv` | `make_report.py`: standing during the interviews, with verdicts |
| `results/outside_interview.csv` | `results/outside_interview.csv` | `make_report.py`: standing in the intros, and sections marked `after the interview` |
| `results/interview_starts.csv` | `results/interview_starts.csv` | `make_report.py`, from `intro.py`: where each video's interview starts and how it was found (`title`, `angles`, `shared`, `short video`) |
| `results/search/<query>/matches.csv` | `results/search/<query>/matches.csv` | `search.py`: each time a described person does the action. `person` numbers the people followed in that video; `match` is `match`, or `possible` with `--possible` (something was not seen well enough); `evidence` lists each part of the query with what was seen and `[yes]`, `[no]` or `[unknown]` |

For the other actions (`make_report.py --action sitting|walking|running|jumping|smiling|talking|silent`) the same files
are written to `results/<action>/`, with `<action>_sections.csv` in place of
`standing_sections.csv` and a `people_<action>` column in place of `people_standing`. Their
manual review goes in `results/<action>/verdicts.csv`, in the same format; until it exists every
section is listed as `not reviewed`. Talking and silent also get `speech.csv` (one row per speech
turn: `upload_date, title, video, start_s, end_s, start, end, duration_s, voice, on_screen,
sound_trusted, youtube`) and `voices.csv` (per video and voice: `seconds, turns,
on_screen_seconds, sound_trusted, thumb`), and their sections `voices` / `voices_heard` and
`judged_by` columns.

Video files are expected to be named the way yt-dlp's `-o "%(upload_date)s_%(id)s.%(ext)s"`
names them (`YYYYMMDD_<11-character id>.<ext>`); the detector builds YouTube links from that.
