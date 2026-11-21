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

For the other actions (`make_report.py --action sitting|jumping|smiling|talking`) the same files
are written to `results/<action>/`, with `<action>_sections.csv` in place of
`standing_sections.csv` and a `people_<action>` column in place of `people_standing`. Their
manual review goes in `results/<action>/verdicts.csv`, in the same format; until it exists every
section is listed as `not reviewed`.

Video files are expected to be named the way yt-dlp's `-o "%(upload_date)s_%(id)s.%(ext)s"`
names them (`YYYYMMDD_<11-character id>.<ext>`); the detector builds YouTube links from that.
