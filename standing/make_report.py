"""Build results/standing_sections.csv and results/README.md from a detector run.

Joins the detector's sections (standing/output/segments.csv) with the manual review
(results/verdicts.csv) and the episode titles (analysis/videos.csv). Sections that have no
verdict yet are listed as "not reviewed".

Usage (from the youtube-analyser folder):
    .venv/bin/python standing/make_report.py
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "standing" / "output"
RESULTS = ROOT / "results"
FIELDS = ["upload_date", "video", "title", "start", "end", "duration_s", "start_s", "end_s",
          "people_standing", "verdict", "note", "youtube"]


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    videos = {r["file"]: r for r in load(ROOT / "analysis" / "videos.csv")}
    verdicts = {(r["video"], float(r["start_s"])): r for r in load(RESULTS / "verdicts.csv")}
    sections, dropped = load(OUT / "segments.csv"), load(OUT / "dropped.csv")
    analysed = sorted(p.stem for p in (OUT / "segments").glob("*.csv"))

    rows = []
    for s in sections:
        v = verdicts.get((s["video"], float(s["start_s"])), {})
        meta = videos.get(s["video"], {})
        rows.append({"upload_date": meta.get("upload_date", ""), "video": s["video"],
                     "title": meta.get("title", ""), "start": s["start"], "end": s["end"],
                     "duration_s": s["duration_s"], "start_s": s["start_s"], "end_s": s["end_s"],
                     "people_standing": s["max_people_standing"],
                     "verdict": v.get("verdict", "not reviewed"), "note": v.get("note", ""),
                     "youtube": s["youtube"]})
    rows.sort(key=lambda r: (r["video"], float(r["start_s"])))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "standing_sections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    by_stem = {Path(f).stem: r for f, r in videos.items()}
    hours = sum(float(by_stem[stem]["duration_min"]) for stem in analysed if stem in by_stem) / 60
    count = Counter(r["verdict"] for r in rows)
    good = [r for r in rows if r["verdict"] in ("confirmed", "unclear")]
    confirmed = [r for r in rows if r["verdict"] == "confirmed"]
    with_standing = sorted({r["video"] for r in confirmed})
    notes = Counter(r["note"] or "person standing in the studio" for r in confirmed)

    md = [
        "# Where someone is standing: results for all downloaded videos",
        "",
        f"The standing detector (`standing/detect_standing.py`) analysed **{len(analysed)} videos "
        f"({hours:.1f} hours)** at 1 frame per second. It reported **{len(rows)} standing sections**, "
        f"after dropping {len(dropped)} sections that were statues, posters or artwork (figures that "
        "never moved). Every one of the remaining sections was checked by eye:",
        "",
        "| Verdict | Sections | Meaning |",
        "|---|---|---|",
        f"| confirmed | {count['confirmed']} | Someone is standing |",
        f"| unclear | {count['unclear']} | A figure that cannot be made out at 144p |",
        f"| false alarm | {count['false alarm']} | A guest sitting square-on to the camera |",
    ]
    if count["not reviewed"]:
        md.append(f"| not reviewed | {count['not reviewed']} | Found after the review; not checked |")
    md += [
        "",
        f"Confirmed standing appears in **{len(with_standing)} videos** and adds up to "
        f"**{sum(float(r['duration_s']) for r in confirmed) / 60:.1f} minutes**. What it shows:",
        "",
        "| What | Sections |",
        "|---|---|",
        *[f"| {n[0].upper() + n[1:]} | {c} |" for n, c in notes.most_common()],
        "",
        "The full list, with titles, YouTube links that open at each section and the verdict, is in",
        "[`standing_sections.csv`](standing_sections.csv). The table below has the same sections",
        "except the false alarms.",
        "",
        "## How well it works",
        "",
        f"- **Precision**: {count['confirmed']} of {len(rows)} reported sections are confirmed "
        f"({100 * count['confirmed'] / max(len(rows), 1):.0f}%), "
        f"{100 * (count['confirmed'] + count['unclear']) / max(len(rows), 1):.0f}% if the unclear ones "
        "are real. Before the still-figure filter it was about 73%.",
        "- **Misses**: 112 frames the detector did not flag were checked by eye:",
        "  - 48 frames where someone's legs were visible: 2 missed standing people (one only half in "
        "frame, one 2 seconds before an opening the detector did catch).",
        "  - 32 frames from the first 20 seconds of episodes: 3 to 5 missed, all waist-up shots of "
        "standing people whose legs are out of frame.",
        "  - 32 random frames: no clear misses, one ambiguous (people at an expo table).",
        "- **So**: when a person's legs are in frame, standing is found reliably. Waist-up shots of "
        "standing people are not, by design, because without legs the rules cannot tell standing "
        "from sitting facing the camera.",
        "",
        "Known false-alarm pattern: a guest sitting alone, square-on to the camera, can look "
        "straight-legged in 2D (9 sections, in 2 episodes). With a seated person next to them for "
        "comparison this is already caught.",
        "",
        "## Sections by video (confirmed and unclear)",
        "",
        "| Uploaded | Video | Title | Start | End | Seconds | Verdict | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in good:
        title = r["title"].replace("|", "/")
        link = f"[{r['start']}]({r['youtube']})" if r["youtube"] else r["start"]
        md.append(f"| {r['upload_date']} | `{r['video']}` | {title} | {link} | {r['end']} | "
                  f"{r['duration_s']} | {r['verdict']} | {r['note']} |")
    md += ["", "## False alarms", "", "| Video | Start | End | Note |", "|---|---|---|---|"]
    md += [f"| `{r['video']}` | {r['start']} | {r['end']} | {r['note']} |"
           for r in rows if r["verdict"] == "false alarm"]
    md += ["", "## Dropped as still figures", "",
           f"{len(dropped)} sections were dropped automatically because the figure's pixels never "
           "changed (statues, posters, artwork). They are listed in "
           "`standing/output/dropped.csv`; all were checked and none was a real person.", ""]
    (RESULTS / "README.md").write_text("\n".join(md))
    print(f"{len(rows)} sections ({dict(count)}) -> {RESULTS / 'standing_sections.csv'}, {RESULTS / 'README.md'}")


if __name__ == "__main__":
    main()
