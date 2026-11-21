"""Build the results/ files from a detector run.

Joins the detector's sections (output/segments.csv, and intro.csv for standing before
the interview starts) with the manual review (results/verdicts.csv) and the episode titles
(analysis/videos.csv). Sections that have no verdict yet are listed as "not reviewed".

Writes results/standing_sections.csv (standing during the interviews), outside_interview.csv
(standing in the intros, found by the detector, and after the interview, marked in the review;
both are left out of the main list), interview_starts.csv and README.md.

With --action (sitting, walking, jumping, smiling, talking, silent) it reads
output/<action>/ and results/<action>/verdicts.csv (optional until you have reviewed
some sections) and writes the same files to results/<action>/, named <action>_sections.csv and
so on. For talking and silent it also writes speech.csv (who spoke when, from the sound) and
voices.csv (each video's voices), with episode titles.

From the command line: `yta report` (standing) or `yta report --action talking`.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

from .actions import ACTIONS as _ACTIONS

OUT = Path("output")
RESULTS = Path("results")
INVENTORY = Path("analysis") / "videos.csv"
FIELDS = ["upload_date", "video", "title", "start", "end", "duration_s", "start_s", "end_s",
          "people_standing", "verdict", "note", "youtube"]
START_FIELDS = ["upload_date", "video", "title", "interview_start", "interview_start_s", "how"]
AFTER = "after the interview"  # verdict for standing in material played after the interview
ACTIONS = list(_ACTIONS)
NOTES = ["voices", "voices_heard", "judged_by"]  # extra section columns some actions have


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def make_report(action="standing", detector_output=OUT, results=RESULTS, inventory=INVENTORY, log=print):
    """Write the results files for an action (see above). Without an inventory, titles and
    upload dates are left empty (and the hours are not counted)."""
    detector_output, results, inventory = Path(detector_output), Path(results), Path(inventory)
    out = detector_output if action == "standing" else detector_output / action
    results = results if action == "standing" else results / action
    if not (out / "segments.csv").exists():
        raise FileNotFoundError(f"no detector output in {out}: run yta detect first")
    fields = [f if f != "people_standing" else f"people_{action}" for f in FIELDS]
    if action != "standing":
        with open(out / "segments.csv", newline="") as f:
            header = csv.DictReader(f).fieldnames or []
        notes = [n for n in NOTES if n in header]
        fields = fields[:fields.index("verdict")] + notes + fields[fields.index("verdict"):]
    if inventory.exists():
        videos = {r["file"]: r for r in load(inventory)}
    else:
        log(f"no inventory at {inventory}: titles and dates are left empty", file=sys.stderr)
        videos = {}
    verdict_file = results / "verdicts.csv"
    if not verdict_file.exists():
        log(f"no review yet ({verdict_file}): every section is listed as not reviewed", file=sys.stderr)
    verdicts = {(r["video"], float(r["start_s"])): r for r in (load(verdict_file) if verdict_file.exists() else [])}
    sections, dropped = load(out / "segments.csv"), load(out / "dropped.csv")
    intro, starts = load(out / "intro.csv"), load(out / "interview_starts.csv")
    analysed = sorted(p.stem for p in (out / "segments").glob("*.csv"))

    def rows_for(found):
        out = []
        for s in found:
            v = verdicts.get((s["video"], float(s["start_s"])), {})
            meta = videos.get(s["video"], {})
            out.append({"upload_date": meta.get("upload_date", ""), "video": s["video"],
                        "title": meta.get("title", ""), "start": s["start"], "end": s["end"],
                        "duration_s": s["duration_s"], "start_s": s["start_s"], "end_s": s["end_s"],
                        f"people_{action}": s[f"max_people_{action}"], **{n: s[n] for n in NOTES if n in s},
                        "verdict": v.get("verdict", "not reviewed"), "note": v.get("note", ""),
                        "youtube": s["youtube"]})
        return sorted(out, key=lambda r: (r["video"], float(r["start_s"])))

    rows, intro_rows = rows_for(sections), rows_for(intro)
    after_rows = [r for r in rows if r["verdict"] == AFTER]
    rows = [r for r in rows if r["verdict"] != AFTER]
    outside = sorted([{**r, "where": "intro"} for r in intro_rows] + [{**r, "where": AFTER} for r in after_rows],
                     key=lambda r: (r["video"], float(r["start_s"])))
    start_rows = sorted(({"upload_date": videos.get(s["video"], {}).get("upload_date", ""), "video": s["video"],
                          "title": videos.get(s["video"], {}).get("title", ""),
                          "interview_start": s["interview_start"], "interview_start_s": s["interview_start_s"],
                          "how": s["how"]} for s in starts), key=lambda r: r["video"])
    results.mkdir(parents=True, exist_ok=True)
    save(results / f"{action}_sections.csv", fields, rows)
    save(results / "outside_interview.csv", ["where"] + fields, outside)
    save(results / "interview_starts.csv", START_FIELDS, start_rows)

    by_stem = {Path(f).stem: r for f, r in videos.items()}
    hours = sum(float(by_stem[stem]["duration_min"] or 0) for stem in analysed if stem in by_stem) / 60
    count = Counter(r["verdict"] for r in rows)
    if action != "standing":
        voices = []
        if (out / "speech.csv").exists():
            def titled(r):
                meta = videos.get(r["video"], {})
                return {"upload_date": meta.get("upload_date", ""), "title": meta.get("title", ""), **r}
            speech = [titled(r) for r in load(out / "speech.csv")]
            voices = [titled(r) for r in load(out / "voices.csv")]
            for name, found in (("speech.csv", speech), ("voices.csv", voices)):
                save(results / name, ["upload_date", "title"] + [k for k in (found[0] if found else {}) if
                                                                 k not in ("upload_date", "title")], found)
        (results / "README.md").write_text(action_readme(action, rows, intro_rows, after_rows, dropped, start_rows,
                                                         len(analysed), hours, count, voices))
        log(f"{len(rows)} sections ({dict(count)}) -> {results / f'{action}_sections.csv'}, {results / 'README.md'}")
        return
    good = [r for r in rows if r["verdict"] in ("confirmed", "unclear")]
    confirmed = [r for r in rows if r["verdict"] == "confirmed"]
    with_standing = sorted({r["video"] for r in confirmed})
    notes = Counter(r["note"] or "person standing in the studio" for r in confirmed)

    hows = Counter(r["how"] for r in start_rows)
    long_starts = sorted(float(r["interview_start_s"]) for r in start_rows if r["how"] != "short video")
    md = [
        "# Where someone is standing during the interviews: results for the downloaded videos",
        "",
        "Generated by `yta report`. Local only: this file names the videos.",
        "",
        f"The standing detector (`yta detect`) analysed **{len(analysed)} videos "
        f"({hours:.1f} hours)** at 1 frame per second. Only standing **during the interviews** is "
        "listed: each episode's intro (cold-open teaser, title sequence, sponsor ad, logo card) is "
        "found automatically and left out, and so is anything played after the interview. After "
        f"dropping {len(dropped)} sections that were statues, posters or artwork (figures that never "
        f"moved), the detector reported **{len(rows)} standing sections** in the interviews. Every "
        "one was checked by eye:",
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
        "## Left out: before and after the interview",
        "",
        f"- **Intros: {len(intro_rows)} sections** in {len({r['video'] for r in intro_rows})} episodes "
        "(title sequences and cold-open teasers). `yta intro` (intro.py) finds where each interview "
        "starts: title sequences, ads and logo cards are shared by many episodes, so the interview "
        "starts after the last stretch of footage the episode shares with 3 or more other episodes. "
        "Episodes whose titles are unique to them use the first stretch of camera angles that recur "
        "later in the episode instead.",
        f"  - The interview starts after a median of {long_starts[len(long_starts) // 2] if long_starts else 0:.0f} s "
        f"(range {long_starts[0] if long_starts else 0:.0f}-{long_starts[-1] if long_starts else 0:.0f} s). "
        "Found from the shared title in "
        f"{hows['title']} episodes, from recurring camera angles in {hows['angles']}, from footage shared "
        f"with one other episode in {hows['shared']}; Shorts ({hows['short video']}) have no intro.",
        "  - Check the starts by eye: look at the frames either side of each one. A sponsor ad that no "
        "other episode shares is counted as interview.",
        f"- **After the interview: {len(after_rows)} section(s)** marked `after the interview` in the "
        "review (for example a trailer played after the end card). The end of the interview is not "
        "detected automatically, so check the sections in the last few minutes of each episode.",
        "",
        "Both are listed in [`outside_interview.csv`](outside_interview.csv), and every episode's "
        "interview start is in [`interview_starts.csv`](interview_starts.csv).",
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
        "  - 32 frames from the first 20 seconds of episodes (mostly intros, now left out): 3 to 5 "
        "missed, all waist-up shots of standing people whose legs are out of frame.",
        "  - 32 random frames: no clear misses, one ambiguous (people at a table).",
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
           "`output/dropped.csv`; all were checked and none was a real person.", ""]
    (results / "README.md").write_text("\n".join(md))
    log(f"{len(rows)} sections ({dict(count)}) -> {results / 'standing_sections.csv'}, {results / 'README.md'}")


WHAT = {  # how each action is found, and what it misses; see docs/actions.md for the details
    "sitting": "the pose model's leg geometry (the standing detector's rules): a thigh level or sideways, "
               "or every visible leg bent, and no leg straight and upright. Only people whose hips and "
               "knees are in view can be judged, so close-ups are never reported.",
    "jumping": "the lowest foot and the hips both rising at least 0.25 torso lengths and coming back down "
               "within a second, with camera movement taken out, in extra samples at 5 per second around "
               "every moment a whole body is in view. Small hops (under about 0.3 torso lengths) are often "
               "missed; people whose feet are out of frame are never reported.",
    "smiling": "a small expression model (trained on FER2013) on each face, cut out around the pose model's "
               "eyes: the median 'happy' probability over 5 frames in 0.8 s is 0.6 or more. Only faces "
               "with 11 px or more between the eyes are judged (close-ups at 144p).",
    "talking": "each visible face's mouth movement (the MediaPipe face landmark model; faces from 6 px between "
               "the eyes) together with the sound: where the sound is clear, a person is talking when speech is "
               "heard and their mouth is the one moving, and nobody is talking when no speech is heard; where it is "
               "not (a loud background, no sound track), when the lips move by 0.12 or more within 0.8 s. Voices "
               "(A, B...) come from the sound, per video.",
    "silent": "the same faces and sound as talking: a visible face that is not talking, either listening to "
              "someone else (voices_heard) or with nobody speaking.",
    "running": "how fast each person's body travels over the ground in extra samples at 5 per second around "
               "every moment a whole body is in view, with the camera's movement (pan, tilt, zoom) taken out: "
               "4 torso lengths a second or more (a brisk walk is up to about 3.5).",
    "walking": "how far each person's body travels between the 1-fps samples, with the camera's movement "
               "taken out: 0.6 torso lengths a second or more (or walking towards or away from the camera), "
               "upright and not seated. People are followed by position, size and the colours of their clothes.",
}


def action_readme(action, rows, intro_rows, after_rows, dropped, start_rows, n_videos, hours, count, voices=()):
    """README.md for an action other than standing: what was found, and the review so far."""
    confirmed = [r for r in rows if r["verdict"] == "confirmed"]
    shown = [r for r in rows if r["verdict"] != "false alarm"]
    hows = Counter(r["how"] for r in start_rows)
    md = [
        f"# Where someone is {action} during the interviews: results for the downloaded videos",
        "",
        f"Generated by `yta report --action {action}`. Local only: this file names the videos.",
        "",
        f"`yta detect --action {action}` analysed **{n_videos} videos ({hours:.1f} hours)**. "
        f"It finds {action} from {WHAT[action]}",
        "",
        f"Only sections **during the interviews** are listed: {len(intro_rows)} in the intros (found "
        f"automatically: {hows['title']} episodes by their title sequence, {hows['angles']} by recurring camera "
        f"angles, {hows['shared']} by shared footage, {hows['short video']} Shorts without an intro) and "
        f"{len(after_rows)} marked `{AFTER}` in the review are in [`outside_interview.csv`](outside_interview.csv)"
        + (f", and {len(dropped)} sections whose figure never moved (statues, posters) were dropped" if dropped else "")
        + ".",
        "",
        f"The detector reported **{len(rows)} {action} sections** in the interviews, adding up to "
        f"{sum(float(r['duration_s']) for r in rows) / 60:.1f} minutes.",
        "",
        "| Verdict | Sections |",
        "|---|---|",
        *[f"| {v} | {c} |" for v, c in sorted(count.items())],
        "",
    ]
    if count["not reviewed"]:
        md += [f"{count['not reviewed']} sections are not reviewed yet. Watch them (for example with "
               f"`yta review --action {action}`) and add a row per section to "
               f"`results/{action}/verdicts.csv` (`video,start_s,verdict,note`; verdicts `confirmed`, "
               f"`unclear`, `false alarm` or `{AFTER}`), then run this again.", ""]
    if confirmed:
        md += [f"Confirmed {action} appears in **{len({r['video'] for r in confirmed})} videos** and adds up to "
               f"**{sum(float(r['duration_s']) for r in confirmed) / 60:.1f} minutes**.", ""]
    md += [f"The full list, with titles, YouTube links that open at each section and the verdict, is in "
           f"[`{action}_sections.csv`]({action}_sections.csv).", ""]
    if voices:
        md += ["## Voices", "",
               "From the sound, per video (A speaks most; the same person has a different letter in another "
               "video). Who spoke when, and whether they were on screen, is in [`speech.csv`](speech.csv).", "",
               "| Video | Title | Voice | Minutes | Turns | Minutes seen talking | Sound clear |", "|---|---|---|---|---|---|---|"]
        md += [f"| `{v['video']}` | {v['title'].replace('|', '/')} | {v['voice']} | {float(v['seconds']) / 60:.1f} | "
               f"{v['turns']} | {float(v['on_screen_seconds']) / 60:.1f} | {float(v['sound_trusted'] or 0):.0%} |"
               for v in voices]
        md += ["", "Where the sound is not clear (a loud background), voices are unreliable and talking was "
               "judged from the faces alone."]
        md += [""]
    md += [
"## Sections by video (all but false alarms)", "",
           "| Uploaded | Video | Title | Start | End | Seconds | Verdict | Note |",
           "|---|---|---|---|---|---|---|---|"]
    for r in shown:
        title = r["title"].replace("|", "/")
        link = f"[{r['start']}]({r['youtube']})" if r["youtube"] else r["start"]
        md.append(f"| {r['upload_date']} | `{r['video']}` | {title} | {link} | {r['end']} | "
                  f"{r['duration_s']} | {r['verdict']} | {r['note']} |")
    if count["false alarm"]:
        md += ["", "## False alarms", "", "| Video | Start | End | Note |", "|---|---|---|---|"]
        md += [f"| `{r['video']}` | {r['start']} | {r['end']} | {r['note']} |"
               for r in rows if r["verdict"] == "false alarm"]
    return "\n".join(md + [""])

