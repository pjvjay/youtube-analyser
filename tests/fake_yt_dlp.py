"""A stand-in for yt-dlp, for testing channel.py without YouTube.

The "channel" is a JSON file (FAKE_CHANNEL): {"videos": [{"id", "title", "upload_date", "clip"}]},
newest first, clip being a local video file. It answers the two calls channel.py makes:

    --flat-playlist --dump-single-json <channel url>      the channel's videos as a playlist
    -f <format> -o <template> ... --print after_move:... <video url>
                                                          "downloads" the clip: copies it (without
                                                          its sound when the format asks for video
                                                          only), honouring --dateafter / --datebefore

Like yt-dlp, it prints `--print pre_process:...` for every video before the date check, and
says nothing about a video it leaves out for its date unless it is told not to be quiet
(--print makes yt-dlp quiet; --no-quiet brings back "upload date is not in range").

Every call is appended to FAKE_LOG (one JSON list of arguments per line), so a test can check the
formats asked for. FAKE_FAIL lists video ids whose download fails.
"""
import json
import os
import re
import subprocess
import sys
import time


def main(argv):
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps(argv) + "\n")
    with open(os.environ["FAKE_CHANNEL"]) as f:
        channel = json.load(f)
    url = argv[-1]
    if "--flat-playlist" in argv:
        m = re.search(r"/(videos|shorts|streams)$", url)
        tab = m.group(1) if m else "videos"
        entries = [{"_type": "url", "id": v["id"], "title": v["title"], "duration": v.get("duration"),
                    "url": f"https://www.youtube.com/watch?v={v['id']}", **({"live_status": v["live"]} if v.get("live") else {}),
                    **({"upload_date": v["upload_date"]} if v.get("listed_date") else {})}
                   for v in channel["videos"] if v.get("tab", "videos") == tab]
        print(json.dumps({"_type": "playlist", "id": "UCfake", "title": f"Fake channel - {tab}", "entries": entries}))
        return 0
    vid = re.search(r"v=([\w-]{11})", url).group(1)
    video = next(v for v in channel["videos"] if v["id"] == vid)
    if vid in os.environ.get("FAKE_FAIL", "").split(","):
        print(f"ERROR: [youtube] {vid}: Video unavailable", file=sys.stderr)
        return 1
    opt = lambda name: argv[argv.index(name) + 1] if name in argv else None
    prints = dict(argv[i + 1].split(":", 1) for i, a in enumerate(argv[:-1]) if a == "--print")   # when: template
    fill = lambda template, values: re.sub(r"%\((\w+)\)j", lambda m: json.dumps(values[m.group(1)]), template)   # JSON fields
    if "pre_process" in prints:
        print(fill(prints["pre_process"], {"upload_date": video["upload_date"], "title": video["title"]}))
    after, before = opt("--dateafter"), opt("--datebefore")
    if (after and video["upload_date"] < after) or (before and video["upload_date"] > before):
        if not prints or "--no-quiet" in argv:
            iso = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}"
            print(f"[download] {iso(video['upload_date'])} upload date is not in range "
                  f"{iso(after or '00010101')} to {iso(before or '99991231')}")
        return 0
    time.sleep(float(os.environ.get("FAKE_DELAY", "0")))
    out = opt("-o").replace("%(upload_date)s", video["upload_date"]).replace("%(id)s", vid).replace("%(ext)s", "mp4")
    sound = "worstaudio" in opt("-f")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video["clip"], "-c", "copy"] + ([] if sound else ["-an"]) + [out],
                   check=True)
    duration = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
                              capture_output=True, text=True).stdout.strip()
    values = {"filepath": out, "upload_date": video["upload_date"], "duration": float(duration), "title": video["title"]}
    print(fill(prints["after_move"], values))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
