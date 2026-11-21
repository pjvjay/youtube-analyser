"""A stand-in for yt-dlp, for testing channel.py without YouTube.

The "channel" is a JSON file (FAKE_CHANNEL): {"videos": [{"id", "title", "upload_date", "clip"}]},
newest first, clip being a local video file. It answers the two calls channel.py makes:

    --flat-playlist --dump-single-json <channel url>      the channel's videos as a playlist
    -f <format> -o <template> ... --print after_move:... <video url>
                                                          "downloads" the clip: copies it (without
                                                          its sound when the format asks for video
                                                          only), honouring --dateafter / --datebefore

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
    channel = json.load(open(os.environ["FAKE_CHANNEL"]))
    url = argv[-1]
    if "--flat-playlist" in argv:
        m = re.search(r"/(videos|shorts|streams)$", url)
        tab = m.group(1) if m else "videos"
        entries = [{"_type": "url", "id": v["id"], "title": v["title"], "duration": v.get("duration"),
                    "url": f"https://www.youtube.com/watch?v={v['id']}"}
                   for v in channel["videos"] if v.get("tab", "videos") == tab]
        print(json.dumps({"_type": "playlist", "id": "UCfake", "title": f"Fake channel - {tab}", "entries": entries}))
        return 0
    vid = re.search(r"v=([\w-]{11})", url).group(1)
    video = next(v for v in channel["videos"] if v["id"] == vid)
    if vid in os.environ.get("FAKE_FAIL", "").split(","):
        print(f"ERROR: [youtube] {vid}: Video unavailable", file=sys.stderr)
        return 1
    opt = lambda name: argv[argv.index(name) + 1] if name in argv else None
    after, before = opt("--dateafter"), opt("--datebefore")
    if (after and video["upload_date"] < after) or (before and video["upload_date"] > before):
        print(f"[download] {video['upload_date']} upload date is not in range {after}-{before}")
        return 0
    time.sleep(float(os.environ.get("FAKE_DELAY", "0")))
    out = opt("-o").replace("%(upload_date)s", video["upload_date"]).replace("%(id)s", vid).replace("%(ext)s", "mp4")
    sound = "worstaudio" in opt("-f")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video["clip"], "-c", "copy"] + ([] if sound else ["-an"]) + [out],
                   check=True)
    template = opt("--print").split(":", 1)[1]
    duration = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
                              capture_output=True, text=True).stdout.strip()
    print(template.replace("%(filepath)s", out).replace("%(upload_date)s", video["upload_date"])
          .replace("%(duration)s", duration).replace("%(title)s", video["title"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
