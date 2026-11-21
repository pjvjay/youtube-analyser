"""Listing a YouTube channel's videos and downloading them, with yt-dlp.

yt-dlp is used as a program (on the PATH, or the yt_dlp package run with this Python), so it can
be updated on its own: YouTube changes often and only a recent yt-dlp keeps up.
"""
import datetime as dt
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from . import media

# The lowest resolution that is still 144p (YouTube's smallest; storyboards are smaller and are
# not videos), with the smallest sound when the sound is needed (talking, silent, voices, gender,
# names). This is the format the original 248 videos were downloaded in.
FORMAT_SOUND = "worstvideo[height>=144]+worstaudio/worst[height>=144]/worst"
FORMAT_SILENT = "worstvideo[height>=144]/worst[height>=144]/worst"
CHANNEL_RE = re.compile(r"^(https?://(?:www\.|m\.)?youtube\.com/(?:@[^/?#]+|channel/[^/?#]+|c/[^/?#]+|user/[^/?#]+))"
                        r"/?(?:featured)?/?$")
# what each download prints, as JSON values (titles can hold tabs and newlines)
PRINTED = "after_move:%(filepath)j\t%(upload_date)j\t%(duration)j\t%(title)j"


OUTSIDE_DATES = "outside the dates asked for"


class YouTubeError(RuntimeError):
    pass


def channel_tabs(url, shorts=False):
    """A channel's address turned into its uploads: /videos (and /shorts); other addresses
    (a playlist, a channel tab, a single video) are used as they are."""
    m = CHANNEL_RE.match(url)
    if not m:
        return [url]
    return [m.group(1) + "/videos"] + ([m.group(1) + "/shorts"] if shorts else [])


def _entries(info):
    """The videos in yt-dlp's JSON for a playlist, a channel (whose entries can be its tabs) or a video."""
    if info.get("_type") in ("playlist", "multi_video"):
        for e in info.get("entries") or []:
            if e:
                yield from _entries(e) if e.get("_type") in ("playlist", "multi_video") else [e]
    else:
        yield info


def entry_date(entry):
    """An entry's upload date (YYYYMMDD) when the listing gives one, else None."""
    if entry.get("upload_date"):
        return str(entry["upload_date"])
    if entry.get("timestamp"):
        return dt.datetime.fromtimestamp(float(entry["timestamp"]), dt.UTC).strftime("%Y%m%d")
    return None


def parse_date(text):
    """A --since / --until date as YYYYMMDD (from YYYYMMDD or YYYY-MM-DD); ValueError otherwise."""
    if text is None or text == "":
        return None
    digits = str(text).strip().replace("-", "")
    try:
        dt.datetime.strptime(digits, "%Y%m%d")
    except ValueError:
        raise ValueError(f"{text!r} is not a date: use YYYYMMDD or YYYY-MM-DD") from None
    return digits


def entry_url(entry):
    """The page to download a listed video from: a playlist entry's address, or a single video's
    page (not its "url", which for a full video description is the address of a media file)."""
    if entry.get("webpage_url"):
        return entry["webpage_url"]
    if entry.get("_type") == "url" and entry.get("url"):
        return entry["url"]
    return f"https://www.youtube.com/watch?v={entry['id']}"


def in_dates(date, since=None, until=None):
    """Whether a date (YYYYMMDD, or YYYY-MM-DD) is within [since, until]; an unknown date is let
    through."""
    if not date:
        return True
    date = str(date).replace("-", "")
    return (not since or date >= since) and (not until or date <= until)


def find_download(folder, vid):
    """A finished download of this video in folder (yt-dlp's YYYYMMDD_<id>.<ext>), or None."""
    for p in sorted(Path(folder).glob(f"*_{_glob_escape(vid)}.*")):
        if media.is_video(p) and media.NAME_RE.match(p.stem):
            return p
    return None


def _glob_escape(s):
    return re.sub(r"([*?\[])", r"[\1]", s)


class YtDlp:
    """yt-dlp with the user's options: command (default: yt-dlp on the PATH, else python -m
    yt_dlp), cookies (a cookies.txt), cookies_from_browser, extra (more options)."""

    def __init__(self, command=None, cookies=None, cookies_from_browser=None, extra=()):
        if command:
            self.base = shlex.split(command) if isinstance(command, str) else list(command)
        elif shutil.which("yt-dlp"):
            self.base = [shutil.which("yt-dlp")]
        elif importlib.util.find_spec("yt_dlp") is not None:
            self.base = [sys.executable, "-m", "yt_dlp"]
        else:
            raise YouTubeError('yt-dlp is not installed: pip install -U yt-dlp (or "youtube-analyser[youtube]")')
        self.options = []
        if cookies:
            self.options += ["--cookies", str(cookies)]
        if cookies_from_browser:
            self.options += ["--cookies-from-browser", cookies_from_browser]
        for a in extra or ():
            self.options += shlex.split(a)

    def _run(self, args):
        return subprocess.run(self.base + self.options + args, capture_output=True, text=True)

    def list(self, source, shorts=False):
        """[{id, title, duration, date, url}] for a channel (newest first, as YouTube lists them),
        a playlist or a video. Live and upcoming streams are left out (they cannot be downloaded
        whole yet)."""
        out, seen = [], set()
        for tab in channel_tabs(source, shorts):
            r = self._run(["--flat-playlist", "--dump-single-json", "--no-warnings", tab])
            if r.returncode:
                raise YouTubeError(f"yt-dlp could not list {tab}:\n{r.stderr.strip()[-2000:]}")
            for e in _entries(json.loads(r.stdout)):
                if not e.get("id") or e["id"] in seen or e.get("live_status") in ("is_live", "is_upcoming"):
                    continue
                seen.add(e["id"])
                out.append({"id": e["id"], "title": e.get("title") or "", "duration": e.get("duration"),
                            "date": entry_date(e), "url": entry_url(e)})
        return out

    def download(self, entry, folder, sound, since=None, until=None, had_sound=False):
        """Download one video at the lowest resolution into folder, named YYYYMMDD_<id>.<ext>.
        A copy already there is used, unless the sound is needed now and the copy was downloaded
        without it (had_sound: it was asked for with its sound; a video with no sound at all is
        not downloaded again and again). Returns (path or None, info, note)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        have = find_download(folder, entry["id"])
        if have is not None and sound and not had_sound and not media.has_audio(have):
            have.unlink()
            have = None
        if have is not None:   # downloaded before, perhaps for a search with other dates
            if not in_dates(media.upload_date(have), since, until):
                return None, {}, OUTSIDE_DATES
            return have, {}, "already downloaded"
        args = ["-f", FORMAT_SOUND if sound else FORMAT_SILENT, "-o", str(folder / "%(upload_date)s_%(id)s.%(ext)s"),
                "--no-playlist", "--no-progress", "--no-warnings", "--concurrent-fragments", "4",
                "--no-simulate", "--print", PRINTED]
        if since:
            args += ["--dateafter", since]
        if until:
            args += ["--datebefore", until]
        r = self._run(args + [entry["url"]])
        for line in reversed(r.stdout.splitlines()):
            parts = line.split("\t")
            if len(parts) == 4:
                try:
                    path, date, duration, title = (None if x == "NA" else json.loads(x) for x in parts)
                except ValueError:
                    continue
                if not path:
                    continue
                return Path(path), {"upload_date": date or "", "title": title or "",
                                    "duration": float(duration) if isinstance(duration, (int, float)) else None}, ""
        have = find_download(folder, entry["id"])
        if have is not None:
            return (have, {}, "") if in_dates(media.upload_date(have), since, until) else (None, {}, OUTSIDE_DATES)
        said = r.stdout + r.stderr
        if "not in range" in said:
            return None, {}, OUTSIDE_DATES
        why = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ["no file"])[-1]
        return None, {}, f"download failed: {why[:300]}"
