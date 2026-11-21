"""Reading videos with ffmpeg: frames at a given rate, single frames, keyframes, the sound.

Everything that decodes goes through here, so the picture a sample shows is the same whichever
step asks for it (see picture_time).
"""
import functools
import json
import re
import subprocess
from pathlib import Path

import numpy as np

VIDEO_EXT = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi")
# yt-dlp names downloads YYYYMMDD_<11-character video id> (see docs); other names get no links
NAME_RE = re.compile(r"^(\d{8})_([A-Za-z0-9_-]{11})$")
SR = 16000                      # the sound is analysed as 16 kHz mono


def hms(t):
    t = int(round(t))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}"


def youtube_id(path):
    """The YouTube video ID in a yt-dlp file name (YYYYMMDD_<id>.<ext>), or None."""
    m = NAME_RE.match(Path(path).stem)
    return m.group(2) if m else None


def upload_date(path):
    """The upload date in a yt-dlp file name (YYYYMMDD), or None."""
    m = NAME_RE.match(Path(path).stem)
    return m.group(1) if m else None


def youtube_url(path, t=None):
    """The video's address on YouTube (opening at t seconds), or "" when its name has no ID."""
    vid = youtube_id(path)
    if vid is None:
        return ""
    return f"https://www.youtube.com/watch?v={vid}" + (f"&t={int(t)}s" if t is not None else "")


def is_format_part(path):
    """One of yt-dlp's unmerged format files (X.f278.webm next to X.webm), not a video to analyse."""
    return bool(re.search(r"\.f\d+$", Path(path).stem))


def is_video(path):
    """A video file to analyse in a folder: a known extension, and not an unmerged format file."""
    path = Path(path)
    return path.suffix.lower() in VIDEO_EXT and not is_format_part(path)


def picture_time(t, fps):
    """When the picture in a sample labelled t was filmed. ffmpeg's fps filter keeps the last
    frame of each 1/fps slot, and its slots are centred on the labels, so a sample shows about
    half a slot later: 0.48 s at 1 fps and 0.08 s at 5 fps on a 25 fps video, measured on the
    test clips (the half frame of the source is left out here). Sections keep the labels, as
    the standing detector always has; this is for lining up samples taken at different rates
    and the sound."""
    return t + 0.5 / fps


STILL_CODECS = {"png", "mjpeg", "bmp", "gif", "webp", "tiff"}   # cover art and thumbnails, not the video


def video_stream(path):
    """(stream index, width, height) of the video stream to read: the first that is a real video
    (not cover art or a thumbnail stored as a video stream). Width and height are as ffmpeg
    decodes the frames: a phone video stored sideways with a rotation tag is decoded upright, so
    they are swapped. ValueError if there is none."""
    path = Path(path)
    try:
        st = path.stat()
        key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    except OSError:
        key = (str(path), None, None)
    return _video_stream(key)


@functools.lru_cache(maxsize=256)
def _video_stream(key):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                        "stream=index,codec_name,width,height:stream_disposition=attached_pic"
                        ":stream_side_data=rotation:stream_tags=rotate", "-of", "json", key[0]],
                       capture_output=True, text=True)
    streams = json.loads(r.stdout or "{}").get("streams") or [] if r.returncode == 0 else []
    real = [s for s in streams if s.get("width") and not s.get("disposition", {}).get("attached_pic")]
    moving = [s for s in real if s.get("codec_name") not in STILL_CODECS]
    if not (moving or real):
        raise ValueError(f"{Path(key[0]).name}: no readable video stream")
    s = (moving or real)[0]
    rotation = s.get("tags", {}).get("rotate")
    for side in s.get("side_data_list", []):
        rotation = side.get("rotation", rotation)
    w, h = s["width"], s["height"]
    if rotation is not None and int(float(rotation)) % 180:
        w, h = h, w
    return s["index"], w, h


def probe(path):
    """(width, height) of the video's frames as ffmpeg decodes them (see video_stream)."""
    return video_stream(path)[1:]


def _map(path):
    """ffmpeg's options to read the video stream video_stream() chose."""
    return ["-map", f"0:{video_stream(path)[0]}"]


def _arg(path):
    """A file name ffprobe cannot take for an option (a name starting with "-")."""
    return str(Path(path).absolute())


def duration(path):
    """Length in seconds (0 if unknown)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", _arg(path)],
                         capture_output=True, text=True).stdout
    try:
        return float(json.loads(out or "{}").get("format", {}).get("duration", 0))
    except (TypeError, ValueError):
        return 0.0


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "json", _arg(path)], capture_output=True, text=True).stdout
    return bool(json.loads(out or "{}").get("streams"))


def read_frames(path, fps, w, h):
    """Yield (seconds, BGR frame) at `fps` frames per second, decoded by ffmpeg."""
    yield from _decode(["ffmpeg", "-v", "error", "-i", _arg(path), *_map(path), "-vf", f"fps={fps}",
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], fps, w, h)


def read_windows(path, windows, fps, w, h):
    """Yield (window number, seconds, BGR frame) at `fps` inside each (t0, t1) window."""
    for n, (t0, t1) in enumerate(windows):
        for t, img in _decode(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", _arg(path), *_map(path), "-t",
                               f"{t1 - t0:.3f}", "-an", "-vf", f"fps={fps}", "-f", "rawvideo",
                               "-pix_fmt", "bgr24", "-"], fps, w, h):
            yield n, round(t0 + t, 3), img


KEYFRAME_EVERY_S = 4            # keyframes(): at most one every 4 s


def keyframes(path, w, h, every=KEYFRAME_EVERY_S):
    """Yield (index, BGR) for the video's keyframes, at most one every `every` seconds. Only the
    keyframes are read and decoded (YouTube's come every few seconds): on a 27-minute test video
    1.8 s, against 10 s for decoding every frame as read_frames() must (and 19 s with
    -skip_frame, which still hands every frame to the decoder)."""
    yield from _decode(["ffmpeg", "-v", "error", "-discard", "nokey", "-i", _arg(path), *_map(path), "-an", "-vf",
                        f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{every})'",
                        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], 1.0, w, h)


def _decode(cmd, fps, w, h):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size, k = w * h * 3, 0
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield k / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def grab_frame(path, t, fps, w, h):
    """Return the frame read_frames() produced at time t (black if there is none).

    Seeking straight to t can land one source frame away from the frame the fps filter
    picked, which at a camera cut is a different shot. So seek a whole number of sample
    intervals earlier, where the fps filter's grid lines up with the original one, and
    run the same filter forward to t.
    """
    k = min(round(t * fps), 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t - k / fps:.6f}", "-i", _arg(path), *_map(path),
                          "-vf", f"fps={fps}", "-frames:v", str(k + 1),
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    size = w * h * 3
    if len(raw) < size:
        return np.zeros((h, w, 3), np.uint8)
    return np.frombuffer(raw[-size:] if len(raw) % size == 0 else raw[:size], np.uint8).reshape(h, w, 3)


def read_range(path, t0, t1, fps, w, h):
    """Grey frames (uint8) that read_frames() produces between t0 and t1 (inclusive), by time."""
    k = min(round(t0 * fps), 2)  # see grab_frame for why it starts k samples early
    n = round((t1 - t0) * fps) + 1
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0 - k / fps:.6f}", "-i", _arg(path), *_map(path),
                          "-vf", f"fps={fps}", "-frames:v", str(k + n), "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True).stdout
    frames = np.frombuffer(raw[:len(raw) // (w * h) * w * h], np.uint8).reshape(-1, h, w)[k:]
    return {round(t0 + i / fps, 3): f for i, f in enumerate(frames)}


def load_audio(path):
    """The sound as 16 kHz mono float32, or None when the video has no sound (or under 1 s)."""
    if not has_audio(path):
        return None
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", _arg(path), "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True).stdout
    wav = np.frombuffer(raw[:len(raw) // 4 * 4], np.float32)
    return wav if len(wav) >= SR else None
