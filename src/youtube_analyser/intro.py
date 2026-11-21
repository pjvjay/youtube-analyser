"""Find where the interview starts in an episode, so standing in the intro can be left out.

Episodes open with some mix of a cold-open teaser, the show's title sequence, a sponsor ad and
a logo card, and the interview starts after the last of these. The title sequences, ads and
logo cards are the same in every episode of a season, so they are found by comparison:

1. Every episode's first 4 minutes are reduced to one tiny grey thumbnail per second.
2. A thumbnail that matches a short moment (at most 15 s long) in 3 or more other episodes is
   title material. Requiring a short moment leaves out interview footage that a re-run or a
   compilation episode repeats, because a static interview shot matches minutes of footage.
   These thumbnails are saved as a "bank" (intro_bank.npz), so a new video can be checked
   against it on its own. The bank is made from the videos, so it is built locally and not
   committed.
3. The interview starts after the last run of title frames that begins in the first 150 s
   (runs up to 15 s apart are joined, which bridges a sponsor read between title and ad).
   Some intros go on to show episode-specific clips inside a fixed frame (a TV set, say), so
   the intro is extended while the edge of the picture stays the same as in the title frames.
4. Episodes whose title is unique to them (a whole season can be like that) have no title
   frames. There the interview starts at the first 10 s in which at least 7 frames show a
   camera angle that is seen again later in the episode. Compilations without such angles
   fall back to frames shared with any one other episode in the first minute.

Checked by eye on 210 test episodes (frames either side of the detected start).

From the command line: `yta intro --build-bank *.mp4` builds the bank, `yta intro video.mp4`
prints where each interview starts.
"""
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .media import hms

DEFAULT_BANK = Path("intro_bank.npz")    # in the folder the commands are run from

W, H = 32, 18           # thumbnail size; enough to recognise a title card, too small to show detail
OPENING_S = 240         # the interview started within the first 4 minutes in every episode
BODY_FROM_S = 270       # "seen again later" compares against keyframes from here on
BODY_STEP_S = 2         # at most one body keyframe every 2 s
MATCH_CORR = 0.95       # thumbnails match: correlation of their grey patterns ...
MATCH_RMS = 6.0         # ... or raw grey levels within 6 (catches pale, low-contrast title cards)
BLANK_STD = 8           # below this contrast the correlation is meaningless (fades, plain cards)
DARK_MEAN = 20          # black frames match each other by grey level, so they are not used for it
TIGHT_S = 15            # a match must be a short moment in the other episode, not a static shot
SHARED_BY = 3           # title material is shared with at least this many other episodes
TITLE_BY_S = 150        # the title run must begin this early
TITLE_GAP_S = 15        # title frames this close together belong to one intro
BORDER_RMS = 12         # picture edge counts as unchanged within this many grey levels
BORDER_MAX_S = 30       # and the intro is extended by at most this much
ANGLE_CORR = 0.75       # an opening frame "is seen again later" at this correlation
ANGLE_WIN, ANGLE_NEED = 10, 7
PAIR_WITHIN_S, PAIR_BY_S, PAIR_GAP_S = 60, 30, 3


def thumbs(path, vf, before=(), after=()):
    """Tiny grey thumbnails from ffmpeg, as a (n, H, W) uint8 array."""
    cmd = ["ffmpeg", "-v", "error", *before, "-i", str(path), "-an", "-vf",
           f"{vf},scale={W}:{H}:flags=area,format=gray", *after, "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw[:len(raw) // (W * H) * W * H], np.uint8).reshape(-1, H, W)


def opening(path):
    """One thumbnail per second for the first OPENING_S seconds; index = second."""
    return thumbs(path, "fps=1", before=["-t", str(OPENING_S)])


def body(path):
    """Keyframe thumbnails after BODY_FROM_S (decoding keyframes only is fast)."""
    return thumbs(path, f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{BODY_STEP_S})'",
                  before=["-skip_frame", "nokey", "-ss", str(BODY_FROM_S)], after=["-fps_mode", "passthrough"])


def _flat(a):
    return a.reshape(len(a), int(np.prod(a.shape[1:]))).astype(np.float32)


def _z(a):
    """Rows scaled so that a dot product is their correlation."""
    f = _flat(a)
    return (f - f.mean(1, keepdims=True)) / np.maximum(f.std(1), 1)[:, None] / np.sqrt(f.shape[1])


def _rms2(fa, fb):
    """Squared RMS grey-level difference between every row of fa and every row of fb."""
    return ((fa ** 2).sum(1)[:, None] + (fb ** 2).sum(1)[None, :] - 2 * fa @ fb.T) / fa.shape[1]


def _match(a, b):
    """Boolean (len(a), len(b)): which thumbnails of a match which of b."""
    fa, fb = _flat(a), _flat(b)
    contrast = (fa.std(1) >= BLANK_STD)[:, None] & (fb.std(1) >= BLANK_STD)[None, :]
    by_corr = (_z(a) @ _z(b).T >= MATCH_CORR) & contrast
    lit = (fa.mean(1) >= DARK_MEAN)[:, None] & (fb.mean(1) >= DARK_MEAN)[None, :]
    by_level = (_rms2(fa, fb) < MATCH_RMS ** 2) & lit
    return by_corr | by_level


def _border(a):
    f = a.astype(np.float32)
    return np.concatenate([f[:, :3].reshape(len(f), -1), f[:, -3:].reshape(len(f), -1),
                           f[:, 3:-3, :4].reshape(len(f), -1), f[:, 3:-3, -4:].reshape(len(f), -1)], 1)


def build_bank(openings):
    """Thumbnails shared with other episodes: {name: opening array} -> bank dict.

    count = how many other episodes show the thumbnail as a short moment (<= TIGHT_S).
    Kept: count >= SHARED_BY anywhere in the opening, or count >= 1 in the first minute.
    Videos under OPENING_S are left out: Shorts are cut from the interviews, so several Shorts
    of one episode would make an interview moment look shared.
    """
    empty = {"thumbs": np.zeros((0, H, W), np.uint8), "count": np.zeros(0, np.int32), "second": np.zeros(0, np.int32)}
    names = sorted(n for n in openings if len(openings[n]) >= OPENING_S)
    if not names:
        return empty
    allt = np.concatenate([openings[n] for n in names])
    vid = np.concatenate([[k] * len(openings[n]) for k, n in enumerate(names)])
    sec = np.concatenate([np.arange(len(openings[n])) for n in names])
    offs = np.cumsum([0] + [len(openings[n]) for n in names])
    count = np.zeros(len(allt), np.int32)
    for i in range(0, len(allt), 1024):
        m = _match(allt[i:i + 1024], allt)
        for v in range(len(names)):
            mv = m[:, offs[v]:offs[v + 1]]
            first = mv.argmax(1)
            last = mv.shape[1] - 1 - mv[:, ::-1].argmax(1)
            count[i:i + 1024] += mv.any(1) & (last - first <= TIGHT_S) & (vid[i:i + 1024] != v)
    keep = (count >= SHARED_BY) | ((count >= 1) & (sec < PAIR_WITHIN_S))
    t, c, s = allt[keep], count[keep], sec[keep]
    if not len(t):
        return empty
    # drop near-duplicates (a static card shown for several seconds), keeping the widest-shared
    order = np.argsort(-c, kind="stable")
    t, c, s = t[order], c[order], s[order]
    ft = _flat(t)
    near = []
    for i in range(0, len(ft), 1024):
        near += [np.where(row)[0] for row in _rms2(ft[i:i + 1024], ft) < 3 ** 2]
    taken = np.zeros(len(t), bool)
    out = []
    for i in range(len(t)):
        if taken[i]:
            continue
        same = near[i][~taken[near[i]]]
        taken[same] = True
        taken[i] = True
        out.append((t[i], c[i], min(s[same].min(initial=s[i]), s[i])))
    if not out:
        return empty
    return {"thumbs": np.stack([o[0] for o in out]), "count": np.array([o[1] for o in out], np.int32),
            "second": np.array([o[2] for o in out], np.int32)}


def title_frames(bank):
    """How many of the bank's thumbnails are title material (shared by SHARED_BY episodes)."""
    return 0 if bank is None else int((bank["count"] >= SHARED_BY).sum())


def load_bank(path=DEFAULT_BANK):
    return dict(np.load(path)) if Path(path).exists() else None


def _chain(idx, begin_by, gap):
    """End (exclusive) of the run of indices starting at idx[0], joining gaps up to `gap`."""
    if not len(idx) or idx[0] > begin_by:
        return 0
    end = idx[0]
    for i in idx[1:]:
        if i - end > gap:
            break
        end = i
    return int(end) + 1


def interview_start(path, bank, first=None):
    """Second at which the interview starts, the end of the title material, and how it was found."""
    if bank is None:
        return {"interview_start_s": 0, "title_end_s": 0, "how": "none"}
    a = opening(path) if first is None else first
    if len(a) < OPENING_S:  # under 4 minutes: a clip or a Short, which has no intro
        return {"interview_start_s": 0, "title_end_s": 0, "how": "short video"}
    m = _match(a, bank["thumbs"])
    title = (m & (bank["count"] >= SHARED_BY)[None, :]).any(1)
    t_end = _chain(np.where(title)[0], TITLE_BY_S, TITLE_GAP_S)
    if t_end:
        edge, ref = _border(a), _border(a[max(0, t_end - 3):t_end])
        start = t_end
        while start < min(t_end + BORDER_MAX_S, len(a)) and \
                np.sqrt(((ref - edge[start]) ** 2).mean(1)).min() <= BORDER_RMS:
            start += 1
        return {"interview_start_s": start, "title_end_s": t_end, "how": "title"}
    later = body(path)
    if len(later):
        seen = (_z(a) @ _z(later).T).max(1) >= ANGLE_CORR
        seen &= _flat(a).std(1) >= BLANK_STD
        for j in range(min(180, len(a) - ANGLE_WIN)):
            if seen[j] and seen[j:j + ANGLE_WIN].sum() >= ANGLE_NEED:
                return {"interview_start_s": j, "title_end_s": 0, "how": "angles"}
    pair = (m & (bank["second"] < PAIR_WITHIN_S)[None, :]).any(1)[:PAIR_WITHIN_S]
    start = _chain(np.where(pair)[0], PAIR_BY_S, PAIR_GAP_S)
    return {"interview_start_s": start, "title_end_s": 0, "how": "shared" if start else "none"}


def build_bank_from(videos, threads=4):
    """The bank from these videos' openings (read a few at a time)."""
    with ThreadPoolExecutor(threads) as ex:
        openings = dict(zip([Path(v).name for v in videos], ex.map(opening, videos)))
    return build_bank(openings), sum(len(o) >= OPENING_S for o in openings.values())


def save_bank(path, bank):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bank)


def start_row(path, bank):
    """interview_start() as a row of interview_starts.csv; the whole video is the interview
    without a bank."""
    r = interview_start(path, bank) if bank is not None else \
        {"interview_start_s": 0, "title_end_s": 0, "how": "intro kept"}
    return {"video": Path(path).name, **r, "interview_start": hms(r["interview_start_s"])}
