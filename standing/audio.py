"""When someone is speaking, and which voice it is, from a video's sound.

Used by the talking and silent actions (see actions.py for how it is combined with the faces):

1. The sound is decoded by ffmpeg as 16 kHz mono.
2. models/voice_activity.onnx (Silero VAD, MIT) gives the probability of speech every 32 ms.
3. The sound is only trusted where speech stands well clear of the background: the median
   level of speech over the quietest 5% of the sound, per minute, must be at least SNR_OK_DB.
   Close-miked studio speech measured 24-41 dB (24 for 9 s of continuous talking, 30-41 for
   meeting recordings); the same conversations with crowd babble 10 dB below the voices
   measured 10-15 dB, where the speech detector also called 84% of the pauses speech. Where the
   sound is not trusted, talking is judged from the faces alone, as before.
4. Every 0.4 s, the 1.6 s of sound around it is turned into a voice fingerprint by
   models/voice_embedding.onnx (Resemblyzer's GE2E speaker encoder, Apache-2.0), if most of it is
   speech. The fingerprints are grouped into voices (spectral clustering, number of voices from
   the eigengap), and then voices that are not clearly different are merged: every pair must be
   at most SPEAKERS_MAX_SIM alike and each fingerprint SPEAKERS_MIN_MARGIN closer to its own voice
   than to any other. So two voices that cannot be told apart are reported as one, rather than
   split at random. On 21 made-up two-person conversations (alternating turns of 7 real people
   from the pyannote.audio test recordings), clean and with crowd noise, this put 2.0% of speech
   (median; mean 4.7%) under the wrong voice; on 30-60 s clips of real 2-4 person meetings,
   where turns are short and overlap, 13.9%.
5. Voices are named A, B, C... in order of how long they speak, separately for each video.
6. Each voice's characteristics are kept for describing and finding people (people.py): its
   median pitch (women in the test recordings spoke at 168-250 Hz, men at 98-119 Hz) and its
   mean fingerprint, which finds the same voice in other videos.
"""
import json
import subprocess

import numpy as np

import actions

SR = 16000
HOP = 512                      # one speech probability per 32 ms
VAD_MODEL = actions.HERE / "models" / "voice_activity.onnx"
VOICE_MODEL = actions.HERE / "models" / "voice_embedding.onnx"

# Speech: probability >= VAD_ON starts it, < VAD_OFF ends it (Silero's own defaults); pauses
# shorter than MIN_GAP_S are bridged and bursts shorter than MIN_SPEECH_S dropped.
VAD_ON, VAD_OFF = 0.5, 0.35
MIN_SPEECH_S, MIN_GAP_S = 0.25, 0.3
# Trusted sound (step 3 above): speech level over the 5th-percentile level, per SNR_WIN_S.
SNR_OK_DB = 20
SNR_WIN_S = 60
# Voice fingerprints (step 4): 1.6 s windows (what the model was trained on) every 0.4 s, kept
# when at least 60% of the window is speech, each scaled to -30 dBFS as the model expects.
EMB_WIN_S, EMB_STEP_S, EMB_MIN_SPEECH = 1.6, 0.4, 0.6
CLUSTER_MAX = 1000             # windows used to find the voices (evenly spaced); all are then labelled
MAX_VOICES = 8
SPEAKERS_MAX_SIM = 0.89        # on the test conversations 0.89-0.91 with a margin of 0.10 gave the
SPEAKERS_MIN_MARGIN = 0.10     # fewest errors and no voice split in two
SMOOTH_WINDOWS = 5             # each window's voice is decided over 2 s, as turns last seconds
# A voice needs at least MIN_VOICE_S of fingerprinted speech; less is merged into the nearest
# voice. A few windows of one voice can look like a voice of their own by chance: on 16 s of one
# speaker, windows from two recordings of him were split into 4 voices of 4-7 windows each.
MIN_VOICE_S = 5


def load(path):
    """The sound as 16 kHz mono float32, or None when the video has no sound."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                        "-of", "json", str(path)], capture_output=True, text=True)
    if not json.loads(r.stdout or "{}").get("streams"):
        return None
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True).stdout
    wav = np.frombuffer(raw[:len(raw) // 4 * 4], np.float32)
    return wav if len(wav) >= SR else None


class VoiceModels:
    """The speech detector and the voice fingerprint model, on OpenVINO or onnxruntime."""

    def __init__(self, runtime="auto", threads=0):
        runtime = runtime if runtime != "auto" else "openvino" if actions.ov else "onnxruntime"
        self.vad = actions.OnnxModel(VAD_MODEL, None, runtime, threads)
        self.voice = actions.OnnxModel(VOICE_MODEL, None, runtime, threads)

    def speech_prob(self, wav, chunk_s=60):
        """Probability of speech for every 32 ms of wav."""
        n = len(wav) // HOP
        x = np.concatenate([np.zeros(64, np.float32), wav[:n * HOP]])
        h = c = np.zeros((1, 1, 128), np.float32)
        out, step = [], int(chunk_s * SR / HOP)
        for i in range(0, n, step):
            rows = np.lib.stride_tricks.sliding_window_view(x[i * HOP:(min(n, i + step)) * HOP + 64], 576)[::HOP]
            p, h, c = self.vad({"input": np.ascontiguousarray(rows), "h": h, "c": c})
            out.append(np.ravel(p))
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def fingerprints(self, wav, starts):
        """Voice fingerprints (n, 256) of the EMB_WIN_S windows starting at these sample offsets."""
        out = []
        for i in range(0, len(starts), 64):
            batch = []
            for s in starts[i:i + 64]:
                seg = wav[s:s + int(EMB_WIN_S * SR)]
                p = _power(wav, s)
                batch.append(p * (10 ** (-30 / 10) / (np.mean(seg ** 2) + 1e-12)))
            out.append(np.asarray(self.voice({"power": np.stack(batch)})[0]))
        return np.concatenate(out) if out else np.zeros((0, 256), np.float32)


def _power(wav, start, frames=160):
    """|STFT|^2 of `frames` 25 ms frames every 10 ms from `start`, as librosa's melspectrogram
    computes them for the whole clip (centred, zero-padded): the model's input."""
    idx = start - 200 + 160 * np.arange(frames)[:, None] + np.arange(400)[None, :]
    x = np.where((idx >= 0) & (idx < len(wav)), wav[np.clip(idx, 0, len(wav) - 1)], 0)
    win = np.hanning(401)[:-1].astype(np.float32)
    return (np.abs(np.fft.rfft(x * win, axis=1)) ** 2).astype(np.float32)


def _runs(m):
    """(value, start, end) for each run of equal values in a boolean array."""
    if not len(m):
        return []
    cuts = np.r_[0, np.flatnonzero(np.diff(m.astype(np.int8))) + 1, len(m)]
    return [(bool(m[a]), a, b) for a, b in zip(cuts[:-1], cuts[1:])]


def speech_mask(p):
    """Speech (True) per 32 ms frame from the probabilities, with hysteresis and minimum lengths."""
    m = np.zeros(len(p), bool)
    on = False
    for i, v in enumerate(p):
        on = bool(v >= (VAD_OFF if on else VAD_ON))
        m[i] = on
    for value, a, b in _runs(m):  # bridge short pauses inside speech
        if not value and a > 0 and b < len(m) and (b - a) * HOP / SR < MIN_GAP_S:
            m[a:b] = True
    for value, a, b in _runs(m):  # drop short bursts
        if value and (b - a) * HOP / SR < MIN_SPEECH_S:
            m[a:b] = False
    return m


def snr_db(wav, p):
    """Median level of speech over the 5th-percentile level, per SNR_WIN_S (NaN: too little speech).
    A last piece shorter than half a window is measured with the window before it (3 s alone
    gave 16 dB, untrusted, on clean speech)."""
    n = min(len(wav) // HOP, len(p))
    e = 10 * np.log10(np.mean(wav[:n * HOP].reshape(n, HOP) ** 2, 1) + 1e-10)
    w = int(SNR_WIN_S * SR / HOP)
    out = []
    for i in range(0, n, w):
        end = n if n - (i + w) < w // 2 else i + w
        x, q = e[i:end], p[i:end]
        speech = x[q >= VAD_ON]
        out.append(np.median(speech) - np.percentile(x, 5) if len(speech) > 10 else np.nan)
        if end == n:
            break
    return np.array(out, np.float32)


def _separation(E, lab):
    """(median margin of each fingerprint to its own voice over the nearest other, the largest
    similarity between two voices' means); NaN for a single voice."""
    ks = np.unique(lab)
    if len(ks) < 2:
        return np.nan, np.nan
    C = _means(E, lab, ks)
    S = E @ C.T
    own = np.searchsorted(ks, lab)
    mine = S[np.arange(len(E)), own]
    S[np.arange(len(E)), own] = -np.inf
    CC = C @ C.T
    np.fill_diagonal(CC, -1)
    return float(np.median(mine - S.max(1))), float(CC.max())


def _means(E, lab, ks):
    C = np.stack([E[lab == k].mean(0) for k in ks])
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)


def _spectral(E, p=0.3):
    """Spectral clustering: each fingerprint keeps its 30% most similar neighbours, the number of
    voices is where the eigenvalues jump most (at most MAX_VOICES), then k-means."""
    n = len(E)
    if n < 3:
        return np.zeros(n, int)
    S = np.clip(E @ E.T, 0, 1)
    keep = max(2, int(np.ceil(p * n)))
    idx = np.argsort(-S, axis=1)[:, :keep]
    A = np.zeros_like(S)
    np.put_along_axis(A, idx, np.take_along_axis(S, idx, 1), 1)
    A = (A + A.T) / 2
    d = 1 / np.sqrt(A.sum(1) + 1e-9)
    w, V = np.linalg.eigh(np.eye(n) - d[:, None] * A * d[None, :])
    k = int(np.argmax(np.diff(w[:MAX_VOICES + 1]))) + 1
    if k == 1:
        return np.zeros(n, int)
    X = V[:, :k] / (np.linalg.norm(V[:, :k], axis=1, keepdims=True) + 1e-9)
    return _kmeans(X, k)


def _kmeans(X, k, rounds=30, tries=8):
    rng = np.random.default_rng(0)
    best = None
    for _ in range(tries):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(rounds):
            lab = (X @ C.T).argmax(1)
            C = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        score = (X * C[lab]).sum()
        if best is None or score > best[0]:
            best = (score, lab)
    return best[1]


def _distinct(E, lab, min_windows=3):
    """Merge voices until every pair is clearly different (step 4) and each has min_windows:
    a voice with too few windows joins its nearest, otherwise the two most alike merge."""
    lab = np.unique(lab, return_inverse=True)[1]
    while len(np.unique(lab)) > 1:
        ks = np.unique(lab)
        sizes = np.array([(lab == k).sum() for k in ks])
        margin, most_alike = _separation(E, lab)
        if most_alike <= SPEAKERS_MAX_SIM and margin >= SPEAKERS_MIN_MARGIN and sizes.min() >= min_windows:
            break
        CC = _means(E, lab, ks) @ _means(E, lab, ks).T
        np.fill_diagonal(CC, -1)
        if sizes.min() < min_windows:
            i = int(np.argmin(sizes))
            j = int(np.argmax(CC[i]))
            i, j = j, i  # the small voice j joins i
        else:
            i, j = np.unravel_index(np.argmax(CC), CC.shape)
        lab[lab == ks[j]] = ks[i]
        for _ in range(2):  # fingerprints move to the nearest remaining voice
            ks = np.unique(lab)
            lab = ks[(E @ _means(E, lab, ks).T).argmax(1)]
        lab = np.unique(lab, return_inverse=True)[1]
    return lab


def speech_windows(wav, m):
    """Start (in samples) of every fingerprint window that is mostly speech (m: speech_mask)."""
    win = int(EMB_WIN_S * SR)
    starts = np.arange(0, len(wav) - win + 1, int(EMB_STEP_S * SR))
    a = (starts / HOP).astype(int)
    b = ((starts + win) / HOP).astype(int)
    cum = np.r_[0, np.cumsum(m)]
    return starts[(cum[np.minimum(b, len(m))] - cum[np.minimum(a, len(m))]) >= EMB_MIN_SPEECH * (b - a)]


def voice_print(wav, models):
    """(fingerprint, pitch) of one person's voice from a recording of them alone (its speech
    windows' fingerprints averaged, as analyse() does for each voice of a video), or None with
    under a window of speech."""
    starts = speech_windows(wav, speech_mask(models.speech_prob(wav)))
    if not len(starts):
        return None
    c = models.fingerprints(wav, starts).mean(0)
    return c / (np.linalg.norm(c) + 1e-9), median_pitch(wav, starts)


def analyse(wav, models):
    """Speech and voices of one video's sound. Returns the arrays saved in audio/<video>.npz."""
    p = models.speech_prob(wav)
    m = speech_mask(p)
    win = int(EMB_WIN_S * SR)
    starts = speech_windows(wav, m)
    E = models.fingerprints(wav, starts)
    centres = (starts + win / 2) / SR
    voice = np.zeros(len(E), int)
    margin = most_alike = np.nan
    if len(E) >= 3:
        pick = np.linspace(0, len(E) - 1, min(len(E), CLUSTER_MAX)).round().astype(int)
        min_windows = max(3, int(np.ceil(MIN_VOICE_S / EMB_STEP_S * len(pick) / len(E))))
        sub = _distinct(E[pick], _spectral(E[pick]), min_windows)
        margin, most_alike = _separation(E[pick], sub)
        C = _means(E[pick], sub, np.unique(sub))
        S = E @ C.T
        h = SMOOTH_WINDOWS // 2  # average each voice's similarity over neighbouring windows (2 s)
        pad = np.pad(S, ((h, h), (0, 0)), mode="edge")
        voice = np.stack([pad[i:i + len(S)] for i in range(SMOOTH_WINDOWS)]).mean(0).argmax(1)
        order = np.argsort(-np.bincount(voice, minlength=len(C)))  # A speaks most
        voice = np.argsort(order)[voice]
    k = int(voice.max()) + 1 if len(E) else 0
    centroid = np.zeros((k, E.shape[1] if len(E) else 256), np.float32)
    f0 = np.full(k, np.nan, np.float32)
    for v in range(k):
        c = E[voice == v].mean(0)
        centroid[v] = c / (np.linalg.norm(c) + 1e-9)
        f0[v] = median_pitch(wav, starts[voice == v])
    return {"speech_prob": p.astype(np.float16), "speech": m, "snr_db": snr_db(wav, p),
            "win_t": centres.astype(np.float32), "win_voice": voice.astype(np.int8),
            "voices": np.int32(len(np.unique(voice)) if len(E) else 0),
            "margin": np.float32(margin), "most_alike": np.float32(most_alike),
            "duration": np.float32(len(wav) / SR), "voice_f0": f0, "voice_centroid": centroid}


PITCH_FRAME, PITCH_HOP = 640, 320     # 40 ms frames every 20 ms
PITCH_RANGE = (60, 400)               # Hz: from a deep man's voice to a child's
PITCH_VOICED = 0.5                    # autocorrelation peak that counts as voiced
PITCH_MAX_FRAMES = 2000


def median_pitch(wav, starts):
    """Median pitch (Hz) of the voiced frames inside windows starting at these sample offsets
    (autocorrelation; at most PITCH_MAX_FRAMES frames); NaN if none is voiced."""
    offsets = np.unique(np.concatenate([s + np.arange(0, int(EMB_WIN_S * SR) - PITCH_FRAME, PITCH_HOP)
                                        for s in starts])) if len(starts) else np.zeros(0, int)
    if len(offsets) > PITCH_MAX_FRAMES:
        offsets = offsets[np.linspace(0, len(offsets) - 1, PITCH_MAX_FRAMES).astype(int)]
    offsets = offsets[offsets + PITCH_FRAME <= len(wav)]
    if not len(offsets):
        return np.nan
    x = wav[offsets[:, None] + np.arange(PITCH_FRAME)[None, :]].astype(np.float64)
    x -= x.mean(1, keepdims=True)
    loud = np.sqrt((x ** 2).mean(1)) >= 1e-3
    x = x[loud]
    if not len(x):
        return np.nan
    spec = np.fft.rfft(x, n=2 * PITCH_FRAME)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:, :PITCH_FRAME]
    ac /= ac[:, :1] + 1e-12
    lo, hi = int(SR / PITCH_RANGE[1]), int(SR / PITCH_RANGE[0])
    lag = lo + np.argmax(ac[:, lo:hi], axis=1)
    voiced = ac[np.arange(len(ac)), lag] > PITCH_VOICED
    return float(np.median(SR / lag[voiced])) if voiced.any() else np.nan


def voice_name(k):
    return chr(ord("A") + int(k)) if k >= 0 else ""


def frame_voices(audio):
    """The voice (index, -1 for none) of every 32 ms frame: speech frames take the voice of the
    nearest fingerprint window within 1 s."""
    m, wt = audio["speech"], audio["win_t"]
    out = np.full(len(m), -1, np.int16)
    if not len(wt):
        return out
    t = (np.arange(len(m)) + 0.5) * HOP / SR
    right = np.clip(np.searchsorted(wt, t), 0, len(wt) - 1)
    left = np.clip(right - 1, 0, len(wt) - 1)
    j = np.where(np.abs(wt[left] - t) < np.abs(wt[right] - t), left, right)
    use = m & (np.abs(wt[j] - t) <= 1.0)
    out[use] = audio["win_voice"][j[use]]
    return out


def heard(audio, t0, t1, voices=None):
    """What the sound says about [t0, t1): (trusted, share of it that is speech, voices heard for
    at least 0.2 s). trusted is False without sound or where the background is too loud."""
    if audio is None:
        return False, 0.0, []
    snr = audio["snr_db"]
    w = min(int(t0 // SNR_WIN_S), len(snr) - 1)  # the last window may take in a short remainder
    ok = w < len(snr) and not np.isnan(snr[w]) and snr[w] >= SNR_OK_DB
    a, b = int(t0 * SR / HOP), max(int(t0 * SR / HOP) + 1, int(t1 * SR / HOP))
    m = audio["speech"][a:b]
    if not ok:
        # a window with too little speech to measure the background is silence if the whole
        # minute is: trust that, so quiet stretches count as silent
        quiet = w < len(snr) and np.isnan(snr[w])
        return bool(quiet), float(m.mean()) if len(m) else 0.0, []
    v = (voices if voices is not None else frame_voices(audio))[a:b]
    counts = np.bincount(v[v >= 0], minlength=1)
    return True, float(m.mean()) if len(m) else 0.0, [k for k in np.argsort(-counts) if counts[k] * HOP / SR >= 0.2]


def turns(audio, voices=None, join_s=0.5):
    """Speech turns: [(start, end, voice)], a voice's speech with pauses under join_s joined."""
    v = voices if voices is not None else frame_voices(audio)
    out = []
    i, n = 0, len(v)
    while i < n:
        if v[i] < 0:
            i += 1
            continue
        j = i
        while j < n and v[j] == v[i]:
            j += 1
        s, e = i * HOP / SR, j * HOP / SR
        if out and out[-1][2] == v[i] and s - out[-1][1] <= join_s:
            out[-1][1] = e
        else:
            out.append([s, e, int(v[i])])
        i = j
    return [(round(s, 2), round(e, 2), k) for s, e, k in out if e - s >= MIN_SPEECH_S]
