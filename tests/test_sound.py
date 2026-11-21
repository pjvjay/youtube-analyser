"""The sound on its own: speech, how clear it is, and how many voices, on real speech.

The clips (see make_fixtures.VOICE_CLIPS) are real recordings of four people (from the
pyannote.audio test set) in alternating turns of 2-6 s: one voice, two, three, three under
crowd babble 5 dB below the voices, silence (room tone), and two voices with a third who says
only 3 s.

    .venv/bin/python -m unittest tests.test_sound -v      (about a minute)
"""
import itertools
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import audio, mf

MODELS = None


def analysed(name):
    global MODELS
    MODELS = MODELS or audio.VoiceModels()
    wav = audio.load(mf.build(name))
    return wav, audio.analyse(wav, MODELS)


def right_voice(a, script):
    """Share of speech (away from turn changes) under the right voice, best naming of the voices."""
    v = audio.frame_voices(a)
    t = (np.arange(len(v)) + 0.5) * audio.HOP / audio.SR
    names = sorted({w["v"] for _, _, w in script})
    truth = np.full(len(v), -1)
    for t0, t1, w in script:
        truth[(t >= t0 + 0.3) & (t < t1 - 0.3)] = names.index(w["v"])
    ok = (v >= 0) & (truth >= 0)
    k = max(int(a["voices"]), len(names))
    return max(np.mean(np.array(p)[truth[ok]] == v[ok]) for p in itertools.permutations(range(k), len(names)))


class Voices(unittest.TestCase):
    def check(self, name, voices, right):
        script, seconds, _ = mf.VOICE_CLIPS[name]
        wav, a = analysed(name)
        self.assertEqual(int(a["voices"]), voices, name)
        self.assertGreaterEqual(right_voice(a, script), right, name)
        self.assertTrue((a["snr_db"] >= audio.SNR_OK_DB).all(), f"{name}: {a['snr_db']}")
        return a

    def test_one_voice(self):
        self.check("voices_one", 1, 0.99)

    def test_two_voices(self):
        self.check("voices_two", 2, 0.95)

    def test_three_voices(self):
        a = self.check("voices_three", 3, 0.95)
        self.assertLessEqual(float(a["most_alike"]), audio.SPEAKERS_MAX_SIM)

    def test_a_brief_third_voice_joins_another(self):
        # 3 s is under MIN_VOICE_S of fingerprints: reported with the nearest voice, not on its own
        self.check("voices_brief_third", 2, 0.90)

    def test_speech_is_found_and_silence_is_not(self):
        script, seconds, _ = mf.VOICE_CLIPS["voices_three"]
        _, a = analysed("voices_three")
        t = (np.arange(len(a["speech"])) + 0.5) * audio.HOP / audio.SR
        inside = np.zeros(len(t), bool)
        for t0, t1, _ in script:
            inside |= (t >= t0 + 0.2) & (t < t1 - 0.2)
        self.assertGreater(a["speech"][inside].mean(), 0.95)
        _, quiet = analysed("voices_silence")
        self.assertEqual(int(quiet["voices"]), 0)
        self.assertLess(quiet["speech"].mean(), 0.01)
        self.assertEqual(audio.heard(quiet, 10, 11), (True, 0.0, []))   # a quiet minute is trusted silence

    def test_turns_follow_the_script(self):
        script, _, _ = mf.VOICE_CLIPS["voices_two"]
        _, a = analysed("voices_two")
        turns = audio.turns(a)
        changes = [t1 for (_, t1, w), (_, _, w2) in zip(script, script[1:]) if w["v"] != w2["v"]]
        found = [t0 for (t0, _, k), (_, _, k0) in zip(turns[1:], turns) if k != k0]
        for c in changes:
            self.assertTrue(any(abs(f - c) <= 1.0 for f in found), f"turn change at {c:.1f}s not found in {found}")


class Noise(unittest.TestCase):
    def test_crowd_babble_is_not_trusted(self):
        _, a = analysed("voices_three_noisy")
        self.assertTrue((a["snr_db"] < audio.SNR_OK_DB).all(), a["snr_db"])
        self.assertFalse(audio.heard(a, 10, 11)[0])


class NoSound(unittest.TestCase):
    def test_a_video_without_sound(self):
        self.assertIsNone(audio.load(mf.build("three_talkers_no_sound")))

    def test_a_sound_too_short_to_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "short.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=64x36:d=1", "-f", "lavfi",
                            "-i", "anullsrc=r=16000:cl=mono", "-t", "0.5", "-shortest", str(clip)], check=True)
            self.assertIsNone(audio.load(clip))


if __name__ == "__main__":
    unittest.main()
