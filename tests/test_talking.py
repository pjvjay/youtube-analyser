"""Up to three people talking, end to end: faces and sound together (and each alone).

three_talkers: three faces side by side (left, middle, right), with real speech from three
people laid under them (see make_fixtures.TALK):
    0-12 s   left talks (voice a)         12-24  middle talks (b)      24-36  right talks (c)
    36-42    nobody speaks                42-47  left and right talk at once (a + c)
    47-53    a fourth voice, off screen   53-59  middle's mouth moves, no speech (chewing)
    59-66    middle talks (b)             66-72  left talks (a)
three_talkers_noisy: the same with crowd babble 5 dB under the voices (the sound is not trusted).
three_talkers_no_sound: the same without a sound track.
cutaways: close-ups of two different people cutting between them every 6 s, each talking in
his own shots with his own voice.

Each sampled second is judged by its picture time and burst (see media.picture_time), away
from the changes in the script (the burst reaches 0.8 s past the sample).

    pytest tests/test_talking.py      (first run about 3 minutes)
"""
import collections
import unittest

from tests import helpers as h
from tests.helpers import media, mf, motion

SCRIPT = mf.TALK["three_talkers"]
CLIPS = ["three_talkers", "three_talkers_noisy", "three_talkers_no_sound", "cutaways"]


def setUpModule():
    h.run("talking", CLIPS)
    h.run("silent", CLIPS)


def seats(name, action="talking"):
    """[(picture time, who the script says talks {seat: voice}, {seat: person})] for steady seconds."""
    timeline, by_frame, _, _ = h.people(name, action)
    out = []
    for t, people in zip(timeline["t"].tolist(), by_frame):
        w = media.picture_time(t, float(timeline["fps"]))
        phase = [(a, b, who) for a, b, who in SCRIPT if a <= w < b]
        if not phase or not (phase[0][0] + 0.3 <= w and w + 1.1 <= phase[0][1]):
            continue
        found = {}
        for p in people:
            if "face_box" not in p:
                continue
            centre = float(p["box"][0] + p["box"][2] / 2)
            seat = min(mf.SEATS, key=lambda k: abs(centre - (44 + 85 * mf.SEATS[k])))
            found[seat] = p
        out.append((w, phase[0][2], found))
    return out


def talking_seats(who):
    return {k for k, v in who.items() if k in mf.SEATS and v is not None}


class ThreeTalkersWithSound(unittest.TestCase):
    def test_all_three_faces_are_judged(self):
        rows = seats("three_talkers")
        for seat in mf.SEATS:
            judged = sum(1 for _, _, found in rows if seat in found and found[seat]["state"] in ("talking", "silent"))
            self.assertGreaterEqual(judged, 0.9 * len(rows), seat)

    def test_nobody_talks_who_is_not_talking(self):
        for w, who, found in seats("three_talkers"):
            for seat, p in found.items():
                if seat not in talking_seats(who):
                    self.assertNotEqual(p["state"], "talking", f"{seat} at {w:.1f}s ({who}), mouth {p['m'].get('mouth_move')}")

    def test_talkers_are_found_talking(self):
        hits = collections.Counter()
        for w, who, found in seats("three_talkers"):
            for seat in talking_seats(who):
                hits[seat, "all"] += 1
                hits[seat, "found"] += found.get(seat, {}).get("state") == "talking"
        for seat in mf.SEATS:
            # observed 83% overall: the source clip's speaker pauses between words
            self.assertGreaterEqual(hits[seat, "found"], 0.75 * hits[seat, "all"], f"{seat}: {hits}")

    def test_no_speech_means_everyone_is_silent(self):
        for w, who, found in seats("three_talkers"):
            if not who or "offscreen" in who or who == {"middle": None}:
                for seat, p in found.items():
                    self.assertEqual(p["state"], "silent", f"{seat} at {w:.1f}s ({who})")
                    if not who or who == {"middle": None}:
                        self.assertEqual(p["m"]["by"], "sound", f"{seat} at {w:.1f}s")

    def test_each_person_keeps_one_voice_and_they_differ(self):
        voices = collections.defaultdict(collections.Counter)
        for w, who, found in seats("three_talkers"):
            for seat in talking_seats(who):
                p = found.get(seat)
                if p is not None and p["state"] == "talking":
                    voices[seat][p["m"].get("voice", "")] += 1
        own = {}
        for seat in mf.SEATS:
            label, n = voices[seat].most_common(1)[0]
            self.assertGreaterEqual(n, 0.9 * sum(voices[seat].values()), f"{seat}: {voices[seat]}")
            self.assertEqual(len(label), 1, f"{seat}: {label}")
            own[seat] = label
        self.assertEqual(len(set(own.values())), 3, own)

    def test_two_at_once_each_with_their_own_voice(self):
        both = 0
        own = {}
        for w, who, found in seats("three_talkers"):
            if len(talking_seats(who)) == 1:
                seat = next(iter(talking_seats(who)))
                if found.get(seat, {}).get("state") == "talking":
                    own.setdefault(seat, collections.Counter())[found[seat]["m"]["voice"]] += 1
        own = {k: c.most_common(1)[0][0] for k, c in own.items()}
        for w, who, found in seats("three_talkers"):
            if talking_seats(who) == {"left", "right"}:
                talkers = {k: p["m"]["voice"] for k, p in found.items() if p["state"] == "talking"}
                self.assertNotIn("middle", talkers, f"at {w:.1f}s")
                for seat, voice in talkers.items():
                    self.assertEqual(voice, own[seat], f"{seat} at {w:.1f}s")
                both += len(talkers) == 2
        self.assertGreaterEqual(both, 1)

    def test_listeners_hear_the_talker(self):
        right = total = 0
        for w, who, found in seats("three_talkers"):
            if len(talking_seats(who)) != 1:
                continue
            talker = found.get(next(iter(talking_seats(who))))
            if talker is None or talker["state"] != "talking":
                continue
            for seat, p in found.items():
                if p["state"] == "silent":
                    total += 1
                    right += talker["m"]["voice"] in p["m"].get("heard", "")
        self.assertGreaterEqual(right, 0.9 * total)

    def test_speech_timeline(self):
        turns = h.csv_rows(h.out_dir("talking") / "speech" / "three_talkers.csv")
        self.assertTrue(turns)
        voices = {r["voice"] for r in turns}
        self.assertGreaterEqual(len(voices), 3)
        for r in turns:
            self.assertEqual(r["sound_trusted"], "yes")
            mid = (float(r["start_s"]) + float(r["end_s"])) / 2
            if 48 <= mid <= 52:
                self.assertEqual(r["on_screen"], "no", r)   # the fourth voice, off screen
            self.assertFalse(54 <= float(r["start_s"]) and float(r["end_s"]) <= 58.5, f"speech while chewing: {r}")
        on = [r for r in turns if r["on_screen"] == "yes"]
        self.assertGreater(sum(float(r["duration_s"]) for r in on), 40)

    def test_talking_sections(self):
        sections = h.sections("three_talkers", "talking")
        for a, b, who in SCRIPT:
            covered = [s for s in sections if float(s["start_s"]) < b - 1 and float(s["end_s"]) > a + 1]
            if talking_seats(who):
                self.assertTrue(covered, f"no section for {a}-{b}")
            else:
                self.assertFalse([s for s in sections if float(s["start_s"]) >= a + 1 and float(s["end_s"]) <= b - 0.5],
                                 f"a section inside {a}-{b}")
        overlap = [s for s in sections if float(s["start_s"]) <= 44 <= float(s["end_s"])]
        self.assertEqual(int(overlap[0]["max_people_talking"]), 2)

    def test_silent_sections(self):
        sections = h.sections("three_talkers", "silent")
        for a, b in ((36, 42), (47, 53), (53, 59)):
            self.assertTrue([s for s in sections if float(s["start_s"]) < b - 1 and float(s["end_s"]) > a + 1], f"{a}-{b}")


class WithoutTrustedSound(unittest.TestCase):
    def test_noisy_sound_is_not_trusted(self):
        _, _, _, extra = h.people("three_talkers_noisy", "talking")
        self.assertTrue((extra["audio"]["snr_db"] < h.audio.SNR_OK_DB).all())
        turns = h.csv_rows(h.out_dir("talking") / "speech" / "three_talkers_noisy.csv")
        self.assertTrue(all(r["sound_trusted"] == "no" for r in turns))

    def test_faces_alone_decide(self):
        for name in ("three_talkers_noisy", "three_talkers_no_sound"):
            hits = collections.Counter()
            for w, who, found in seats(name):
                moving = set(who) & set(mf.SEATS)        # with the sound off, chewing looks like talking
                for seat, p in found.items():
                    if p["state"] in ("talking", "silent"):
                        self.assertEqual(p["m"]["by"], "face")
                        self.assertNotIn("voice", p["m"])
                    if seat not in moving:
                        self.assertNotEqual(p["state"], "talking", f"{name} {seat} at {w:.1f}s")
                    else:
                        hits["all"] += 1
                        hits["found"] += p["state"] == "talking"
            # observed 70%: the face alone misses the speaker's quieter seconds (mouth moving under 0.12)
            self.assertGreaterEqual(hits["found"], 0.6 * hits["all"], f"{name}: {hits}")

    def test_no_sound_track(self):
        _, _, _, extra = h.people("three_talkers_no_sound", "talking")
        self.assertIsNone(extra["audio"])
        self.assertEqual(h.csv_rows(h.out_dir("talking") / "speech" / "three_talkers_no_sound.csv"), [])


class Cutaways(unittest.TestCase):
    def test_each_person_keeps_his_voice_across_cuts(self):
        timeline, by_frame, _, extra = h.people("cutaways", "talking")
        voices = collections.defaultdict(collections.Counter)
        for t, people in zip(timeline["t"].tolist(), by_frame):
            w = media.picture_time(t, 1.0)
            shot = [(a, b, who) for a, b, who in mf.CUTAWAYS if a + 0.3 <= w and w + 1.1 <= b]
            for p in people:
                if shot and p["state"] == "talking":
                    voices[next(iter(shot[0][2]))][p["m"]["voice"]] += 1
        x, y = voices["X"].most_common(1)[0], voices["Y"].most_common(1)[0]
        self.assertNotEqual(x[0], y[0])
        self.assertEqual(x[1], sum(voices["X"].values()), voices)
        self.assertEqual(y[1], sum(voices["Y"].values()), voices)

    def test_cuts_are_seen(self):
        timeline, by_frame, _, extra = h.people("cutaways", "talking")
        cuts = [t for t, r in zip(timeline["t"].tolist(), extra["camera"]["response"]) if t > 0 and r < motion.CUT_RESPONSE]
        self.assertEqual(len(cuts), 5, cuts)


if __name__ == "__main__":
    unittest.main()
