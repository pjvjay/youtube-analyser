"""Fast tests of the decision rules, on made-up measurements (no models, no video): seconds to run.

Each test builds what the models would have measured (people's body points, the camera's
movement, mouth movement, what the sound heard, voice fingerprints) and checks the rule's
verdict, including the edge cases the rules were written for.

    .venv/bin/python -m unittest tests.test_rules -v
"""
import itertools
import unittest

import numpy as np
import cv2

from tests.helpers import actions, audio, mf

SOURCES = mf.SOURCES

W, H = 256, 144
TORSO = 20.0


# ---- made-up people ------------------------------------------------------------------------

def body(x, hip_y=70.0, torso=TORSO, foot_lift=0.0, seen=0.9):
    """17 keypoints of an upright person whose mid-hip is at (x, hip_y); foot_lift raises the feet
    (torso lengths). Returns (keypoints, box x y w h)."""
    kp = np.zeros((17, 3), np.float32)
    sho_y = hip_y - torso
    kp[0] = (x, sho_y - 0.45 * torso, seen)                                   # nose
    kp[1], kp[2] = (x + 0.12 * torso, sho_y - 0.55 * torso, seen), (x - 0.12 * torso, sho_y - 0.55 * torso, seen)
    kp[5], kp[6] = (x + 0.35 * torso, sho_y, seen), (x - 0.35 * torso, sho_y, seen)
    kp[11], kp[12] = (x + 0.25 * torso, hip_y, seen), (x - 0.25 * torso, hip_y, seen)
    knee, ankle = hip_y + 0.9 * torso, hip_y + 1.8 * torso
    kp[13], kp[14] = (x + 0.25 * torso, knee, seen), (x - 0.25 * torso, knee, seen)
    kp[15], kp[16] = (x + 0.25 * torso, ankle, seen), (x - 0.25 * torso, ankle, seen)
    kp[[13, 14, 15, 16], 1] -= foot_lift * torso
    top, bottom = sho_y - 0.9 * torso, ankle - foot_lift * torso + 0.2 * torso
    box = np.array([x - 0.6 * torso, top, 1.2 * torso, bottom - top], np.float32)
    return kp, box


def poses_of(frames, fps=1.0):
    """frames: per sample, a list of (keypoints, box). -> a poses dict like infer() returns."""
    t, frame, kp, box = [], [], [], []
    for f, people in enumerate(frames):
        t.append(f / fps)
        for k, b in people:
            frame.append(f)
            kp.append(k)
            box.append(b)
    return {"t": np.array(t, np.float64), "frame": np.array(frame, np.int32),
            "conf": np.full(len(frame), 0.9, np.float32), "kp": np.array(kp, np.float32).reshape(-1, 17, 3),
            "box": np.array(box, np.float32).reshape(-1, 4), "size": np.array([W, H]), "fps": np.float32(fps)}


def camera_of(poses, shift=None, scale=None, response=None, looks=None):
    """A camera_motion() result: still camera unless given per-sample shift / scale / response,
    and per-detection looks (default: every person in the same clothes)."""
    n = len(poses["t"])
    sh = np.zeros((n, 2), np.float32) if shift is None else np.array(shift, np.float32).reshape(n, 2)
    sc = np.ones(n, np.float32) if scale is None else np.array(scale, np.float32)
    rs = np.ones(n, np.float32) if response is None else np.array(response, np.float32)
    sh[0], rs[0] = np.nan, 0.0
    lk = np.full((len(poses["frame"]), 24), 100.0, np.float32) if looks is None else np.array(looks, np.float32)
    return {"shift": sh, "scale": sc, "response": rs, "look": lk}


def dense_of(frames, fps=5.0, **camera):
    d = poses_of(frames, fps)
    d["window"] = np.zeros(len(d["t"]), np.int32)
    d.update(camera_of(d, **camera))
    return d


def zoomed(p, s, cx=W / 2, cy=H / 2):
    return s * (p - np.array([cx, cy])) + np.array([cx, cy])


# ---- walking between 1-fps samples ------------------------------------------------------------

class Walking(unittest.TestCase):
    def speeds(self, frames, **camera):
        poses = poses_of(frames)
        return actions.label_motion(poses, camera_of(poses, **camera)), poses

    def test_standing_still_is_still(self):
        speed, _ = self.speeds([[body(120 + 0.5 * (f % 2))] for f in range(6)])  # half-pixel jitter
        for s in speed[1:-1]:
            self.assertEqual(actions.motion_state(s), "still")

    def test_walking_is_walking_at_the_right_speed(self):
        speed, _ = self.speeds([[body(40 + 1.5 * TORSO * f)] for f in range(5)])
        for s in speed[1:-1]:
            self.assertEqual(actions.motion_state(s), "walking")
            self.assertAlmostEqual(s, 1.5, delta=0.05)

    def test_brisk_walk_is_walking_not_running(self):
        speed, _ = self.speeds([[body(20 + 3.0 * TORSO * f)] for f in range(3)])
        self.assertEqual(actions.motion_state(speed[1]), "walking")

    def test_a_runner_between_samples_is_not_called_walking(self):
        # 6 torso lengths a second is too far to follow between 1-fps samples: unknown, not walking
        speed, _ = self.speeds([[body(20 + 6 * TORSO * f), body(200)] for f in range(2)] + [[body(200)]])
        self.assertTrue(np.isnan(speed[0]))

    def test_setting_off_is_not_walking(self):
        # still, then running: the sample in between saw one still second and one too far to follow
        xs = [30, 30, 30 + 0.5 * 6 * TORSO, 30 + 1.5 * 6 * TORSO]
        speed, poses = self.speeds([[body(x)] for x in xs])
        self.assertNotEqual(actions.motion_state(speed[2]), "walking")

    def test_camera_panning_past_someone_standing(self):
        # the camera pans right 40 px a second: the person slides left in the picture
        frames = [[body(200 - 40 * f)] for f in range(4)]
        speed, _ = self.speeds(frames, shift=[(-40, 0)] * 4)
        for s in speed[1:-1]:
            self.assertEqual(actions.motion_state(s), "still")

    def test_camera_following_a_walker(self):
        # he stays put in the picture while the picture moves 1.5 torso lengths a second
        speed, _ = self.speeds([[body(120)] for f in range(4)], shift=[(-1.5 * TORSO, 0)] * 4)
        for s in speed[1:-1]:
            self.assertEqual(actions.motion_state(s), "walking")
            self.assertAlmostEqual(s, 1.5, delta=0.05)

    def test_zooming_in_on_someone_standing(self):
        frames, x, y, torso = [], 150.0, 70.0, TORSO
        for f in range(4):
            frames.append([body(x, y, torso)])
            x, y = zoomed(np.array([x, y]), 1.1)
            torso *= 1.1
        speed, _ = self.speeds(frames, scale=[1.1] * 4)
        for s in speed[1:-1]:
            self.assertEqual(actions.motion_state(s), "still")

    def test_walking_towards_the_camera(self):
        # 1.5 torso lengths a second from 10 torso lengths away: grows as distance shrinks
        focal = actions.FOCAL_H * H
        frames = [[body(128, 70, focal / (10.0 - 1.5 * f))] for f in range(3)]
        speed, _ = self.speeds(frames)
        self.assertEqual(actions.motion_state(speed[1]), "walking")

    def test_size_jitter_is_not_walking_towards_the_camera(self):
        speed, _ = self.speeds([[body(128, 70, TORSO * (1.06 if f % 2 else 1.0))] for f in range(5)])
        for s in speed[1:-1]:
            self.assertEqual(actions.motion_state(s), "still")

    def test_nobody_is_followed_across_a_cut(self):
        speed, _ = self.speeds([[body(60)], [body(60 + 1.5 * TORSO)], [body(60 + 3 * TORSO)]], response=[0, 0.0, 0.0])
        self.assertTrue(np.isnan(speed).all())

    def test_crossing_people_are_not_swapped(self):
        # two people walk towards each other; by look (clothes) they stay themselves
        frames = [[body(60 + 1.5 * TORSO * f), body(200 - 1.5 * TORSO * f)] for f in range(4)]
        poses = poses_of(frames)
        looks = [[100.0] * 24 if i % 2 == 0 else [160.0] * 24 for i in range(len(poses["frame"]))]
        speed = actions.label_motion(poses, camera_of(poses, looks=looks))
        for j in range(2, 6):
            self.assertAlmostEqual(speed[j], 1.5, delta=0.1)

    def test_people_at_the_edge_are_not_measured(self):
        frames = [[body(10 + 0.1 * f)] for f in range(3)]  # box runs off the left edge
        speed, _ = self.speeds(frames)
        self.assertTrue(np.isnan(speed).all())

    def test_people_side_by_side_are_measured(self):
        # their boxes overlap by a third: still two people, each measured
        frames = [[body(100 + 1.5 * TORSO * f), body(100 + 0.8 * TORSO + 1.5 * TORSO * f)] for f in range(3)]
        poses = poses_of(frames)
        looks = [[100.0] * 24 if i % 2 == 0 else [160.0] * 24 for i in range(len(poses["frame"]))]
        speed = actions.label_motion(poses, camera_of(poses, looks=looks))
        self.assertEqual(actions.motion_state(speed[2]), "walking")
        self.assertEqual(actions.motion_state(speed[3]), "walking")

    def test_someone_hidden_behind_another_is_not_measured(self):
        frames = [[body(100), body(105)] for f in range(3)]
        speed, _ = self.speeds(frames)
        self.assertTrue(np.isnan(speed).all())

    def test_speed_from_one_side_counts_when_the_other_side_is_empty(self):
        # he walks for two samples, then the shot ends (nobody left unmatched): the last link counts
        speed, _ = self.speeds([[body(60 + 1.5 * TORSO * f)] for f in range(2)])
        self.assertEqual(actions.motion_state(speed[1]), "walking")


# ---- running and jumping in the 5-fps samples -------------------------------------------------

class Running(unittest.TestCase):
    def states(self, xs, seated=None, **camera):
        people = actions.label_runs(dense_of([[body(x)] for x in xs], **camera), seated=seated)
        return [p[0]["state"] for p in people], [p[0]["m"].get("speed", np.nan) for p in people]

    def test_running(self):
        states, speeds = self.states([10 + 6 * TORSO * 0.2 * f for f in range(8)])
        self.assertEqual(states[2:-2], ["running"] * 4)
        self.assertAlmostEqual(float(np.nanmedian(speeds)), 6.0, delta=0.3)

    def test_walking_and_standing_are_not_running(self):
        states, _ = self.states([40 + 2 * TORSO * 0.2 * f for f in range(10)])
        self.assertEqual(set(states[2:-2]), {"walking"})
        states, _ = self.states([120.0] * 10)
        self.assertEqual(set(states[2:-2]), {"still"})

    def test_the_running_threshold(self):
        for speed, expect in ((3.6, "walking"), (4.4, "running")):
            states, _ = self.states([10 + speed * TORSO * 0.2 * f for f in range(10)])
            self.assertEqual(set(states[2:-2]), {expect}, speed)

    def test_camera_following_a_runner(self):
        states, _ = self.states([120.0] * 10, shift=[(-6 * TORSO * 0.2, 0)] * 10)
        self.assertEqual(set(states[2:-2]), {"running"})

    def test_a_seated_rider_is_not_running(self):
        states, _ = self.states([10 + 6 * TORSO * 0.2 * f for f in range(8)], seated=lambda kp: True)
        self.assertNotIn("running", states)

    def test_a_measurement_glitch_does_not_make_a_runner(self):
        xs = [120.0] * 10
        xs[5] = 120 + 2 * TORSO  # one frame where the points jump
        states, _ = self.states(xs)
        self.assertNotIn("running", states)


class Jumping(unittest.TestCase):
    @staticmethod
    def hop(t, t0, height, air=0.45):
        u = (t - t0) / air
        return 4 * height * u * (1 - u) if 0 <= u <= 1 else 0.0

    def jumps(self, rises, xs=None, **camera):
        """rises[f]: body lift in torso lengths per 5-fps sample. -> states per sample."""
        xs = xs or [128.0] * len(rises)
        frames = [[body(x, 70 - r * TORSO)] for x, r in zip(xs, rises)]
        return [p[0]["state"] for p in actions.label_jumps(dense_of(frames, **camera))]

    def test_a_jump_is_found_once(self):
        rises = [self.hop(f * 0.2, 1.0, 0.45) for f in range(15)]
        states = self.jumps(rises)
        self.assertIn("jumping", states)
        self.assertTrue(all(s == "jumping" for s, r in zip(states, rises) if r >= 0.3))
        self.assertTrue(all(s != "jumping" for s, r in zip(states, rises) if r == 0))

    def test_a_tiny_hop_is_not_a_jump(self):
        self.assertNotIn("jumping", self.jumps([self.hop(f * 0.2, 1.0, 0.12) for f in range(15)]))

    def test_stepping_up_is_not_a_jump(self):
        self.assertNotIn("jumping", self.jumps([0.45 * min(1, max(0, (f * 0.2 - 1) / 0.4)) for f in range(15)]))

    def test_the_camera_bobbing_is_not_a_jump(self):
        rises = [self.hop(f * 0.2, 1.0, 0.45) for f in range(15)]
        # the whole picture moves up and down with him: the camera did it
        shift = [(0, -(rises[f] - rises[f - 1]) * TORSO) if f else (0, 0) for f in range(15)]
        self.assertNotIn("jumping", self.jumps(rises, shift=shift))

    def test_jumping_while_walking(self):
        rises = [self.hop(f * 0.2, 1.0, 0.45) for f in range(15)]
        xs = [40 + 1.5 * TORSO * 0.2 * f for f in range(15)]
        self.assertIn("jumping", self.jumps(rises, xs=xs))

    def test_two_people_jumping_at_once(self):
        rises = [self.hop(f * 0.2, 1.0, 0.45) for f in range(15)]
        frames = [[body(70, 70 - r * TORSO), body(190, 70 - r * TORSO)] for r in rises]
        looks = [[100.0] * 24 if i % 2 == 0 else [160.0] * 24 for i in range(2 * len(rises))]
        people = actions.label_jumps(dense_of(frames, looks=looks))
        for who in (0, 1):
            self.assertIn("jumping", [p[who]["state"] for p in people])


# ---- frame_motion: the camera -----------------------------------------------------------------

class Camera(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        street = cv2.imread(str(SOURCES / "street.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        cls.scene = cv2.resize(street, (W * 3, int(street.shape[0] * W * 3 / street.shape[1])), interpolation=cv2.INTER_AREA)
        cls.hann = cv2.createHanningWindow((W, H), cv2.CV_32F)

    def view(self, dx=0.0, scale=1.0, person_at=None):
        """The camera looking at the street; dx moves the picture's content right, scale zooms in."""
        cx, cy = self.scene.shape[1] / 2, self.scene.shape[0] / 2
        m = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        m[:, 2] += (W / 2 - cx + dx, H / 2 - cy)
        img = cv2.warpAffine(self.scene, m, (W, H))
        if person_at is not None:  # a figure that stays put in the picture (the camera follows it)
            img[30:130, person_at:person_at + 40] = 240
            img[60:70, person_at + 10:person_at + 30] = 20
        return img

    def test_pan_and_zoom_are_measured(self):
        for dx, s in ((0, 1.0), (12, 1.0), (-30, 1.0), (0, 1.08), (8, 1.05)):
            got = actions.frame_motion(self.view(), self.view(dx, s), self.hann)
            self.assertAlmostEqual(got[3], s, delta=0.01, msg=(dx, s))
            self.assertAlmostEqual(got[0], dx, delta=1.0, msg=(dx, s))
            self.assertGreater(got[2], actions.CUT_RESPONSE)

    def test_a_followed_person_does_not_hide_the_pan(self):
        got = actions.frame_motion(self.view(person_at=100), self.view(30, person_at=100), self.hann,
                                   [(100, 30, 40, 100)])
        self.assertAlmostEqual(got[0], 30, delta=1.0)

    def test_a_cut_has_a_low_response(self):
        other = cv2.GaussianBlur(np.random.default_rng(1).uniform(0, 255, (H, W)).astype(np.float32), (0, 0), 2)
        self.assertLess(actions.frame_motion(self.view(), other, self.hann)[2], actions.CUT_RESPONSE)

    def test_a_step_out_of_line_is_smoothed(self):
        steps = np.array([[-32, 0, 0.5, 1], [-32, 0, 0.5, 1], [-1, 0, 0.5, 1], [-33, 0, 0.5, 1], [-32, 0, 0.5, 1]], float)
        self.assertAlmostEqual(actions._smooth_steps(steps)[2, 0], -32, delta=1)

    def test_a_cut_is_not_smoothed_away(self):
        steps = np.array([[0, 0, 0.8, 1], [np.nan, np.nan, 0.0, 1], [40, 0, 0.8, 1]], float)
        self.assertTrue(np.isnan(actions._smooth_steps(steps)[1, 0]))


# ---- talking and silent -------------------------------------------------------------------

def face(move, track=0):
    return {"state": "unknown", "m": {"mouth_move": move}, "track": track}


class Talkers(unittest.TestCase):
    def test_without_trusted_sound_the_mouth_decides(self):
        people = [face(0.2), face(0.05)]
        actions._talkers(people, (False, 0.0, []))
        self.assertEqual([p["state"] for p in people], ["talking", "silent"])
        self.assertEqual(people[0]["m"]["by"], "face")

    def test_no_speech_means_silent_whatever_the_mouth_does(self):
        people = [face(0.3), face(0.2)]
        actions._talkers(people, (True, 0.05, []))
        self.assertEqual([p["state"] for p in people], ["silent", "silent"])

    def test_one_voice_three_faces(self):
        people = [face(0.09), face(0.03), face(0.02)]
        actions._talkers(people, (True, 0.9, [0]))
        self.assertEqual([p["state"] for p in people], ["talking", "silent", "silent"])

    def test_the_one_moving_most_talks(self):
        people = [face(0.09), face(0.11), face(0.02)]
        actions._talkers(people, (True, 0.9, [0]))
        self.assertEqual([p["state"] for p in people], ["silent", "talking", "silent"])

    def test_two_voices_two_talkers(self):
        people = [face(0.09), face(0.10), face(0.02)]
        actions._talkers(people, (True, 0.9, [0, 1]))
        self.assertEqual([p["state"] for p in people], ["talking", "talking", "silent"])

    def test_two_people_talking_over_each_other_with_one_voice_heard(self):
        people = [face(0.25), face(0.20), face(0.02)]
        actions._talkers(people, (True, 0.9, [0]))
        self.assertEqual([p["state"] for p in people], ["talking", "talking", "silent"])

    def test_a_listener_nodding_is_not_talking(self):
        people = [face(0.25), face(0.06)]
        actions._talkers(people, (True, 0.9, [0]))
        self.assertEqual(people[1]["state"], "silent")

    def test_each_face_keeps_its_own_voice(self):
        # A talks alone with voice 0, then B alone with voice 1, then both at once while the sound
        # only makes out voice 0: each keeps its own
        frames, sounds = [], []
        for _ in range(3):
            frames.append([face(0.2, 0), face(0.01, 1)])
            sounds.append((True, 0.9, [0]))
        for _ in range(3):
            frames.append([face(0.01, 0), face(0.2, 1)])
            sounds.append((True, 0.9, [1]))
        frames.append([face(0.2, 0), face(0.2, 1)])
        sounds.append((True, 0.9, [0]))
        for people, sound in zip(frames, sounds):
            actions._talkers(people, sound)
        actions._name_voices(frames, sounds)
        self.assertEqual(frames[0][0]["m"]["voice"], "A")
        self.assertEqual(frames[3][1]["m"]["voice"], "B")
        self.assertEqual([p["m"]["voice"] for p in frames[-1]], ["A", "B"])
        self.assertEqual(frames[0][1]["m"]["heard"], "A")     # the listener hears the talker
        self.assertEqual(frames[3][0]["m"]["heard"], "B")

    def test_face_tracks_break_at_a_cut(self):
        poses = poses_of([[body(120)], [body(120)], [body(120)]])
        judged = [[{"j": 0, "state": "talking", "m": {}}], [{"j": 1, "state": "talking", "m": {}}],
                  [{"j": 2, "state": "talking", "m": {}}]]
        cam = camera_of(poses, response=[0, 0.9, 0.0])
        actions._face_tracks(judged, poses, cam)
        self.assertEqual(judged[0][0]["track"], judged[1][0]["track"])
        self.assertNotEqual(judged[1][0]["track"], judged[2][0]["track"])


# ---- the sound -----------------------------------------------------------------------------

class Sound(unittest.TestCase):
    def test_speech_mask_bridges_pauses_and_drops_clicks(self):
        frame = audio.HOP / audio.SR
        p = np.zeros(200)
        p[10:60] = 0.9                 # speech
        p[60:65] = 0.1                 # a 0.16 s pause inside it: bridged
        p[65:100] = 0.9
        p[150:153] = 0.9               # a 0.1 s click: dropped
        m = audio.speech_mask(p)
        self.assertTrue(m[10:100].all())
        self.assertFalse(m[150:153].any())
        self.assertLess(5 * frame, audio.MIN_GAP_S)

    def test_speech_mask_hysteresis(self):
        p = np.r_[np.zeros(10), 0.6 * np.ones(20), 0.4 * np.ones(20), 0.2 * np.ones(20)]
        m = audio.speech_mask(p)
        self.assertTrue(m[10:50].all())   # 0.4 keeps speech going once it has started
        self.assertFalse(m[50:].any())

    def fake_voices(self, sizes, spread=0.2, seed=0):
        rng = np.random.default_rng(seed)
        centres = rng.normal(0, 1, (len(sizes), 256))
        centres /= np.linalg.norm(centres, axis=1, keepdims=True)
        E, lab = [], []
        for k, n in enumerate(sizes):
            x = centres[k] + spread * rng.normal(0, 1, (n, 256)) / np.sqrt(256)
            E.append(x / np.linalg.norm(x, axis=1, keepdims=True))
            lab += [k] * n
        return np.concatenate(E), np.array(lab)

    def agreement(self, got, truth):
        """Share of fingerprints with the right voice, under the best naming of the voices found."""
        k = max(got.max(), truth.max()) + 1
        return max(np.mean(np.array(perm)[truth] == got) for perm in itertools.permutations(range(k)))

    def test_one_two_and_three_voices_are_found(self):
        for sizes in ([60], [60, 40], [50, 40, 30]):
            E, truth = self.fake_voices(sizes)
            got = audio._distinct(E, audio._spectral(E), 3)
            self.assertEqual(len(set(got)), len(sizes), sizes)
            self.assertGreater(self.agreement(got, truth), 0.95)

    def test_voices_that_cannot_be_told_apart_are_one(self):
        rng = np.random.default_rng(3)
        base = rng.normal(0, 1, 256)
        E = np.stack([base + 0.3 * rng.normal(0, 1, 256) for _ in range(80)])
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        forced = np.r_[np.zeros(40, int), np.ones(40, int)]     # split in two at random
        self.assertEqual(len(set(audio._distinct(E, forced, 3))), 1)

    def test_a_voice_with_too_little_speech_joins_the_nearest(self):
        E, truth = self.fake_voices([60, 50, 4])
        got = audio._distinct(E, truth.copy(), 12)
        self.assertEqual(len(set(got)), 2)

    def sound(self, speech, snr, win_t=(), win_voice=()):
        return {"speech": np.array(speech, bool), "snr_db": np.array(snr, np.float32),
                "win_t": np.array(win_t, np.float32), "win_voice": np.array(win_voice, np.int8)}

    def test_heard_trusts_only_clear_sound(self):
        n = int(60 * audio.SR / audio.HOP)
        self.assertFalse(audio.heard(self.sound(np.ones(n), [8.0]), 10, 11)[0])    # a crowd
        self.assertTrue(audio.heard(self.sound(np.ones(n), [30.0], [10.5], [0]), 10, 11)[0])
        ok, share, voices = audio.heard(self.sound(np.zeros(n), [np.nan]), 10, 11)
        self.assertTrue(ok)                                                      # a quiet minute
        self.assertEqual((share, voices), (0.0, []))
        self.assertFalse(audio.heard(None, 0, 1)[0])                              # no sound track

    def test_heard_lists_each_voice_once(self):
        n = int(60 * audio.SR / audio.HOP)
        s = self.sound(np.ones(n), [30.0], [10.2, 10.6], [0, 1])
        self.assertEqual(sorted(audio.heard(s, 10, 11)[2]), [0, 1])

    def test_turns_join_a_voice_across_short_pauses(self):
        v = np.full(100, -1)
        v[0:20], v[25:40], v[50:70] = 0, 0, 1          # a 0.16 s pause inside voice 0's turn
        self.assertEqual([k for _, _, k in audio.turns(None, v)], [0, 1])

    def test_a_short_last_piece_of_sound_joins_the_minute_before(self):
        rng = np.random.default_rng(0)
        wav = (0.001 * rng.normal(0, 1, int(63 * audio.SR))).astype(np.float32)
        p = np.zeros(len(wav) // audio.HOP)
        for s in range(0, 63, 2):  # speech bursts every other second
            a = int(s * audio.SR)
            wav[a:a + audio.SR] += 0.05 * np.sin(np.arange(audio.SR) * 0.3).astype(np.float32)
            p[a // audio.HOP:(a + audio.SR) // audio.HOP] = 0.9
        self.assertEqual(len(audio.snr_db(wav, p)), 1)


if __name__ == "__main__":
    unittest.main()
