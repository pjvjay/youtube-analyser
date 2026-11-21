"""Standing must give exactly what it gave before the other actions were added.

The golden file (fixtures/standing_golden.json) holds a fingerprint of what the ORIGINAL
standing/detect_standing.py (git commit 7bf602d) made of 240 made-up pose sequences: people
standing, sitting or cut off at the knees, at random, with the default and other section rules
and interview starts. The test runs the current code on the same sequences and compares.

    .venv/bin/python -m unittest tests.test_standing -v
    .venv/bin/python -m tests.test_standing --make-golden    # rebuild the file from commit 7bf602d
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import REPO, ds, mf

GOLDEN = mf.ROOT / "fixtures" / "standing_golden.json"
BASELINE = "7bf602d"
RULES = [(2, 2), (1, 0), (3, 4)]     # (min_frames, max_gap)


def person(rng, kind, x, scale):
    """17 keypoints of someone standing, sitting or cut off at the knees, with noise."""
    kp = np.zeros((17, 3), np.float32)
    sho_y, hip_y = 40, 40 + 30 * scale
    kp[5], kp[6] = (x - 8, sho_y, .9), (x + 8, sho_y, .9)
    kp[11], kp[12] = (x - 6, hip_y, .9), (x + 6, hip_y, .9)
    if kind == "stand":
        kp[13], kp[14] = (x - 6, hip_y + 28 * scale, .9), (x + 6, hip_y + 28 * scale, .9)
        kp[15], kp[16] = (x - 6, hip_y + 55 * scale, .8), (x + 6, hip_y + 55 * scale, .8)
    elif kind == "sit":
        kp[13], kp[14] = (x - 30, hip_y + 3, .9), (x + 30, hip_y + 3, .9)
        kp[15], kp[16] = (x - 30, hip_y + 30, .8), (x + 30, hip_y + 30, .8)
    kp[:, :2] += rng.normal(0, 2.5, (17, 2))
    kp[:, 2] = np.clip(kp[:, 2] + rng.normal(0, .15, 17), 0, 1)
    return kp


def sequence(seed):
    rng = np.random.default_rng(seed)
    t, frame, conf, box, kps = [], [], [], [], []
    for f in range(int(rng.integers(20, 120))):
        t.append(f)
        for _ in range(rng.choice([0, 1, 1, 2, 3])):
            kp = person(rng, rng.choice(["stand", "sit", "crop"], p=[.5, .3, .2]), rng.uniform(30, 220), rng.uniform(.6, 1.4))
            frame.append(f)
            conf.append(rng.uniform(.35, 1))
            kps.append(kp)
            box.append([kp[:, 0].min(), kp[:, 1].min(), np.ptp(kp[:, 0]) + 1, np.ptp(kp[:, 1]) + 1])
    start = float(rng.choice([0, rng.integers(1, 60)]))
    poses = {"t": np.array(t, np.float32), "frame": np.array(frame, np.int32), "conf": np.array(conf, np.float32),
             "box": np.array(box, np.float32).reshape(-1, 4), "kp": np.array(kps, np.float32).reshape(-1, 17, 3),
             "size": np.array([256, 144]), "fps": np.float32(1)}
    return poses, start


def fingerprint(find_sections, poses, start, rules):
    """A hash of everything find_sections returns that ends up in the output."""
    frames, sections, intro = find_sections(poses, "v.mp4", *rules, start)

    def clean(rows):
        out = []
        for r in rows:
            row = {k: v for k, v in r.items() if k not in ("best_people", "boxes", "last_t", "n", "most", "best")}
            if "best_people" in r:
                row["best_people"] = [(p["state"], [round(float(v), 3) for v in p["box"]]) for p in r["best_people"]]
            out.append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
        return out
    blob = json.dumps([clean(frames), clean(sections), clean(intro)], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16], len(sections) + len(intro)


def cases():
    for seed in range(80):
        poses, start = sequence(seed)
        for rules in RULES:
            yield seed, rules, poses, start


def make_golden():
    """Write the golden file from the original code at commit BASELINE."""
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{BASELINE}:standing/detect_standing.py"],
                         capture_output=True, text=True, check=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "detect_standing_original.py"
        path.write_text(src)
        sys.path.insert(0, str(REPO / "standing"))
        spec = importlib.util.spec_from_file_location("detect_standing_original", path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
    golden = {f"{seed}/{rules[0]}/{rules[1]}": fingerprint(original.find_sections, poses, start, rules)
              for seed, rules, poses, start in cases()}
    GOLDEN.write_text(json.dumps({"baseline": BASELINE, "cases": golden}, indent=0, sort_keys=True))
    print(f"{len(golden)} cases, {sum(n for _, n in golden.values())} sections -> {GOLDEN}")


class StandingUnchanged(unittest.TestCase):
    def test_sections_are_exactly_as_before(self):
        golden = json.loads(GOLDEN.read_text())["cases"]
        self.assertEqual(len(golden), 240)
        for seed, rules, poses, start in cases():
            key = f"{seed}/{rules[0]}/{rules[1]}"
            self.assertEqual(list(fingerprint(ds.find_sections, poses, start, rules)), golden[key], key)

    def test_the_golden_file_has_sections_to_compare(self):
        golden = json.loads(GOLDEN.read_text())["cases"]
        self.assertGreater(sum(n for _, n in golden.values()), 1000)

    def test_the_csv_columns_are_as_before(self):
        self.assertEqual(ds.FRAME_FIELDS, ["t", "people", "standing", "sitting", "unknown", "best_drop", "best_knee"])
        self.assertEqual(ds.SEGMENT_FIELDS, ["video", "start_s", "end_s", "start", "end", "duration_s", "standing_frames",
                                             "max_people_standing", "best_drop", "still_diff", "youtube", "thumb"])

    def test_the_command_line_output_is_byte_for_byte_as_before(self):
        """The original detect_standing.py (from git) and the current one, on the same clips into
        fresh folders: every file they write must be the same, and nothing more."""
        clips = [str(mf.build(n)) for n in ("walk_stop_walk", "two_jumpers")]
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{BASELINE}:standing/detect_standing.py"],
                             capture_output=True, text=True)
        if src.returncode:
            self.skipTest("git history not available")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "code").mkdir()
            (tmp / "code" / "detect_standing.py").write_text(src.stdout)
            (tmp / "code" / "intro.py").write_text((REPO / "standing" / "intro.py").read_text())
            (tmp / "code" / "models").symlink_to(REPO / "standing" / "models")
            for code, out in ((tmp / "code" / "detect_standing.py", tmp / "before"),
                              (REPO / "standing" / "detect_standing.py", tmp / "after")):
                r = subprocess.run([sys.executable, str(code), *clips, "--out", str(out), "--keep-intro"],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            before = sorted(p.relative_to(tmp / "before") for p in (tmp / "before").rglob("*") if p.is_file())
            after = sorted(p.relative_to(tmp / "after") for p in (tmp / "after").rglob("*") if p.is_file())
            self.assertEqual(after, before)
            with open(tmp / "after" / "segments.csv", newline="") as f:
                self.assertTrue(list(ds.csv.DictReader(f)), "no sections to compare")
            for rel in before:
                self.assertEqual((tmp / "after" / rel).read_bytes(), (tmp / "before" / rel).read_bytes(), str(rel))


if __name__ == "__main__":
    if "--make-golden" in sys.argv:
        make_golden()
    else:
        unittest.main()
