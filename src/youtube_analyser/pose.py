"""Standing, sitting or unknown for one person, from the pose model's body points.

These are the original standing detector's rules, unchanged. Each person is labelled from their
leg geometry, measured in torso lengths (mid-shoulder to mid-hip); see the thresholds below.
"""
import math

import numpy as np

# COCO keypoint indices used by YOLO pose models.
NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHO, R_SHO, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 5, 6, 11, 12, 13, 14, 15, 16
LEGS = ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK))
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]

# Thresholds for one leg, in units of the person's torso length (mid-shoulder to mid-hip).
# A standing thigh hangs below the hip by roughly a torso length; a seated thigh points
# sideways or at the camera, so the knee sits level with the hip or only a little below it.
# Someone seated square-on to the camera is the hard case: their thigh points at the lens,
# so in 2D the knee still lands below the hip. What gives them away is the bent knee
# (78-121 degrees in the seated examples checked, 160-180 standing) and a thigh that looks
# much shorter than the shin. So standing needs an ankle; without one a leg can only vote
# sitting or unknown.
STAND_DROP, SIT_DROP = 0.55, 0.35      # vertical hip-to-knee drop / torso length
STAND_TILT, SIT_TILT = 35, 55          # thigh angle away from vertical, degrees
STAND_KNEE, SIT_KNEE = 150, 125        # hip-knee-ankle angle, degrees (180 = straight leg)
MAX_SHIN_THIGH = 1.35                  # shin / thigh length; higher means a foreshortened thigh
VIS = 0.5                              # keypoint confidence needed for shoulders, hips, knees
ANKLE_VIS = 0.3                        # ankles of walking people at 144p often score 0.3-0.5

# A guest seated square-on to the camera can match every leg rule above (in one test video a
# seated guest did, for 20 sections). The giveaway is the seated person
# next to them: at the same size in frame, someone standing has their shoulders well above the
# sitter's. So a standing vote is overruled when a sitting person in the same frame has a torso
# within SAME_SCALE of theirs and shoulders less than ABOVE_SITTER torso lengths lower.
SAME_SCALE = 0.35                      # torso lengths within 35% of each other
ABOVE_SITTER = 0.35                    # standing shoulders must be this far above a sitter's


def angle(a, b, c):
    """Angle at b between b->a and b->c, in degrees."""
    u, v = a - b, c - b
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def classify(kp, img_h):
    """Return (state, measurements) for one person: standing, sitting or unknown."""
    inside = kp[:, 1] < img_h - 1  # keypoints guessed below the bottom edge are not real
    ok = (kp[:, 2] >= VIS) & inside
    ok_ank = (kp[:, 2] >= ANKLE_VIS) & inside
    pt = kp[:, :2]

    def mid(a, b):
        return (pt[a] + pt[b]) / 2 if ok[a] and ok[b] else pt[a] if ok[a] else pt[b] if ok[b] else None

    sho, hip = mid(L_SHO, R_SHO), mid(L_HIP, R_HIP)
    if sho is None or hip is None:
        return "unknown", {}
    torso = float(np.linalg.norm(hip - sho))
    if torso < 3:
        return "unknown", {}
    votes, why, drops, knees = [], [], [], []
    for h, k, a in LEGS:
        if not (ok[h] and ok[k]):
            continue
        thigh = pt[k] - pt[h]
        drop = thigh[1] / torso
        tilt = math.degrees(math.atan2(abs(thigh[0]), max(thigh[1], 1e-9)))
        knee = angle(pt[h], pt[k], pt[a]) if ok_ank[a] else None
        shin_thigh = (np.linalg.norm(pt[a] - pt[k]) / max(np.linalg.norm(thigh), 1e-9)
                      if ok_ank[a] else None)
        drops.append(drop)
        if knee is not None:
            knees.append(knee)
        if (knee is not None and knee >= STAND_KNEE and drop >= STAND_DROP
                and tilt <= STAND_TILT and shin_thigh <= MAX_SHIN_THIGH):
            votes.append("standing")
            why.append("")
        elif drop < SIT_DROP or tilt > SIT_TILT or (knee is not None and knee < SIT_KNEE):
            votes.append("sitting")
            why.append("thigh" if drop < SIT_DROP or tilt > SIT_TILT else "knee")
        else:
            votes.append("unknown")
            why.append("")
    m = {"drop": max(drops) if drops else None, "knee": max(knees) if knees else None,
         "torso": torso, "sho_y": float(sho[1]), "votes": list(zip(votes, why))}
    if "sitting" in votes:
        return "sitting", m
    if "standing" in votes:
        return "standing", m
    return "unknown", m


def seated(m):
    """Stricter sitting, for --action sitting: no leg looks standing, and a thigh is level or
    sideways, or every leg seen is bent. The standing rules count a single bent knee as sitting,
    which also catches the back leg of someone walking (seen on a test clip)."""
    legs = m.get("votes", [])
    if any(v == "standing" for v, _ in legs):
        return False
    sitting = [w for v, w in legs if v == "sitting"]
    return "thigh" in sitting or len(legs) >= 2 and len(sitting) == len(legs)


def level_with(stander, sitter):
    """True when a sitter of about the same size has shoulders nearly as high as the stander's."""
    a, b = stander["torso"], sitter["torso"]
    if abs(a - b) > SAME_SCALE * max(a, b):
        return False
    return sitter["sho_y"] - stander["sho_y"] < ABOVE_SITTER * a  # image y grows downwards


def is_seated(kp, img_h):
    """A person the stricter sitting rules call seated (see seated())."""
    state, m = classify(kp, img_h)
    return state == "sitting" and seated(m)

