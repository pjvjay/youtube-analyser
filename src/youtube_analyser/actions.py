"""The actions (--action), and who is doing one in each sampled frame.

    standing, sitting   leg geometry (pose.py); --motion still (or moving) also uses how far
                        people travel between samples (sitting: still by default)
    walking             upright people whose body travels 0.6-4 torso lengths a second between
                        the 1-fps samples, with the camera's pan, tilt and zoom taken out (motion.py)
    running             4 torso lengths a second or more, in 5-fps samples taken again around
                        every moment a whole body is in view (motion.py)
    jumping             the lowest foot and the hips rise and come back down, in the same
                        5-fps samples (motion.py)
    smiling             each face, found from the pose model's eye and nose points, is measured
                        by a small expression model over a short burst of frames (faces.py)
    talking, silent     the faces' mouth movement together with the sound, where it is clear:
                        whether anyone speaks and which voice (audio.py); each face keeps its
                        own voice
"""
from . import audio, faces, motion
from .media import picture_time
from .pose import classify, is_seated, level_with, seated

# What each --action looks for. states: the labels counted in frames/<video>.csv, the action's
# own label first. stats: measurements reported per frame as best_<name> (largest in the frame).
# score: the measurement whose largest value picks a section's thumbnail, reported as
# best_<score>. min_frames / max_gap: the default section rules, in sampled frames.
# still: drop sections whose figures never move (statues, posters); jumping and talking are
# movement already.
ACTIONS = {
    "standing": {"states": ("standing", "sitting", "unknown"), "stats": ("drop", "knee"), "score": "drop",
                 "min_frames": 2, "max_gap": 2, "still": True},
    "sitting": {"states": ("sitting", "standing", "unknown"), "stats": ("drop", "knee"), "score": "conf",
                "min_frames": 2, "max_gap": 2, "still": True},
    # in JUMP_FPS samples: one airborne frame is a jump; jumps up to 1 s apart are one section
    "jumping": {"states": ("jumping", "on the ground", "unknown"), "stats": ("rise",), "score": "rise",
                "min_frames": 1, "max_gap": 4, "still": False},
    "smiling": {"states": ("smiling", "not smiling", "unknown"), "stats": ("happy", "eyes"), "score": "happy",
                "min_frames": 2, "max_gap": 2, "still": True},
    "talking": {"states": ("talking", "silent", "unknown"), "stats": ("mouth_move", "eyes"), "score": "mouth_move",
                "min_frames": 2, "max_gap": 2, "still": False, "notes": ("voice", "by")},
    # a visible face that is not talking: listening, or nobody speaking
    "silent": {"states": ("silent", "talking", "unknown"), "stats": ("mouth_move", "eyes"), "score": "eyes",
               "min_frames": 2, "max_gap": 1, "still": True, "notes": ("heard", "by")},  # talking ends it
    # moving across the ground upright, from how far the body travels between 1-fps samples
    "walking": {"states": ("walking", "running", "still", "unknown"), "stats": ("speed",), "score": "speed",
                "min_frames": 2, "max_gap": 1, "still": False},
    # in jumping's 5-fps samples: 0.6 s of running, with gaps up to 0.6 s
    "running": {"states": ("running", "walking", "still", "unknown"), "stats": ("speed",), "score": "speed",
                "min_frames": 3, "max_gap": 2, "still": False},
}
ROUND = {"knee": 0, "eyes": 1}  # decimals in the CSVs; everything else has 2
# notes: what the people showing the action say about it, listed per section (voices heard
# while talking or while silent, and whether it was judged from the sound and the face or the
# face alone), in these columns
NOTE_COLUMNS = {"voice": "voices", "heard": "voices_heard", "by": "judged_by"}
FACE_ACTIONS = ("smiling", "talking", "silent")
SOUND_ACTIONS = ("talking", "silent")
MOTION = {"standing": "any", "sitting": "still", "walking": "moving"}  # default --motion per action
DENSE_ACTIONS = ("jumping", "running")  # judged in the 5-fps samples around whole bodies


def frame_fields(action="standing"):
    a = ACTIONS[action]
    return ["t", "people", *a["states"], *[f"best_{m}" for m in a["stats"]]]


def segment_fields(action="standing"):
    notes = [NOTE_COLUMNS[n] for n in ACTIONS[action].get("notes", ())]
    return ["video", "start_s", "end_s", "start", "end", "duration_s", f"{action}_frames",
            f"max_people_{action}", f"best_{ACTIONS[action]['score']}", *notes, "still_diff", "youtube", "thumb"]


SPEECH_FIELDS = ["video", "start_s", "end_s", "start", "end", "duration_s", "voice", "on_screen", "sound_trusted",
                 "youtube"]
VOICE_FIELDS = ["video", "voice", "seconds", "turns", "on_screen_seconds", "sound_trusted", "thumb"]


FRAME_FIELDS, SEGMENT_FIELDS = frame_fields(), segment_fields()  # standing's


def rounded(name, v):
    d = ROUND.get(name, 2)
    return round(v) if d == 0 else round(float(v), d)


def label_people(poses, action="standing", extra=None):
    """Everyone in every sampled frame, with a state for the action.

    Returns (timeline, people per frame): timeline is poses, except for jumping, where it is the
    dense samples. extra is what analysis.VideoAnalysis.inputs() gives: jumping's dense samples; smiling's face
    measurements; for talking and silent {"faces", "audio"} (audio None without sound); for
    standing, sitting and walking {"camera", "motion"}, or None to ignore movement (standing's
    default, which keeps its results as they were).
    """
    h = int(poses["size"][1])
    if action == "jumping":
        return extra, motion.label_jumps(extra)
    if action == "running":
        return extra, motion.label_runs(extra, seated=lambda kp: is_seated(kp, h))
    by_frame = [[] for _ in poses["t"]]
    for j, f in enumerate(poses["frame"]):
        state, meas = classify(poses["kp"][j], h)
        by_frame[f].append({"j": j, "kp": poses["kp"][j], "box": poses["box"][j], "state": state,
                            "m": {**meas, "conf": float(poses["conf"][j])}})
    for people in by_frame:
        sitters = [p["m"] for p in people if p["state"] == "sitting"]
        for p in people:
            if p["state"] == "standing" and any(level_with(p["m"], s) for s in sitters):
                p["state"] = "sitting"
            elif action == "sitting" and p["state"] == "sitting" and not seated(p["m"]):
                p["state"] = "unknown"
    if action == "smiling":
        faces.label_faces(by_frame, poses, extra, action)
    elif action in SOUND_ACTIONS:
        sound, heard = extra.get("audio"), None
        if sound is not None:
            voices = audio.frame_voices(sound)
            burst, fps = faces.BURST_N * faces.BURST_STEP, float(poses["fps"])
            heard = [audio.heard(sound, picture_time(t, fps), picture_time(t, fps) + burst, voices)
                     for t in poses["t"].tolist()]
        faces.label_faces(by_frame, poses, extra["faces"], action, heard, extra.get("camera"))
    elif extra is not None:
        label_movement(by_frame, poses, action, extra["camera"], extra["motion"])
    return poses, by_frame


def label_movement(by_frame, poses, action, camera, wanted):
    """Walking: upright people whose body travels 0.6-4 torso lengths a second between samples
    (motion.label_motion) are walking; faster is running (the running action judges that in the
    5-fps samples); slower is still. Standing and sitting with --motion still (or moving): people
    whose movement says otherwise are relabelled walking or running (or still); people who could
    not be followed keep their label."""
    speed = motion.label_motion(poses, camera)
    for people in by_frame:
        for p in people:
            state = motion.motion_state(float(speed[p["j"]]))
            if state is not None:
                p["m"]["speed"] = float(speed[p["j"]])
            if action == "walking":
                upright = not (p["state"] == "sitting" and seated(p["m"]))
                p["state"] = "unknown" if state is None else state if upright else "still"
            elif p["state"] == action and state is not None and wanted != "any" \
                    and (state != "still") != (wanted == "moving"):
                p["state"] = state

