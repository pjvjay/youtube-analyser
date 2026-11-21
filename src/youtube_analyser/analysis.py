"""Everything computed about one video, made once and (with a cache folder) saved for next time.

VideoAnalysis is the pipeline's single source of model output. Each part is computed only when
a step asks for it, from the parts it depends on:

    poses       the pose model at `fps` (every person, 17 body points)      poses/<video>.npz
    camera      the camera's pan, tilt and zoom between the samples,          camera/<video>.npz
                and each person's clothes colours
    dense       the pose model again at `jump_fps`, around every moment a     poses/<video>.dense.npz
                whole body is in view (running and jumping)
    faces       every usable face measured over a burst of frames              faces/<video>.npz
    sound       speech, how clear it is, and the voices (None: no sound)      audio/<video>.npz
    appearance  hair and clothes colours, apparent gender and age             appearance/<video>.npz

Each saved file records what it was made from (the pose model, its input size, the sampling
rate and confidence, the video file's size, and the versions of the steps), and is made again
when any of that has changed. Files saved by earlier versions of this code, which recorded less,
are used when what they did record still fits and the settings are the defaults they were made
with.
"""
import json
import time
import zipfile
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np

from . import assets, audio, faces, media, motion, people
from .actions import DENSE_ACTIONS, MOTION, SOUND_ACTIONS, label_people
from .inference import POSE_MODEL, Models

POSES_VERSION = 1
AUDIO_VERSION = 2


@dataclass(frozen=True)
class Settings:
    """How videos are sampled and which pose model looks at them."""
    pose_model: str = POSE_MODEL    # a file name in the models folder, or a path
    imgsz: int = 640                # longest side fed to the pose model
    fps: float = 1.0                # frames sampled per second
    conf: float = 0.35              # minimum person confidence
    jump_fps: float = motion.JUMP_FPS
    runtime: str = "auto"           # auto, openvino or onnxruntime
    threads: int = 0                # inference threads (0: automatic)

    @property
    def model_path(self):
        p = Path(self.pose_model)
        return p if p.is_absolute() or p.exists() or len(p.parts) > 1 else assets.model_path(self.pose_model)

    def is_default(self):
        d = Settings()
        return (Path(self.pose_model).name, self.imgsz, self.conf) == (Path(d.pose_model).name, d.imgsz, d.conf)

    def pose_key(self):
        path = self.model_path
        return {"model": path.name, "model_bytes": path.stat().st_size if path.exists() else 0,
                "imgsz": self.imgsz, "fps": float(self.fps), "conf": float(self.conf), "version": POSES_VERSION}

    def with_(self, **changes):
        return replace(self, **changes)

    def models(self, gender_age=True):
        """The models for these settings (one set per process, see inference.Models.shared)."""
        return Models.shared(self.model_path, self.imgsz, self.runtime, self.threads, gender_age)


def save_npz(path, arrays):
    """Save atomically: a reader never sees a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_npz(path):
    """The arrays saved at path, or None when there is no file or it cannot be read (a file cut
    short by a crash is made again rather than failing the video for good)."""
    if path is None or not Path(path).exists():
        return None
    try:
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return None


def infer(path, model, fps, conf, windows=None):
    """Run the pose model on every sampled frame. Returns the arrays saved in poses/*.npz.

    With windows (a list of (start, end) seconds), only frames inside them are sampled, and
    each frame's window number, the camera's movement from the frame before (shift, scale, and
    response, which drops at a cut) and each person's look are added (motion.camera_motion).
    """
    w, h = media.probe(path)
    times, frame, confs, boxes, kps, window = [], [], [], [], [], []

    def images():
        if windows is None:
            for t, img in media.read_frames(path, fps, w, h):
                times.append(t)
                yield img
            return
        for n, t, img in media.read_windows(path, windows, fps, w, h):
            times.append(t)
            window.append(n)
            yield img

    for i, (c, b, k) in enumerate(model.run(images(), conf) if model is not None else ()):
        frame += [i] * len(c)
        confs.append(c)
        boxes.append(b)
        kps.append(k)
    cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape, np.float32)
    out = {"t": np.array(times, np.float32), "frame": np.array(frame, np.int32),
           "conf": cat(confs, (0,)), "box": cat(boxes, (0, 4)), "kp": cat(kps, (0, 17, 3)),
           "size": np.array([w, h]), "fps": np.float32(fps)}
    if windows is not None:  # times like 7.4 s are kept exact (float32 would print 7.400000095)
        out.update(t=np.array(times, np.float64), window=np.array(window, np.int32),
                   windows=np.array(windows, np.float32).reshape(-1, 2))
        out.update(motion.camera_motion(media.read_windows(path, windows, fps, w, h), out),
                   camera_version=np.int32(motion.CAMERA_VERSION))
    return out


def infer_images(images, model, conf, w, h):
    """The pose model on a list of pictures (the quick look's keyframes), as infer() returns it."""
    frame, confs, boxes, kps = [], [], [], []
    for i, (c, b, k) in enumerate(model.run(iter(images), conf)):
        frame += [i] * len(c)
        confs.append(c)
        boxes.append(b)
        kps.append(k)
    cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape, np.float32)
    return {"t": np.arange(len(images), dtype=np.float32), "frame": np.array(frame, np.int32), "conf": cat(confs, (0,)),
            "box": cat(boxes, (0, 4)), "kp": cat(kps, (0, 17, 3)), "size": np.array([w, h]), "fps": np.float32(1)}


class VideoAnalysis:
    """What has been (or will be) computed about one video.

    cache: the folder model output is saved in (None: kept in memory only); force: make
    everything again. models: an inference.Models, shared between videos."""

    def __init__(self, path, settings=None, models=None, cache=None, force=False):
        self.path = Path(path)
        self.settings = settings or Settings()
        self.models = models or self.settings.models()
        self.cache = Path(cache) if cache is not None else None
        self.force = force
        self.notes = []              # what was computed (for progress lines)
        self.pose_note = ""
        self.timings = {}            # seconds spent per part

    # ---- the cache ----------------------------------------------------------------------------

    def _file(self, folder, suffix=".npz"):
        return None if self.cache is None else self.cache / folder / f"{self.path.stem}{suffix}"

    def _key(self, **parts):
        return json.dumps({"video_bytes": self.path.stat().st_size if self.path.exists() else 0, **parts}, sort_keys=True)

    def _cached(self, folder, suffix, key, make, legacy=None):
        """The arrays saved for this key, or make() them (and save them)."""
        path = self._file(folder, suffix)
        saved = load_npz(path) if path is not None and not self.force else None
        if saved is not None:
            mark = saved.pop("_key", None)
            if (mark is not None and str(mark) == key) or (mark is None and legacy is not None and legacy(saved)):
                return saved
        t0 = time.perf_counter()
        data = make()
        self.timings[folder + suffix] = time.perf_counter() - t0
        if path is not None:
            save_npz(path, {**data, "_key": np.array(key)})
        return data

    # ---- the parts ----------------------------------------------------------------------------

    @cached_property
    def size(self):
        return media.probe(self.path)

    @cached_property
    def poses(self):
        s = self.settings
        started = time.perf_counter()
        made = []

        def make():
            made.append(True)
            return infer(self.path, self.models.pose, s.fps, s.conf)
        data = self._cached("poses", ".npz", self._key(poses=s.pose_key()), make,
                            legacy=lambda d: s.is_default() and float(d["fps"]) == s.fps)
        self.pose_note = (f"{1000 * (time.perf_counter() - started) / max(len(data['t']), 1):.0f} ms/frame"
                          if made else "saved poses")
        return data

    def _pose_key(self):
        return {"poses": self.settings.pose_key(), "n": int(len(self.poses["frame"]))}

    @cached_property
    def camera(self):
        poses = self.poses

        def make():
            w, h = (int(x) for x in poses["size"])
            steps = motion.CAMERA_STEPS
            cam = motion.camera_motion(media.read_frames(self.path, float(poses["fps"]) * steps, w, h), poses, steps)
            cam.update(poses_t=np.int32(len(poses["t"])), steps=np.int32(steps), camera_version=np.int32(motion.CAMERA_VERSION))
            return cam
        return self._cached("camera", ".npz", self._key(**self._pose_key(), camera=motion.CAMERA_VERSION,
                                                          steps=motion.CAMERA_STEPS), make,
                            legacy=lambda d: self.settings.is_default() and int(d["poses_t"]) == len(poses["t"]) and "steps" in d
                            and int(d.get("camera_version", 0)) == motion.CAMERA_VERSION)

    @cached_property
    def dense(self):
        poses, s = self.poses, self.settings

        def make():
            _, by_frame = label_people(poses)
            states = {p["j"]: p["state"] for people_ in by_frame for p in people_}
            windows = motion.jump_windows(poses, [states[j] for j in range(len(poses["frame"]))])
            data = infer(self.path, self.models.pose if windows else None, s.jump_fps, s.conf, windows)
            self.notes.append(f"{sum(b - a for a, b in data['windows']):.0f}s sampled again at {s.jump_fps:g} fps")
            return data
        return self._cached("poses", ".dense.npz", self._key(**self._pose_key(), jump_fps=float(s.jump_fps),
                                                               camera=motion.CAMERA_VERSION), make,
                            legacy=lambda d: s.is_default() and float(d["fps"]) == s.jump_fps
                            and int(d.get("camera_version", 0)) == motion.CAMERA_VERSION)

    @cached_property
    def faces(self):
        poses = self.poses

        def make():
            w, h = (int(x) for x in poses["size"])
            data = faces.analyse_faces(media.read_frames(self.path, 1 / faces.BURST_STEP, w, h), poses, self.models.faces)
            data["poses_n"] = np.int32(len(poses["frame"]))
            self.notes.append(f"{len(data['det'])} faces measured")
            return data
        return self._cached("faces", ".npz", self._key(**self._pose_key(), burst_step=faces.BURST_STEP), make,
                            legacy=lambda d: self.settings.is_default() and int(d["poses_n"]) == len(poses["frame"])
                            and "aligned" in d
                            and float(d["burst_step"]) == np.float32(faces.BURST_STEP))

    @cached_property
    def sound(self):
        """Speech and voices (audio.analyse), or None when the video has no sound."""
        def make():
            wav = media.load_audio(self.path)
            if wav is None:
                self.notes.append("no sound")
                return {"has_sound": np.bool_(False)}
            data = {**audio.analyse(wav, self.models.voices), "has_sound": np.bool_(True)}
            self.notes.append(f"{int(data['voices'])} voices, sound trusted "
                              f"{np.mean(data['snr_db'] >= audio.SNR_OK_DB) if len(data['snr_db']) else 0:.0%} of the time")
            return data
        data = self._cached("audio", ".npz", self._key(audio=AUDIO_VERSION), make,
                            legacy=lambda d: not bool(d["has_sound"]) or "voice_f0" in d)
        return data if bool(data["has_sound"]) else None

    def appearance(self, config, looks=True, faces_to_judge=()):
        """The appearance arrays (people.describe), measuring what is asked for and not yet
        measured: looks (hair and clothes of everyone) and the faces to judge for gender and age."""
        poses = self.poses
        key = self._key(**self._pose_key(), appearance=people.APPEARANCE_VERSION)
        if not hasattr(self, "_appearance"):
            path = self._file("appearance")
            saved = load_npz(path) if path is not None and not self.force else None
            if saved is not None and str(saved.pop("_key", "")) != key:
                saved = None
            self._appearance = saved if saved is not None else people.empty(poses)
        data = self._appearance
        gender_age = self.models.appearance.gender_age is not None if faces_to_judge else False
        todo = [j for j in faces_to_judge if not data["faced"][j]] if gender_age else []
        if (looks and not int(data["looked"])) or todo:
            w, h = (int(x) for x in poses["size"])
            t0 = time.perf_counter()
            data = people.describe(media.read_frames(self.path, float(poses["fps"]), w, h), poses,
                                   self.models.appearance, config, data, looks, todo)
            self.timings["appearance"] = self.timings.get("appearance", 0) + time.perf_counter() - t0
            self._appearance = data
            if self.cache is not None:
                save_npz(self._file("appearance"), {**data, "_key": np.array(key)})
        return data

    # ---- for the actions ----------------------------------------------------------------------

    def inputs(self, action, motion_filter=None, use_sound=True):
        """What an action needs besides the poses (see actions.label_people): jumping's and
        running's dense samples; smiling's faces; talking's and silent's {"faces", "audio",
        "camera"}; {"camera", "motion"} for walking and for standing or sitting still or moving;
        None otherwise (standing's default, which ignores movement)."""
        if action in DENSE_ACTIONS:
            return self.dense
        if action == "smiling":
            return self.faces
        if action in SOUND_ACTIONS:
            return {"faces": self.faces, "audio": self.sound if use_sound else None, "camera": self.camera}
        wanted = motion_filter or MOTION.get(action, "any")
        if wanted == "any" and action != "walking":
            return None
        return {"camera": self.camera, "motion": wanted}

    def label(self, action, motion_filter=None, use_sound=True):
        """(timeline, people per sampled frame) labelled for the action."""
        return label_people(self.poses, action, self.inputs(action, motion_filter, use_sound))

    def interview_start(self, bank):
        from .intro import start_row
        if not hasattr(self, "_starts"):
            self._starts = {}
        if id(bank) not in self._starts:
            self._starts[id(bank)] = start_row(self.path, bank)
        return self._starts[id(bank)]
