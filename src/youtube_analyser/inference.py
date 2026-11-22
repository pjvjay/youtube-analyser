"""The neural networks, on the CPU through OpenVINO (when installed) or onnxruntime.

Both runtimes give the same detections (the pose model's raw outputs agree to within 0.006, the
face models' to 2e-6, the voice models' exactly); OpenVINO is about 1.4-1.8x faster on Intel
CPUs. Models are loaded the first time a step needs them (Models), once per process.
"""
import math
import threading

import cv2
import numpy as np
import onnxruntime as ort

from . import assets

try:
    import openvino as ov
except ImportError:  # pragma: no cover - onnxruntime alone still works, just slower
    ov = None

POSE_MODEL = "yolo11s-pose.onnx"          # Ultralytics YOLO11s-pose, AGPL-3.0
LANDMARK_MODEL = "face_landmark.onnx"     # MediaPipe face mesh, Apache-2.0
EXPRESSION_MODEL = "face_expression.onnx"  # mini-Xception on FER2013, MIT
VAD_MODEL = "voice_activity.onnx"         # Silero VAD, MIT
VOICE_MODEL = "voice_embedding.onnx"      # Resemblyzer GE2E, Apache-2.0
PERSON_MASK_MODEL = "person_mask.onnx"    # MediaPipe selfie segmentation, Apache-2.0
GENDER_AGE_MODEL = "gender_age.onnx"      # yu4u WideResNet on IMDB-WIKI (research-only data, see people.toml)
HAPPY = 3                                 # expression classes: angry disgust fear happy sad surprise neutral


def runtime_name(runtime="auto"):
    """The runtime to use: "auto" is OpenVINO when installed, else onnxruntime."""
    if runtime == "auto":
        return "openvino" if ov is not None else "onnxruntime"
    if runtime == "openvino" and ov is None:
        raise RuntimeError("openvino is not installed; use --runtime onnxruntime")
    if runtime not in ("openvino", "onnxruntime"):
        raise ValueError(f"unknown runtime {runtime!r}")
    return runtime


class OnnxModel:
    """An ONNX model on the CPU. shape fixes the input's shape (faster on OpenVINO); None keeps the
    model's own, possibly variable, shapes. Called with an array (the only input) or a dict of
    named inputs; returns the outputs in order."""

    def __init__(self, path, shape, runtime="auto", threads=0):
        self.runtime = runtime_name(runtime)
        if self.runtime == "openvino":
            core = ov.Core()
            m = core.read_model(str(path))
            if shape is not None:
                m.reshape(list(shape))
            config = {"PERFORMANCE_HINT": "LATENCY"}
            if threads:
                config["INFERENCE_NUM_THREADS"] = threads
            self.compiled = core.compile_model(m, "CPU", config)
            self._local = threading.local()   # an inference request per thread: they are not shared safely
        else:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            if threads:
                so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name

    def __call__(self, x):
        if self.runtime == "openvino":
            request = getattr(self._local, "request", None)
            if request is None:
                request = self._local.request = self.compiled.create_infer_request()
            r = request.infer(x if isinstance(x, dict) else {0: x})
            return [np.array(r[o]) for o in self.compiled.outputs]   # copies: the request's buffers are reused
        return self.sess.run(None, x if isinstance(x, dict) else {self.inp: x})


class PoseModel:
    """YOLO pose model on the CPU, through OpenVINO when it is installed, else onnxruntime.

    Both give the same detections (raw outputs agree to within 0.006). On this Intel Mac
    OpenVINO is about 1.4x faster, mostly from keeping several frames in flight at once.
    """

    def __init__(self, path, imgsz, threads=0, runtime="auto"):
        self.path, self.imgsz = str(path), imgsz
        self.runtime = runtime_name(runtime)
        if self.runtime == "openvino":
            self.core = ov.Core()
            self.config = {"PERFORMANCE_HINT": "THROUGHPUT"}
            if threads:
                self.config["INFERENCE_NUM_THREADS"] = threads
            self.compiled = {}  # one per input shape: landscape videos and vertical Shorts differ
            self._lock = threading.Lock()
            self._local = threading.local()
        else:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            if threads:
                so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name

    def _compiled(self, shape):
        with self._lock:
            if shape not in self.compiled:
                m = self.core.read_model(self.path)
                m.reshape(list(shape))
                self.compiled[shape] = self.core.compile_model(m, "CPU", self.config)
            return self.compiled[shape]

    def _prep(self, img):
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)
        nw, nh = round(w * r), round(h * r)
        pw, ph = -(-nw // 32) * 32, -(-nh // 32) * 32  # letterbox to a multiple of 32
        dx, dy = (pw - nw) // 2, (ph - nh) // 2
        canvas = np.full((ph, pw, 3), 114, np.uint8)
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], np.float32) / 255
        return x, (r, dx, dy)

    @staticmethod
    def _post(raw, meta, conf, iou):
        """Return (conf (n,), box (n, 4) x y w h, kp (n, 17, 3) x y visibility) in image pixels."""
        r, dx, dy = meta
        out = raw[0].T  # (anchors, 4 box + 1 conf + 17*3)
        out = out[out[:, 4] > conf]
        if not len(out):
            return np.zeros(0, np.float32), np.zeros((0, 4), np.float32), np.zeros((0, 17, 3), np.float32)
        xywh = np.c_[out[:, 0] - out[:, 2] / 2, out[:, 1] - out[:, 3] / 2, out[:, 2], out[:, 3]]
        keep = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), out[:, 4].tolist(), conf, iou)).flatten()
        out, xywh = out[keep], xywh[keep]
        kp = out[:, 5:].reshape(-1, 17, 3).copy()
        kp[:, :, 0] = (kp[:, :, 0] - dx) / r
        kp[:, :, 1] = (kp[:, :, 1] - dy) / r
        box = np.c_[(xywh[:, 0] - dx) / r, (xywh[:, 1] - dy) / r, xywh[:, 2] / r, xywh[:, 3] / r]
        return out[:, 4].copy(), box.astype(np.float32), kp

    def __call__(self, img, conf, iou=0.45):
        """Detect people in one image (see _post for the result)."""
        x, meta = self._prep(img)
        if self.runtime == "openvino":
            requests = self._local.__dict__.setdefault("requests", {})
            if x.shape not in requests:
                requests[x.shape] = self._compiled(x.shape).create_infer_request()
            raw = np.array(requests[x.shape].infer({0: x})[0])
        else:
            raw = self.sess.run(None, {self.inp: x})[0]
        return self._post(raw, meta, conf, iou)

    def run(self, images, conf, iou=0.45):
        """Yield detections for each image of one video, in order."""
        if self.runtime != "openvino":
            for img in images:
                yield self(img, conf, iou)
            return
        done, nxt, count, queue = {}, 0, 0, None

        def finished(request, userdata):
            i, meta = userdata
            done[i] = self._post(request.get_output_tensor(0).data, meta, conf, iou)

        for i, img in enumerate(images):
            x, meta = self._prep(img)
            if queue is None:
                queue = ov.AsyncInferQueue(self._compiled(x.shape))
                queue.set_callback(finished)
            queue.start_async({0: x}, (i, meta))  # waits while every request is busy
            count = i + 1
            while nxt in done:
                yield done.pop(nxt)
                nxt += 1
        if queue is not None:
            queue.wait_all()
        while nxt in done:
            yield done.pop(nxt)
            nxt += 1
        if nxt != count:
            raise RuntimeError(f"pose model returned {nxt} of {count} frames")



# ---- faces ------------------------------------------------------------------------------------

def below(mid, d, ang, k):
    """The point k eye-distances below the middle of the eyes, along the face's own vertical."""
    a = math.radians(ang)
    return mid + k * d * np.array([-math.sin(a), math.cos(a)])


def crop(img, center, side, ang, out):
    """A side x side square around center, turned by ang degrees, scaled to out x out pixels."""
    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), ang, out / side)
    m[:, 2] += np.array([out / 2, out / 2]) - np.asarray(center, np.float64)
    return cv2.warpAffine(img, m, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class FaceModels:
    """The face landmark and expression models. Faces are cut out around the pose model's eyes
    (geo: middle of the eyes, eye distance d, roll): the landmark model gets a 4d square centred
    0.6d below the eyes (what MediaPipe itself uses for these faces), the expression model a 2.6d
    square centred 0.5d below the eyes, about how tightly the FER2013 training faces are cropped."""

    def __init__(self, runtime="auto", threads=0):
        self.landmark = OnnxModel(assets.model_path(LANDMARK_MODEL), (1, 192, 192, 3), runtime, threads)
        self.expression = OnnxModel(assets.model_path(EXPRESSION_MODEL), (1, 64, 64, 1), runtime, threads)

    def face(self, img, geo):
        """(468 x 2 landmarks in the 192-px crop, face score, happy probability) for one face."""
        mid, d, ang = geo
        lm, score = self.landmark(np.ascontiguousarray(crop(img, below(mid, d, ang, 0.6), 4 * d, ang, 192)[None, :, :, ::-1],
                                                       np.float32) / 255)
        grey = cv2.cvtColor(crop(img, below(mid, d, ang, 0.5), 2.6 * d, ang, 64), cv2.COLOR_BGR2GRAY)
        probs = self.expression(((grey.astype(np.float32) / 255 - 0.5) * 2)[None, :, :, None])[0]
        return (np.asarray(lm).reshape(468, 3)[:, :2], 1 / (1 + math.exp(-float(np.ravel(score)[0]))),
                float(np.ravel(probs)[HAPPY]))

    def landmarks(self, img, geo):
        """(468 x 2 landmarks in picture pixels, face score)."""
        mid, d, ang = geo
        centre = below(mid, d, ang, 0.6)
        m = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), ang, 192 / (4 * d))
        m[:, 2] += np.array([96, 96]) - centre
        lm, score, _ = self.face(img, geo)
        back = cv2.invertAffineTransform(m)
        return lm @ back[:, :2].T + back[:, 2], score


# ---- the sound --------------------------------------------------------------------------------

class VoiceModels:
    """The speech detector (a probability of speech every 32 ms) and the voice fingerprint model
    (256 numbers for 1.6 s of speech)."""

    HOP = 512                    # one speech probability per 32 ms at 16 kHz
    WINDOW = 25600               # 1.6 s of 16 kHz sound per fingerprint

    def __init__(self, runtime="auto", threads=0):
        self.vad = OnnxModel(assets.model_path(VAD_MODEL), None, runtime, threads)
        self.voice = OnnxModel(assets.model_path(VOICE_MODEL), None, runtime, threads)

    def speech_prob(self, wav, chunk_s=60, sr=16000):
        """Probability of speech for every 32 ms of wav."""
        hop = self.HOP
        n = len(wav) // hop
        x = np.concatenate([np.zeros(64, np.float32), wav[:n * hop]])
        h = c = np.zeros((1, 1, 128), np.float32)
        out, step = [], int(chunk_s * sr / hop)
        for i in range(0, n, step):
            rows = np.lib.stride_tricks.sliding_window_view(x[i * hop:(min(n, i + step)) * hop + 64], 576)[::hop]
            p, h, c = self.vad({"input": np.ascontiguousarray(rows), "h": h, "c": c})
            out.append(np.ravel(p))
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def fingerprints(self, wav, starts):
        """Voice fingerprints (n, 256) of the 1.6 s windows starting at these sample offsets,
        each scaled to -30 dBFS as the model expects."""
        out = []
        for i in range(0, len(starts), 64):
            batch = []
            for s in starts[i:i + 64]:
                seg = wav[s:s + self.WINDOW]
                batch.append(_power(wav, s) * (10 ** (-30 / 10) / (np.mean(seg ** 2) + 1e-12)))
            out.append(np.asarray(self.voice({"power": np.stack(batch)})[0]))
        return np.concatenate(out) if out else np.zeros((0, 256), np.float32)


def _power(wav, start, frames=160):
    """|STFT|^2 of `frames` 25 ms frames every 10 ms from `start`, as librosa's melspectrogram
    computes them for the whole clip (centred, zero-padded): the voice model's input."""
    idx = start - 200 + 160 * np.arange(frames)[:, None] + np.arange(400)[None, :]
    x = np.where((idx >= 0) & (idx < len(wav)), wav[np.clip(idx, 0, len(wav) - 1)], 0)
    win = np.hanning(401)[:-1].astype(np.float32)
    return (np.abs(np.fft.rfft(x * win, axis=1)) ** 2).astype(np.float32)


# ---- appearance -------------------------------------------------------------------------------

class AppearanceModels:
    """The person mask, the face landmark model (shared) and, if enabled and present, gender and age."""

    def __init__(self, faces, runtime="auto", threads=0, gender_age=True):
        self.faces = faces
        self.mask_model = OnnxModel(assets.model_path(PERSON_MASK_MODEL), (1, 144, 256, 3), runtime, threads)
        path = assets.model_path(GENDER_AGE_MODEL)
        self.gender_age = OnnxModel(path, (1, 64, 64, 3), runtime, threads) if gender_age and path.exists() else None

    def person_mask(self, img):
        """Which pixels are people (True), at the picture's size."""
        h, w = img.shape[:2]
        x = cv2.resize(img, (256, 144), interpolation=cv2.INTER_AREA)[None, :, :, ::-1].astype(np.float32) / 255
        m = np.asarray(self.mask_model(x)[0])[0, :, :, 0]
        return cv2.resize(m, (w, h)) >= 0.5

    def female_age(self, img, geo):
        """(probability the face looks female, apparent age) from a 4d square around the face."""
        mid, d, ang = geo
        face = crop(img, below(mid, d, ang, 0.5), 4 * d, ang, 64)
        female, ages = self.gender_age(face[None].astype(np.float32))
        return float(np.ravel(female)[0]), float(np.ravel(ages) @ np.arange(101))


class Models:
    """Every model a run may need, each loaded the first time it is asked for and then shared:
    pose (the settings' pose model and input size), faces, voices, appearance. Models.shared()
    gives one set per process for the same settings (loading takes seconds)."""

    _shared = {}
    _shared_lock = threading.Lock()

    def __init__(self, pose_model=None, imgsz=640, runtime="auto", threads=0, gender_age=True):
        self.pose_model = pose_model or assets.model_path(POSE_MODEL)
        self.imgsz, self.runtime, self.threads, self.gender_age_on = imgsz, runtime_name(runtime), threads, gender_age
        self._loaded = {}
        self._lock = threading.RLock()

    @classmethod
    def shared(cls, pose_model=None, imgsz=640, runtime="auto", threads=0, gender_age=True):
        key = (str(pose_model or assets.model_path(POSE_MODEL)), imgsz, runtime_name(runtime), threads, gender_age)
        with cls._shared_lock:
            if key not in cls._shared:
                cls._shared[key] = cls(pose_model, imgsz, runtime, threads, gender_age)
            return cls._shared[key]

    def _get(self, name, make):
        with self._lock:
            if name not in self._loaded:
                self._loaded[name] = make()
            return self._loaded[name]

    @property
    def pose(self):
        return self._get("pose", lambda: PoseModel(self.pose_model, self.imgsz, self.threads, self.runtime))

    @property
    def faces(self):
        return self._get("faces", lambda: FaceModels(self.runtime, self.threads))

    @property
    def voices(self):
        return self._get("voices", lambda: VoiceModels(self.runtime, self.threads))

    @property
    def appearance(self):
        return self._get("appearance", lambda: AppearanceModels(self.faces, self.runtime, self.threads, self.gender_age_on))
