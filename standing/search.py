"""Search the videos for a described person doing an action.

    .venv/bin/python standing/search.py "woman with purple hair talking" *.mp4 *.mkv *.webm
    .venv/bin/python standing/search.py "man in a blue coat" *.mp4 --action walking
    .venv/bin/python standing/search.py "Jane talking" *.mp4          (Jane enrolled in people.toml)

The query names a person by what they look like (hair and clothes colours, apparent gender and
age), how they sound (a deep or high voice) or who they are (a name enrolled in people.toml, found
by their voice), and optionally an action (standing, sitting, walking, running, jumping, smiling,
talking, silent; without one, any time they are in view). The words it understands are in
people.toml; any other word is listed as ignored.

For each video, each person followed through a shot (people.py) is checked against every part of
the query: yes, no, or unknown (not seen well enough: a face too small for its hair colour, say).
People matching every part are found ("match"); with --possible, so are people for whom nothing
is a "no" but something is unknown ("possible"). Then the times they do the action (the
detector's rules, detect_standing.py) become the results. --explain prints what was seen of
every person and why they matched or not.

It reuses everything detect_standing.py has saved in --out (the pose model's output, faces, sound,
camera movement) and saves what it adds (appearance/<video>.npz), so a second search is quick.

Writes --results/<query>/matches.csv (video, person, start, end, action, match, evidence,
YouTube link at the start), index.html with a thumbnail of each (the person boxed), and prints a
summary.
"""
import argparse
import contextlib
import csv
import html
import re
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import actions  # noqa: E402
import audio  # noqa: E402
import detect_standing as ds  # noqa: E402
import intro  # noqa: E402
import people  # noqa: E402

FIELDS = ["video", "person", "start_s", "end_s", "start", "end", "duration_s", "action", "match", "evidence",
          "youtube", "thumb"]
PARTS = ("gender", "age", "hair", "top", "bottom", "clothes", "voice", "name")


# ---- the query -------------------------------------------------------------------------------

def parse(query, config, names=()):
    """What a query asks for: {gender, age, hair, top, bottom, clothes, voice, name, action,
    ignored}. Colours are sets of palette names (a group or synonym becomes its names)."""
    w = config["words"]
    colours = set(config["colours"]["hair"]) | set(config["colours"]["clothes"])
    req = {k: None for k in PARTS}
    req.update(action=None, ignored=[])
    lower = {n.lower(): n for n in names}
    text = query.lower()
    for name in sorted(lower, key=len, reverse=True):  # names may have spaces
        if re.search(rf"\b{re.escape(name)}\b", text):
            req["name"] = lower[name]
            text = re.sub(rf"\b{re.escape(name)}\b", " ", text)
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", text)
    tokens = []
    for word in words:  # "purple-haired" -> "purple", "haired"; "grey-haired" stays a word of its own
        parts = word.split("-")
        tokens += [word] if word in _all_words(w) or len(parts) == 1 else parts
    pending, wearing = [], False
    i = 0
    while i < len(tokens):
        t = tokens[i]
        pair = f"{t} {tokens[i + 1]}" if i + 1 < len(tokens) else None
        if pair in colours:  # "light blue"
            pending.append({pair})
            i += 2
            continue
        colour = _colour(t, config)
        action = next((a for a, ws in w["actions"].items() if t in ws), None)
        if colour is not None and not (t in w["low_voice"] + w["high_voice"] and _voice_next(tokens, i)):
            pending.append(colour)
        elif t in w["hair"]:
            req["hair"] = _join(req["hair"], pending)
            pending = []
        elif t in w["top"]:
            req["top"] = _join(req["top"], pending)
            pending, wearing = [], False
        elif t in w["bottom"]:
            req["bottom"] = _join(req["bottom"], pending)
            pending, wearing = [], False
        elif t in w["wearing"]:
            wearing = True
        elif t in w["female"]:
            req["gender"] = "female"
        elif t in w["male"]:
            req["gender"] = "male"
        elif t in w["young"]:
            req["age"] = "young"
        elif t in w["old"]:
            req["age"] = "old"
            if t.endswith("haired"):
                req["hair"] = _join(req["hair"], [set(config["words"]["groups"]["grey"])])
        elif t in w["low_voice"] + w["high_voice"] and (_voice_next(tokens, i) or "voic" in t or "pitch" in t):
            req["voice"] = "low" if t in w["low_voice"] else "high"
            i += 1 if _voice_next(tokens, i) else 0
        elif t in ("voice", "voiced"):
            pass
        elif action is not None:
            req["action"] = action
        elif t in w["anyone"] or t in w["ignore"]:
            pass
        else:
            req["ignored"].append(t)
        if pending and (t in w["female"] + w["male"] + w["anyone"]) and not wearing:
            # "blonde woman", "redhead": a colour before the person is their hair
            hairy = [c for c in pending if c <= set(config["colours"]["hair"])]
            if hairy:
                req["hair"] = _join(req["hair"], hairy)
                pending = [c for c in pending if c not in hairy]
        i += 1
    if pending:  # "in red", "wearing blue": colours left over are clothes (or hair for "redhead")
        hairy = [c for c in pending if c <= set(config["colours"]["hair"]) and not wearing and
                 any(x in query.lower() for x in ("blond", "brunette", "redhead", "ginger", "auburn"))]
        rest = [c for c in pending if c not in hairy]
        if hairy:
            req["hair"] = _join(req["hair"], hairy)
        if rest:
            req["clothes"] = _join(req["clothes"], rest)
    return req


def _all_words(w):
    out = set()
    for v in w.values():
        if isinstance(v, list):
            out |= set(v)
        elif isinstance(v, dict):
            for x in v.values():
                out |= set(x) if isinstance(x, list) else {x}
    return out


def _voice_next(tokens, i):
    return i + 1 < len(tokens) and tokens[i + 1] in ("voice", "voiced", "voices")


def _colour(t, config):
    """The palette names a word stands for, or None."""
    w = config["words"]
    t = w["colour"].get(t, t)
    if t in w["groups"]:
        return set(w["groups"][t])
    if t in config["colours"]["hair"] or t in config["colours"]["clothes"]:
        return {t}
    return None


def _join(have, pending):
    if not pending:
        return have
    got = set().union(*pending)
    return got if have is None else have | got


def describe_query(req):
    parts = []
    for k in PARTS:
        if req[k]:
            v = req[k]
            parts.append(f"{k} {'/'.join(sorted(v)) if isinstance(v, set) else v}")
    return ", ".join(parts) or "anyone"


# ---- checking a person -----------------------------------------------------------------------

LOOKS = ("hair", "top", "bottom", "clothes")        # the parts the appearance pass alone decides


def check(person, req, config, enrolled_req=None, parts=PARTS):
    """(verdict, evidence): "match", "possible" or "no", and what decided each part (only these
    parts: the search checks the looks first, before the slower passes)."""
    look = config["look"]
    verdicts, evidence = [], []
    for part in parts:
        want = req[part]
        if not want:
            continue
        v, why = _part(person, part, want, look, config)
        if part == "name" and v == "unknown" and enrolled_req:
            described = [_part(person, p, enrolled_req[p], look, config) for p in PARTS if enrolled_req.get(p)]
            if described and all(x[0] == "yes" for x in described):
                v, why = "yes", f"looks like {want}: " + "; ".join(x[1] for x in described)
        verdicts.append(v)
        evidence.append(f"{why} [{v}]")
    verdict = "no" if "no" in verdicts else "match" if all(v == "yes" for v in verdicts) else "possible"
    return verdict, "; ".join(evidence)


def _part(person, part, want, look, config):
    if part == "gender":
        got, by = person["gender"], person["gender_by"]
        if got is None:
            return "unknown", f"gender not seen (faces {person['faces']})"
        return ("yes" if got == want else "no"), f"{got} ({by})"
    if part == "age":
        if person["age"] is None:
            return "unknown", "age not seen"
        young, old = person["age"] < look["young_below"], person["age"] > look["old_above"]
        ok = young if want == "young" else old
        return ("yes" if ok else "no"), f"looks about {person['age']:.0f}"
    if part in ("hair", "top", "bottom"):
        name, share, n = person[part]
        if name is None or share < look["min_share"]:
            return "unknown", f"{part} colour not seen ({n} samples)"
        return ("yes" if name in want else "no"), f"{part} {name} ({share:.0%} of {n})"
    if part == "clothes":
        seen = [(person[p][0], person[p][1]) for p in ("top", "bottom") if person[p][0] is not None and person[p][1] >= look["min_share"]]
        if not seen:
            return "unknown", "clothes not seen"
        ok = any(n in want for n, _ in seen)
        return ("yes" if ok else "no"), "clothes " + "/".join(n for n, _ in seen)
    if part == "voice":
        f0 = person["voice_f0"]
        if f0 is None or np.isnan(f0):
            return "unknown", "voice not heard"
        v = config["voice"]
        got = "low" if f0 <= v["low_below_hz"] else "high" if f0 >= v["high_above_hz"] else "middle"
        return ("yes" if got == want else "no"), f"{got} voice ({f0:.0f} Hz)"
    if part == "name":
        if person["voice"] is None:
            return "unknown", "not heard talking"
        got = person["voice_name"]
        return ("yes" if got == want else "no"), f"voice of {got}" if got else f"voice not {want}'s"
    raise KeyError(part)


def describe_person(person, config):
    """One line of what was seen of a person (for --explain)."""
    parts = [f"{ds.hms(person['first_t'])}-{ds.hms(person['last_t'])}, {len(person['detections'])} samples"]
    if person["gender"]:
        parts.append(f"{person['gender']} ({person['gender_by']})")
    if person["age"] is not None:
        parts.append(f"about {person['age']:.0f}")
    for part in ("hair", "top", "bottom"):
        name, share, n = person[part]
        parts.append(f"{part} {name} ({share:.0%} of {n})" if name else f"{part} not seen")
    if person["voice_f0"] is not None and not np.isnan(person["voice_f0"]):
        parts.append(f"voice {person['voice_f0']:.0f} Hz" + (f" ({person['voice_name']})" if person["voice_name"] else ""))
    return ", ".join(parts)


# ---- searching one video ---------------------------------------------------------------------

def options(out, **over):
    """The command line's settings (for using Searcher from other code): out is the cache folder."""
    args = argparse.Namespace(out=Path(out), fps=1.0, conf=0.35, jump_fps=actions.JUMP_FPS, force=False,
                              keep_intro=False, intro_bank=intro.DEFAULT_BANK, runtime="auto", threads=0,
                              model=ds.DEFAULT_MODEL, imgsz=640, possible=False, explain=False, videos=[])
    for k, v in over.items():
        setattr(args, k, v)
    return args


class Searcher:
    """Searches videos with the detector's saved inputs in args.out (computing what is missing)."""

    def __init__(self, args, config, models=None, bank="load"):
        """args: as main()'s; models: already loaded ones ({"pose", "faces", "voices",
        "appearance"}: a model, or a function that loads it); bank: the intro bank (None: keep
        intros), by default loaded as args say."""
        self.args, self.config = args, config
        self.given = dict(models or {})
        self.loaded = {}
        self.models = {n: (lambda n=n: self.model(n)) for n in ("pose", "faces", "voices")}
        self.enrolled = None
        self.unsure = 0   # people left out who might match (--possible)
        self.starts = {}
        self.label_intros = False   # True: keep results in the intros, marked in_intro (channel.py)
        self.timings = {}           # seconds spent on each step, added up over the videos searched
        self.note = ""
        self.bank = bank if bank != "load" else None if args.keep_intro else intro.load_bank(args.intro_bank)

    def model(self, name):
        if name not in self.loaded and name in self.given:
            m = self.given[name]
            self.loaded[name] = m() if isinstance(m, types.FunctionType) else m
        if name not in self.loaded:
            a = self.args
            self.loaded[name] = (ds.PoseModel(a.model, a.imgsz, a.threads, a.runtime) if name == "pose"
                                 else actions.FaceModels(a.runtime, a.threads) if name == "faces"
                                 else audio.VoiceModels(a.runtime, a.threads) if name == "voices"
                                 else people.AppearanceModels(a.runtime, a.threads, self.config["models"]["gender_age"]))
        return self.loaded[name]

    def inputs(self, v, poses, action):
        a = self.args
        sub = argparse.Namespace(out=a.out, action=action, force=False, no_sound=False, motion=None,
                                 jump_fps=a.jump_fps, conf=a.conf)
        return ds.load_input(v, poses, sub, self.models)[0]

    def appearance(self, v, poses, looks=True, faces=(), saved=None):
        """The appearance arrays (appearance/<video>.npz), measuring what is asked for and not
        yet measured: looks (hair and clothes of everyone) and faces (detections to judge for
        gender and age)."""
        path = self.args.out / "appearance" / f"{v.stem}.npz"
        if saved is None:
            saved = dict(np.load(path)) if path.exists() else None
            if saved is None or int(saved["version"]) != people.APPEARANCE_VERSION \
                    or int(saved["poses_n"]) != len(poses["frame"]):
                saved = people.empty(poses)
        todo_faces = [j for j in faces if not saved["faced"][j]] if self.config["models"]["gender_age"] else []
        if not (looks and not int(saved["looked"])) and not todo_faces:
            return saved
        w, h = (int(x) for x in poses["size"])
        out = people.describe(ds.read_frames(v, float(poses["fps"]), w, h), poses, self.model("appearance"),
                              self.config, saved, looks, todo_faces)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_npz(path, out)
        return out

    @contextlib.contextmanager
    def timed(self, step):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[step] = self.timings.get(step, 0.0) + time.perf_counter() - t0

    def enrol(self):
        """{name: (voice fingerprint, pitch)} of the people in the config with a voice sample."""
        if self.enrolled is not None:
            return self.enrolled
        self.enrolled = {}
        for p in self.config.get("people", []):
            src = p.get("voice_from") or {}
            path = Path(p["voice_file"]) if "voice_file" in p else Path(src.get("video", "")) if src else None
            if path is None:
                continue
            if not path.exists() and src:
                path = next((v for v in self.args.videos if v.name == src["video"]), path)
            wav = audio.load(path) if path.exists() else None
            if wav is None:
                print(f"{p['name']}: no sound in {path}, not enrolled", file=sys.stderr)
                continue
            a, b = int(src.get("start", 0) * audio.SR), int(src.get("end", len(wav) / audio.SR) * audio.SR)
            voice = audio.voice_print(wav[a:b], self.model("voices"))
            if voice is None:
                print(f"{p['name']}: too little speech in the voice sample (at least 2 s needed)", file=sys.stderr)
                continue
            self.enrolled[p["name"]] = voice
        return self.enrolled

    def search(self, v, req, action):
        """Results for one video: [row dict], each with best_t, best_j (the 1-fps detection) and
        best_box for the thumbnail. self.note says why there were none.

        The work is done cheapest first, and stops as soon as nobody can match: the pose model;
        then hair and clothes colours, if asked for (nobody with purple hair: done); then
        following people and checking their looks; then their voices, if needed; then the face
        model for gender and age, on each remaining person's few clearest faces; then the action.
        Without a description ("anyone jumping"), the detector's own sections are the results."""
        a, cfg = self.args, self.config
        self.note = ""
        with self.timed("pose model"):
            poses, _ = ds.load_poses(v, a, self.models["pose"])
        if not len(poses["frame"]):
            self.note = "nobody in view"
            return []
        if not any(req[k] for k in PARTS):
            return self.anyone(v, poses, action)
        enrolled_req = None
        if req["name"]:
            described = next((p.get("describe") for p in cfg.get("people", []) if p["name"] == req["name"]), None)
            enrolled_req = parse(described, cfg) if described else None
        either = [r for r in (req, enrolled_req) if r]
        looks = any(r[k] for r in either for k in LOOKS)
        faces = any(r[k] for r in either for k in ("gender", "age"))
        with self.timed("hair and clothes"):
            look = self.appearance(v, poses, looks=looks)
        if not a.possible and not req["name"] and not self.could_match(look, req):
            self.note = "nobody looks right: " + self.note
            return []
        with self.timed("camera"):
            camera = ds.load_camera(v, poses, a)
        track = people.tracks(poses, camera, camera["look"], look["hair"])
        persons = people.summarise(poses, track, look, cfg)
        first = [p for p in LOOKS if req[p]]
        alive = {tr for tr, p in persons.items() if check(p, req, cfg, parts=first)[0] != "no"}
        if not alive:
            self.note = "nobody looks right"
            return []
        own, voices = {}, None
        if bool(req["voice"] or req["name"] or req["gender"]) or action in ds.SOUND_ACTIONS:
            with self.timed("faces and voices"):
                own, voices = self.voices(v, poses)
            persons = people.summarise(poses, track, look, cfg, own, voices)
            heard = lambda tr: [p for p in ("voice",) if req[p]] + (["gender"] if req["gender"] and persons[tr]["gender_by"] == "voice" else [])
            alive = {tr for tr in alive if check(persons[tr], req, cfg, parts=heard(tr))[0] != "no"}
        if faces and alive:
            undecided = [tr for tr in alive if persons[tr]["gender_by"] != "voice" or any(r["age"] for r in either)]
            with self.timed("gender and age"):
                look = self.appearance(v, poses, looks=looks, faces=people.face_picks(look, track, undecided, cfg), saved=look)
            persons = people.summarise(poses, track, look, cfg, own, voices)
        wanted = {}
        for tr, person in persons.items():
            verdict, why = check(person, req, cfg, enrolled_req) if tr in alive else ("no", "ruled out by their looks")
            if a.explain:
                print(f"  {v.name} person {tr}: {describe_person(person, cfg)} -> {verdict}: {why}", flush=True)
            if verdict == "match" or (verdict == "possible" and a.possible):
                wanted[tr] = (verdict, why)
            elif verdict == "possible":
                self.unsure += 1
        if not wanted:
            self.note = "nobody matches"
            return []
        with self.timed(action or "visible"):
            timeline, states = self.states(v, poses, track, action)
        rows = self.sections(v, poses, timeline, states, wanted, action)
        if not rows:
            self.note = f"found, but not {action}"
        return rows

    def scout(self, v, req):
        """A quick look before the full search: the pose model and the hair and clothes colours
        on the video's keyframes only (see keyframes()). False when nobody there has the colours
        asked for (self.note says which); True when someone might, or when the query names no
        colours (nothing to look for)."""
        if not any(req[k] for k in LOOKS):
            return True
        w, h = ds.probe(v)
        with self.timed("quick look"):
            return self._scout(v, req, w, h)

    def _scout(self, v, req, w, h):
        frames = list(keyframes(v, w, h))
        n = len(frames)
        confs, boxes, kps, frame = [], [], [], []
        for i, (c, b, k) in enumerate(self.model("pose").run((img for _, img in frames), self.args.conf)):
            frame += [i] * len(c)
            confs.append(c)
            boxes.append(b)
            kps.append(k)
        cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape, np.float32)
        poses = {"t": np.arange(n, dtype=np.float32), "frame": np.array(frame, np.int32), "conf": cat(confs, (0,)),
                 "box": cat(boxes, (0, 4)), "kp": cat(kps, (0, 17, 3)), "size": np.array([w, h]), "fps": np.float32(1)}
        if not len(poses["frame"]):
            self.note = f"quick look: nobody in {n} keyframes"
            return False
        look = people.describe(iter(frames), poses, self.model("appearance"), self.config, looks=True)
        if self.could_match(look, req):
            return True
        self.note = f"quick look ({n} keyframes): {self.note}"
        return False

    def could_match(self, look, req):
        """Whether anyone in the video could have the hair and clothes colours asked for: each
        colour must be the name of at least one detection's (a person's colour is the most
        common of theirs). Sets self.note to what was missing."""
        palettes = self.config["colours"]
        names = {}
        for part, palette in (("hair", "hair"), ("top", "clothes"), ("bottom", "clothes")):
            names[part] = {people.colour_name(x, palettes[palette])[0] for x in look[part] if not np.isnan(x).any()}
        names["clothes"] = names["top"] | names["bottom"]
        for part in LOOKS:
            if req[part] and not (req[part] & names[part]):
                self.note = f"no {'/'.join(sorted(req[part]))} {part} seen"
                return False
        return True

    def voices(self, v, poses):
        """({detection: its own voice}, {"f0", "names"}) from the talking rules and the sound."""
        own, voices = {}, None
        extra = self.inputs(v, poses, "talking")
        _, heard = ds.label_people(poses, "talking", extra)
        for frame in heard:
            for p in frame:
                voice = p["m"].get("voice", "")
                if p["state"] == "talking" and len(voice) == 1:
                    own[p["j"]] = ord(voice) - ord("A")
        sound = extra["audio"]
        if sound is not None and "voice_f0" in sound:
            names = people.name_voices(sound["voice_centroid"], sound["voice_f0"], self.enrol(), self.config)
            voices = {"f0": sound["voice_f0"], "names": names}
        return own, voices

    def anyone(self, v, poses, action):
        """Anyone doing the action: the detector's own sections (as detect_standing.py finds them,
        still figures dropped), one row each with everyone doing it boxed."""
        if action in (None, "visible"):
            raise ValueError("a search needs a person or an action")
        with self.timed(action):
            extra = self.inputs(v, poses, action)
        rules = ds.ACTIONS[action]
        _, sections, in_intro, _ = ds.detect(v, poses, action, rules["min_frames"], rules["max_gap"],
                                             self.interview_start(v), extra)
        for s in in_intro:
            s["in_intro"] = True
        m = ds.NAME_RE.match(v.stem)
        url = f"https://www.youtube.com/watch?v={m.group(2)}" if m else ""
        rows = []
        for s in sorted((in_intro if self.label_intros else []) + sections, key=lambda s: s["start_s"]):
            boxes = [p["box"] for p in s["best_people"] if p["state"] == action]
            n = s[f"max_people_{action}"]
            rows.append({"video": v.name, "person": "", "start_s": s["start_s"], "end_s": s["end_s"], "start": s["start"],
                         "end": s["end"], "duration_s": s["duration_s"], "action": action, "match": "match",
                         "evidence": f"{n} {'person' if n == 1 else 'people'} {action}", "youtube":
                         f"{url}&t={int(s['start_s'])}s" if url else "", "best_t": s["best_t"], "best_j": None,
                         "best_box": [float(x) for x in max(boxes, key=lambda b: float(b[2] * b[3]))],
                         "best_boxes": [[float(x) for x in b] for b in boxes], "in_intro": s.get("in_intro", False)})
        if not rows:
            self.note = f"nobody {action}"
        return rows

    def states(self, v, poses, track, action):
        """(timeline, [(t, track, detection, doing it, box)] for every sample of a tracked person;
        detection indexes the 1-fps poses, box is where they are in that sample)."""
        if action in (None, "visible"):
            return poses, [(float(poses["t"][f]), int(track[j]), j, True, poses["box"][j]) for j, f in enumerate(poses["frame"])]
        extra = self.inputs(v, poses, action)
        timeline, by_frame = ds.label_people(poses, action, extra)
        out = []
        if action not in ds.DENSE_ACTIONS:
            for t, frame in zip(timeline["t"].tolist(), by_frame):
                out += [(t, int(track[p["j"]]), p["j"], p["state"] == action, p["box"]) for p in frame]
            return timeline, out
        # 5-fps samples: each detection belongs to the track of the nearest 1-fps sample's box
        fps = float(poses["fps"])
        for t, frame in zip(timeline["t"].tolist(), by_frame):
            f = int(round(t * fps))
            near = np.flatnonzero(poses["frame"] == min(f, len(poses["t"]) - 1))
            for p in frame:
                best = max(near, key=lambda j: people._iou(people._xyxy(poses["box"][j]), people._xyxy(p["box"])), default=None)
                if best is not None and people._iou(people._xyxy(poses["box"][best]), people._xyxy(p["box"])) >= 0.2:
                    out.append((t, int(track[best]), best, p["state"] == action, p["box"]))
        return timeline, out

    def interview_start(self, v):
        """Where the interview starts (0 when intros are kept), found once per video."""
        if v.name not in self.starts:
            self.starts[v.name] = ds.find_interview_start(v, self.bank)["interview_start_s"]
        return self.starts[v.name]

    def sections(self, v, poses, timeline, states, wanted, action):
        rules = ds.ACTIONS.get(action, {"min_frames": 1, "max_gap": 2}) if action not in (None, "visible") else {"min_frames": 1, "max_gap": 2}
        fps = float(timeline["fps"])
        start = self.interview_start(v)
        since = 0 if self.label_intros else start
        m = ds.NAME_RE.match(v.stem)
        url = f"https://www.youtube.com/watch?v={m.group(2)}" if m else ""
        rows = []
        for tr, (verdict, why) in wanted.items():
            times = sorted(((t, j, box) for t, k, j, doing, box in states if k == tr and doing and t >= since),
                           key=lambda x: x[0])
            cur = None
            for t, j, box in times:
                if cur and t - cur["last"] <= (rules["max_gap"] + 1) / fps:
                    cur["last"], cur["n"] = t, cur["n"] + 1
                    cur["samples"].append((t, j, box))
                else:
                    cur = {"first": t, "last": t, "n": 1, "samples": [(t, j, box)]}
                    rows.append((tr, verdict, why, cur))
        out = []
        for tr, verdict, why, s in rows:
            if s["n"] < rules["min_frames"]:
                continue
            end = s["last"] + 1 / fps
            best_t, best_j, best_box = max(s["samples"], key=lambda x: float(x[2][2] * x[2][3]))  # the largest view
            out.append({"video": v.name, "person": tr, "start_s": round(s["first"], 3), "end_s": round(end, 3),
                        "start": ds.hms(s["first"]), "end": ds.hms(end), "duration_s": round(end - s["first"], 1),
                        "action": action or "visible", "match": verdict, "evidence": why,
                        "youtube": f"{url}&t={int(s['first'])}s" if url else "", "best_t": best_t, "best_j": best_j,
                        "best_box": [float(x) for x in best_box], "in_intro": bool(s["first"] < start)})
        return sorted(out, key=lambda r: (r["start_s"], str(r["person"])))


# ---- output ----------------------------------------------------------------------------------

def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "search"


def write(rows, query, req, videos, out_dir):
    """matches.csv, index.html and thumbnails (the person boxed) in out_dir."""
    by_name = {v.name: v for v in videos}
    for r in rows:
        r["thumb"] = thumbnail(by_name[r["video"]], r, out_dir / "thumbs")
    with open(out_dir / "matches.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, FIELDS, extrasaction="ignore", lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    cards = "\n".join(
        f'<figure><a href="thumbs/{html.escape(r["thumb"])}"><img src="thumbs/{html.escape(r["thumb"])}" loading="lazy"></a>'
        f'<figcaption>{html.escape(r["video"])} person {r["person"]}<br>{r["start"]} to {r["end"]} ({r["duration_s"]}s, '
        f'{r["action"]}, {r["match"]})<br><small>{html.escape(r["evidence"])}</small>'
        + (f' · <a href="{html.escape(r["youtube"])}">YouTube</a>' if r["youtube"] else "") + "</figcaption></figure>"
        for r in rows)
    ignored = f"<p>Ignored words: {html.escape(', '.join(req['ignored']))}</p>" if req["ignored"] else ""
    (out_dir / "index.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>{html.escape(query)}</title>"
        "<style>body{font:14px system-ui;margin:16px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}"
        "figure{margin:0}img{width:100%;border-radius:4px}</style>"
        f"<h1>{html.escape(query)}</h1><p>Looking for: {html.escape(describe_query(req))}; action: "
        f"{html.escape(req['action'] or 'any (in view)')}. {len(rows)} results.</p>{ignored}<main>{cards}</main>")


def thumbnail(v, r, folder):
    """Save a result's thumbnail (its best sample, 3x enlarged, the person boxed) in folder;
    returns its file name."""
    folder.mkdir(parents=True, exist_ok=True)
    w, h = ds.probe(v)
    big = cv2.resize(_frame_at(v, r["best_t"], w, h), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    for x, y, bw, bh in r.get("best_boxes") or [r["best_box"]]:
        cv2.rectangle(big, (int(3 * x), int(3 * y)), (int(3 * (x + bw)), int(3 * (y + bh))), (60, 200, 60), 2)
    name = f"{v.stem}_{int(r['start_s']):05d}_{r['person']}.jpg"
    cv2.imwrite(str(folder / name), big)
    return name


SCOUT_EVERY_S = 4   # the quick look takes at most one keyframe every 4 s


def keyframes(path, w, h):
    """Yield (index, BGR) for the video's keyframes, at most one every SCOUT_EVERY_S. Only the
    keyframes are read and decoded (YouTube's come every few seconds): on the 27-minute test
    video, 1.8 s against 10 s for decoding every frame as read_frames() must (and 19 s with
    -skip_frame, which still hands every frame to the decoder)."""
    yield from ds._decode(["ffmpeg", "-v", "error", "-discard", "nokey", "-i", str(path), "-an", "-vf",
                           f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{SCOUT_EVERY_S})'",
                           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], 1.0, w, h)


def _frame_at(v, t, w, h):
    """The 1-fps sample at t (read_frames' picture), or the 5-fps one for fractional times."""
    fps = 1.0 if abs(t - round(t)) < 1e-6 else actions.JUMP_FPS
    return ds.grab_frame(v, t, fps, w, h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", help='who to look for, e.g. "woman with purple hair talking"')
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--action", choices=list(ds.ACTIONS) + ["visible"],
                    help="what they should be doing (default: from the query, else any time they are in view)")
    ap.add_argument("--out", type=Path, default=HERE / "output", help="detect_standing.py's --out (saved model output)")
    ap.add_argument("--results", type=Path, default=HERE.parent / "results" / "search")
    ap.add_argument("--config", type=Path, help="your people.toml (merged over the defaults)")
    ap.add_argument("--possible", action="store_true", help="also list people who might match (some part not seen)")
    ap.add_argument("--keep-intro", action="store_true", help="search the intros too")
    ap.add_argument("--explain", action="store_true", help="print what was seen of every person and why they matched or not")
    ap.add_argument("--quick", action="store_true", help="with a colour in the query, look at each video's keyframes first "
                    "and skip the video if nobody there has it (can miss someone on screen for under about 10 s)")
    ap.add_argument("--intro-bank", type=Path, default=intro.DEFAULT_BANK)
    ap.add_argument("--model", type=Path, default=ds.DEFAULT_MODEL)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--jump-fps", type=float, default=actions.JUMP_FPS)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--runtime", choices=["auto", "openvino", "onnxruntime"], default="auto")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    args.force = False
    args.videos = [v for v in args.videos if not re.search(r"\.f\d+$", v.stem)]
    config = people.load_config(args.config)
    names = [p["name"] for p in config.get("people", [])]
    req = parse(args.query, config, names)
    action = args.action or req["action"]
    req["action"] = action
    print(f"looking for: {describe_query(req)}; action: {action or 'any (in view)'}"
          + (f"; ignored words: {', '.join(req['ignored'])}" if req["ignored"] else ""), flush=True)
    if not any(req[k] for k in PARTS) and not action:
        sys.exit("the query names nobody and no action: nothing to search for")
    searcher = Searcher(args, config)
    rows = []
    for i, v in enumerate(args.videos, 1):
        try:
            found = searcher.search(v, req, action) if not args.quick or searcher.scout(v, req) else []
        except Exception as e:  # a bad file should not stop the search
            print(f"[{i}/{len(args.videos)}] {v.name}: FAILED {e}", file=sys.stderr, flush=True)
            continue
        rows += found
        print(f"[{i}/{len(args.videos)}] {v.name}: {len(found)} results"
              + (f" ({searcher.note})" if searcher.note and not found else ""), flush=True)
    rows.sort(key=lambda r: (r["match"] != "match", r["video"], r["start_s"]))
    out_dir = args.results / slug(args.query if not args.action or args.action in args.query.lower()
                                  else f"{args.query} {args.action}")
    write(rows, args.query, req, args.videos, out_dir)
    print(f"{len(rows)} results ({sum(r['match'] == 'match' for r in rows)} matches) -> {out_dir / 'matches.csv'}, "
          f"{out_dir / 'index.html'}")
    if searcher.unsure:
        print(f"{searcher.unsure} more people might match (not seen well enough to tell); --possible lists them, "
              "--explain says why")


if __name__ == "__main__":
    main()
