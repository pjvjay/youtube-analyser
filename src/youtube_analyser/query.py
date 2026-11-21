"""Reading a search query ("woman with purple hair talking") and checking a person against it.

A query names a person by what they look like (hair and clothes colours, apparent gender and
age), how they sound (a deep or high voice) or who they are (a name enrolled in people.toml,
found by their voice), and optionally an action. The words it understands are in people.toml;
any other word is listed as ignored.

Each person (people.summarise) is checked against every part of the query: yes, no, or unknown
(not seen well enough). Everyone with all yes is a "match"; nothing "no" but something unknown is
"possible".
"""
import re

import numpy as np

from .media import hms

PARTS = ("gender", "age", "hair", "top", "bottom", "clothes", "voice", "name")
LOOKS = ("hair", "top", "bottom", "clothes")        # the parts the appearance pass alone decides


def parse(query, config, names=()):
    """What a query asks for: {gender, age, hair, top, bottom, clothes, voice, name, action,
    ignored}. Colours are sets of palette names (a group or synonym becomes its names)."""
    w = config["words"]
    colours = set(config["colours"]["hair"]) | set(config["colours"]["clothes"])
    req = dict.fromkeys(PARTS)
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
    parts = [f"{hms(person['first_t'])}-{hms(person['last_t'])}, {len(person['detections'])} samples"]
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

