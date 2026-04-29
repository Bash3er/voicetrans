import re
import numpy as np

FILLERS = {
    "uh", "uhh", "uhhhh", "um", "umm", "hmm", "hm", "ah", "ahh",
    "er", "erm", "like", "you know", "okay so", "so yeah",
}


def clean_text(text: str) -> str:
    t = text.strip()
    t = re.sub(r"[.!?,;]{2,}", ".", t)
    for filler in FILLERS:
        t = re.sub(r"\b" + re.escape(filler) + r"\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s{2,}", " ", t).strip()
    if len(t) < 2 or not any(c.isalpha() for c in t):
        return ""
    return t


def fix_grammar(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    t = re.sub(r"([.!?])\s*([a-z])", lambda m: m.group(1) + " " + m.group(2).upper(), t)
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t


def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return audio
    ratio = to_sr / from_sr
    new_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
