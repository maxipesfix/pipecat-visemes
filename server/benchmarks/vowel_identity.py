"""Vowel-identity scoring: does the signal round /u o/ and spread /i e/?

L2 correlates openness and width against per-clip-normalized Praat F1/F2,
which bakes in the analyzer's own 1-D mapping and cannot see rounding at all;
the vowel-probe bars only ask that *some* hop in a rounded sentence round.
Neither notices a voice whose /i/ renders rounded or whose /o/ renders
spread. This scores the rendered parameters against the vowel each word
should carry instead — phone identity is voice-independent, and the fixtures
already hold the TTS word timings, so no forced aligner is needed.

Only single-vowel dictionary words are labelled, so the vowel is the word's
nucleus without guessing stress, and only the loud core of each word is
scored (Praat-voiced hops at least half the word's peak energy), which keeps
consonant transitions and loose word boundaries out. The score is an AUC
pooled per voice: the probability that a hop of a rounded vowel reads more
rounded than a hop of a spread one (0.5 = no information, 1.0 = perfect).
"""

from dataclasses import dataclass

import numpy as np

from lipsync.pronunciation import pronounce, tokenize

# Rounded vs spread for `rounding`; back vs front for `width`. AO, AA, AH,
# ER and the diphthongs are left out: their rounding is dialect-dependent
# (AO/AA merge in most American English) or moves within the vowel.
ROUNDED_VOWELS = frozenset(("UW", "UH", "OW"))
SPREAD_VOWELS = frozenset(("IY", "IH", "EY", "EH", "AE"))
BACK_VOWELS = frozenset(("UW", "UH", "OW", "AO", "AA"))
FRONT_VOWELS = frozenset(("IY", "IH", "EY", "EH", "AE"))
VOWELS = frozenset("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())

# A hop is in a word's vowel when it is at least this loud relative to the
# word's own peak.
_CORE_FRACTION = 0.5


@dataclass
class VowelHops:
    """Rendered parameters on labelled vowel hops of one clip."""

    rounded: list[float]  # rounding on rounded-vowel hops
    spread: list[float]  # rounding on spread-vowel hops
    back: list[float]  # width on back-vowel hops
    front: list[float]  # width on front-vowel hops

    def extend(self, other: "VowelHops"):
        self.rounded += other.rounded
        self.spread += other.spread
        self.back += other.back
        self.front += other.front


def word_vowel(word: str) -> str | None:
    """The vowel phone of a single-vowel dictionary word, else None."""
    tokens = tokenize(word)
    if len(tokens) != 1:
        return None
    pron = pronounce(tokens[0])
    if not pron.certain:
        return None
    vowels = [p for p in pron.phones if p in VOWELS]
    return vowels[0] if len(vowels) == 1 else None


def word_spans(text_timing: dict | None, duration: float) -> list[tuple[str, float, float]]:
    """(vowel, start, end) for each labelled word; a word ends where the next starts."""
    if not text_timing:
        return []
    words = [(w, pts) for w, pts, _ in text_timing.get("words", []) if pts is not None]
    spans = []
    for i, (word, start) in enumerate(words):
        end = words[i + 1][1] if i + 1 < len(words) else duration
        vowel = word_vowel(word)
        if vowel and end > start:
            spans.append((vowel, start, end))
    return spans


def vowel_hops(
    spans: list[tuple[str, float, float]],
    offsets: np.ndarray,
    rms: np.ndarray,
    voiced: np.ndarray,
    rounding: np.ndarray,
    width: np.ndarray,
) -> VowelHops:
    """Collect rendered rounding/width on the loud voiced core of each word."""
    out = VowelHops([], [], [], [])
    for vowel, start, end in spans:
        inside = (offsets >= start) & (offsets < end) & voiced
        if inside.sum() == 0:
            continue
        core = inside & (rms >= _CORE_FRACTION * rms[inside].max())
        if vowel in ROUNDED_VOWELS:
            out.rounded += rounding[core].tolist()
        elif vowel in SPREAD_VOWELS:
            out.spread += rounding[core].tolist()
        if vowel in BACK_VOWELS:
            out.back += width[core].tolist()
        elif vowel in FRONT_VOWELS:
            out.front += width[core].tolist()
    return out


def auc(positive: list[float], negative: list[float]) -> float:
    """P(positive > negative), ties counted half (Mann-Whitney U / n1 n2)."""
    if len(positive) < 8 or len(negative) < 8:
        return float("nan")
    pos, neg = np.asarray(positive), np.asarray(negative)
    ranks = _ranks(np.concatenate([pos, neg]))
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def _ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def scores(hops: VowelHops) -> dict[str, float]:
    """AUCs plus the means behind them, for reporting."""

    def mean(values):
        return float(np.mean(values)) if values else float("nan")

    return {
        "rounding_auc": auc(hops.rounded, hops.spread),
        "width_auc": auc(hops.front, hops.back),
        "rounding_on_rounded": mean(hops.rounded),
        "rounding_on_spread": mean(hops.spread),
        "width_on_front": mean(hops.front),
        "width_on_back": mean(hops.back),
        "rounded_hops": float(len(hops.rounded)),
        "spread_hops": float(len(hops.spread)),
    }
