#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Per-voice vowel profile: where one TTS voice puts its vowels.

The analyzer's default mapping reads lip rounding from F2 alone, assuming a
rounded vowel has the voice's lowest F2. Many TTS voices front /u/: on the
bot's default Cartesia voice, "new" and "two" sit at F2 ≈ 2000-2300 Hz, often
above the F2 band, exactly where its /i/ sits, while "boot" is at ≈ 1500 Hz.
What still tells them apart is F3 (lip rounding lowers it: ≈ 2500 vs 3000 Hz
on that voice), and where *this voice* puts each vowel in each context.

A profile holds one exemplar per calibration word: its median F1, F2 and F3,
and how often F2 went unfound (above the band, the /i/ signature). Each hop's
formants are scored against every exemplar with one pooled covariance, and
the posterior-weighted articulation of the exemplars' vowels gives a width
and rounding target. Exemplars are per *word*, not per vowel, because one
voice can say the same vowel in two places ("who'd" vs "boo").

Profiles are measured from the voice itself (see :mod:`lipsync.calibration`),
not trained across voices; numpy only, a few hundred floats per voice.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# (width, rounding) per vowel, the articulatory targets of
# plans/experiments/l4_proto.py. Openness stays on the F1 mapping.
VOWEL_TARGETS: dict[str, tuple[float, float]] = {
    "IY": (0.95, 0.0),
    "IH": (0.8, 0.0),
    "EY": (0.8, 0.0),
    "EH": (0.7, 0.0),
    "AE": (0.65, 0.0),
    "AA": (0.5, 0.0),
    "AH": (0.5, 0.0),
    "OW": (0.2, 0.7),
    "UH": (0.15, 0.85),
    "UW": (0.15, 0.85),
}
ROUNDED_VOWELS = frozenset(("OW", "UH", "UW"))

# F2 assumed for an exemplar whose F2 was never found: above the F2 band.
_F2_ABOVE_BAND_HZ = 2900.0
# Floor on the pooled per-formant spread (Hz), so a tight calibration does not
# make every other hop look impossibly far from all exemplars.
_SPREAD_FLOOR_HZ = (25.0, 50.0, 60.0)
# An exemplar needs at least this many voiced hops to be kept.
MIN_EXEMPLAR_HOPS = 4


@dataclass(frozen=True)
class VowelExemplar:
    """One calibration word as this voice says it.

    Parameters:
        word: The calibration word.
        vowel: Its vowel (ARPAbet, a key of ``VOWEL_TARGETS``).
        f1: Median F1 in Hz.
        f2: Median F2 in Hz over hops where F2 was found; ``_F2_ABOVE_BAND_HZ``
            if it never was.
        f3: Median F3 in Hz.
        f2_missing: Smoothed fraction of hops with no F2 found.
    """

    word: str
    vowel: str
    f1: float
    f2: float
    f3: float
    f2_missing: float


@dataclass(frozen=True)
class FormantSample:
    """One voiced hop of a calibration word: raw formants, 0.0 = not found."""

    word: str
    vowel: str
    f1: float
    f2: float
    f3: float


class VoiceProfile:
    """Exemplar vowels of one voice, scored with a pooled covariance."""

    def __init__(self, exemplars: list[VowelExemplar], covariance: np.ndarray, f3_fill: float):
        if not exemplars:
            raise ValueError("a voice profile needs at least one exemplar")
        self.exemplars = exemplars
        self.covariance = np.asarray(covariance, dtype=float)
        self.f3_fill = float(f3_fill)
        self._means = np.array([[e.f1, e.f2, e.f3] for e in exemplars])
        self._inv = np.linalg.inv(self.covariance)
        self._sd1 = float(np.sqrt(self.covariance[0, 0]))
        self._sd3 = float(np.sqrt(self.covariance[2, 2]))
        self._log_found = np.log([1.0 - e.f2_missing for e in exemplars])
        self._log_missing = np.log([e.f2_missing for e in exemplars])
        self._targets = np.array([VOWEL_TARGETS[e.vowel] for e in exemplars])
        self._rounded = np.array([e.vowel in ROUNDED_VOWELS for e in exemplars])

    @classmethod
    def fit(cls, samples: list[FormantSample]) -> "VoiceProfile":
        """Build a profile from calibration hops (see :mod:`lipsync.calibration`)."""
        by_word: dict[str, list[FormantSample]] = {}
        for s in samples:
            if s.f1 > 0.0 and s.vowel in VOWEL_TARGETS:
                by_word.setdefault(s.word, []).append(s)
        exemplars, residuals, f3s = [], [], []
        for word, hops in by_word.items():
            if len(hops) < MIN_EXEMPLAR_HOPS:
                continue
            f1 = np.array([h.f1 for h in hops])
            f2 = np.array([h.f2 for h in hops])
            f3 = np.array([h.f3 for h in hops])
            found = f2 > 0.0
            has_f3 = f3 > 0.0
            f3s.extend(f3[has_f3])
            mean = (
                float(np.median(f1)),
                float(np.median(f2[found])) if found.sum() >= 2 else _F2_ABOVE_BAND_HZ,
                float(np.median(f3[has_f3])) if has_f3.any() else 0.0,
            )
            exemplars.append(
                VowelExemplar(
                    word=word,
                    vowel=hops[0].vowel,
                    f1=mean[0],
                    f2=mean[1],
                    f3=mean[2],
                    f2_missing=float((np.mean(~found) + 0.05) / 1.1),
                )
            )
            full = found & has_f3
            if full.sum() >= 2:
                residuals.append(np.c_[f1[full], f2[full], f3[full]] - np.array(mean))
        if not exemplars:
            raise ValueError("no calibration word had enough voiced hops")
        f3_fill = float(np.median(f3s)) if f3s else 2700.0
        exemplars = [
            e if e.f3 > 0.0 else VowelExemplar(e.word, e.vowel, e.f1, e.f2, f3_fill, e.f2_missing)
            for e in exemplars
        ]
        pooled = np.cov(np.vstack(residuals).T) if residuals else np.zeros((3, 3))
        return cls(exemplars, pooled + np.diag(_SPREAD_FLOOR_HZ) ** 2, f3_fill)

    def posterior(self, f1: float, f2: float, f3: float) -> np.ndarray:
        """Probability of each exemplar for one hop; f2 = 0.0 means not found."""
        f3 = f3 if f3 > 0.0 else self.f3_fill
        if f2 > 0.0:
            d = np.array([f1, f2, f3]) - self._means
            ll = -0.5 * np.einsum("ij,jk,ik->i", d, self._inv, d) + self._log_found
        else:
            z1 = (f1 - self._means[:, 0]) / self._sd1
            z3 = (f3 - self._means[:, 2]) / self._sd3
            ll = -0.5 * (z1 * z1 + z3 * z3) + self._log_missing
        p = np.exp(ll - ll.max())
        return p / p.sum()

    def articulation(self, f1: float, f2: float, f3: float) -> tuple[float, float, float]:
        """(width, rounding, probability the vowel is rounded) for one hop."""
        p = self.posterior(f1, f2, f3)
        width, rounding = p @ self._targets
        return float(width), float(rounding), float(p[self._rounded].sum())

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "exemplars": [asdict(e) for e in self.exemplars],
            "covariance": self.covariance.tolist(),
            "f3_fill": self.f3_fill,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceProfile":
        if data.get("version") != 1:
            raise ValueError(f"unsupported voice profile version {data.get('version')!r}")
        return cls(
            [VowelExemplar(**e) for e in data["exemplars"]],
            np.array(data["covariance"]),
            data["f3_fill"],
        )

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "VoiceProfile":
        return cls.from_dict(json.loads(Path(path).read_text()))


# Online learning: word timestamps (TTS word clock) run early against the audio
# by about this much on Cartesia, enough to put a word's window on the
# previous word's vowel.
_ONLINE_WORD_SHIFT_SEC = 0.1
# A word is taken once the analysis is this far past its end (the hops there
# are final).
_ONLINE_SETTLE_SEC = 0.1
# Most recent hops kept per learned word, and the minimum audio between refits.
_ONLINE_MAX_HOPS_PER_WORD = 60
_ONLINE_REFIT_SEC = 1.0
# A profile learned without calibration is used only once it has this many
# words, with both rounded and unrounded vowels among them.
_ONLINE_MIN_WORDS = 6
# Hops kept for the current utterance (about 8 s at 20 ms).
_ONLINE_MAX_HOPS = 400


def _single_vowel(word: str) -> str | None:
    """The vowel of a single-vowel dictionary word with a profile target, else None."""
    from lipsync.pronunciation import pronounce, tokenize

    tokens = tokenize(word)
    if len(tokens) != 1 or tokens[0] in _REDUCIBLE:
        return None
    pron = pronounce(tokens[0])
    if not pron.certain:
        return None
    vowels = [p for p in pron.phones if p in VOWEL_TARGETS or p in _OTHER_VOWELS]
    return vowels[0] if len(vowels) == 1 and vowels[0] in VOWEL_TARGETS else None


_OTHER_VOWELS = frozenset(("AO", "AW", "AY", "ER", "OY"))
# Function words usually spoken reduced (to -> "tuh"): their dictionary vowel
# is not what is in the audio, so they would teach the profile that schwa is
# rounded, or spread.
_REDUCIBLE = frozenset(
    "a an the to of and or for from at as in on by is it its be been was were are am "
    "you your do does did would could should can will just than then that this them "
    "he she we me my us our her his him has have had but so if not".split()
)


class ProfileLearner:
    """Refines a voice profile from the bot's own words as it speaks.

    The TTS's word timestamps label the audio: each finished single-vowel
    dictionary word ("you", "two", "see") adds its loud voiced hops under that
    word, and the profile is refit from them. Calibration exemplars stay as
    measured (their labels are cleaner); learned words are added beside them.
    Without a calibration profile, the learned one is used once it covers
    enough words. One learner per voice session, carried across utterances.
    """

    def __init__(self, base: VoiceProfile | None = None):
        self.base = base
        self.profile = base
        self.words_learned = 0
        self._samples: dict[str, list[FormantSample]] = {}
        self._hops: list[tuple[float, float, float, float, float]] = []
        self._taken = 0
        self._last_refit: float | None = None
        self._audio_seen = 0.0
        self._utterance_start = 0.0

    def begin_utterance(self):
        """A new TTS context: offsets restart at zero."""
        self._audio_seen += self._utterance_start
        self._utterance_start = 0.0
        self._hops.clear()
        self._taken = 0

    def observe(self, offset: float, f1: float, f2: float, f3: float, rms: float):
        """One voiced hop with a measured F1 (raw formants, 0.0 = not found)."""
        self._hops.append((offset, f1, f2, f3, rms))
        if len(self._hops) > _ONLINE_MAX_HOPS:
            del self._hops[: len(self._hops) - _ONLINE_MAX_HOPS]
        self._utterance_start = max(self._utterance_start, offset)

    def update(self, prior, cursor: float):
        """Take the words whose audio has been analyzed; refit if due."""
        if prior is None or prior.word_start_pts is None:
            return
        times = [(w.text, (w.pts - prior.word_start_pts) / 1e9) for w in prior.words]
        added = False
        for i in range(self._taken, len(times)):
            end = times[i + 1][1] if i + 1 < len(times) else prior.audio_end
            if end is None or end + _ONLINE_WORD_SHIFT_SEC + _ONLINE_SETTLE_SEC > cursor:
                break
            self._taken = i + 1
            text, start = times[i]
            vowel = _single_vowel(text)
            if vowel is None:
                continue
            lo, hi = start + _ONLINE_WORD_SHIFT_SEC, end + _ONLINE_WORD_SHIFT_SEC
            inside = [h for h in self._hops if lo <= h[0] < hi]
            if not inside:
                continue
            peak = max(h[4] for h in inside)
            word = text.strip(".,!?;:\"'").lower()
            kept = self._samples.setdefault(word, [])
            kept += [
                FormantSample(word, vowel, h[1], h[2], h[3]) for h in inside if h[4] >= 0.5 * peak
            ]
            del kept[: max(0, len(kept) - _ONLINE_MAX_HOPS_PER_WORD)]
            self.words_learned += 1
            added = True
        now = self._audio_seen + cursor
        if added and (self._last_refit is None or now - self._last_refit >= _ONLINE_REFIT_SEC):
            self._last_refit = now
            self._refit()

    def _refit(self):
        samples = [s for kept in self._samples.values() for s in kept]
        try:
            learned = VoiceProfile.fit(samples)
        except ValueError:
            return
        if self.base is None:
            vowels = {e.vowel for e in learned.exemplars}
            if (
                len(learned.exemplars) < _ONLINE_MIN_WORDS
                or not vowels & ROUNDED_VOWELS
                or not vowels - ROUNDED_VOWELS
            ):
                return
            self.profile = learned
            return
        known = {e.word for e in self.base.exemplars}
        self.profile = VoiceProfile(
            self.base.exemplars + [e for e in learned.exemplars if e.word not in known],
            self.base.covariance,
            self.base.f3_fill,
        )
