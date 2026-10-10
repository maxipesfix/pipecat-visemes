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
