#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Measure a voice's vowels from a short calibration recording.

The TTS voice speaks :data:`CALIBRATION_WORDS` as a list ("Heed. Tea. Bead.
..."), about 30 s of audio. Each word is its own burst of speech, so words
are found by energy alone: the k-th burst is the k-th word. If the TTS ran
two words together, the provider's word timestamps (when given) assign
bursts instead. Each word's vowel is the loud voiced core of its burst, and
the analyzer's own formant measurements there become the voice's
:class:`~lipsync.voice_profile.VoiceProfile`, so the profile matches what the
analyzer sees live.

The words cover every vowel the mapping targets, each after both a
lip-closing and a tongue-front consonant where it matters: one voice can say
/u/ at two places ("boo" vs "two").
"""

import numpy as np

from lipsync.base_lipsync_analyzer import LipsyncAnalysisContext
from lipsync.dsp import ANALYSIS_SAMPLE_RATE
from lipsync.formant_lipsync_analyzer import FormantLipsyncAnalyzer
from lipsync.voice_profile import FormantSample, VoiceProfile

CALIBRATION_WORDS: dict[str, str] = {
    "heed": "IY", "tea": "IY", "bead": "IY", "knee": "IY",
    "hid": "IH", "tip": "IH", "bid": "IH",
    "hayed": "EY", "day": "EY", "babe": "EY", "date": "EY",
    "head": "EH", "debt": "EH", "bed": "EH",
    "had": "AE", "tap": "AE", "bad": "AE",
    "hod": "AA", "dot": "AA", "pod": "AA",
    "hud": "AH", "dud": "AH", "bud": "AH",
    "hoed": "OW", "toe": "OW", "dough": "OW", "bode": "OW",
    "hood": "UH", "took": "UH", "put": "UH",
    "who'd": "UW", "two": "UW", "new": "UW", "do": "UW", "boo": "UW", "moo": "UW",
}  # fmt: skip

# Bursts: energy above this fraction of the recording's peak, bridging dips
# up to this many hops (stop closures), lasting at least this many hops.
_BURST_FRACTION = 0.1
_BURST_BRIDGE_HOPS = 3
_BURST_MIN_HOPS = 4
# A word's vowel: the part of its burst at least this loud relative to the
# burst's own peak.
_VOWEL_FRACTION = 0.5


def calibration_text(words: dict[str, str] = CALIBRATION_WORDS) -> str:
    """What the TTS should say: the words as a list of one-word sentences."""
    return " ".join(f"{w[0].upper()}{w[1:]}." for w in words)


def _runs_above(x: np.ndarray, threshold: float, bridge: int) -> list[tuple[int, int]]:
    """[start, end) index runs where x >= threshold, bridging short dips."""
    runs, start, gap = [], None, 0
    for i, above in enumerate(x >= threshold):
        if above:
            start = i if start is None else start
            gap = 0
        elif start is not None:
            gap += 1
            if gap > bridge:
                runs.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((start, len(x) - gap))
    return runs


async def measure(pcm: np.ndarray) -> list:
    """Run a fresh analyzer over 16 kHz float32 audio; its per-hop debug frames."""
    analyzer = FormantLipsyncAnalyzer(collect_debug=True)
    await analyzer.start(ANALYSIS_SAMPLE_RATE)
    context = LipsyncAnalysisContext(context_id="calibration", sample_rate=ANALYSIS_SAMPLE_RATE)
    # The analyzer's ingest buffer holds about 2 s per call.
    step = ANALYSIS_SAMPLE_RATE // 2
    for start in range(0, len(pcm), step):
        await analyzer.analyze(pcm[start : start + step], context)
    await analyzer.flush(context)
    return analyzer.debug_features


def samples_from_debug(
    debug: list,
    words: dict[str, str] = CALIBRATION_WORDS,
    word_times: list[float] | None = None,
) -> list[FormantSample]:
    """Each word's vowel-core hops, words found as bursts of speech.

    Args:
        debug: The analyzer's debug frames for the whole recording.
        words: The words spoken, in order, with their vowels.
        word_times: Start time (s) of each word, e.g. TTS word timestamps;
            only used when the burst count does not match the word count.

    Raises:
        ValueError: If words cannot be matched to bursts.
    """
    rms = np.array([d.rms for d in debug])
    offsets = np.array([d.offset for d in debug])
    bursts = [
        r
        for r in _runs_above(rms, _BURST_FRACTION * rms.max(), _BURST_BRIDGE_HOPS)
        if r[1] - r[0] >= _BURST_MIN_HOPS
    ]
    names = list(words)
    if len(bursts) == len(names):
        assigned = list(zip(names, bursts))
    elif word_times is not None and len(word_times) == len(names):
        ends = list(word_times[1:]) + [float(offsets[-1])]
        assigned = []
        for name, start, end in zip(names, word_times, ends):
            burst = max(
                bursts, key=lambda r: min(offsets[r[1] - 1], end) - max(offsets[r[0]], start)
            )
            if min(offsets[burst[1] - 1], end) > max(offsets[burst[0]], start):
                assigned.append((name, burst))
    else:
        raise ValueError(
            f"found {len(bursts)} bursts of speech for {len(names)} words and no word timings"
        )
    samples = []
    for name, (a, b) in assigned:
        burst = rms[a:b]
        for j in range(a, b):
            d = debug[j]
            if d.voiced and rms[j] >= _VOWEL_FRACTION * burst.max():
                samples.append(FormantSample(name, words[name], d.f1, d.f2, d.f3))
    return samples


async def calibrate(
    pcm: np.ndarray,
    words: dict[str, str] = CALIBRATION_WORDS,
    word_times: list[float] | None = None,
) -> VoiceProfile:
    """A voice profile from the calibration words spoken by that voice.

    Args:
        pcm: The recording of :func:`calibration_text`, 16 kHz float32.
        words: The words spoken, in order, with their vowels.
        word_times: Optional start time (s) of each word.
    """
    return VoiceProfile.fit(samples_from_debug(await measure(pcm), words, word_times))
