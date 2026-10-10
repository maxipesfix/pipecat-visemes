"""Per-voice vowel profiles and their calibration."""

import asyncio
from dataclasses import fields

import numpy as np
import pytest

from lipsync.base_lipsync_analyzer import LipsyncAnalysisContext
from lipsync.calibration import CALIBRATION_WORDS, calibration_text, samples_from_debug
from lipsync.dsp import ANALYSIS_SAMPLE_RATE
from lipsync.formant_lipsync_analyzer import FormantLipsyncAnalyzer, LipsyncDebugFrame
from lipsync.text_prior import TextPrior, TextWord
from lipsync.voice_profile import FormantSample, ProfileLearner, VoiceProfile
from tests.synth import synth_vowel

rng = np.random.default_rng(0)


def hops(word, vowel, f1, f2, f3, n=12, f2_missing=0.0):
    """Noisy calibration hops; a fraction of them with F2 unfound (0.0)."""
    out = []
    for i in range(n):
        found = i >= round(n * f2_missing)
        out.append(
            FormantSample(
                word,
                vowel,
                f1 + rng.normal(0, 20),
                f2 + rng.normal(0, 40) if found else 0.0,
                f3 + rng.normal(0, 50),
            )
        )
    return out


# A voice like the bot's: /i/ F2 above the band, /u/ fronted after a tongue-front
# consonant ("new") but not after a lip-closing one ("boo"), told apart by F3.
FRONTED = (
    hops("heed", "IY", 300, 2900, 3050, f2_missing=0.95)
    + hops("new", "UW", 400, 2250, 2550, f2_missing=0.8)
    + hops("boo", "UW", 350, 1600, 2700)
    + hops("head", "EH", 780, 2250, 3100)
    + hops("hod", "AA", 760, 1100, 2950)
)


def test_fronted_oo_and_ee_are_told_apart_by_f3_when_f2_is_missing():
    profile = VoiceProfile.fit(FRONTED)
    # Same F1, no F2 found: only F3 separates "new" from "heed".
    _, round_oo, p_oo = profile.articulation(390.0, 0.0, 2560.0)
    _, round_ee, p_ee = profile.articulation(305.0, 0.0, 3040.0)
    assert p_oo > 0.8 and round_oo > 0.6
    assert p_ee < 0.2 and round_ee < 0.2


def test_back_oo_and_open_vowels_keep_their_shapes():
    profile = VoiceProfile.fit(FRONTED)
    width, rounding, _ = profile.articulation(350.0, 1600.0, 2700.0)  # "boo"
    assert rounding > 0.7 and width < 0.3
    width, rounding, _ = profile.articulation(770.0, 1100.0, 2950.0)  # "hod"
    assert rounding < 0.1


def test_a_word_with_too_few_hops_is_left_out_and_round_trips():
    profile = VoiceProfile.fit(FRONTED + hops("hid", "IH", 400, 2400, 3150, n=2))
    assert "hid" not in {e.word for e in profile.exemplars}
    again = VoiceProfile.from_dict(profile.to_dict())
    assert again.articulation(390.0, 0.0, 2560.0) == pytest.approx(
        profile.articulation(390.0, 0.0, 2560.0)
    )
    with pytest.raises(ValueError):
        VoiceProfile.fit(hops("hid", "IH", 400, 2400, 3150, n=2))


def debug_frames(bursts, gap=8, rms=0.3):
    """Fake debug frames: one voiced burst of speech per (f1, f2, f3, length)."""
    blank = {f.name: 0.0 for f in fields(LipsyncDebugFrame)} | {
        "voiced": False,
        "nasal_active": False,
    }
    frames, t = [], 0.0
    for f1, f2, f3, length in bursts:
        for k in range(length + gap):
            speaking = k < length
            frame = blank | {"offset": t, "rms": rms if speaking else 0.001, "voiced": speaking}
            if speaking:
                frame |= {"f1": f1, "f2": f2, "f3": f3}
            frames.append(LipsyncDebugFrame(**frame))
            t += 0.02
    return frames


def test_calibration_finds_one_burst_per_word():
    words = {"heed": "IY", "boo": "UW", "hod": "AA"}
    debug = debug_frames([(300, 0, 3050, 10), (350, 1600, 2700, 10), (760, 1100, 2950, 10)])
    samples = samples_from_debug(debug, words)
    assert {s.word for s in samples} == set(words)
    assert all(s.f1 == 350 for s in samples if s.word == "boo")


def test_calibration_uses_word_times_when_two_words_run_together():
    words = {"heed": "IY", "boo": "UW", "hod": "AA"}
    debug = debug_frames([(300, 0, 3050, 10), (350, 1600, 2700, 10), (760, 1100, 2950, 10)])
    # Glue the first two bursts: three words, two bursts.
    for d in debug[10:18]:
        d.rms = 0.3
        d.voiced = False
    with pytest.raises(ValueError):
        samples_from_debug(debug, words)
    samples = samples_from_debug(debug, words, word_times=[0.0, 0.2, 0.72])
    assert {"heed", "hod"} <= {s.word for s in samples}
    assert all(s.f1 == 760 for s in samples if s.word == "hod")


def test_calibration_text_lists_every_word():
    text = calibration_text()
    assert text.startswith("Heed. Tea.") and text.count(".") == len(CALIBRATION_WORDS)


async def analyze(pcm, profile=None):
    analyzer = FormantLipsyncAnalyzer(voice_profile=profile)
    await analyzer.start(ANALYSIS_SAMPLE_RATE)
    context = LipsyncAnalysisContext(context_id="t", sample_rate=ANALYSIS_SAMPLE_RATE)
    keyframes = (await analyzer.analyze(pcm, context)).keyframes
    keyframes += (await analyzer.flush(context)).keyframes
    return keyframes


def test_a_profile_rounds_a_fronted_oo_the_f2_mapping_reads_as_spread():
    # A fronted /u/: F2 mid-high, F3 low. The F2 mapping reads it as spread.
    pcm = synth_vowel(380, 2000, f3=2500, secs=1.0)
    plain = asyncio.run(analyze(pcm))
    profile = VoiceProfile.fit(
        hops("two", "UW", 380, 2000, 2500)
        + hops("tip", "IH", 420, 2050, 3100)
        + hops("hod", "AA", 760, 1100, 2950)
    )
    calibrated = asyncio.run(analyze(pcm, profile))

    def late(kfs, name):
        return np.mean([getattr(k, name) for k in kfs if k.offset > 0.4])

    assert late(plain, "rounding") < 0.3
    assert late(calibrated, "rounding") > 0.6
    # Openness and width stay on the formant mapping.
    for name in ("openness", "width"):
        assert late(calibrated, name) == pytest.approx(late(plain, name), abs=0.02)


def prior(words, audio_end=None):
    """A text prior with word times in seconds from the word clock's start."""
    return TextPrior(
        words=tuple(TextWord(w, round(t * 1e9)) for w, t in words),
        word_start_pts=0,
        audio_end=audio_end,
    )


def speak(learner, segments):
    """Observe hops: (start s, end s, f1, f2, f3) voiced stretches at 20 ms."""
    for start, end, f1, f2, f3 in segments:
        for t in np.arange(start, end, 0.02):
            learner.observe(float(t), f1, f2, f3, 0.3)


def test_learner_takes_finished_dictionary_words_but_not_reduced_ones():
    learner = ProfileLearner()
    # Word times run ~0.1 s early: "two" is timed at 0.0 but voiced 0.1-0.4.
    speak(
        learner,
        [(0.1, 0.4, 380, 2000, 2500), (0.6, 0.9, 300, 0, 3050), (1.1, 1.3, 500, 1500, 2600)],
    )
    learner.update(prior([("two", 0.0), ("tea", 0.5), ("to", 1.0)], audio_end=1.3), cursor=2.0)
    assert set(learner._samples) == {"two", "tea"}  # "to" is usually said "tuh"
    assert all(s.vowel == "UW" and s.f1 == 380 for s in learner._samples["two"])
    taken = len(learner._samples["two"])
    # Each word is taken once.
    learner.update(prior([("two", 0.0), ("tea", 0.5), ("to", 1.0)], audio_end=1.3), cursor=3.0)
    assert len(learner._samples["two"]) == taken


def test_learner_waits_for_a_word_to_be_analyzed():
    learner = ProfileLearner()
    speak(learner, [(0.1, 0.4, 380, 2000, 2500)])
    learner.update(prior([("two", 0.0), ("tea", 0.5)]), cursor=0.5)
    assert not learner._samples  # "two" ends at 0.6 (with the shift): not final yet


def test_learner_keeps_calibrated_words_and_adds_new_ones():
    base = VoiceProfile.fit(FRONTED)
    learner = ProfileLearner(base)
    speak(learner, [(0.1, 0.4, 380, 2000, 2500), (0.6, 0.9, 760, 1100, 2950)])
    learner.update(prior([("two", 0.0), ("hod", 0.5)], audio_end=0.8), cursor=2.0)
    words = [e.word for e in learner.profile.exemplars]
    assert words[: len(base.exemplars)] == [e.word for e in base.exemplars]
    assert "two" in words and words.count("hod") == 1


def test_an_uncalibrated_learner_waits_for_enough_words():
    learner = ProfileLearner()
    speak(learner, [(0.1, 0.4, 380, 2000, 2500)])
    learner.update(prior([("two", 0.0)], audio_end=0.4), cursor=2.0)
    assert learner.profile is None  # one rounded word is not a profile


def test_processor_loads_a_profile_and_learning_collects_text(tmp_path):
    from lipsync.lipsync_processor import LipsyncParams, LipsyncProcessor

    path = tmp_path / "voice.json"
    VoiceProfile.fit(FRONTED).save(path)
    plain = LipsyncProcessor()
    assert plain._analyzer._voice_profile is None and not plain._collect_text
    calibrated = LipsyncProcessor(params=LipsyncParams(voice_profile_path=str(path)))
    assert len(calibrated._analyzer._voice_profile.exemplars) == 5
    learning = LipsyncProcessor(params=LipsyncParams(voice_profile_learning=True))
    assert learning._collect_text and learning._analyzer._profile_learner is not None


def test_a_profile_with_hums_tells_a_dark_oo_from_a_hum():
    profile = VoiceProfile.fit(FRONTED + hops("hmm", "M", 280, 0, 2600, f2_missing=0.9))
    assert profile.has_nasal
    assert profile.nasal(285.0, 0.0, 2600.0) > 0.7  # this voice's hum
    assert profile.nasal(350.0, 1600.0, 2700.0) < 0.1  # its "boo"
    # Hums shape no vowel: rounding comes from the vowel exemplars only.
    _, rounding, _ = profile.articulation(350.0, 1600.0, 2700.0)
    assert rounding > 0.7
    assert not VoiceProfile.fit(FRONTED).has_nasal
