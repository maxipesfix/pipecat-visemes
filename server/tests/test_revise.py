"""Revising held keyframes once word timings arrive, and anchoring them on onset."""

import unittest

from lipsync.base_lipsync_analyzer import (
    BaseLipsyncAnalyzer,
    LipsyncAnalysisContext,
    LipsyncFrameResult,
)
from lipsync.formant_lipsync_analyzer import FormantLipsyncAnalyzer
from lipsync.frames import TTSLipsyncFrame
from lipsync.lipsync_processor import LipsyncProcessor, _Context
from lipsync.text_events import ONSET_MAX_SHIFT, TextEvents
from lipsync.text_prior import TextWord
from lipsync.types import LipsyncKeyframe
from tests.test_text_events import prior


def kf(offset, energy=0.8, rounding=0.0, width=0.8):
    return LipsyncKeyframe(offset, 0.3, width, rounding, energy, 0.5, 0.2)


class TestOnsetAnchoring(unittest.TestCase):
    def spans(self, onset):
        model = TextEvents()
        text = prior("Bob sees.", TextWord("Bob", 0), TextWord("sees", 300_000_000))
        model.prepare(text, 0.0, onset)
        return [(s.phone, round(s.start, 3)) for s in model._spans]

    def test_word_times_are_anchored_on_the_speech_onset(self):
        self.assertEqual(self.spans(None)[0], ("B", 0.0))
        self.assertEqual(self.spans(0.1)[0], ("B", 0.1))
        # The whole word moves with its onset: B AA B, 0.1 s each.
        self.assertEqual(self.spans(0.1), [("B", 0.1), ("AA", 0.2), ("B", 0.3)])

    def test_an_implausible_onset_is_ignored(self):
        self.assertEqual(self.spans(ONSET_MAX_SHIFT + 0.1), self.spans(None))


class TestReviseKeyframes(unittest.IsolatedAsyncioTestCase):
    async def revise(self, keyframes, start=0.0):
        analyzer = FormantLipsyncAnalyzer(text_events_enabled=True)
        await analyzer.start(16_000)
        text = prior("Who sees.", TextWord("Who", 0), TextWord("sees", 400_000_000))
        context = LipsyncAnalysisContext("test", 16_000, text_prior=text)
        return analyzer.revise_keyframes(context, keyframes, start)

    async def test_rounds_the_loud_core_of_a_timed_rounded_word(self):
        # "Who" spans 0.0-0.4: a quiet /h/, then a loud vowel at 0.1-0.3.
        held = [kf(0.0, energy=0.2), kf(0.1), kf(0.2), kf(0.3), kf(0.45), kf(0.6)]
        added = await self.revise(held)
        for k in held[1:4]:
            self.assertGreaterEqual(k.rounding, 0.7, k.offset)
            self.assertLessEqual(k.width, 0.25, k.offset)
        # The /h/ and the next word keep their shape.
        for k in (held[0], held[4], held[5]):
            self.assertEqual((k.rounding, k.width), (0.0, 0.8), k.offset)
        # Guards just outside the stretch stop client interpolation from
        # carrying the rounding into the neighbours.
        outside = [k for k in added if k.offset < 0.1 or k.offset >= 0.32]
        self.assertTrue(outside)
        self.assertTrue(all(k.rounding == 0.0 for k in outside))

    async def test_released_keyframes_are_out_of_reach(self):
        held = [kf(0.25), kf(0.3), kf(0.45)]
        added = await self.revise(held, start=0.25)
        self.assertTrue(all(k.offset >= 0.25 for k in added))

    async def test_without_the_text_tier_nothing_changes(self):
        analyzer = FormantLipsyncAnalyzer()
        await analyzer.start(16_000)
        held = [kf(0.1), kf(0.2)]
        context = LipsyncAnalysisContext("test", 16_000)
        self.assertEqual(analyzer.revise_keyframes(context, held, 0.0), [])
        self.assertEqual([k.rounding for k in held], [0.0, 0.0])


class _StubAnalyzer(BaseLipsyncAnalyzer):
    """Records what it is asked to revise and adds fixed keyframes."""

    def __init__(self, add):
        self.add = add
        self.seen: list[float] = []

    async def start(self, sample_rate):
        pass

    async def analyze(self, pcm, context):
        return LipsyncFrameResult()

    async def flush(self, context):
        return LipsyncFrameResult()

    async def reset(self):
        pass

    def revise_keyframes(self, context, keyframes, start):
        self.seen = [k.offset for k in keyframes]
        self.start = start
        return [kf(t) for t in self.add]


class TestProcessorRevision(unittest.TestCase):
    def test_held_keyframes_are_revised_and_additions_land_in_their_batch(self):
        analyzer = _StubAnalyzer(add=[0.15, 0.25])
        processor = LipsyncProcessor(analyzer=analyzer)
        context = _Context("ctx")
        context.window_start = 0.2
        context.pending_keyframes = [kf(0.22)]
        held = TTSLipsyncFrame(
            context_id="ctx", window_start=0.0, window_end=0.2, keyframes=[kf(0.05), kf(0.18)]
        )
        other = TTSLipsyncFrame(context_id="other", window_start=0.0, window_end=0.2)
        processor._scheduled = [(1, 1, held), (2, 2, other)]

        processor._revise_held(context)

        # Only this context's held keyframes, from its oldest held window on.
        self.assertEqual(analyzer.seen, [0.05, 0.18, 0.22])
        self.assertEqual(analyzer.start, 0.0)
        self.assertEqual([k.offset for k in held.keyframes], [0.05, 0.15, 0.18])
        self.assertEqual([k.offset for k in context.pending_keyframes], [0.22, 0.25])
        self.assertEqual(other.keyframes, [])
