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
from lipsync.text_prior import TextAnchor, TextPrior, TextWord
from lipsync.types import LipsyncEventKind, LipsyncKeyframe
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


def counting(*words):
    """A streamed turn: three sentence anchors in one TTS context."""
    return TextPrior(
        anchors=(TextAnchor("Sure!"), TextAnchor("One, two, three."), TextAnchor("Bob.")),
        words=tuple(words),
        word_start_pts=0,
    )


TIMED = (
    TextWord("Sure", 0),
    TextWord("One", 500_000_000),
    TextWord("two", 1_000_000_000),
    TextWord("three", 1_400_000_000),
    TextWord("Bob", 2_000_000_000),
)


class TestStreamedTurns(unittest.TestCase):
    def test_sentences_are_aligned_by_their_first_word(self):
        model = TextEvents()
        model.prepare(counting(*TIMED))
        self.assertEqual(model.segment_at(0.2).words, ("sure",))
        self.assertEqual(model.segment_at(1.1).words, ("one", "two", "three"))
        self.assertEqual(model.segment_at(2.1).words, ("bob",))

    def test_a_sentence_without_timing_gets_no_inventory(self):
        model = TextEvents()
        model.prepare(counting(*TIMED[:1]))
        # "Sure" is timed but its end is not: nothing is bounded yet, so the
        # DSP alone decides, as before streamed turns were supported.
        self.assertIsNone(model.segment_at(0.2))
        self.assertIsNone(model.segment_at(1.1))

    def test_words_ahead_of_their_sentence_wait_for_it(self):
        prior = TextPrior(anchors=(TextAnchor("Sure!"),), words=TIMED[:3], word_start_pts=0)
        model = TextEvents()
        model.prepare(prior)
        self.assertEqual(model.stats["mismatched_contexts"], 0)


class TestEventVetoes(unittest.TestCase):
    def setUp(self):
        self.model = TextEvents()
        self.model.prepare(counting(*TIMED))

    def test_a_murmur_heard_in_a_close_vowel_is_dropped(self):
        # "two" (1.0-1.4 s): its vowel, far from the /n/ ending "one".
        self.assertTrue(self.model.vetoes(LipsyncEventKind.NASAL, 1.3))

    def test_a_nasal_near_a_nasal_phone_is_kept(self):
        # "one" is W AH N over 0.5-1.0 s: its /n/ spans ~0.83-1.0 s.
        self.assertFalse(self.model.vetoes(LipsyncEventKind.NASAL, 0.9))

    def test_closures_only_go_where_the_sentence_has_no_bilabial(self):
        self.assertTrue(self.model.vetoes(LipsyncEventKind.CLOSURE, 1.1))  # One, two, three.
        self.assertFalse(self.model.vetoes(LipsyncEventKind.CLOSURE, 2.05))  # Bob.

    def test_hums_keep_their_nasals(self):
        model = TextEvents()
        model.prepare(
            TextPrior(
                anchors=(TextAnchor("Hmm."),),
                words=(TextWord("Hmm", 0),),
                word_start_pts=0,
                audio_end=0.5,
            )
        )
        self.assertFalse(model.vetoes(LipsyncEventKind.NASAL, 0.2))
