import numpy as np

from benchmarks.vowel_identity import auc, scores, vowel_hops, word_spans, word_vowel


def test_word_vowel_labels_single_vowel_dictionary_words():
    assert word_vowel("moon") == "UW"
    assert word_vowel("Road.") == "OW"
    assert word_vowel("see") == "IY"
    # Two vowels: which one carries the word is a stress question; skip it.
    assert word_vowel("believe") is None
    # Out of vocabulary: no guessed labels.
    assert word_vowel("zxqv") is None


def test_auc_is_rank_based_and_counts_ties_half():
    assert auc([1.0] * 8, [0.0] * 8) == 1.0
    assert auc([0.0] * 8, [1.0] * 8) == 0.0
    assert auc([0.5] * 8, [0.5] * 8) == 0.5
    # Too few hops on either side to mean anything.
    assert np.isnan(auc([1.0] * 3, [0.0] * 8))


def test_word_spans_end_each_word_at_the_next():
    timing = {"words": [["Joe", 0.1, 0.0], ["see", 0.5, 0.0], ["the", 0.8, 0.0]]}
    assert word_spans(timing, 1.2) == [("OW", 0.1, 0.5), ("IY", 0.5, 0.8), ("AH", 0.8, 1.2)]
    assert word_spans(None, 1.2) == []


def test_vowel_hops_score_only_the_loud_voiced_core():
    offsets = np.arange(0.0, 1.0, 0.02)
    rms = np.where((offsets > 0.2) & (offsets < 0.4), 1.0, 0.1)
    rms[(offsets > 0.6) & (offsets < 0.8)] = 1.0
    voiced = np.ones_like(offsets, dtype=bool)
    rounding = np.where(offsets < 0.5, 0.9, 0.1)
    width = np.where(offsets < 0.5, 0.2, 0.8)
    hops = vowel_hops([("OW", 0.0, 0.5), ("IY", 0.5, 1.0)], offsets, rms, voiced, rounding, width)
    # Only the loud core of each word: no quiet onset or tail hops.
    assert len(hops.rounded) == len(hops.spread) == 9
    assert set(hops.rounded) == {0.9} and set(hops.spread) == {0.1}
    result = scores(hops)
    assert result["rounding_auc"] == 1.0
    assert result["width_auc"] == 1.0
