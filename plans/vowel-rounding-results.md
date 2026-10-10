# Vowel identity and the text tier, live — 2026-10-09

Branch: `fix/vowel-rounding` on [maxipesfix/pipecat-visemes](https://github.com/maxipesfix/pipecat-visemes),
starting at upstream `9c85db4`. Nine commits, listed below. The client side of this work (VRM avatar, vowel probe) is in
[OpenHRIai/vrm-lipsync](https://github.com/OpenHRIai/vrm-lipsync), `docs/lipsync-progress.md`.

## The problem

Asked to say the VRM vowels, the bot pronounced them but the avatar's mouth didn't match:

1. **"ee" rendered as a closed "oo", the mouth fluttering.** The bot voice's /i/ has F2 ≈ 3000 Hz,
   above the 2600 Hz search band. The missing F2 counted as evidence of a nasal murmur and kept
   shutting the lips.
2. **"oo" and "oh" rendered spread.** This Cartesia voice says /u o/ with F2 ≈ 1600–2100 Hz, as far
   forward as /i e/. The analyzer measures it correctly (it agrees with Praat), but rounding is
   derived from F2 alone. Sound alone can't fix this; the words can.
3. **The text that could fix it was almost never used live.** Word timestamps arrive after the
   audio is analyzed, and every multi-sentence bot turn was rejected by the text tier.

## Results

| What | Before | After |
|---|---|---|
| "two" in a live-style counting turn: width / rounding (0–1; "oo" wants low width, high rounding) | 0.87 / 0.01 | **0.25 / 0.70** |
| False nasal events in that turn (each shuts the lips ≥ 50 ms) | 2 | **0** |
| Rounded vs spread AUC as delivered, streamed turns¹ | 0.66 | **0.78** |
| Mean rounding on "oo" words as delivered¹ | 0.26 | **0.54** |
| "ee" hops latched as nasal, bot voice² | 19% | **1%** |
| "oo" hops latched as nasal, bot voice² | 51% | **33%** |
| Real hums detected, range across 7 voices² | 0–76% | **74–98%** |
| Accuracy composite, 7 voices, text tier on³ | 84.4 | **84.7** |
| Nasal / closure checks³ | 100/112, 55/56 (sound only) | **111/112, 54/56** |
| "two" after a long comma pause ("Of course! One, two, three.")⁴ | 1.00 / 0.00 | **0.25 / 0.70** |

¹ `benchmarks.record --reanalyze` over the 19-utterance eval recording split into sentence anchors,
as the live bot delivers turns. "Before" is the text tier as first shipped; in live use nearly
every turn is multi-sentence, so the live "before" was closer to sound alone (0.58).
² Seven Cartesia voices, single-vowel dictionary words.
³ `benchmarks.accuracy --offline --text-events` on our fixtures (7 voices × 13 sentences × 2 takes
= 182 clips). TTS isn't deterministic, so upstream's committed baseline (86.0) was scored on other
audio; unmodified upstream reads 84.1 on these fixtures. **With the text tier off, output is
identical to upstream on all 182 clips.**
⁴ Headless repro: real Cartesia TTS through the bot's output path, three streamed turns.

## Commits

| Commit | Change | Measured effect |
|---|---|---|
| `fa5fe13` | **Benchmark: vowel-identity metric.** AUC of rendered rounding (and width) on rounded vs spread dictionary vowels, per voice; the existing metrics couldn't see rounding. Corpus gains "Joe rode the old boat…" (it had almost no /o/) | Baseline rounding AUC 0.72 pooled, **0.63 on the bot voice** (worst of 7); width AUC 0.72 |
| `0e21a32` | **F3 rounding cue, off by default** (review item ROUND-1, kept as an experiment) | Rejected: rounding AUC 0.72 → 0.67 (`max`) / 0.68 (`mean`). It rounds /i e a/ more than /u o/ |
| `cdfef1b` | **Nasal: a missing F2 no longer counts as nasal where trusted text covers the hop.** Hums come from the text instead. Without text, output is unchanged | Nasal checks 110 → 111/112. "ee" latched 19% → 1%, "oo" 51% → 33% (bot voice) |
| `8219e54` | **Bot: text tier on** (`LipsyncParams(text_events_enabled=True)`) | Composite 84.1 → 84.4, nasal checks 100 → 110/112 |
| `bb745af` + `e7a2d6c` | **Word times anchored on the utterance's speech onset** (`ONSET_MAX_SHIFT` 0.3 s). Cartesia's word times landed ~0.1 s early, enough to put a word's shape on the previous word | Composite 84.4 → 84.7, closure checks 53 → 54/56 |
| `e7a2d6c` | **Revise held keyframes when word timings arrive** (`revise_keyframes` / `revise_events`). Keyframes still in the delivery queue get the "oo/oh" rounding hint, placed on the word's loud core, with guard keyframes so it doesn't bleed into neighbours. No extra latency | Delivered rounding AUC 0.58 → 0.785; "oo" words 0.26 → 0.54 |
| `744640c` | **Streamed turns:** later sentences of a shared context are bounded by their first word's timestamp instead of rejected. **Held events revised:** a nasal far from any m/n/ng, or a closure in a sentence with no b/p/m, is dropped before release | Streamed AUC 0.657 → 0.783 (= single-sentence 0.785). Closures 103 → 98, nasals 44 → 27; every b/p/m closure and every hum kept |
| `20e74e2` | Eval corpus: a counting turn ("Sure! One, two, three. Four, five, six.") | "two": width 0.87 → 0.25, rounding 0.01 → 0.70, nasal events 2 → 0 |
| `3e81d9e` | **A word before a long pause keeps its timing** (`text_events.py`, `MIN_PHONE_STEP` / `MAX_PHONE_STEP`). See below | "two" after a long pause rounds; benchmark output unchanged |

`bb745af` also carries part of the onset anchoring that its message doesn't describe.

### The comma-pause fix (`3e81d9e`)

Live, "two" in "count one two three" rounded on the first turn and rendered spread on later ones.
The cause was the length of the pause after "two,", not the turn. `_build_spans` spreads a word's
phones over its interval, which runs to the next word's timestamp, pause included. Above 0.35 s
per phone the whole word was dropped, so it got no timing and no rounding hint:

| Turn | "two" → "three" | Per phone (T, UW) | "two" before the fix |
|---|---|---|---|
| "Sure! One, two, three." | 0.61 s | 0.30 s, kept | 0.25 / 0.70 |
| "Of course! One, two, three." | 0.86 s | 0.43 s, **dropped** | 1.00 / 0.00 |

The "good" turns sat just under the limit, which is why it looked random live. Phones are now
capped at 0.35 s and the rest of the gap is left untimed, where DSP alone decides, as before. A
single-vowel word's hint still goes on its loud core.

- `tests/test_revise.py::test_a_word_before_a_long_pause_is_still_rounded` fails without the fix,
  passes with it. 111 tests pass.
- `benchmarks.accuracy --offline --text-events`: composite 84.7, nasal 111/112, closures 54/56;
  output **identical on all 182 clips** to the run without the fix.
- Streamed real-time replay of the 19-utterance recording: rounding AUC 0.838 vs 0.839 (scored with
  a stand-in for the delivered-AUC script, which isn't committed; it reads higher than the figures
  above but ranks runs the same). Neither recording has a pause long enough to trigger the bug.

## Tried and rejected

| Experiment | Result |
|---|---|
| F3 rounding cue (`max` / `mean` with F2) | Rounding AUC −0.04 on 7 voices |
| Widen F2/F3 bands to 3200/4000 Hz (for /i/) | Composite **−2.8**, F2 error +21 Hz, width correlation −0.05 |
| 2–4 kHz energy as a murmur-vs-vowel cue | Doesn't transfer across voices; the bot voice's /u/ is darker than its own hums |
| Closure veto at phone level (held events) | Dropped real /m/ closures (7 → 4 on "Mama made more…"); replaced by a sentence-level veto |

## Seen from the client: the vowel probe

vrm-lipsync's vowel probe (`tools/vowel-probe/capture.py`) records clips through this bot's output
path and scores them two ways: the analyzer fed the audio directly, and what `LipsyncProcessor`
delivers after the delivery queue ("correction buffer"). The avatar shows the right vowel for
**7/25** vowels without the buffer and **14/25** with it (10 single words, the same words
mid-sentence, a 5-word sequence). In the sequence "He. Ha. Who. Hoe. Heh.", "Who" goes from
rounding 0.07 (shown `ih`) to 0.70 (shown `ou`).

## Still open

- **First word of each answer:** its timing arrives ~0.1 s after its batch is sent, so it's never
  revised.
- **"oo/oh" from sound alone:** unsolved for this voice; everything above relies on the text. Next
  step: predict each vowel from the TTS input text as its loud burst begins (counting bursts against
  the dictionary's vowel sequence), instead of waiting for word timestamps.
- **Word-timing alignment:** one shift per utterance puts the first word on its speech onset, but
  Cartesia's error isn't constant (0.1–0.25 s within one sentence). Later words can land late:
  mid-sentence "who" gets its rounding after its vowel, and "Hoe"'s /o/ can reach into the next word.
- **Bleed across shared timestamps:** "help" before "you" still picks up rounding.
- **Spellings and digits** ("oo", "1, 2, 3") switch the text tier off for that sentence.
- **Upstream:** none of this has been offered to jptaylor/pipecat-visemes.

## Reproduction

```bash
cd server
uv run pytest -q
uv run python -m benchmarks.accuracy --offline --text-events      # seven-voice benchmark
uv run python -m benchmarks.record --reanalyze --text-events      # real-time replay, newest recording
```
