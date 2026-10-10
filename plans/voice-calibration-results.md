# Voice calibration — 2026-10-09

Branch: `voice-calibration` on [maxipesfix/pipecat-visemes](https://github.com/maxipesfix/pipecat-visemes),
starting at upstream `9c85db4`, plus the benchmark's vowel-identity score (`f921ccd`, cherry-picked
from `fix/vowel-rounding`). Independent of the fork's text-tier work, so it can be proposed upstream
on its own. Opt-in: with no profile the analyzer's output is unchanged.

## The problem

The analyzer reads lip rounding from F2 alone: rounding rises as F2 falls below the midpoint of the
voice's F2 range. That assumes a rounded vowel has the voice's lowest F2. Many TTS voices front /u/,
and the bot's default Cartesia voice (`71a7ad14`) renders "oo" spread, the avatar showing "ee".

## What the data shows

Analyzer measurements on the 7 Cartesia voices: the benchmark sentences, a clean word list per voice,
and a clean held-out set ("Now say boot, please.", different words from calibration).

- **/u/ depends on the consonant before it.** On `71a7ad14`: "boot" F2 ≈ 1520 Hz, "boo" 1640 (low
  enough for the F2 mapping); "who'd" 1790, "two" 2010, "new" 2310 with F2 unfound 82% of the time
  (above the 2600 Hz band, the signature of its /i/). The F2 mapping reads the second group as spread.
  Four of the seven voices have a back /u/ (F2 1150-1450) and are already mostly fine.
- **F3 separates fronted /u/ from /i/** (lip rounding lowers it): on `71a7ad14`, "two"/"new" F3 ≈
  2460-2580 Hz vs ≈ 3030 for "heed"/"tea". /u/-vs-/i/ AUC on the analyzer's F3 alone: 0.71-0.94 on
  six voices. The rejected F3 experiment (`0e21a32` on `fix/vowel-rounding`) applied F3 to every
  vowel and rounded /a e/ too; here it only decides between neighbours in the voice's own vowel space.
- **A per-voice classifier trained on that voice's labels** reaches ≈ 0.70-0.96 on running speech,
  depending on the voice: the ceiling for anything built on F1-F3. Labels in the benchmark sentences
  come from word timestamps that are 0.1-0.25 s off, which caps every method there.

## What was built

| Piece | What it does |
|---|---|
| `lipsync/voice_profile.py` | `VoiceProfile`: one exemplar per calibration *word* (median F1/F2/F3, how often F2 went unfound), scored with a pooled covariance; a missing F2 is evidence in itself. Rounding = posterior-weighted rounding of the exemplars' vowels (targets from `plans/experiments/l4_proto.py`). Per word, not per vowel, because one voice says /u/ in two places. |
| `lipsync/calibration.py` | 36 words plus two hums as a list ("Heed. Tea. Bead. ... Hmm. Mmm."), every vowel, /u/ after lip-closing and tongue-front consonants. Each word is its own burst of speech, found by energy (13 of 14 recordings matched the word count exactly; word timestamps are the fallback). The analyzer's own measurements become the profile. |
| Analyzer | `FormantLipsyncAnalyzer(voice_profile=…)`: rounding from the profile on hops with a measured F1. Width and openness stay on the formant mapping (`_PROFILE_WIDTH_WEIGHT = 0`, see below). About +10 µs per hop (151 → 161 µs). |
| Nasal decision | The hums become exemplars too. Below 0.1 probability of being this voice's hum, a hop cannot latch NASAL (`_PROFILE_NASAL_VETO`); at 0.7 or above it counts as nasal even if the fixed thresholds miss it (`_PROFILE_NASAL_SUPPORT`). Text, when it votes, wins. |
| Online learning (experimental) | `ProfileLearner`: adds the voice's own finished single-vowel dictionary words (TTS word timestamps, shifted 0.1 s) as exemplars during a session; calibration exemplars stay as measured; reducible function words ("to", "you", "the") are skipped. |
| Processor | `LipsyncParams(voice_profile_path=…, voice_profile_learning=False)`. |
| Benchmark | `python -m benchmarks.calibrate` records the list per corpus voice (cached under `fixtures/<provider>/<voice>/calibration/`) and writes profiles; `accuracy --voice-profiles DIR [--online-profiles]` scores with them. |

## Results

Seven Cartesia voices, sound only (no text tier), this branch's analyzer with and without profiles.
Rounding AUC = vowel-identity score (rendered rounding on rounded vs spread dictionary vowels).

| Run | Composite | Checks: closure / nasal / other | Rounding AUC pooled | Bot voice | Rounding on rounded / spread |
|---|---|---|---|---|---|
| Baseline (2 takes) | 84.1 | 55/56, 100/112, 41/42 | 0.72 | 0.63 | 0.38 / 0.14 |
| **Calibration** | **84.1** | 55/56, 100/112, **42/42** | **0.77** | **0.73** | 0.60 / 0.30 |
| Calibration, width too (`_PROFILE_WIDTH_WEIGHT=1`) | 78.0 | 55/56, 100/112, 40/42 | 0.77 | 0.73 | — |

Taking width from the profile costs 6.1 composite points: `width_r` scores width against Praat's F2
track, which a categorical width does not follow. Width stays on the F2 mapping.

**Online learning, with one take per sentence.** With both takes, the learner hears each sentence
before being scored on its second take, and looks better than it is (0.80 pooled). One take:

| Run (1 take) | Rounding AUC pooled | 71a7 | 5ee9 | ef19 | 47c3 | d1d9 | 62ae | db6b |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0.72 | 0.62 | 0.76 | 0.71 | 0.76 | 0.80 | 0.70 | 0.73 |
| **Calibration** | **0.77** | 0.74 | 0.78 | **0.65** | 0.80 | 0.86 | 0.77 | 0.80 |
| Online only | 0.71 | 0.71 | 0.67 | 0.70 | 0.75 | 0.77 | 0.69 | 0.71 |
| Calibration + online | 0.73 | **0.78** | 0.71 | 0.67 | 0.75 | 0.79 | 0.73 | 0.65 |

Composite 83.9 in all four. Calibration improves six of seven voices. Online learning helps the bot
voice and hurts others: learned exemplars come from words whose timestamps are 0.1-0.25 s off, so it
stays off by default.

### The nasal detector

The fixed nasal thresholds read the dark, quiet tail of a fronted /u/ ("who") as a hum and shut the
lips; on deep voices they miss real hums. Per voice, a hum is separable from that voice's vowels on
F1/F2/F3 alone (cross-validated AUC 0.82-0.99; adding centroid and band energies does not help
consistently), so the profile records the voice's hums and votes. Seven voices, two takes, sound
only, calibrated:

| Run | Composite | Nasal checks | Nasal time (share of voiced hops) |
|---|---|---|---|
| Baseline, no profile | 84.1 | 100/112 | 0.06 |
| Profile, no nasal vote | 84.1 | 100/112 | 0.06 |
| Veto below 0.1 | 84.1 | 101/112 | 0.02 |
| Veto + support at 0.9 / 0.8 | 84.2 / 84.3 | 103 / 104 | 0.04 / 0.05 |
| **Veto + support at 0.7** (default) | **84.4** | **105/112** | 0.05 |
| Veto + support at 0.6 | 84.4 | 105/112 | 0.06 |

The check that flips with the veto is the bot voice's "Soon the new moon grew blue..." (too much
nasal); support adds detected hums ("Hmm. Hmm, hmm.") on deep voices. On the vowel probe (vrm-lipsync),
false nasal latches across the 21 clips drop with calibration: 17 -> 5 on `71a7ad14` (6 -> 1 on the
"oo" words), 3 -> 0 on `47c38ca4`, 4 -> 2, 5 -> 2, 5 -> 3 on three others; vowels rendered right are
unchanged on five voices and one lower on two.

## Still open

- **`ef191366` gets worse with calibration** (0.71 → 0.65); not yet diagnosed.
- **Spread vowels pick up rounding** (0.14 → 0.30 on average). Whether that reads as rounded on the
  avatar is untested: the vowel probe in vrm-lipsync and the live bot are the next checks.
- **Online labels** need something better than raw word timestamps (onset anchoring, or burst
  segmentation as in calibration) before learning can be on by default.
- **One calibration recording ran two words together** (`71a7ad14` take 1): its "had/tap/bad" were
  lost and "day" looks misassigned. Timestamp fallback assigns bursts by overlap, which can still
  mislabel; two takes per voice limit the damage.
- **Width** stays on the F2 mapping, so a fronted /u/ is rounded but still fairly wide.
- **Deepgram** not yet run (no cached fixtures here).
- One test failed once in 13 runs of the suite and could not be reproduced (name not captured).

## Reproduction

```bash
cd server
uv run python -m benchmarks.calibrate                       # records the word list (and hums) once per voice (CARTESIA_API_KEY)
uv run python -m benchmarks.accuracy --offline --tag base
uv run python -m benchmarks.accuracy --offline --voice-profiles benchmarks/results/profiles/cartesia --compare benchmarks/results/accuracy-base.json
uv run python -m benchmarks.accuracy --offline --takes 1 --voice-profiles benchmarks/results/profiles/cartesia --online-profiles
```
