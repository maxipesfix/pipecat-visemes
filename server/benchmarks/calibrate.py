"""Measure per-voice vowel profiles from a calibration recording.

Each corpus voice speaks lipsync.calibration's word list (about 30 s), cached
under fixtures/<provider>/<voice_id>/calibration/take-<n>.wav like the other
fixtures; the analyzer's own measurements of each word's vowel become the
voice's profile. Run the accuracy benchmark with --voice-profiles on the
output directory to score with them.

Run (from server/):
    uv run python -m benchmarks.calibrate                 # synthesizes once (CARTESIA_API_KEY)
    uv run python -m benchmarks.calibrate --offline       # cached recordings only
    uv run python -m benchmarks.accuracy --offline --voice-profiles benchmarks/results/profiles/cartesia
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from benchmarks.common import (
    FIXTURES_DIR,
    RESULTS_DIR,
    chunks_16k_float32,
    load_corpus,
    read_wav,
    synthesize,
    write_wav,
)
from lipsync.calibration import CALIBRATION_WORDS, calibration_text, measure, samples_from_debug
from lipsync.voice_profile import VoiceProfile


async def recording(provider: str, voice_id: str, take: int, offline: bool):
    """(pcm bytes, sample rate, text timing) of one calibration take, cache-first."""
    base = FIXTURES_DIR / provider / voice_id / "calibration"
    wav, meta = base / f"take-{take}.wav", base / f"take-{take}.json"
    text = calibration_text()
    if wav.exists() and meta.exists() and json.loads(meta.read_text()).get("text") == text:
        pcm, rate = read_wav(wav)
        return pcm, rate, json.loads(meta.read_text()).get("text_timing")
    if offline:
        return None
    pcm, rate, timing = await synthesize(provider, voice_id, text, with_text=True)
    base.mkdir(parents=True, exist_ok=True)
    write_wav(wav, pcm, rate)
    meta.write_text(json.dumps({"text": text, "text_timing": timing}, indent=1) + "\n")
    return pcm, rate, timing


async def profile_for(
    provider: str, voice_id: str, takes: int, offline: bool
) -> VoiceProfile | None:
    samples = []
    for take in range(1, takes + 1):
        got = await recording(provider, voice_id, take, offline)
        if got is None:
            continue
        pcm, rate, timing = got
        audio = np.concatenate([c async for c in chunks_16k_float32(pcm, rate)])
        times = [pts for _, pts, _ in (timing or {}).get("words", []) if pts is not None]
        try:
            samples += samples_from_debug(
                await measure(audio),
                CALIBRATION_WORDS,
                times if len(times) == len(CALIBRATION_WORDS) else None,
            )
        except ValueError as e:
            print(f"  ! {voice_id[:8]} take {take}: {e}")
    return VoiceProfile.fit(samples) if samples else None


async def main(args):
    load_dotenv(override=True)
    _, voice_map = load_corpus()
    voices = [v.id for v in voice_map.get(args.provider, [])]
    if args.voices:
        voices = args.voices.split(",")
    out = args.out or RESULTS_DIR / "profiles" / args.provider
    out.mkdir(parents=True, exist_ok=True)
    for voice_id in voices:
        profile = await profile_for(args.provider, voice_id, args.takes, args.offline)
        if profile is None:
            print(f"  ! {voice_id[:8]}: no calibration recording")
            continue
        profile.save(out / f"{voice_id}.json")
        found = {e.vowel for e in profile.exemplars}
        print(
            f"  {voice_id[:8]}: {len(profile.exemplars)} words, {len(found)} vowels -> {out / voice_id}.json"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", default="cartesia", choices=["cartesia", "deepgram"])
    parser.add_argument("--voices", help="comma-separated voice ids (default: the corpus voices)")
    parser.add_argument("--takes", type=int, default=2)
    parser.add_argument("--offline", action="store_true", help="never synthesize; cache only")
    parser.add_argument(
        "--out", type=Path, help="profile directory (default: results/profiles/<provider>)"
    )
    sys.exit(asyncio.run(main(parser.parse_args())))
