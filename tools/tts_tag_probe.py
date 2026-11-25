"""Measure which Gemini-TTS inline tags the pinned voice honors silently.

A tag is honored silently when the audio changes as the tag says and the tag
itself is not spoken. The transcript decides the second half: a tag whose
word turns up in it would also turn up in the captions, which are
transcribed from the same audio. The first half is read from the gap
between the two sentences around the tag, against the same pair with no tag.

Usage:
    python tools/tts_tag_probe.py [--profile charon] [--tags "[short pause]" ...]
    python tools/tts_tag_probe.py --gemini-api [--model M] [--voice V] --repeat 3

`--gemini-api` voices through the Gemini API speech model instead (REQ-CNT-182)
and probes its angle-bracket tags. The `own_word` column says whether the
tag's own word was heard, which keeps a tag off the performed allowlist; any
other extra word is what the caption filter drops (REQ-CNT-183).

Writes one WAV per tag under the output directory and prints a table:
tag, gap in seconds, whether the transcript carries anything beyond the two
sentences, and the transcript itself. Costs one TTS call per tag.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from pathlib import Path

import whisper  # type: ignore[import-untyped]
from dotenv import load_dotenv

from src.utils.outputs_paths import get_project_root
from src.video.config.audio_models import GeminiApiTTSSettings
from src.video.config_adapter import load_video_config_modular
from src.video.tts import _generate_gemini_speech

BEFORE = "This charger is tiny."
AFTER = "It fits in any pocket and still charges a laptop."
DEFAULT_TAGS = [
    "",
    "[short pause]",
    "[medium pause]",
    "[long pause]",
    "[sigh]",
    "[uhm]",
    "[laughing]",
    "[breath]",
    "[slow]",
    "[fast]",
]
GEMINI_API_TAGS = [
    "",
    "<short pause>",
    "<long pause>",
    "<sigh>",
    "<laugh>",
    "<breath>",
    "<gasp>",
    "<cough>",
    "<throat-clearing>",
]
EXPECTED_WORDS = set(re.findall(r"[a-z]+", (BEFORE + " " + AFTER).lower()))


def own_word_heard(tag: str, extra: list[str]) -> bool:
    """Whether the transcript carries the tag's own word ("sigh" for <sigh>)."""
    own = {_stem(w) for w in re.findall(r"[a-z]+", tag.lower()) if len(w) > 2}
    return any(any(w.startswith(o) for o in own) for w in extra)


def _stem(word: str) -> str:
    """The word without "ing" and a final "e", so "clears" matches "clearing"
    and "pausing" matches "pause".
    """
    word = word.removesuffix("ing")
    return word.removesuffix("e") if len(word) > 3 else word


def _gap_and_extra(result: dict) -> tuple[float | None, list[str]]:
    words = [w for seg in result["segments"] for w in seg.get("words", [])]
    gap = None
    for prev, nxt in zip(words, words[1:], strict=False):
        if prev["word"].strip().lower().startswith("tiny"):
            gap = round(nxt["start"] - prev["end"], 3)
            break
    heard = re.findall(r"[a-z]+", result["text"].lower())
    extra = [w for w in heard if w not in EXPECTED_WORDS]
    return gap, extra


def _gemini_api_voice(
    model_name: str, voice: str | None, settings: GeminiApiTTSSettings
):
    async def voice_text(text: str, path: Path) -> Path | None:
        from src.video.delivery_plan import DeliverySegment
        from src.video.gemini_api_tts import GeminiApiTTSError, synthesize_segments

        try:
            return await synthesize_segments(
                [DeliverySegment(text=text)],
                path,
                api_key=os.environ.get(settings.api_key_env_var, ""),
                model=model_name,
                voice=voice,
                endpoint=settings.endpoint,
                timeout_sec=settings.api_timeout_sec,
                retries=settings.api_max_retries,
                retry_delay_sec=settings.api_retry_delay_sec,
            )
        except GeminiApiTTSError as e:
            print(f"  failed: {e}")
            return None

    return voice_text


async def _probe(
    profile_name: str,
    tags: list[str],
    out_dir: Path,
    repeat: int,
    gemini_api: tuple[str, str | None] | None = None,
) -> None:
    config = load_video_config_modular()
    tts = config.tts_config
    model = whisper.load_model(config.whisper_settings.model_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    if gemini_api:
        if not os.environ.get(tts.gemini_api.api_key_env_var):
            raise SystemExit(f"{tts.gemini_api.api_key_env_var} is not set")
        voice_text = _gemini_api_voice(gemini_api[0], gemini_api[1], tts.gemini_api)
        print(f"gemini_api model={gemini_api[0]} voice={gemini_api[1]}")
    else:
        profile = tts.voice_profiles[profile_name]

        async def voice_text(text: str, path: Path) -> Path | None:
            out, _ = await _generate_gemini_speech(
                text, path, tts.google_cloud, profile
            )
            return out

        print(f"profile={profile_name} model={profile.gemini_model_name}")
    print(f"{'tag':<18} gaps_s  own_word  extra_words  transcript (last sample)")
    for tag in tags:
        text = f"{BEFORE} {tag} {AFTER}" if tag else f"{BEFORE} {AFTER}"
        slug = re.sub(r"[^a-z]+", "_", tag.lower()).strip("_") or "none"
        gaps: list[str] = []
        extras: set[str] = set()
        transcript = ""
        for n in range(repeat):
            path = await voice_text(text, out_dir / f"{slug}_{n}.wav")
            if not path:
                gaps.append("fail")
                continue
            result = model.transcribe(str(path), word_timestamps=True, language="en")
            gap, extra = _gap_and_extra(result)
            gaps.append(str(gap) if gap is not None else "?")
            extras.update(extra)
            transcript = result["text"].strip()
        own = "yes" if own_word_heard(tag, sorted(extras)) else "no"
        print(
            f"{tag or '(none)':<18} {'/'.join(gaps)}  {own:<8}  "
            f"{','.join(sorted(extras)) or '-':<11}  {transcript}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--profile", default="charon")
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument(
        "--gemini-api",
        action="store_true",
        help="probe the Gemini API speech model and its angle-bracket tags",
    )
    parser.add_argument("--model", default="gemini-3.8-flash-tts")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=get_project_root() / "outputs" / "temp" / "tts_tag_probe",
    )
    args = parser.parse_args()
    load_dotenv(get_project_root() / ".env")
    tags = args.tags or (GEMINI_API_TAGS if args.gemini_api else DEFAULT_TAGS)
    gemini_api = (args.model, args.voice) if args.gemini_api else None
    asyncio.run(_probe(args.profile, tags, args.out_dir, args.repeat, gemini_api))


if __name__ == "__main__":
    main()
