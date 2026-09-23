"""Measure which Gemini-TTS inline tags the pinned voice honors silently.

A tag is honored silently when the audio changes as the tag says and the tag
itself is not spoken. The transcript decides the second half: a tag whose
word turns up in it would also turn up in the captions, which are
transcribed from the same audio. The first half is read from the gap
between the two sentences around the tag, against the same pair with no tag.

Usage:
    python tools/tts_tag_probe.py [--profile charon] [--tags "[short pause]" ...]

Writes one WAV per tag under the output directory and prints a table:
tag, gap in seconds, whether the transcript carries anything beyond the two
sentences, and the transcript itself. Costs one TTS call per tag.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

import whisper  # type: ignore[import-untyped]
from dotenv import load_dotenv

from src.utils.outputs_paths import get_project_root
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
EXPECTED_WORDS = set(re.findall(r"[a-z]+", (BEFORE + " " + AFTER).lower()))


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


async def _probe(
    profile_name: str, tags: list[str], out_dir: Path, repeat: int
) -> None:
    config = load_video_config_modular()
    tts = config.tts_config
    profile = tts.voice_profiles[profile_name]
    model = whisper.load_model(config.whisper_settings.model_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"profile={profile_name} model={profile.gemini_model_name}")
    print(f"{'tag':<18} gaps_s  extra_words  transcript (last sample)")
    for tag in tags:
        text = f"{BEFORE} {tag} {AFTER}" if tag else f"{BEFORE} {AFTER}"
        slug = re.sub(r"[^a-z]+", "_", tag.lower()).strip("_") or "none"
        gaps: list[str] = []
        extras: set[str] = set()
        transcript = ""
        for n in range(repeat):
            path, _ = await _generate_gemini_speech(
                text, out_dir / f"{slug}_{n}.wav", tts.google_cloud, profile
            )
            if not path:
                gaps.append("fail")
                continue
            result = model.transcribe(str(path), word_timestamps=True, language="en")
            gap, extra = _gap_and_extra(result)
            gaps.append(str(gap) if gap is not None else "?")
            extras.update(extra)
            transcript = result["text"].strip()
        print(
            f"{tag or '(none)':<18} {'/'.join(gaps)}  "
            f"{','.join(sorted(extras)) or '-':<11}  {transcript}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--profile", default="charon")
    parser.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=get_project_root() / "outputs" / "temp" / "tts_tag_probe",
    )
    args = parser.parse_args()
    load_dotenv(get_project_root() / ".env")
    asyncio.run(_probe(args.profile, args.tags, args.out_dir, args.repeat))


if __name__ == "__main__":
    main()
