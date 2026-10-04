"""Measure which numbers, units and model names the pinned voice misreads.

Each case is voiced twice in the same sentence: as written ("5000mAh") and
spelled out the way a person would say it ("5000 milliamp hours"). Whisper
transcribes both. When the two transcripts agree, the voice already reads
the written form as intended; when they differ, the written form is misread
and belongs in the `tts_normalisation` table. Comparing the pair rather than
reading one transcript sidesteps Whisper writing digits and units back in
their written form.

Usage:
    python tools/tts_normalisation_probe.py [--profile charon] [--repeat 1]

Writes the WAVs under the output directory and prints one row per case:
whether the transcripts agree, then both transcripts. Costs two TTS calls
per case and repeat.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

import whisper
from dotenv import load_dotenv

from src.utils.outputs_paths import get_project_root
from src.video.config_adapter import load_video_config_modular
from src.video.tts import _generate_gemini_speech

FRAME = "It has {} and fits in a pocket."
# (written form, spoken form)
CASES = [
    ("5000mAh", "5000 milliamp hours"),
    ("10,000 mAh", "10,000 milliamp hours"),
    ("65W", "65 watts"),
    ("2.4 GHz", "2.4 gigahertz"),
    ("5GHz", "5 gigahertz"),
    ("1.83-inch", "1.83 inch"),
    ("USB-C", "USB C"),
    ("IP68", "I P 68"),
    ("Wi-Fi 6E", "Wi-Fi 6 E"),
    ("128GB", "128 gigabytes"),
    ("3.5mm", "3.5 millimetre"),
    ("SKU A2337", "SKU A 2337"),
]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace(",", ""))


async def _probe(profile_name: str, out_dir: Path, repeat: int) -> None:
    config = load_video_config_modular()
    tts = config.tts_config
    profile = tts.voice_profiles[profile_name]
    if tts.google_cloud is None:
        raise SystemExit("No google_cloud TTS settings in the config.")
    model = whisper.load_model(config.whisper_settings.model_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"profile={profile_name} model={profile.gemini_model_name}")
    print(f"{'written':<12} agree  written transcript | spoken transcript")
    for written, spoken in CASES:
        slug = re.sub(r"[^a-z0-9]+", "_", written.lower()).strip("_")
        for n in range(repeat):
            heard = []
            for kind, text in (("written", written), ("spoken", spoken)):
                path, _ = await _generate_gemini_speech(
                    FRAME.format(text),
                    out_dir / f"{slug}_{kind}_{n}.wav",
                    tts.google_cloud,
                    profile,
                )
                if not path:
                    heard.append("(failed)")
                    continue
                result = model.transcribe(str(path), language="en")
                heard.append(result["text"].strip())
            agree = len(heard) == 2 and _words(heard[0]) == _words(heard[1])
            print(f"{written:<12} {'yes' if agree else 'NO ':<5}  {' | '.join(heard)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--profile", default="charon")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=get_project_root() / "outputs" / "temp" / "tts_normalisation_probe",
    )
    args = parser.parse_args()
    load_dotenv(get_project_root() / ".env")
    asyncio.run(_probe(args.profile, args.out_dir, args.repeat))


if __name__ == "__main__":
    main()
