"""The motif plays under the first word and the turn (REQ-CNT-192).

Asserted against real audio like the sting: a long tone mixed into silence,
measured in windows around each placement, so the trim and the fade show.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.video.assembler.audio_builder import AudioFilterBuilder
from src.video.config import load_video_config_modular
from src.video.config.audio_models import Motif

pytestmark = pytest.mark.req("REQ-CNT-192")
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")


def _run(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stderr


def _source(path: Path, spec: str, duration: float) -> Path:
    _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", spec,
          "-t", str(duration), str(path)])  # fmt: skip
    return path


def _max_volume(path: Path, start: float, length: float) -> float:
    out = _run(["ffmpeg", "-v", "info", "-ss", str(start), "-t", str(length),
                "-i", str(path), "-af", "volumedetect", "-f", "null", "-"])  # fmt: skip
    match = re.search(r"max_volume: (-?[\d.]+|-inf) dB", out)
    assert match, out
    return float("-inf") if match.group(1) == "-inf" else float(match.group(1))


def _render(tmp_path: Path, motif: Motif | None, times: list[float]) -> Path:
    import asyncio

    from src.video.assembler.media_inspector import MediaInspector

    config = load_video_config_modular()
    config.audio_settings.motif = motif
    config.audio_settings.signature_sting = None
    config.audio_settings.loudness_normalization_enabled = False
    builder = AudioFilterBuilder(config)
    voice = _source(tmp_path / "voice.wav", "anullsrc=r=48000:cl=mono", 6.0)
    parts: list[str] = []
    filters, label = asyncio.run(
        builder.build_mix(parts, voice, None, 6.0, MediaInspector(), motif_times=times)
    )
    out = tmp_path / "mix.wav"
    _run(["ffmpeg", "-y", "-v", "error", *parts,
          "-filter_complex", ";".join(filters), "-map", label, str(out)])  # fmt: skip
    return out


@pytest.fixture
def tone(tmp_path: Path) -> Path:
    # Four seconds long, so the 0.5 s trim is what stops it.
    return _source(tmp_path / "motif.wav", "sine=frequency=1000:sample_rate=48000", 4)


@needs_ffmpeg
def test_each_copy_lands_at_its_time_and_is_trimmed(tmp_path, tone):
    mix = _render(tmp_path, Motif(path=tone, max_sec=0.5, volume_db=0), [1.0, 3.0])
    assert _max_volume(mix, 0.0, 0.9) < -60
    assert _max_volume(mix, 1.05, 0.3) > -40
    assert _max_volume(mix, 1.6, 1.3) < -60
    assert _max_volume(mix, 3.05, 0.3) > -40
    assert _max_volume(mix, 3.6, 2.0) < -60


@needs_ffmpeg
def test_no_motif_or_a_missing_file_mixes_nothing(tmp_path, tone):
    assert _max_volume(_render(tmp_path, None, [1.0]), 0.0, 5.5) < -60
    missing = Motif(path=tmp_path / "absent.wav")
    assert _max_volume(_render(tmp_path, missing, [1.0]), 0.0, 5.5) < -60


def test_the_identity_motif_reaches_the_record(tmp_path):
    from src.scraper.amazon.models import ProductData
    from src.scraper.base.models import Platform
    from src.video.config import config
    from src.video.config.identity_models import NarratorIdentity
    from src.video.producer.orchestration import config_for_record

    cfg = config.model_copy(deep=True)
    motif = Motif(path=tmp_path / "m.wav")
    cfg.narrator_identity.product = NarratorIdentity(motif=motif)
    product = ProductData(
        title="A stand", price="$1", url="https://www.amazon.com/dp/B0TEST0192",
        platform=Platform.AMAZON, asin="B0TEST0192", description="d",
    )  # fmt: skip
    assert config_for_record(cfg, product).audio_settings.motif == motif
    assert config.audio_settings.motif is None


def _ctx(tmp_path, motif, structure=None, words=None):
    from src.video.config import config

    cfg = config.model_copy(deep=True)
    cfg.audio_settings.motif = motif
    path = tmp_path / "w.json"
    if words is not None:
        path.write_text(json.dumps({"segments": [{"words": words}]}))
    ctx = MagicMock()
    ctx.config = cfg
    ctx.script = "One two three. Four five."
    ctx.state = {"structure": json.dumps(structure)} if structure else {}
    ctx.run_paths = {"whisper_transcript_file": path}
    return ctx


WORDS = [{"word": w, "start": 0.5 + i, "end": 0.9 + i} for i, w in enumerate("abcde")]


def test_times_come_from_the_first_word_and_the_turn(tmp_path):
    from src.video.producer.steps import _motif_times

    ctx = _ctx(tmp_path, Motif(path=tmp_path / "m.wav"),
               {"labels": {"turn": 1, "payoff": None}}, WORDS)  # fmt: skip
    assert _motif_times(ctx) == [0.5, 3.5]


def test_no_turn_label_skips_that_placement(tmp_path, caplog):
    from src.video.producer.steps import _motif_times

    caplog.set_level("INFO")
    ctx = _ctx(tmp_path, Motif(path=tmp_path / "m.wav"), None, WORDS)
    assert _motif_times(ctx) == [0.5]
    assert "no turn sentence" in caplog.text


def test_no_motif_gives_no_times(tmp_path):
    from src.video.producer.steps import _motif_times

    assert _motif_times(_ctx(tmp_path, None, None, WORDS)) is None


def test_a_stale_turn_label_is_skipped(tmp_path, caplog):
    from src.video.producer.steps import _motif_times, script_digest

    caplog.set_level("INFO")
    ctx = _ctx(
        tmp_path,
        Motif(path=tmp_path / "m.wav"),
        {"labels": {"turn": 1}, "script": script_digest("An older script.")},
        WORDS,
    )
    assert _motif_times(ctx) == [0.5]
    assert "no turn sentence" in caplog.text


def test_the_turn_label_is_read_from_the_step_entry(tmp_path):
    from src.video.producer.steps import _motif_times

    ctx = _ctx(tmp_path, Motif(path=tmp_path / "m.wav"), None, WORDS)
    ctx.state = {"generate_script": {"structure": json.dumps({"labels": {"turn": 1}})}}
    assert _motif_times(ctx) == [0.5, 3.5]


@needs_ffmpeg
def test_motif_at_records_only_what_was_mixed(tmp_path, tone):
    import asyncio

    from src.video.assembler.media_inspector import MediaInspector

    config = load_video_config_modular()
    config.audio_settings.signature_sting = None
    builder = AudioFilterBuilder(config)
    voice = _source(tmp_path / "voice.wav", "anullsrc=r=48000:cl=mono", 6.0)

    config.audio_settings.motif = Motif(path=tmp_path / "absent.wav")
    asyncio.run(
        builder.build_mix([], voice, None, 6.0, MediaInspector(), motif_times=[1.0])
    )
    assert builder.motif_placed == []

    config.audio_settings.motif = Motif(path=tone)
    asyncio.run(
        builder.build_mix([], voice, None, 6.0, MediaInspector(), motif_times=[1.0])
    )
    assert builder.motif_placed == [1.0]
