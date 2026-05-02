"""Integration tests for the pycaps subtitle engine.

These tests run the real ``PycapsRenderer`` against a small fixture video +
transcript. They are skipped automatically when ``pycaps`` is not installed,
so they stay out of the default CI path. Locally:

.. code-block:: bash

    poetry install --with pycaps
    poetry run playwright install chromium  # only needed for 'css' renderer
    poetry run pytest tests/video/test_pycaps_integration.py -v -m integration

The fixture (~30s portrait video + matching whisper_json transcript) lives
under ``tests/fixtures/pycaps/`` and was originally produced during the
library reality-check spike — see ``/home/user/tmp/pycaps-test`` for the
source scripts.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

pycaps = pytest.importorskip(
    "pycaps",
    reason="pycaps optional group is not installed (poetry install --with pycaps)",
)

from src.video.config.subtitle_models import PycapsSettings  # noqa: E402
from src.video.pycaps_engine import PycapsRenderer  # noqa: E402
from src.video.subtitle_positioning import VisualBounds  # noqa: E402

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "pycaps"
SAMPLE_VIDEO = FIXTURE_DIR / "sample_30s.mp4"
SAMPLE_TRANSCRIPT = FIXTURE_DIR / "transcript_30s.json"


def _ffprobe_stream_info(video: Path) -> dict:
    """Return ``{codec, width, height, duration}`` for the first video stream."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height:format=duration",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return {
        "codec": stream["codec_name"],
        "width": stream["width"],
        "height": stream["height"],
        "duration": float(data["format"]["duration"]),
    }


@pytest.mark.integration
def test_pictex_renderer_burns_captions_on_fixture(tmp_path: Path):
    """Pictex renderer path — no Chromium needed, suitable for CI when the group is installed."""
    if not SAMPLE_VIDEO.exists():
        pytest.skip(f"Fixture video missing: {SAMPLE_VIDEO}")
    if not SAMPLE_TRANSCRIPT.exists():
        pytest.skip(f"Fixture transcript missing: {SAMPLE_TRANSCRIPT}")

    output_video = tmp_path / "sample_with_captions.mp4"
    settings = PycapsSettings(
        template_name="word-focus",
        template_pool=[],
        renderer="pictex",
        max_width_ratio=0.85,
        max_number_of_lines=2,
    )
    # Mimic an assembler-style top-heavy product image: 10% top, 75% height.
    bounds = VisualBounds(x=0.075, y=0.10, width=0.85, height=0.75)

    renderer = PycapsRenderer()
    result = renderer.render(
        input_video=SAMPLE_VIDEO,
        transcript_path=SAMPLE_TRANSCRIPT,
        output_video=output_video,
        product_id="B0INTEG",
        visual_bounds=bounds,
        settings=settings,
    )

    assert result.success, f"pycaps render failed: {result.error}"
    assert output_video.exists()
    assert output_video.stat().st_size > 0

    info = _ffprobe_stream_info(output_video)
    assert info["codec"] == "h264"
    assert info["width"] == 1080
    assert info["height"] == 1920
    # Allow 0.5s tolerance because pycaps may trim the trailing silence.
    assert abs(info["duration"] - 30.0) < 0.5


@pytest.mark.integration
def test_pycaps_ai_tagging_with_mocked_gemini(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """End-to-end render with AI tagging on a built-in template + mocked Gemini.

    Uses ``neo-minimal`` because it ships a ``type: ai`` tagger rule. We
    register our ``GeminiLlm`` adapter against pycaps' ``LlmProvider`` (the
    same wiring ``step_burn_pycaps_subtitles`` does in production) and assert
    Gemini is called and the burned video is produced.
    """
    if not SAMPLE_VIDEO.exists():
        pytest.skip(f"Fixture video missing: {SAMPLE_VIDEO}")
    if not SAMPLE_TRANSCRIPT.exists():
        pytest.skip(f"Fixture transcript missing: {SAMPLE_TRANSCRIPT}")

    # Install a fake google.genai module so the adapter doesn't reach the
    # real Gemini API. Returns a fixed response that pycaps' AI tagger
    # accepts as a tagged-words list.
    import sys
    import types
    from unittest.mock import MagicMock

    fake_genai = types.ModuleType("google.genai")
    client_instance = MagicMock(name="GeminiClient")
    response = MagicMock(text="impactful")  # any non-empty string keeps tagger happy
    client_instance.models.generate_content.return_value = response
    fake_genai.Client = MagicMock(return_value=client_instance)
    fake_google_pkg = types.ModuleType("google")
    fake_google_pkg.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    from pycaps.ai import LlmProvider  # noqa: E402

    from src.video.pycaps_engine import GeminiLlm  # noqa: E402

    adapter = GeminiLlm(api_key="test-key", on_error="skip")
    LlmProvider.set(adapter)

    output_video = tmp_path / "sample_with_ai_captions.mp4"
    settings = PycapsSettings(
        template_name="neo-minimal",  # ships a `type: ai` tagger rule
        template_pool=[],
        renderer="pictex",  # avoid Chromium dependency in this test
        max_width_ratio=0.85,
        max_number_of_lines=2,
        enable_ai_tagging=True,
        llm_model="gemini-2.5-flash",
        ai_tagging_on_error="skip",
    )

    renderer = PycapsRenderer()
    result = renderer.render(
        input_video=SAMPLE_VIDEO,
        transcript_path=SAMPLE_TRANSCRIPT,
        output_video=output_video,
        product_id="B0AITAG1",
        visual_bounds=None,
        settings=settings,
    )

    assert result.success, f"pycaps render failed: {result.error}"
    assert output_video.exists()
    assert output_video.stat().st_size > 0
    # Pycaps' AI tagger calls the LLM at least once for the segment-aware rule.
    assert adapter.call_count >= 1, (
        "Expected GeminiLlm to be invoked by pycaps' AI tagger, "
        f"got call_count={adapter.call_count}"
    )
