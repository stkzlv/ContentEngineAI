"""Per-product choices are the same in every process (#582)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = """
from src.video.config import load_video_config_modular
from src.video.config.subtitle_models import SubtitleSettings
from src.video.producer.utils import select_profile_for_product
from src.video.subtitle_positioning import StylePreset, get_style_config
from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator

c = load_video_config_modular()
for pid in ("B0STABLE01", "B0STABLE02", "B0STABLE03", "B0STABLE04"):
    print(select_profile_for_product(pid, list(c.video_profiles)[:8], c))
    print(get_style_config(StylePreset.RANDOM, SubtitleSettings(), pid, c)["effects"])
    g = UnifiedSubtitleGenerator(
        SubtitleSettings(randomize_effects=True), (1080, 1920), pid, c
    )
    print(sorted(k for k, v in g._select_effects().items() if v))
"""


def _choices(hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    return subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    ).stdout


@pytest.mark.req("REQ-BAT-040", "REQ-VID-072")
def test_each_choice_is_the_same_in_two_processes() -> None:
    first = _choices("1")

    assert first.strip()
    assert _choices("2") == first
    assert _choices("3") == first


@pytest.mark.req("REQ-BAT-040", "REQ-VID-072")
def test_no_code_reseeds_the_global_generator() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "src").rglob("*.py")
        if re.search(r"\brandom\.seed\(|=\s*hash\(product_id", path.read_text())
    ]

    assert offenders == []
