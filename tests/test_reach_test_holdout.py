"""Features that change the reach test's treatment arm ship switched off.

The format-vs-format reach comparison needs the script and the voice held
constant until its durability readout; a humanisation feature switched on
mid-test becomes a confound on the arm the comparison exists to measure. Each
such feature is built and merged off, and this file fails if the shipped
config turns one on. Delete a check here when its feature is deliberately
enabled after the readout.
"""

from __future__ import annotations

from src.video.config import load_video_config_modular


def test_script_naturalism_is_off() -> None:
    settings = load_video_config_modular().llm_settings
    assert settings.script_templates.naturalism.intensity == 0
