"""Configuration precedence: CLI, machine environment, profile, YAML.

Decision 0003 limits the environment to secrets and machine settings. These
drive the real loaders; an earlier version of this file tested a hand-written
copy of the config manager, which kept the old variable list after the code
changed.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from src.config_manager import MACHINE_ENV_SETTINGS, UnifiedConfigManager
from src.video.config.core_models import VideoProfile

REPO = Path(__file__).resolve().parents[2]

# Behaviour settings the environment used to carry. None may change a run now.
REMOVED_BEHAVIOUR_VARS = {
    "DEBUG_MODE": "true",
    "CONTENT_ENGINE_DEBUG": "true",
    "CONTENT_ENGINE_TIMEOUT": "17",
    "SUBTITLE_ANCHOR": "top",
    "SUBTITLE_MARGIN": "0.31",
    "SUBTITLE_CONTENT_AWARE": "false",
    "SUBTITLE_STYLE_PRESET": "minimal",
    "SUBTITLE_FONT_SIZE_SCALE": "1.7",
    "SUBTITLE_ALIGNMENT": "left",
    "SUBTITLE_MAX_WIDTH_FRACTION": "0.33",
    "SUBTITLE_RANDOMIZE_FONTS": "true",
    "SUBTITLE_RANDOMIZE_COLORS": "true",
    "SUBTITLE_RANDOMIZE_EFFECTS": "true",
    "SUBTITLE_MAX_LINE_LENGTH": "11",
    "SUBTITLE_MAX_WORDS_PER_LINE": "2",
    "SUBTITLE_MAX_DURATION": "9.5",
    "SUBTITLE_MIN_DURATION": "0.2",
}


def video_config(env: dict[str, str], cli: dict | None = None) -> dict:
    with patch.dict(os.environ, env):
        return UnifiedConfigManager().get_video_config(cli)


@pytest.mark.req("REQ-OPS-001")
def test_a_machine_setting_from_the_environment_beats_the_yaml() -> None:
    config = video_config({"OUTPUTS_DIR": "/srv/renders", "FFMPEG_THREADS": "3"})

    assert config["global_output_directory"] == "/srv/renders"
    assert config["ffmpeg_settings"]["encoding"]["threads"] == 3


@pytest.mark.req("REQ-OPS-001")
def test_the_cli_beats_the_environment() -> None:
    config = video_config(
        {"OUTPUTS_DIR": "/srv/renders"}, cli={"output_dir": "/srv/from-cli"}
    )

    assert config["global_output_directory"] == "/srv/from-cli"


@pytest.mark.req("REQ-OPS-001")
def test_no_environment_setting_is_one_a_profile_can_set() -> None:
    """What keeps the environment above the profile without a second merge.

    The profile merges after the environment is applied, so a key both could
    set would let the profile win. None is shared, so the order holds.
    """
    profile_keys = set(VideoProfile.model_fields) | {"video_settings"}
    for paths in MACHINE_ENV_SETTINGS.values():
        for path in paths:
            assert path.split(".")[0] not in profile_keys, path


@pytest.mark.req("REQ-OPS-003")
def test_behaviour_variables_no_longer_change_the_config() -> None:
    baseline = video_config({})
    with_vars = video_config(REMOVED_BEHAVIOUR_VARS)

    assert with_vars == baseline


@pytest.mark.req("REQ-OPS-003")
def test_the_publisher_reads_only_secrets_from_the_environment(
    tmp_path: Path,
) -> None:
    from src.publisher.config import load_publisher_config

    env = {
        "LATE_API_KEY": "sk_test_env",
        "PUBLISHER_PROVIDER": "nonexistent",
        "PUBLISHER_IMMEDIATE": "true",
        "PUBLISHER_MAX_RETRIES": "9",
        "PUBLISHER_TIMEOUT": "1",
        "PUBLISHER_DEFAULT_PLATFORMS": "tiktok",
    }
    with patch.dict(os.environ, {"LATE_API_KEY": "sk_test_env"}):
        baseline = load_publisher_config()
    with patch.dict(os.environ, env):
        config = load_publisher_config()

    assert config.api_key == "sk_test_env"
    assert config.provider == baseline.provider
    assert config.immediate_publish == baseline.immediate_publish
    assert config.max_retries == baseline.max_retries
    assert config.timeout == baseline.timeout
    assert config.default_platforms == baseline.default_platforms


@pytest.mark.req("REQ-OPS-009")
def test_the_env_example_lists_only_secrets_and_machine_settings() -> None:
    from src.utils.secrets import is_secret_key

    # Machine or account values: where outputs go and how this machine runs,
    # and the operator's own links and program state, which a public YAML
    # must not carry.
    allowed = set(MACHINE_ENV_SETTINGS) | {
        "COQUI_TTS_GPU",
        "PIPELINE_TOPICS_FILE",
        "AMAZON_ASSOCIATE_TAG",
        "AMAZON_AFFILIATE_LINKS_ENABLED",
        "SUBTITLE_BUSINESS_URL",
        "LINK_IN_BIO_URL",
        "JAMENDO_CLIENT_ID",
        "FREESOUND_CLIENT_ID",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }
    text = (REPO / ".env.example").read_text(encoding="utf-8")
    names = re.findall(r"^([A-Z][A-Z0-9_]*)=", text, flags=re.M)

    assert names
    unexpected = [n for n in names if n not in allowed and not is_secret_key(n)]
    assert unexpected == []


@pytest.mark.req("REQ-OPS-002")
def test_the_batch_takes_its_outputs_dir_from_the_yaml(tmp_path: Path) -> None:
    from src.pipeline.cli import create_argument_parser
    from src.pipeline.config import load_global_batch_config

    path = tmp_path / "pipeline.yaml"
    path.write_text(
        yaml.dump(
            {"global_batch": {"product_ids": ["B0CONFIG01"], "outputs_dir": "custom"}}
        )
    )
    args = create_argument_parser().parse_args([])

    config = load_global_batch_config(args, config_path=path)

    assert config.outputs_dir.name == "custom"


@pytest.mark.req("REQ-OPS-002")
@pytest.mark.asyncio
async def test_the_producer_scans_the_configured_outputs_dir(tmp_path: Path) -> None:
    from src.video.config import load_video_config_modular

    product = tmp_path / "B0CONFIG02"
    product.mkdir()
    (product / "data.json").write_text(
        json.dumps(
            {
                "asin": "B0CONFIG02",
                "title": "Lamp",
                "price": "$1",
                "url": "https://example.com",
                "platform": "amazon",
            }
        )
    )
    config = load_video_config_modular()
    config.global_output_directory = str(tmp_path)
    argv = ["producer", "--batch", "--batch-profile", "slideshow_images1"]

    with (
        patch.object(sys, "argv", argv),
        patch("src.video.producer.cli.load_video_config_modular", return_value=config),
        patch("src.video.producer.cli.setup_logging", return_value=Path("t.log")),
        patch("src.video.producer.cli.validate_config_and_exit_on_error"),
        patch("src.video.producer.cli.load_dotenv"),
        patch("os.getenv", return_value="dummy_key"),
        patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        patch(
            "src.video.producer.cli.create_video_for_product", new_callable=AsyncMock
        ) as create,
        patch("asyncio.sleep", return_value=None),
        patch("src.utils.connection_pool.close_global_pool", new_callable=AsyncMock),
    ):
        from src.video.producer.cli import main

        create.return_value = Path("video.mp4")
        await main()

    assert create.call_count == 1
    assert create.call_args.args[1].asin == "B0CONFIG02"


@pytest.mark.req("REQ-OPS-002")
@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], "price-asc-rank"), (["--sort", "relevance"], "relevanceblender")],
)
def test_the_scraper_sort_follows_the_yaml_unless_passed(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: str
) -> None:
    from src.scraper.amazon import cli as scraper_cli
    from src.scraper.amazon.models import SearchParameters

    monkeypatch.setattr(
        scraper_cli,
        "get_default_search_parameters",
        lambda: SearchParameters(sort_order="price-asc-rank"),
    )
    args = scraper_cli.build_argument_parser().parse_args(argv)

    result = scraper_cli._build_search_params(args)

    assert result is not None
    assert result[0].sort_order == expected
