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


@pytest.mark.req("REQ-OPS-001")
@pytest.mark.parametrize(
    ("cli", "env", "configured", "expected"),
    [
        ("/srv/cli", {"OUTPUTS_DIR": "/srv/env"}, "yaml", "/srv/cli"),
        (None, {"OUTPUTS_DIR": "/srv/env"}, "yaml", "/srv/env"),
        (None, {"CONTENT_ENGINE_OUTPUT": "/srv/alt"}, "yaml", "/srv/alt"),
        (
            None,
            {"OUTPUTS_DIR": "/srv/env", "CONTENT_ENGINE_OUTPUT": "/srv/alt"},
            None,
            "/srv/env",
        ),
        (None, {}, "/srv/yaml", "/srv/yaml"),
        (None, {}, None, str(REPO / "outputs")),
        (None, {"OUTPUTS_DIR": "renders"}, None, str(REPO / "renders")),
    ],
)
def test_the_outputs_root_resolves_cli_then_machine_then_config(
    monkeypatch: pytest.MonkeyPatch,
    cli: str | None,
    env: dict[str, str],
    configured: str | None,
    expected: str,
) -> None:
    from src.utils.outputs_paths import resolve_outputs_dir

    for name in ("OUTPUTS_DIR", "CONTENT_ENGINE_OUTPUT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert resolve_outputs_dir(cli, configured) == Path(expected)


@pytest.mark.req("REQ-OPS-001")
def test_the_batch_takes_outputs_dir_from_the_machine_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.pipeline.cli import create_argument_parser
    from src.pipeline.config import load_global_batch_config

    path = tmp_path / "pipeline.yaml"
    path.write_text(
        yaml.dump(
            {"global_batch": {"product_ids": ["B0CONFIG01"], "outputs_dir": "custom"}}
        )
    )
    monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "machine"))
    parser = create_argument_parser()

    from_env = load_global_batch_config(parser.parse_args([]), config_path=path)
    from_cli = load_global_batch_config(
        parser.parse_args(["--outputs-dir", str(tmp_path / "cli")]), config_path=path
    )

    assert from_env.outputs_dir == tmp_path / "machine"
    assert from_cli.outputs_dir == tmp_path / "cli"


def _publisher_main(
    argv: list[str], dotenv_sets: dict[str, str], monkeypatch: pytest.MonkeyPatch
):
    """Run the publisher's main up to its command, with `.env` setting vars."""
    from src.publisher.late import cli as publisher_cli

    def fake_dotenv(*_args, **_kwargs):
        # Through monkeypatch, so the variable does not outlive the test.
        for name, value in dotenv_sets.items():
            monkeypatch.setenv(name, value)

    return (
        publisher_cli,
        [
            patch.object(sys, "argv", ["publisher", *argv]),
            patch.object(publisher_cli, "load_dotenv", side_effect=fake_dotenv),
            patch.object(publisher_cli, "setup_debug_logging"),
            patch.object(publisher_cli, "load_publisher_config"),
        ],
    )


@pytest.mark.req("REQ-OPS-001")
@pytest.mark.asyncio
async def test_the_publisher_reads_outputs_dir_set_in_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OUTPUTS_DIR", raising=False)
    monkeypatch.delenv("CONTENT_ENGINE_OUTPUT", raising=False)
    publisher_cli, patches = _publisher_main(
        ["schedule"], {"OUTPUTS_DIR": str(tmp_path)}, monkeypatch
    )
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patch.object(publisher_cli, "cmd_schedule_auto", new_callable=AsyncMock) as run,
    ):
        await publisher_cli.main()

    assert run.call_args.args[0].outputs_dir == tmp_path


@pytest.mark.req("REQ-OPS-001")
@pytest.mark.asyncio
async def test_the_publisher_calendar_reads_the_schedule_under_that_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from src.publisher.late import cli as publisher_cli

    class StopError(Exception):
        pass

    seen: list[Path] = []

    def manager(schedule_path=None, **_kwargs):
        seen.append(schedule_path)
        raise StopError

    monkeypatch.setattr(publisher_cli, "ScheduleManager", manager)
    args = argparse.Namespace(outputs_dir=tmp_path)

    with pytest.raises(StopError):
        await publisher_cli.cmd_calendar(args, None, None)

    assert seen == [tmp_path / "state" / "schedule.json"]


@pytest.mark.req("REQ-OPS-001")
@pytest.mark.asyncio
async def test_a_single_publish_looks_for_the_product_under_that_root(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import argparse

    from src.publisher.late import cli as publisher_cli

    args = argparse.Namespace(outputs_dir=tmp_path, product_id="B0MISSING1")

    with pytest.raises(SystemExit):
        await publisher_cli.cmd_single(args, None, None)

    assert str(tmp_path / "B0MISSING1") in caplog.text


@pytest.mark.req("REQ-OPS-001")
@pytest.mark.parametrize(
    "env",
    [
        {},
        {"OUTPUTS_DIR": "/srv/env"},
        {"OUTPUTS_DIR": "renders"},
        {"OUTPUTS_DIR": "~/renders"},
        {"OUTPUTS_DIR": "", "CONTENT_ENGINE_OUTPUT": "/srv/alt"},
        {"OUTPUTS_DIR": "/srv/env", "CONTENT_ENGINE_OUTPUT": "/srv/alt"},
    ],
)
def test_the_publisher_and_batch_root_matches_the_producers(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    from src.utils.outputs_paths import get_project_root, resolve_outputs_dir

    for name in ("OUTPUTS_DIR", "CONTENT_ENGINE_OUTPUT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    producer = get_project_root() / video_config({})["global_output_directory"]

    assert resolve_outputs_dir(None) == producer


@pytest.mark.req("REQ-OPS-001")
def test_an_override_does_not_outlive_its_load() -> None:
    """A later load without the override sees the YAML value again.

    Through the shared manager, whose adapter caches the merged YAML: the
    batch loads with overrides and other phases load without them.
    """
    from src.config_manager import get_unified_config_manager

    manager = get_unified_config_manager()
    before = manager.get_video_config(None)["description_settings"]["metadata_mode"]
    manager.get_video_config({"description_settings.metadata_mode": "optimized"})

    after = manager.get_video_config(None)["description_settings"]["metadata_mode"]
    assert after == before != "optimized"


@pytest.mark.parametrize(
    ("block", "key"),
    [
        ("audio_settings", "freesound_download_chunk_size"),
        ("media_settings", "temp_media_dir"),
        ("video_settings", "default_max_chars_per_line"),
    ],
)
def test_the_unread_video_keys_are_gone(block: str, key: str) -> None:
    """Nothing read them, so setting them changed nothing."""
    from src.video.config import load_video_config_modular

    config = load_video_config_modular()

    assert not hasattr(getattr(config, block), key)


def test_an_old_yaml_setting_the_removed_video_key_fails_the_load() -> None:
    """The bundled settings validate; the same plus the old key does not."""
    from pydantic import ValidationError

    from src.video.config import load_video_config_modular
    from src.video.config.visual_models import VideoSettings

    current = load_video_config_modular().video_settings.model_dump()
    VideoSettings.model_validate(current)

    with pytest.raises(ValidationError, match="default_max_chars_per_line"):
        VideoSettings.model_validate({**current, "default_max_chars_per_line": 20})


def test_an_old_yaml_setting_script_paths_fails_the_load() -> None:
    """Nothing read it; a config still setting it is told so."""
    from pydantic import ValidationError

    from src.video.config import load_video_config_modular
    from src.video.config.subtitle_models import SubtitleSettings

    current = load_video_config_modular().subtitle_settings
    current = current if isinstance(current, dict) else current.model_dump()
    SubtitleSettings.model_validate(current)

    with pytest.raises(ValidationError, match="script_paths"):
        SubtitleSettings.model_validate({**current, "script_paths": ["x.txt"]})
