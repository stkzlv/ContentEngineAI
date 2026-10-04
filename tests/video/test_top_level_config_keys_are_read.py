"""Every top-level block in the merged video config reaches something.

`VideoConfig` ignores unknown top-level keys, so a block with no field loads
into nothing and its values never apply. `batch.profile_pool` was one: the
producer looked for it with `hasattr(config, "batch")`, which was always false,
so the YAML pool was never drawn from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from pydantic import ValidationError

from src.config_manager import UnifiedConfigManager, get_unified_config_manager
from src.video.config import VideoConfig, load_video_config_modular
from src.video.config.core_models import BatchSettings

# Blocks read straight from their YAML file rather than through `VideoConfig`.
READ_ELSEWHERE = {
    "circuit_breaker": "src/utils/circuit_breaker.py reads performance.yaml",
}


def test_every_top_level_block_is_a_field_or_read_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The YAML files only: an exported OUTPUTS_DIR writes
    # scraper paths into this dict, which is not a YAML block.
    monkeypatch.setattr(
        UnifiedConfigManager, "_apply_env_overrides", lambda self, config: None
    )
    merged = get_unified_config_manager().get_video_config(None)

    unread = set(merged) - set(VideoConfig.model_fields) - set(READ_ELSEWHERE)

    assert unread == set()


@pytest.mark.req("REQ-BAT-057")
def test_the_bundled_pool_keeps_every_eligible_profile() -> None:
    data = yaml.safe_load(Path("config/video_production.yaml").read_text())

    assert data["batch"]["profile_pool"] == []


def test_the_batch_block_refuses_an_unknown_key() -> None:
    with pytest.raises(ValidationError, match="profile_pol"):
        BatchSettings.model_validate({"profile_pol": ["slideshow_images1"]})


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    for asin in ("B0POOL0001", "B0POOL0002", "B0POOL0003"):
        (tmp_path / asin).mkdir()
        (tmp_path / asin / "data.json").write_text(
            json.dumps(
                {
                    "asin": asin,
                    "title": asin,
                    "price": "$10",
                    "url": "https://example.com",
                    "platform": "amazon",
                }
            )
        )
    return tmp_path


@pytest.mark.req("REQ-BAT-057")
@pytest.mark.asyncio
async def test_a_random_batch_draws_from_the_yaml_pool(outputs: Path) -> None:
    config = load_video_config_modular()
    config.batch.profile_pool = ["slideshow_images2"]
    argv = ["producer", "--batch", "--random-profile", "--outputs-dir", str(outputs)]

    with (
        patch.object(sys, "argv", argv),
        patch("src.video.producer.cli.load_video_config_modular", return_value=config),
        patch("src.video.producer.cli.setup_logging", return_value=Path("test.log")),
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

    assert create.call_count == 3
    assert {c.args[2] for c in create.call_args_list} == {"slideshow_images2"}


@pytest.mark.req("REQ-BAT-058")
@pytest.mark.asyncio
async def test_a_yaml_pool_naming_a_missing_profile_stops_the_batch(
    outputs: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = load_video_config_modular()
    config.batch.profile_pool = ["no_such_profile"]
    argv = ["producer", "--batch", "--random-profile", "--outputs-dir", str(outputs)]

    with (
        patch.object(sys, "argv", argv),
        patch("src.video.producer.cli.load_video_config_modular", return_value=config),
        patch("src.video.producer.cli.setup_logging", return_value=Path("test.log")),
        patch("src.video.producer.cli.validate_config_and_exit_on_error"),
        patch("src.video.producer.cli.load_dotenv"),
        patch("os.getenv", return_value="dummy_key"),
        patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("src.utils.connection_pool.close_global_pool", new_callable=AsyncMock),
        pytest.raises(SystemExit) as stopped,
    ):
        from src.video.producer.cli import main

        await main()

    assert stopped.value.code == 1
    assert "Invalid profile pool configuration" in caplog.text
    assert "no_such_profile" in caplog.text
