"""Tests for the shared post-publish link-in-bio hook."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.publisher.link_in_bio.manager import update_link_in_bio_safe
from src.publisher.models import LinkInBioConfig


@pytest.mark.asyncio
async def test_defaults_to_enabled():
    """No config means enabled: the manager is created and update runs."""
    with patch(
        "src.publisher.link_in_bio.manager.create_link_in_bio_manager"
    ) as factory:
        mgr = factory.return_value
        mgr.update = AsyncMock(return_value={"success": True})

        await update_link_in_bio_safe("B0TEST", Path("outputs"))

        factory.assert_called_once()
        mgr.update.assert_awaited_once_with("B0TEST", Path("outputs"))


@pytest.mark.asyncio
async def test_disabled_config_is_a_noop():
    with patch(
        "src.publisher.link_in_bio.manager.create_link_in_bio_manager"
    ) as factory:
        await update_link_in_bio_safe(
            "B0TEST", Path("outputs"), LinkInBioConfig(enabled=False)
        )
        factory.assert_not_called()


@pytest.mark.asyncio
async def test_config_values_reach_the_factory():
    cfg = LinkInBioConfig(provider="lnkbio", max_links=5, max_title_length=40)
    with patch(
        "src.publisher.link_in_bio.manager.create_link_in_bio_manager"
    ) as factory:
        factory.return_value.update = AsyncMock(return_value={"success": True})

        await update_link_in_bio_safe("B0TEST", Path("outputs"), cfg)

        factory.assert_called_once_with(
            provider_name="lnkbio", max_links=5, max_title_length=40
        )


@pytest.mark.asyncio
async def test_provider_errors_never_raise():
    with patch(
        "src.publisher.link_in_bio.manager.create_link_in_bio_manager"
    ) as factory:
        factory.return_value.update = AsyncMock(side_effect=RuntimeError("API down"))

        # Must not raise; failures only WARN so publishing is never blocked
        await update_link_in_bio_safe("B0TEST", Path("outputs"))


@pytest.mark.asyncio
async def test_unsuccessful_result_never_raises():
    with patch(
        "src.publisher.link_in_bio.manager.create_link_in_bio_manager"
    ) as factory:
        factory.return_value.update = AsyncMock(
            return_value={"success": False, "reason": "no_data"}
        )

        await update_link_in_bio_safe("B0TEST", Path("outputs"))
