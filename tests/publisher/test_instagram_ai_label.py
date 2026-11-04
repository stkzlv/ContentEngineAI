"""Instagram's AI-generated-media label reaches every Instagram payload.

The label can only be set when a post is created, so a payload branch that
drops it publishes a post that can never be labelled.
"""

import pytest

from src.publisher.late.client import LatePublisher
from src.publisher.models import PublisherConfig

INSTAGRAM = [{"platform": "instagram", "account_id": "acc1"}]


def _payload(label: bool, platform_contents=None, cover=None):
    publisher = LatePublisher(api_key="k" * 40, instagram_ai_label=label)
    platforms, _ = publisher._build_sdk_platforms(
        INSTAGRAM,
        "media_1",
        platform_contents=platform_contents,
        cover=cover,
    )
    return platforms[0].get("platformSpecificData", {})


@pytest.mark.req("REQ-CMP-025")
def test_it_is_off_by_default():
    assert PublisherConfig.__dataclass_fields__["instagram_ai_label"].default is False
    assert LatePublisher(api_key="k" * 40).instagram_ai_label is False


@pytest.mark.req("REQ-CMP-025")
@pytest.mark.parametrize("label", [True, False])
def test_the_shared_content_branch_sends_it(label):
    assert _payload(label)["isAiGenerated"] is label


@pytest.mark.req("REQ-CMP-025")
@pytest.mark.parametrize("label", [True, False])
def test_the_per_platform_branch_sends_it_beside_the_first_comment(label):
    payload = _payload(
        label,
        platform_contents={"instagram": {"content": "c", "first_comment": "fc"}},
    )

    assert payload["isAiGenerated"] is label
    assert payload["firstComment"] == "fc"


@pytest.mark.req("REQ-CMP-025")
def test_the_cover_merges_with_it():
    payload = _payload(True, cover={"instagram_url": "https://example.test/c.jpg"})

    assert payload["isAiGenerated"] is True
    assert payload["instagramThumbnail"] == "https://example.test/c.jpg"


@pytest.mark.req("REQ-CMP-025")
def test_other_platforms_do_not_carry_it():
    publisher = LatePublisher(api_key="k" * 40, instagram_ai_label=True)
    platforms, _ = publisher._build_sdk_platforms(
        [{"platform": "tiktok", "account_id": "acc2"}],
        "media_1",
        platform_contents=None,
    )

    psd = platforms[0]["platformSpecificData"]
    assert isinstance(psd, dict)
    assert "isAiGenerated" not in psd


@pytest.mark.req("REQ-CMP-025")
def test_the_shared_factory_passes_the_configured_value():
    """Both the publisher CLI and the batch build through this factory."""
    from unittest.mock import patch

    from src.publisher.registry import create_publisher_from_config

    config = PublisherConfig(provider="late", api_key="k" * 40, instagram_ai_label=True)
    with patch("src.publisher.registry.create_publisher") as create:
        create_publisher_from_config(config)

    assert create.call_args.kwargs["instagram_ai_label"] is True


@pytest.mark.req("REQ-CMP-025")
def test_a_non_boolean_value_is_refused(tmp_path, monkeypatch):
    from src.publisher.config import load_publisher_config

    monkeypatch.setenv("CONFIG_OVERLAYS", "off")
    path = tmp_path / "publisher.yaml"
    path.write_text('api_key: "sk_test_abc"\ninstagram_ai_label: "yes"\n')

    with pytest.raises(ValueError, match="instagram_ai_label"):
        load_publisher_config(path)
