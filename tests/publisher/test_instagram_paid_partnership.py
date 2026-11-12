"""Instagram's paid-partnership label follows the render's material connection.

REQ-CMP-026: with `instagram_paid_partnership` on, every Instagram payload
carries `isPaidPartnership`, true for a render with a material connection and
false for one without; an account the provider refuses it for is posted to
without it.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from late.client.exceptions import LateAPIError

from src.publisher.late.client import (
    PAID_PARTNERSHIP_LOGIN_ERROR,
    LatePublisher,
    drop_paid_partnership,
)
from src.publisher.models import PublisherConfig

pytestmark = pytest.mark.req("REQ-CMP-026")

INSTAGRAM = [{"platform": "instagram", "account_id": "acc1"}]


def _payload(on: bool, carries: bool, platform_contents=None):
    publisher = LatePublisher(api_key="k" * 40, instagram_paid_partnership=on)
    platforms, _ = publisher._build_sdk_platforms(
        INSTAGRAM,
        "media_1",
        platform_contents=platform_contents,
        carries_affiliate_content=carries,
    )
    return platforms[0].get("platformSpecificData", {})


def test_it_is_off_by_default():
    fields = PublisherConfig.__dataclass_fields__
    assert fields["instagram_paid_partnership"].default is False
    assert LatePublisher(api_key="k" * 40).instagram_paid_partnership is False
    assert "isPaidPartnership" not in _payload(on=False, carries=True)


@pytest.mark.parametrize("carries", [True, False])
def test_on_it_follows_the_material_connection(carries):
    assert _payload(on=True, carries=carries)["isPaidPartnership"] is carries


def test_it_sits_beside_the_first_comment():
    payload = _payload(
        on=True,
        carries=True,
        platform_contents={"instagram": {"content": "c", "first_comment": "fc"}},
    )

    assert payload["isPaidPartnership"] is True
    assert payload["firstComment"] == "fc"


def test_other_platforms_do_not_carry_it():
    publisher = LatePublisher(api_key="k" * 40, instagram_paid_partnership=True)
    platforms, _ = publisher._build_sdk_platforms(
        [{"platform": "tiktok", "account_id": "acc2"}],
        "media_1",
        platform_contents=None,
    )

    psd = platforms[0]["platformSpecificData"]
    assert isinstance(psd, dict)
    assert "isPaidPartnership" not in psd


def test_the_shared_factory_passes_the_configured_value():
    from src.publisher.registry import create_publisher_from_config

    config = PublisherConfig(
        provider="late", api_key="k" * 40, instagram_paid_partnership=True
    )
    with patch("src.publisher.registry.create_publisher") as create:
        create_publisher_from_config(config)

    assert create.call_args.kwargs["instagram_paid_partnership"] is True


def test_only_the_login_refusal_drops_the_flag():
    entries = [
        {"platform": "instagram", "platformSpecificData": {"isPaidPartnership": True}}
    ]

    assert not drop_paid_partnership(entries, LateAPIError("other", 400))
    assert entries[0]["platformSpecificData"] == {"isPaidPartnership": True}
    assert drop_paid_partnership(
        entries, LateAPIError(PAID_PARTNERSHIP_LOGIN_ERROR, 400)
    )
    assert entries[0]["platformSpecificData"] == {}


@pytest.mark.asyncio
async def test_a_refused_label_posts_again_without_it(caplog):
    publisher = LatePublisher(
        api_key="sk_test_abc123", max_retries=1, instagram_paid_partnership=True
    )
    publisher.client = MagicMock()
    sent: list[dict] = []

    def create(**kwargs):
        sent.append(dict(kwargs["platforms"][0]["platformSpecificData"]))
        if len(sent) == 1:
            raise LateAPIError(PAID_PARTNERSHIP_LOGIN_ERROR, 400)
        return MagicMock(post=MagicMock(field_id="post_1"))

    publisher.client.posts.create.side_effect = create

    with patch("src.publisher.late.client.asyncio.sleep", new_callable=AsyncMock):
        result = await publisher.publish(
            media_id="https://blob.test/v.mp4",
            platforms=INSTAGRAM,
            content="A caption",
            carries_affiliate_content=True,
        )

    assert sent[0]["isPaidPartnership"] is True
    assert "isPaidPartnership" not in sent[1]
    assert result["post_id"] == "post_1"
    assert "Set the label by hand" in caplog.text
