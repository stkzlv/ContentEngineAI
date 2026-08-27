"""TikTok's AI-generated-content label reaches every TikTok payload.

TikTok requires the label for AI-generated speech and extends it to AI
voiceover even when the footage is real; every render here carries an AI TTS
voiceover. Undisclosed AI content is auto-labelled from C2PA credentials and
an auto-flag suppresses distribution, so the label protects reach rather than
costing it.
"""

import pytest

from src.publisher.models import TikTokContentSettings


class TestTheLabelIsOnByDefault:
    def test_a_stock_install_declares_it(self):
        """An install that configures nothing still discloses.

        The opposite default from YouTube's synthetic-media disclosure, on
        purpose: TikTok names AI speech, YouTube excludes voiceover.
        """
        assert TikTokContentSettings().video_made_with_ai is True

    def test_it_can_be_turned_off(self):
        assert (
            TikTokContentSettings(video_made_with_ai=False).video_made_with_ai is False
        )

    def test_it_survives_the_non_affiliate_rewrite(self):
        """`for_render` rewrites the commercial fields and must not touch this.

        A topic video with no affiliate link is not commercial content, but it
        is still AI-voiced.
        """
        settings = TikTokContentSettings().for_render(carries_affiliate_content=False)

        assert settings.commercial_content_type == "none"
        assert settings.video_made_with_ai is True


class TestItIsSentWhereTheSdkModelsIt:
    """Flat beside `tiktokSettings`, not inside it.

    The SDK types `platformSpecificData` as a flat `TikTokPlatformData` and
    models no `tiktokSettings` key at all. The nested block this project sends
    is a legacy shape the API still tolerates, so a field the SDK does model
    goes where the SDK models it.
    """

    def test_the_flat_payload_carries_the_camel_case_key(self):
        data = TikTokContentSettings().to_platform_data()

        assert data == {"videoMadeWithAi": True}

    def test_the_key_matches_the_sdk_field_name(self):
        """Guards against a rename or a snake_case slip.

        `platformSpecificData` is passed through as a raw dict, so a key the
        API does not recognise is dropped in silence and the post publishes
        undisclosed.
        """
        from late.models._generated.models import TikTokPlatformData

        key = next(iter(TikTokContentSettings().to_platform_data()))
        assert key in TikTokPlatformData.model_fields

    def test_the_nested_block_does_not_carry_it(self):
        """Sending it twice in two shapes would invite them to disagree."""
        assert "videoMadeWithAi" not in TikTokContentSettings().to_sdk_dict()
        assert "video_made_with_ai" not in TikTokContentSettings().to_sdk_dict()


class TestEveryTikTokPayloadIncludesIt:
    """Both build sites, because a post that skips one publishes undisclosed."""

    @pytest.mark.parametrize("marker", ["to_platform_data()"])
    def test_both_client_sites_spread_the_flat_payload(self, marker):
        from pathlib import Path

        text = Path("src/publisher/late/client.py").read_text()
        assert text.count(f"**tiktok_settings.{marker}") == 2, (
            "both platformSpecificData builders must carry the label; "
            "one that does not publishes an undisclosed AI post"
        )
