"""End-to-end regression suite for the affiliate-content disclosure stack.

This file is the single canonical entry point for verifying the four disclosure
surfaces the pipeline produces. Individual unit tests for each surface live in
their own files (overlay_builder, test_models, test_publisher_integration);
this suite asserts the surfaces still cooperate when something downstream
changes. If any disclosure layer regresses, this is the test that should fail
loudly enough to catch it before the regression ships.

Surfaces covered:

1. On-frame overlay (FFmpeg drawtext, configured via DisclosureSettings).
2. Caption first-line text (PublishMetadata.format_content prepends #ad).
3. TikTok branded-content flags (commercialContentType, isBrandOrganicPost).
4. YouTube AI-content flag (containsSyntheticMedia).

See ``docs/compliance.md`` for the regulatory framing each surface satisfies.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.publisher.models import Platform, PublishMetadata
from src.video.assembler.overlay_builder import apply_disclosure_overlay
from src.video.config.visual_models import DisclosureSettings

# ---------------------------------------------------------------------------
# Minimal mock publisher for payload assertions
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_late_sdk():
    """SDK stub that records the last posts.create call for assertions."""
    client = MagicMock()
    upload_response = MagicMock()
    upload_response.url = "https://storage.late.dev/media_test.mp4"
    upload_response.files = []
    client.media.upload = MagicMock(return_value=upload_response)
    post_response = MagicMock()
    post_response.post = MagicMock(
        field_id="post_xyz",
        status=MagicMock(value="scheduled"),
        platforms=[],
    )
    client.posts.create = MagicMock(return_value=post_response)
    return client


@pytest.fixture
def mock_publisher(mock_late_sdk):
    """LatePublisher with the SDK stubbed."""
    with patch("src.publisher.late.client.Late", return_value=mock_late_sdk):
        from src.publisher.late.client import LatePublisher

        publisher = LatePublisher(
            api_key="sk_test_disclosure_stack",
            vercel_token="vercel_token_test",  # noqa: S106
            timeout=30.0,
            max_retries=1,
        )
        publisher.client = mock_late_sdk
        return publisher


# ---------------------------------------------------------------------------
# Surface 1: on-frame overlay
# ---------------------------------------------------------------------------


class TestOnFrameOverlay:
    """Surface 1: persistent #ad burned in a fixed corner of every render."""

    def test_assembler_filter_chain_gains_drawtext_overlay(self, tmp_path):
        # The assembler always ends with a "[stream]copy[v_out]" no-op. The
        # overlay rewrites that to a drawtext filter producing the same
        # [v_out] label so the rest of the FFmpeg command needs no changes.
        chain = [
            "[0:v]scale=1080:1920[v0]",
            "[v0]copy[v_subtitle]",
            "[v_subtitle]copy[v_out]",
        ]
        out = apply_disclosure_overlay(chain, DisclosureSettings(), 80, tmp_path)

        assert len(out) == len(chain)
        assert out[-1].startswith("[v_subtitle]drawtext=")
        assert out[-1].endswith("[v_out]")
        # Text goes through textfile=, not inline text=, so a localized value
        # with an apostrophe can't corrupt the multi-filter chain.
        assert "textfile=" in out[-1]
        assert (tmp_path / "disclosure_text.txt").read_text() == "#ad"

    def test_overlay_can_be_disabled_for_non_affiliate_renders(self, tmp_path):
        # The Phase 2.2 "non_affiliate" pillar mode (educational track) needs
        # an escape hatch so educational videos don't ship with #ad.
        chain = ["[v_subtitle]copy[v_out]"]
        out = apply_disclosure_overlay(
            chain, DisclosureSettings(enabled=False), 80, tmp_path
        )
        assert out == chain

    def test_overlay_text_propagates_through_to_filter(self, tmp_path):
        # Phase 0.4 will inject Spanish #publi via this same DisclosureSettings
        # path; verify the text override reaches the rendered filter today.
        chain = ["[v_subtitle]copy[v_out]"]
        apply_disclosure_overlay(chain, DisclosureSettings(text="#publi"), 80, tmp_path)
        assert (tmp_path / "disclosure_text.txt").read_text() == "#publi"

    def test_localized_text_with_apostrophe_survives(self, tmp_path):
        # The disclosure sits inside the assembler's multi-filter chain, where
        # an inline text= carrying an apostrophe made FFmpeg swallow the
        # filter's own trailing args and drop the disclosure entirely.
        chain = ["[v_subtitle]copy[v_out]"]
        out = apply_disclosure_overlay(
            chain, DisclosureSettings(text="Pub d'affiliation"), 80, tmp_path
        )
        assert "text='" not in out[-1]
        assert (tmp_path / "disclosure_text.txt").read_text() == "Pub d'affiliation"


# ---------------------------------------------------------------------------
# Surface 2: caption first-line disclosure
# ---------------------------------------------------------------------------


class TestCaptionFirstLine:
    """Surface 2: every formatted caption opens with the disclosure on its own line."""

    @pytest.mark.parametrize(
        "platform",
        [Platform.TIKTOK, Platform.INSTAGRAM, Platform.YOUTUBE],
    )
    def test_caption_leads_with_ad_disclosure(self, platform):
        kwargs = {
            "platform": platform,
            "title": "T" if platform == Platform.YOUTUBE else None,
            "description": "Body of the caption.",
            "hashtags": ["techfinds"],
        }
        content = PublishMetadata(**kwargs).format_content()

        assert content.split("\n", 1)[0] == "#ad"

    def test_disclosure_not_duplicated_when_present_in_hashtags(self):
        # Generators that auto-append #ad to hashtags shouldn't make the
        # disclosure appear twice in the published caption.
        meta = PublishMetadata(
            platform=Platform.TIKTOK,
            title=None,
            description="Body.",
            hashtags=["ad", "techfinds"],
        )
        content = meta.format_content()
        assert content.count("#ad") == 1
        assert content.startswith("#ad\n")

    def test_custom_disclosure_propagates(self):
        # Same Spanish-render scenario as the overlay test above; the caption
        # surface honors the same config knob.
        meta = PublishMetadata(
            platform=Platform.INSTAGRAM,
            title=None,
            description="Texto.",
            hashtags=["gadgets"],
            disclosure="#publi",
        )
        content = meta.format_content()
        assert content.startswith("#publi\n")


# ---------------------------------------------------------------------------
# Surface 3 + 4: platform-policy flags in publish payloads
# ---------------------------------------------------------------------------


class TestPlatformFlags:
    """Surfaces 3 and 4: platform-policy disclosure tags on every publish payload."""

    @pytest.mark.asyncio
    async def test_tiktok_payload_carries_brand_organic_flag(self, mock_publisher):
        await mock_publisher.publish(
            media_id="https://storage.late.dev/media.mp4",
            platforms=[{"platform": "tiktok", "account_id": "acc_tt"}],
            content="#ad\n\nBody.",
        )

        call = mock_publisher.client.posts.create.call_args
        sdk_platforms = call.kwargs.get("platforms", [])
        tiktok = next(p for p in sdk_platforms if p["platform"] == "tiktok")
        settings = tiktok["platformSpecificData"]["tiktokSettings"]
        assert settings["commercial_content_type"] == "brand_organic"
        assert settings["is_brand_organic_post"] is True

    @pytest.mark.asyncio
    async def test_tiktok_declares_none_without_a_material_connection(
        self, mock_publisher
    ):
        """A topic post promotes nothing and earns nothing.

        `brand_organic` tells viewers the creator is promoting their own
        business, which for a topic render is simply untrue. TikTok has a
        value for "not commercial content", and it is sent explicitly rather
        than by omitting the settings, since absence reads as a payload that
        forgot them.
        """
        await mock_publisher.publish(
            media_id="https://storage.late.dev/media.mp4",
            platforms=[{"platform": "tiktok", "account_id": "acc_tt"}],
            content="Body.",
            carries_affiliate_content=False,
        )

        call = mock_publisher.client.posts.create.call_args
        sdk_platforms = call.kwargs.get("platforms", [])
        tiktok = next(p for p in sdk_platforms if p["platform"] == "tiktok")
        settings = tiktok["platformSpecificData"]["tiktokSettings"]
        assert settings["commercial_content_type"] == "none"
        assert settings["is_brand_organic_post"] is False

    @pytest.mark.asyncio
    async def test_tiktok_gate_applies_on_the_platform_contents_branch(
        self, mock_publisher
    ):
        """The builder attaches TikTok settings at two sites.

        The `schedule` path always supplies platform_contents, so that branch
        is the live one for scheduled posts, while the other branch is what
        the simplest tests exercise. They must not drift apart.
        """
        await mock_publisher.publish(
            media_id="https://storage.late.dev/media.mp4",
            platforms=[{"platform": "tiktok", "account_id": "acc_tt"}],
            content="Body.",
            platform_contents={"tiktok": {"content": "Body."}},
            carries_affiliate_content=False,
        )

        call = mock_publisher.client.posts.create.call_args
        sdk_platforms = call.kwargs.get("platforms", [])
        tiktok = next(p for p in sdk_platforms if p["platform"] == "tiktok")
        settings = tiktok["platformSpecificData"]["tiktokSettings"]
        assert settings["commercial_content_type"] == "none"
        assert settings["is_brand_organic_post"] is False

    @pytest.mark.asyncio
    async def test_the_top_level_tiktok_block_agrees_with_the_platform_one(
        self, mock_publisher
    ):
        """The settings are sent twice, and both copies must say the same thing.

        A payload that declares "not commercial" per-platform and
        "brand_organic" at the top level is worse than either alone.
        """
        await mock_publisher.publish(
            media_id="https://storage.late.dev/media.mp4",
            platforms=[{"platform": "tiktok", "account_id": "acc_tt"}],
            content="Body.",
            carries_affiliate_content=False,
        )

        call = mock_publisher.client.posts.create.call_args
        top = call.kwargs["tiktok_settings"]
        sdk_platforms = call.kwargs.get("platforms", [])
        tiktok = next(p for p in sdk_platforms if p["platform"] == "tiktok")
        per_platform = tiktok["platformSpecificData"]["tiktokSettings"]
        assert top["commercialContentType"] == per_platform["commercial_content_type"]
        assert top["commercialContentType"] == "none"

    @pytest.mark.asyncio
    async def test_youtube_does_not_self_declare_synthetic_media_by_default(
        self, mock_publisher
    ):
        """YouTube's policy targets realistic content that could mislead about
        real people or events, and explicitly excludes AI narration, AI
        scripts and stock footage. Declaring it anyway applies a viewer-facing
        label the policy does not ask for.
        """
        await mock_publisher.publish(
            media_id="https://storage.late.dev/media.mp4",
            platforms=[{"platform": "youtube", "account_id": "acc_yt"}],
            content="#ad\n\nBody.",
        )

        call = mock_publisher.client.posts.create.call_args
        sdk_platforms = call.kwargs.get("platforms", [])
        youtube = next(p for p in sdk_platforms if p["platform"] == "youtube")
        psd = youtube["platformSpecificData"]
        assert psd["containsSyntheticMedia"] is False

    @pytest.mark.asyncio
    async def test_youtube_declares_synthetic_media_when_configured(
        self, mock_publisher
    ):
        """The flag is gated, not removed. Output that does meet the bar --
        AI-generated music, or AI footage of a real place -- must still be
        able to declare it.
        """
        mock_publisher.synthetic_media_disclosure = True
        await mock_publisher.publish(
            media_id="https://storage.late.dev/media.mp4",
            platforms=[{"platform": "youtube", "account_id": "acc_yt"}],
            content="#ad\n\nBody.",
        )

        call = mock_publisher.client.posts.create.call_args
        sdk_platforms = call.kwargs.get("platforms", [])
        youtube = next(p for p in sdk_platforms if p["platform"] == "youtube")
        assert youtube["platformSpecificData"]["containsSyntheticMedia"] is True


# ---------------------------------------------------------------------------
# Cross-surface invariant: the four surfaces never disagree silently
# ---------------------------------------------------------------------------


class TestStackInvariants:
    """Cross-layer assertions that catch silent drift between disclosure surfaces."""

    def test_default_disclosure_is_consistent_across_surfaces(self):
        # The on-frame overlay default and the caption-text default must agree
        # so a render and a published caption don't ship contradictory tags.
        overlay_default = DisclosureSettings().text
        meta = PublishMetadata(
            platform=Platform.TIKTOK,
            title=None,
            description="Body.",
        )
        caption_default = meta.disclosure
        assert overlay_default == caption_default == "#ad"

    def test_custom_disclosure_propagates_to_both_surfaces(self, tmp_path):
        # When a future Phase 0.4 wires Spanish renders, setting the disclosure
        # value on both DisclosureSettings and PublishMetadata is what the
        # pipeline needs to do consistently. Verify the propagation today.
        spanish_overlay = DisclosureSettings(text="#publi")
        meta = PublishMetadata(
            platform=Platform.INSTAGRAM,
            title=None,
            description="Texto.",
            disclosure="#publi",
        )
        assert spanish_overlay.text == meta.disclosure == "#publi"

        apply_disclosure_overlay(
            ["[v_subtitle]copy[v_out]"], spanish_overlay, 80, tmp_path
        )
        assert (tmp_path / "disclosure_text.txt").read_text() == "#publi"
        assert meta.format_content().startswith("#publi\n")
