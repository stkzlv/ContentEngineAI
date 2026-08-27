"""TikTok's AI-generated-content label reaches every TikTok payload.

TikTok requires the label for AI-generated speech and extends it to AI
voiceover even when the footage is real; every render here carries an AI TTS
voiceover. Undisclosed AI content is auto-labelled from C2PA credentials and
an auto-flag suppresses distribution, so the label protects reach rather than
costing it.
"""

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


class TestBothPathsHonourTheConfiguredSettings:
    """Same YAML, same payload, whichever entry point publishes.

    #255: the batch built its own publisher and passed no settings, so it ran
    on the dataclass defaults. A deliberate opt-out applied on the `single` and
    `schedule` paths and was silently ignored on the one `CLAUDE.md` names as
    the default for batch runs.

    Each producer of the settings is driven through its own real code here.
    Calling the shared parser twice and comparing the results would assert one
    deterministic function against itself, which no change can falsify -- and
    the loader dropping the section is #255 happening on the other path.
    """

    def test_the_default_is_on(self):
        """If this ever flips, every post goes out undisclosed by default."""
        assert TikTokContentSettings().video_made_with_ai is True

    def test_the_two_producers_agree_on_a_configured_opt_out(self):
        from src.publisher.config import (
            _parse_schedule_and_cleanup_config,
            parse_tiktok_settings,
        )

        section = {"video_made_with_ai": False}
        # The publisher CLI reaches its settings through the loader; the batch
        # reads publisher.yaml inline and calls the parser directly.
        cli_settings = _parse_schedule_and_cleanup_config(
            {"tiktok_settings": dict(section)}
        )["tiktok_settings"]
        batch_settings = parse_tiktok_settings(dict(section))

        assert cli_settings == batch_settings
        assert cli_settings.to_platform_data() == {"videoMadeWithAi": False}

    def test_the_batch_passes_what_the_yaml_section_says(self):
        """Reads the call site, because nothing else proves it is wired.

        The parser being shared is not the fix on its own -- the defect was a
        call site that never called any parser at all.

        Pinned to the call that also carries `synthetic_media_disclosure`, so
        moving the kwarg onto the slot-occupancy publisher a few lines above
        fails here rather than keeping a bare count at one.
        """
        import ast
        from pathlib import Path

        tree = ast.parse(Path("src/pipeline/global_batch.py").read_text())
        wired = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "create_publisher"
            and {"tiktok_settings", "synthetic_media_disclosure"}
            <= {kw.arg for kw in node.keywords}
        ]
        assert len(wired) == 1, (
            "the batch's publishing publisher must pass tiktok_settings; the "
            "slot-occupancy publisher never publishes and must not need it"
        )


class TestTheConfiguredValueSurvivesIntoThePayload:
    """The seam between a settings object and what TikTok receives.

    Everything above stops at the settings object. That is one link short:
    `_build_sdk_platforms` could ignore it and hardcode the flag, and every
    other test here would stay green while a configured opt-out published as
    a disclosure.
    """

    @staticmethod
    def _tiktok_payload(settings):
        from src.publisher.late.client import LatePublisher

        publisher = LatePublisher(api_key="k" * 40, tiktok_settings=settings)
        platforms, _ = publisher._build_sdk_platforms(
            [{"platform": "tiktok", "account_id": "acc1"}],
            "media_1",
            platform_contents=None,
            carries_affiliate_content=True,
        )
        return platforms[0]["platformSpecificData"]

    def test_an_opt_out_publishes_as_an_opt_out(self):
        payload = self._tiktok_payload(TikTokContentSettings(video_made_with_ai=False))

        assert payload["videoMadeWithAi"] is False

    def test_a_stock_install_publishes_the_label(self):
        payload = self._tiktok_payload(TikTokContentSettings())

        assert payload["videoMadeWithAi"] is True

    def test_a_neighbouring_field_travels_too(self):
        """The label is not the only thing #255 was dropping."""
        payload = self._tiktok_payload(TikTokContentSettings(allow_duet=True))

        assert payload["tiktokSettings"]["allow_duet"] is True
