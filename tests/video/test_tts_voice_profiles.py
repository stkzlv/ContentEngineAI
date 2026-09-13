"""Unit tests for TTS voice profile selection, markup, and metadata."""

import hashlib
import random
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video.config import VideoConfig
from src.video.config.audio_models import (
    GoogleCloudTTSSettings,
    GoogleCloudVoiceCriteria,
    TextMarkupRule,
    TTSConfig,
    VoiceProfileConfig,
)
from src.video.tts import TTSManager


def _make_tts_config(
    profiles: dict[str, VoiceProfileConfig] | None = None,
    pool: list[str] | None = None,
    enabled: bool = True,
    default_voice_profile: str | None = None,
) -> TTSConfig:
    """Build a minimal TTSConfig with voice profiles for testing."""
    return TTSConfig(
        provider_order=["google_cloud"],
        google_cloud=GoogleCloudTTSSettings(
            language_code="en-US",
            voice_selection_criteria=[
                GoogleCloudVoiceCriteria(language_code="en-US", ssml_gender="FEMALE")
            ],
        ),
        voice_profiles_enabled=enabled,
        voice_profiles=profiles or {},
        voice_profile_pool=pool or [],
        default_voice_profile=default_voice_profile,
    )


SAMPLE_PROFILES = {
    "warm": VoiceProfileConfig(
        provider="gemini",
        style_prompt="Speak warmly.",
        speaking_rate=1.05,
        markup_rules=[
            TextMarkupRule(pattern=r"\.\s+", insert_after="[short pause] "),
        ],
    ),
    "calm": VoiceProfileConfig(
        provider="gemini",
        style_prompt="Speak calmly.",
        speaking_rate=0.95,
    ),
    "chirp": VoiceProfileConfig(
        provider="google_cloud",
        voice_criteria=[
            GoogleCloudVoiceCriteria(
                language_code="en-US", name_contains="Chirp3", ssml_gender="FEMALE"
            ),
        ],
    ),
}


class TestVoiceProfileSelection:
    """Tests for _select_voice_profile deterministic selection."""

    def test_disabled_returns_none(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, enabled=False)
        mgr = TTSManager(cfg, {}, product_id="B0TEST001")
        name, profile = mgr._select_voice_profile()
        assert name is None
        assert profile is None

    def test_empty_profiles_returns_none(self):
        cfg = _make_tts_config(profiles={}, enabled=True)
        mgr = TTSManager(cfg, {})
        name, profile = mgr._select_voice_profile()
        assert name is None
        assert profile is None

    def test_deterministic_by_product_id(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        mgr = TTSManager(cfg, {}, product_id="B09S8WMJY9")
        name1, _ = mgr._select_voice_profile()
        name2, _ = mgr._select_voice_profile()
        assert name1 == name2, "Same product_id should always pick the same profile"

    def test_different_products_can_get_different_profiles(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        results = set()
        for asin in ["B0A", "B0B", "B0C", "B0D", "B0E", "B0F", "B0G", "B0H"]:
            mgr = TTSManager(cfg, {}, product_id=asin)
            name, _ = mgr._select_voice_profile()
            results.add(name)
        assert len(results) > 1, "Different products should produce variety"

    def test_pool_restricts_selection(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, pool=["chirp"])
        for asin in ["B0A", "B0B", "B0C", "B0D"]:
            mgr = TTSManager(cfg, {}, product_id=asin)
            name, _ = mgr._select_voice_profile()
            assert name == "chirp"

    def test_pool_filters_invalid_names(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, pool=["nonexistent", "warm"])
        mgr = TTSManager(cfg, {}, product_id="B0TEST001")
        name, profile = mgr._select_voice_profile()
        assert name == "warm"
        assert profile is not None

    def test_no_product_id_uses_random(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        mgr = TTSManager(cfg, {}, product_id=None)
        name, profile = mgr._select_voice_profile()
        assert name in SAMPLE_PROFILES
        assert profile is not None


class TestVoiceProfileOverride:
    """Tests for CLI voice profile override."""

    def test_override_selects_specified_profile(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        mgr = TTSManager(cfg, {}, product_id="B0TEST001", voice_profile_override="calm")
        name, profile = mgr._select_voice_profile()
        assert name == "calm"
        assert profile is not None
        assert profile.style_prompt == "Speak calmly."

    def test_override_ignores_product_hash(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        # Without override, different products get different profiles
        mgr1 = TTSManager(cfg, {}, product_id="B0A", voice_profile_override="warm")
        mgr2 = TTSManager(cfg, {}, product_id="B0B", voice_profile_override="warm")
        name1, _ = mgr1._select_voice_profile()
        name2, _ = mgr2._select_voice_profile()
        assert name1 == name2 == "warm"

    def test_invalid_override_falls_back(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        mgr = TTSManager(
            cfg, {}, product_id="B0TEST001", voice_profile_override="nonexistent"
        )
        name, profile = mgr._select_voice_profile()
        # Should fall back to normal selection, not None
        assert name in SAMPLE_PROFILES
        assert profile is not None


class TestDefaultVoiceProfile:
    """Tests for tts_config.default_voice_profile precedence."""

    def test_default_pins_voice_when_pool_empty(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, default_voice_profile="calm")
        for asin in ["B0A", "B0B", "B0C", "B0D"]:
            mgr = TTSManager(cfg, {}, product_id=asin)
            name, profile = mgr._select_voice_profile()
            assert name == "calm"
            assert profile is not None

    def test_cli_override_beats_default(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, default_voice_profile="calm")
        mgr = TTSManager(cfg, {}, product_id="B0TEST001", voice_profile_override="warm")
        name, _ = mgr._select_voice_profile()
        assert name == "warm"

    def test_non_empty_pool_beats_default(self):
        # Pool non-empty means user opted into random selection (testing path);
        # default_voice_profile should not pin.
        cfg = _make_tts_config(
            profiles=SAMPLE_PROFILES,
            pool=["chirp"],
            default_voice_profile="calm",
        )
        for asin in ["B0A", "B0B", "B0C"]:
            mgr = TTSManager(cfg, {}, product_id=asin)
            name, _ = mgr._select_voice_profile()
            assert name == "chirp"

    def test_invalid_default_falls_back_to_random(self):
        cfg = _make_tts_config(
            profiles=SAMPLE_PROFILES, default_voice_profile="nonexistent"
        )
        mgr = TTSManager(cfg, {}, product_id="B0TEST001")
        name, profile = mgr._select_voice_profile()
        assert name in SAMPLE_PROFILES
        assert profile is not None

    def test_no_default_keeps_random_back_compat(self):
        # default_voice_profile not set, pool empty, no override.
        # Behavior should match the prior random-across-all path.
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, default_voice_profile=None)
        results = set()
        for asin in ["B0A", "B0B", "B0C", "B0D", "B0E", "B0F", "B0G", "B0H"]:
            mgr = TTSManager(cfg, {}, product_id=asin)
            name, _ = mgr._select_voice_profile()
            results.add(name)
        assert len(results) > 1, "Variety preserved when no default is set"


class TestMarkupRules:
    """Tests for _apply_markup_rules and _strip_markup."""

    def test_apply_markup_inserts_after_sentences(self):
        rules = [TextMarkupRule(pattern=r"\.\s+", insert_after="[short pause] ")]
        text = "First sentence. Second sentence. Third."
        result = TTSManager._apply_markup_rules(text, rules)
        assert "[short pause]" in result
        assert result.count("[short pause]") == 2  # after 1st and 2nd periods

    def test_apply_markup_inserts_before(self):
        rules = [TextMarkupRule(pattern=r"!\s+", insert_before="[excited] ")]
        text = "Wow! That's great! Done."
        result = TTSManager._apply_markup_rules(text, rules)
        assert result.count("[excited]") == 2

    def test_apply_markup_empty_rules(self):
        text = "No changes expected."
        result = TTSManager._apply_markup_rules(text, [])
        assert result == text

    def test_apply_markup_multiple_rules(self):
        rules = [
            TextMarkupRule(pattern=r"\.\s+", insert_after="[pause] "),
            TextMarkupRule(pattern=r"!\s+", insert_after="[short pause] "),
        ]
        text = "Sentence one. Wow! Sentence three."
        result = TTSManager._apply_markup_rules(text, rules)
        assert "[pause]" in result
        assert "[short pause]" in result

    def test_strip_markup_removes_pause_tags(self):
        text = "Hello. [short pause] World. [pause] Done."
        result = TTSManager._strip_markup(text)
        assert "[short pause]" not in result
        assert "[pause]" not in result
        assert "Hello" in result
        assert "World" in result

    def test_strip_markup_removes_long_pause(self):
        text = "Wait[long pause]here"
        result = TTSManager._strip_markup(text)
        assert "[long pause]" not in result

    def test_strip_markup_no_tags_unchanged(self):
        text = "Plain text without any tags."
        result = TTSManager._strip_markup(text)
        assert result == text


class TestGenerateSpeechRouting:
    """Tests for generate_speech profile routing and metadata capture."""

    @pytest.mark.asyncio
    async def test_empty_text_returns_none(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        mgr = TTSManager(cfg, {})
        result = await mgr.generate_speech("", Path("/tmp/out.wav"))  # noqa: S108
        assert result is None

    @pytest.mark.asyncio
    async def test_whitespace_text_returns_none(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        mgr = TTSManager(cfg, {})
        result = await mgr.generate_speech("   \n  ", Path("/tmp/out.wav"))  # noqa: S108
        assert result is None

    @pytest.mark.asyncio
    async def test_gemini_profile_tries_gemini_first(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, pool=["warm"])
        mgr = TTSManager(cfg, {}, product_id="B0TEST")

        fake_path = Path("/tmp/test.wav")  # noqa: S108
        with patch(
            "src.video.tts._generate_gemini_speech", new_callable=AsyncMock
        ) as mock_gemini:
            mock_gemini.return_value = (fake_path, "Kore")
            result = await mgr.generate_speech("Test text.", fake_path)

            mock_gemini.assert_called_once()
            assert result == fake_path
            assert mgr.selected_profile_name == "warm"
            assert mgr.selected_voice_name == "Kore"

    @pytest.mark.asyncio
    async def test_gemini_failure_falls_back_to_google_cloud(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, pool=["warm"])
        mgr = TTSManager(cfg, {}, product_id="B0TEST")

        fake_path = Path("/tmp/test.wav")  # noqa: S108
        with (
            patch(
                "src.video.tts._generate_gemini_speech", new_callable=AsyncMock
            ) as mock_gemini,
            patch(
                "src.video.tts._generate_google_cloud_speech", new_callable=AsyncMock
            ) as mock_gc,
        ):
            mock_gemini.side_effect = Exception("Gemini unavailable")
            mock_gc.return_value = (fake_path, "en-US-Chirp3-HD-Achird")

            result = await mgr.generate_speech("Test text.", fake_path)

            assert result == fake_path
            assert mgr.selected_voice_name == "en-US-Chirp3-HD-Achird"

    @pytest.mark.asyncio
    async def test_google_cloud_profile_skips_gemini(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, pool=["chirp"])
        mgr = TTSManager(cfg, {}, product_id="B0TEST")

        fake_path = Path("/tmp/test.wav")  # noqa: S108
        with (
            patch(
                "src.video.tts._generate_gemini_speech", new_callable=AsyncMock
            ) as mock_gemini,
            patch(
                "src.video.tts._generate_google_cloud_speech", new_callable=AsyncMock
            ) as mock_gc,
        ):
            mock_gc.return_value = (fake_path, "en-US-Chirp3-HD-Achird")

            await mgr.generate_speech("Test text.", fake_path)

            mock_gemini.assert_not_called()
            mock_gc.assert_called_once()
            assert mgr.selected_profile_name == "chirp"

    @pytest.mark.asyncio
    async def test_markup_stripped_on_fallback(self):
        """When Gemini fails and fallback to google_cloud, markup should be stripped."""
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, pool=["warm"])
        mgr = TTSManager(cfg, {}, product_id="B0TEST")

        fake_path = Path("/tmp/test.wav")  # noqa: S108
        with (
            patch(
                "src.video.tts._generate_gemini_speech", new_callable=AsyncMock
            ) as mock_gemini,
            patch(
                "src.video.tts._generate_google_cloud_speech", new_callable=AsyncMock
            ) as mock_gc,
        ):
            mock_gemini.side_effect = Exception("Gemini down")
            mock_gc.return_value = (fake_path, "en-US-Chirp3-HD-Achird")

            await mgr.generate_speech("First sentence. Second sentence.", fake_path)

            # The text passed to google_cloud should not contain markup
            call_args = mock_gc.call_args
            text_arg = call_args[0][0]  # first positional arg is text
            assert "[short pause]" not in text_arg

    @pytest.mark.asyncio
    async def test_no_providers_returns_none(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, pool=["chirp"])
        cfg.provider_order = []
        mgr = TTSManager(cfg, {}, product_id="B0TEST")

        result = await mgr.generate_speech("Test.", Path("/tmp/test.wav"))  # noqa: S108
        assert result is None


class TestGenerateGeminiSpeechErrors:
    """Error-path branches of _generate_gemini_speech.

    Routing-level tests mock _generate_gemini_speech wholesale. These
    reach into the helper and exercise the catalog guards (empty Gemini
    voice list, no matching voice) and the retry/exception handling
    branches (GoogleAPIError, TimeoutError, DefaultCredentialsError,
    generic exception).
    """

    @staticmethod
    def _settings() -> GoogleCloudTTSSettings:
        return GoogleCloudTTSSettings(
            language_code="en-US",
            voice_selection_criteria=[
                GoogleCloudVoiceCriteria(language_code="en-US", ssml_gender="FEMALE")
            ],
            api_max_retries=1,
            api_retry_delay_sec=0,
            api_timeout_sec=5,
        )

    @staticmethod
    def _profile() -> VoiceProfileConfig:
        return SAMPLE_PROFILES["calm"]

    @staticmethod
    def _fake_voice() -> MagicMock:
        v = MagicMock()
        v.name = "Kore"
        v.language_codes = ["en-US"]
        return v

    @pytest.mark.asyncio
    async def test_empty_gemini_voice_catalog_returns_none(self, tmp_path):
        """All voices have hyphens in their names → Gemini filter yields []."""
        from src.video import tts as tts_module
        from src.video.tts import _generate_gemini_speech

        non_gemini = MagicMock()
        non_gemini.name = "en-US-Neural2-A"
        non_gemini.language_codes = ["en-US"]

        with (
            patch.object(tts_module, "GOOGLE_CLOUD_AVAILABLE", True),
            patch.object(tts_module, "AIOFILES_AVAILABLE", True),
            patch.object(tts_module, "_global_google_cloud_client", MagicMock()),
            patch.object(
                tts_module,
                "_fetch_available_voices",
                AsyncMock(return_value=[non_gemini]),
            ),
        ):
            out_path = tmp_path / "out.wav"
            result = await _generate_gemini_speech(
                "Hello.", out_path, self._settings(), self._profile()
            )
        assert result == (None, None)
        assert not out_path.exists()

    @pytest.mark.asyncio
    async def test_no_matching_voice_returns_none(self, tmp_path):
        """Gemini voice list non-empty but filter returns None."""
        from src.video import tts as tts_module
        from src.video.tts import _generate_gemini_speech

        with (
            patch.object(tts_module, "GOOGLE_CLOUD_AVAILABLE", True),
            patch.object(tts_module, "AIOFILES_AVAILABLE", True),
            patch.object(tts_module, "_global_google_cloud_client", MagicMock()),
            patch.object(
                tts_module,
                "_fetch_available_voices",
                AsyncMock(return_value=[self._fake_voice()]),
            ),
            patch.object(tts_module, "_filter_and_select_voice", return_value=None),
        ):
            out_path = tmp_path / "out.wav"
            result = await _generate_gemini_speech(
                "Hello.", out_path, self._settings(), self._profile()
            )
        assert result == (None, None)
        assert not out_path.exists()

    @pytest.mark.asyncio
    async def test_google_api_error_exhausts_retries(self, tmp_path):
        """GoogleAPIError on every attempt → (None, None) after retries."""
        from src.video import tts as tts_module
        from src.video.tts import _generate_gemini_speech

        fake_client = MagicMock()
        fake_client.synthesize_speech = AsyncMock(
            side_effect=tts_module.GoogleAPIError("boom")
        )

        with (
            patch.object(tts_module, "GOOGLE_CLOUD_AVAILABLE", True),
            patch.object(tts_module, "AIOFILES_AVAILABLE", True),
            patch.object(tts_module, "_global_google_cloud_client", fake_client),
            patch.object(
                tts_module,
                "_fetch_available_voices",
                AsyncMock(return_value=[self._fake_voice()]),
            ),
            patch.object(
                tts_module, "_filter_and_select_voice", return_value=self._fake_voice()
            ),
            patch("asyncio.sleep", AsyncMock()),
        ):
            out_path = tmp_path / "out.wav"
            result = await _generate_gemini_speech(
                "Hello.", out_path, self._settings(), self._profile()
            )
        # 1 retry allowed → 2 attempts total
        assert fake_client.synthesize_speech.call_count == 2
        assert result == (None, None)
        assert not out_path.exists()

    @pytest.mark.asyncio
    async def test_timeout_exhausts_retries(self, tmp_path):
        """asyncio.wait_for timeout on every attempt → (None, None).

        On Python 3.11+, TimeoutError is a subclass of OSError, so the
        earlier `except (OSError, ...)` clause in _generate_gemini_speech
        handles the timeout — not the explicit `except TimeoutError:`
        block below it. This test still exercises the real retry/cleanup
        behaviour; the TimeoutError-specific branch is effectively dead
        code on modern Python.
        """
        from src.video import tts as tts_module
        from src.video.tts import _generate_gemini_speech

        fake_client = MagicMock()
        fake_client.synthesize_speech = AsyncMock(side_effect=TimeoutError("slow"))

        with (
            patch.object(tts_module, "GOOGLE_CLOUD_AVAILABLE", True),
            patch.object(tts_module, "AIOFILES_AVAILABLE", True),
            patch.object(tts_module, "_global_google_cloud_client", fake_client),
            patch.object(
                tts_module,
                "_fetch_available_voices",
                AsyncMock(return_value=[self._fake_voice()]),
            ),
            patch.object(
                tts_module, "_filter_and_select_voice", return_value=self._fake_voice()
            ),
            patch("asyncio.sleep", AsyncMock()),
        ):
            out_path = tmp_path / "out.wav"
            result = await _generate_gemini_speech(
                "Hello.", out_path, self._settings(), self._profile()
            )
        assert fake_client.synthesize_speech.call_count == 2
        assert result == (None, None)
        assert not out_path.exists()

    @pytest.mark.asyncio
    async def test_default_credentials_error_breaks_immediately(self, tmp_path):
        """DefaultCredentialsError is non-retryable: break after first attempt."""
        from src.video import tts as tts_module
        from src.video.tts import _generate_gemini_speech

        fake_client = MagicMock()
        fake_client.synthesize_speech = AsyncMock(
            side_effect=tts_module.DefaultCredentialsError("no creds")
        )

        with (
            patch.object(tts_module, "GOOGLE_CLOUD_AVAILABLE", True),
            patch.object(tts_module, "AIOFILES_AVAILABLE", True),
            patch.object(tts_module, "_global_google_cloud_client", fake_client),
            patch.object(
                tts_module,
                "_fetch_available_voices",
                AsyncMock(return_value=[self._fake_voice()]),
            ),
            patch.object(
                tts_module, "_filter_and_select_voice", return_value=self._fake_voice()
            ),
            patch("asyncio.sleep", AsyncMock()) as mock_sleep,
        ):
            out_path = tmp_path / "out.wav"
            result = await _generate_gemini_speech(
                "Hello.", out_path, self._settings(), self._profile()
            )
        # Non-retryable: exactly one attempt, no sleep
        assert fake_client.synthesize_speech.call_count == 1
        mock_sleep.assert_not_called()
        assert result == (None, None)

    @pytest.mark.asyncio
    async def test_generic_exception_exhausts_retries(self, tmp_path):
        """Catch-all Exception branch also retries up to the configured limit."""
        from src.video import tts as tts_module
        from src.video.tts import _generate_gemini_speech

        fake_client = MagicMock()
        fake_client.synthesize_speech = AsyncMock(
            side_effect=RuntimeError("unexpected")
        )

        with (
            patch.object(tts_module, "GOOGLE_CLOUD_AVAILABLE", True),
            patch.object(tts_module, "AIOFILES_AVAILABLE", True),
            patch.object(tts_module, "_global_google_cloud_client", fake_client),
            patch.object(
                tts_module,
                "_fetch_available_voices",
                AsyncMock(return_value=[self._fake_voice()]),
            ),
            patch.object(
                tts_module, "_filter_and_select_voice", return_value=self._fake_voice()
            ),
            patch("asyncio.sleep", AsyncMock()),
        ):
            out_path = tmp_path / "out.wav"
            result = await _generate_gemini_speech(
                "Hello.", out_path, self._settings(), self._profile()
            )
        assert fake_client.synthesize_speech.call_count == 2
        assert result == (None, None)


class TestTextMarkupRuleModel:
    """Tests for TextMarkupRule Pydantic model."""

    def test_defaults(self):
        rule = TextMarkupRule(pattern=r"\.\s+")
        assert rule.insert_before == ""
        assert rule.insert_after == ""

    def test_full_config(self):
        rule = TextMarkupRule(
            pattern=r"!\s+",
            insert_before="[excited] ",
            insert_after="[pause] ",
        )
        assert rule.pattern == r"!\s+"
        assert rule.insert_before == "[excited] "
        assert rule.insert_after == "[pause] "


class TestVoiceProfileConfigModel:
    """Tests for VoiceProfileConfig Pydantic model."""

    def test_defaults(self):
        profile = VoiceProfileConfig()
        assert profile.provider == "google_cloud"
        assert profile.style_prompt is None
        assert profile.gemini_model_name == "gemini-2.5-flash-tts"
        assert profile.voice_criteria is None
        assert profile.speaking_rate is None
        assert profile.pitch is None
        assert profile.markup_rules == []

    def test_gemini_profile(self):
        profile = VoiceProfileConfig(
            provider="gemini",
            style_prompt="Speak warmly.",
            gemini_model_name="gemini-2.5-pro-tts",
            speaking_rate=1.1,
        )
        assert profile.provider == "gemini"
        assert profile.gemini_model_name == "gemini-2.5-pro-tts"

    def test_google_cloud_profile_with_criteria(self):
        profile = VoiceProfileConfig(
            provider="google_cloud",
            voice_criteria=[
                GoogleCloudVoiceCriteria(language_code="en-US", name_contains="Chirp3"),
            ],
        )
        assert profile.voice_criteria is not None
        assert len(profile.voice_criteria) == 1
        assert profile.voice_criteria[0].name_contains == "Chirp3"


class TestTTSConfigVoiceProfiles:
    """Tests for TTSConfig voice profile fields."""

    def test_voice_profiles_default_empty(self):
        cfg = _make_tts_config()
        assert cfg.voice_profiles == {}
        assert cfg.voice_profile_pool == []
        assert cfg.voice_profiles_enabled is True

    def test_voice_profiles_loaded(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        assert len(cfg.voice_profiles) == 3
        assert "warm" in cfg.voice_profiles
        assert cfg.voice_profiles["warm"].provider == "gemini"

    def test_voice_profiles_disabled(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, enabled=False)
        assert cfg.voice_profiles_enabled is False
        # Profiles still loaded, just not used
        assert len(cfg.voice_profiles) == 3
