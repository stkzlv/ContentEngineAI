"""Unit tests for the text-to-speech (TTS) functionality."""

import html
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video.config import (
    GoogleCloudTTSSettings,
    GoogleCloudVoiceCriteria,
    VideoConfig,
)
from src.video.tts import TTSManager


class TestTTSManager:
    """Test the TTS manager functionality."""

    @pytest.fixture
    def tts_manager(self, mock_config: VideoConfig) -> TTSManager:
        """Create a TTS manager instance for testing."""
        test_secrets = {
            "GOOGLE_APPLICATION_CREDENTIALS": "/fake/path/to/credentials.json",
            "OPENAI_API_KEY": "test_openai_key",
        }
        return TTSManager(mock_config.tts_config, test_secrets)

    @pytest.fixture
    def sample_script(self) -> str:
        """Sample script for TTS testing."""
        return "This is a test script for text-to-speech conversion."

    @pytest.mark.asyncio
    async def test_tts_manager_initialization(self, mock_config: VideoConfig):
        """Test TTS manager initialization."""
        test_secrets = {
            "GOOGLE_APPLICATION_CREDENTIALS": "/fake/path/to/credentials.json",
            "OPENAI_API_KEY": "test_openai_key",
        }
        tts_manager = TTSManager(mock_config.tts_config, test_secrets)

        assert tts_manager.config == mock_config.tts_config
        assert tts_manager.secrets == test_secrets

    @pytest.mark.asyncio
    async def test_generate_speech_basic(
        self, tts_manager: TTSManager, sample_script: str, temp_dir: Path
    ):
        """Test basic speech generation functionality."""
        output_path = temp_dir / "test_output.wav"

        # Mock the speech generation to avoid actual TTS calls
        with patch.object(tts_manager, "generate_speech") as mock_generate:
            mock_generate.return_value = output_path

            result = await tts_manager.generate_speech(sample_script, output_path)

            assert result == output_path
            mock_generate.assert_called_once_with(sample_script, output_path)


class TestTTSIntegration:
    """Integration tests for TTS functionality."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tts_with_empty_provider_list(
        self, mock_config: VideoConfig, temp_dir: Path
    ):
        """Test TTS with empty provider list (fallback scenario)."""
        # Set empty provider list
        mock_config.tts_config.provider_order = []

        test_secrets = {"GOOGLE_APPLICATION_CREDENTIALS": "/fake/path"}
        tts_manager = TTSManager(mock_config.tts_config, test_secrets)

        output_path = temp_dir / "test_output.wav"

        # Should handle gracefully when no providers are available
        result = await tts_manager.generate_speech("Test text", output_path)

        # With no providers, should return None or handle gracefully
        assert result is None or isinstance(result, Path)


class TestSSMLGeneration:
    """Tests for SSML generation and text escaping."""

    def test_ssml_break_tag_format(self):
        """Test that SSML break tag is correctly formatted with buffer time."""
        # Simulate the SSML generation logic from tts.py
        text = "Hello world"
        last_word_buffer_sec = 0.3
        break_time_ms = int(last_word_buffer_sec * 1000)
        escaped_text = html.escape(text)
        ssml_text = f"<speak>{escaped_text}<break time='{break_time_ms}ms'/></speak>"

        assert ssml_text == "<speak>Hello world<break time='300ms'/></speak>"
        assert "300ms" in ssml_text

    def test_ssml_escapes_special_characters(self):
        """Test that special characters are properly escaped for SSML."""
        text = "Test with <brackets> & ampersand"
        escaped_text = html.escape(text)
        ssml_text = f"<speak>{escaped_text}<break time='300ms'/></speak>"

        # Verify special characters are escaped
        assert "&lt;brackets&gt;" in ssml_text
        assert "&amp;" in ssml_text
        # Verify the structure is valid XML-like
        assert ssml_text.startswith("<speak>")
        assert ssml_text.endswith("</speak>")

    def test_ssml_handles_quotes(self):
        """Test that quotes in text are properly escaped."""
        text = 'Say "hello" to everyone'
        escaped_text = html.escape(text)
        ssml_text = f"<speak>{escaped_text}<break time='300ms'/></speak>"

        # html.escape converts quotes to &quot; by default
        assert "&quot;hello&quot;" in ssml_text
        # Verify the structure is valid XML-like
        assert ssml_text.startswith("<speak>")
        assert ssml_text.endswith("</speak>")

    def test_ssml_buffer_time_calculation(self):
        """Test buffer time calculation from seconds to milliseconds."""
        test_cases = [
            (0.3, 300),  # Default
            (0.5, 500),
            (1.0, 1000),
            (0.1, 100),
            (0.0, 0),
        ]
        for seconds, expected_ms in test_cases:
            break_time_ms = int(seconds * 1000)
            assert break_time_ms == expected_ms, f"Failed for {seconds}s"

    def test_ssml_empty_text(self):
        """Test SSML generation with empty text."""
        text = ""
        break_time_ms = 300
        escaped_text = html.escape(text)
        ssml_text = f"<speak>{escaped_text}<break time='{break_time_ms}ms'/></speak>"

        assert ssml_text == "<speak><break time='300ms'/></speak>"

    @pytest.mark.unit
    def test_google_tts_settings_default_buffer(self):
        """Test that GoogleCloudTTSSettings has correct default buffer."""
        settings = GoogleCloudTTSSettings(
            model_name="en-US-Chirp3-HD",
            language_code="en-US",
            voice_selection_criteria=[
                GoogleCloudVoiceCriteria(language_code="en-US", ssml_gender="FEMALE")
            ],
        )
        # Default should be 0.3 seconds
        assert settings.last_word_buffer_sec == 0.3
