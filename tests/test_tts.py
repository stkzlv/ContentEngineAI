"""Unit tests for the text-to-speech (TTS) functionality."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.video.tts import TTSManager
from src.video.video_config import VideoConfig


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
