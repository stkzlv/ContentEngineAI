"""The Gemini style prompt reaches the voice only through its own field."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video import tts as tts_module
from src.video.config.audio_models import (
    GoogleCloudTTSSettings,
    GoogleCloudVoiceCriteria,
    TTSConfig,
    VoiceProfileConfig,
)
from src.video.tts import TTSManager, _generate_gemini_speech

pytestmark = pytest.mark.req("REQ-CNT-185")


def _settings() -> GoogleCloudTTSSettings:
    return GoogleCloudTTSSettings(
        language_code="en-US",
        voice_selection_criteria=[
            GoogleCloudVoiceCriteria(language_code="en-US", ssml_gender="FEMALE")
        ],
        api_max_retries=0,
        api_retry_delay_sec=0,
        api_timeout_sec=5,
    )


INSTRUCTION = "Say the following calmly:"


def _profile(
    model: str = "gemini-2.5-flash-tts", prompt: str = INSTRUCTION
) -> VoiceProfileConfig:
    return VoiceProfileConfig(
        provider="gemini", style_prompt=prompt, gemini_model_name=model
    )


def _voice() -> MagicMock:
    v = MagicMock()
    v.name = "Kore"
    v.language_codes = ["en-US"]
    return v


async def _sent_input(tmp_path: Path, profile: VoiceProfileConfig, send: bool):
    """Run the Gemini path and return the SynthesisInput it sent."""
    response = MagicMock()
    response.audio_content = b"RIFFdata"
    client = MagicMock()
    client.synthesize_speech = AsyncMock(return_value=response)
    with (
        patch.object(tts_module, "GOOGLE_CLOUD_AVAILABLE", True),
        patch.object(tts_module, "AIOFILES_AVAILABLE", True),
        patch.object(tts_module, "_global_google_cloud_client", client),
        patch.object(
            tts_module, "_fetch_available_voices", AsyncMock(return_value=[_voice()])
        ),
        patch.object(tts_module, "_filter_and_select_voice", return_value=_voice()),
    ):
        path, _ = await _generate_gemini_speech(
            "Hello there.",
            tmp_path / "out.wav",
            _settings(),
            profile,
            send_style_prompt=send,
        )
    assert path is not None
    return client.synthesize_speech.call_args.kwargs["request"].input


@pytest.mark.asyncio
async def test_switch_on_sends_the_prompt_in_its_own_field(tmp_path):
    sent = await _sent_input(tmp_path, _profile(), send=True)
    assert sent.prompt == INSTRUCTION
    assert sent.text == "Hello there."


@pytest.mark.asyncio
async def test_switch_off_sends_the_text_alone(tmp_path):
    sent = await _sent_input(tmp_path, _profile(), send=False)
    assert sent.prompt == ""
    assert sent.text == "Hello there."


@pytest.mark.asyncio
async def test_model_without_prompt_support_gets_no_prompt(tmp_path):
    sent = await _sent_input(tmp_path, _profile("gemini-3.8-flash-tts"), send=True)
    assert sent.prompt == ""


@pytest.mark.asyncio
async def test_descriptive_prompt_is_not_sent(tmp_path, caplog):
    profile = _profile(prompt="Warm but grounded. Like a friend.")
    sent = await _sent_input(tmp_path, profile, send=True)
    assert sent.prompt == ""
    assert sent.text == "Hello there."
    assert "doesn't end in a colon" in caplog.text


def test_switch_ships_off():
    config = TTSConfig(provider_order=["google_cloud"], google_cloud=_settings())
    assert config.send_style_prompt is False


@pytest.mark.asyncio
async def test_manager_passes_the_switch_to_the_gemini_path(tmp_path):
    config = TTSConfig(
        provider_order=["google_cloud"],
        google_cloud=_settings(),
        voice_profiles={"calm": _profile()},
        default_voice_profile="calm",
        send_style_prompt=True,
    )
    manager = TTSManager(config, {}, product_id="B0TEST0001")
    gemini = AsyncMock(return_value=(tmp_path / "v.wav", "Kore"))
    with patch("src.video.tts._generate_gemini_speech", gemini):
        await manager.generate_speech("Hello there.", tmp_path / "v.wav")
    assert gemini.call_args.kwargs["send_style_prompt"] is True
