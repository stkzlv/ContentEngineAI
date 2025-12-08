# src/video/config/audio_models.py
"""Audio-related configuration models for TTS, STT, and audio processing."""

import logging
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from src.video.config.constants import (
    DEFAULT_WHISPER_MODEL_DIR,
    FREESOUND_DEFAULT_DOWNLOAD_TIMEOUT_SEC,
    FREESOUND_DEFAULT_SEARCH_TIMEOUT_SEC,
    FREESOUND_DOWNLOAD_CHUNK_SIZE,
    FREESOUND_TOKEN_EXPIRY_SEC,
    FREESOUND_TOKEN_REFRESH_BUFFER_SEC,
    LLM_MODEL_FETCH_TIMEOUT_SEC,
    TTS_PITCH_MAX,
    TTS_PITCH_MIN,
    TTS_SPEAKING_RATE_MAX,
    TTS_SPEAKING_RATE_MIN,
)

logger = logging.getLogger(__name__)


class AudioSettings(BaseModel):
    music_volume_db: float
    voiceover_volume_db: float
    audio_mix_duration: str = Field("longest")
    background_music_paths: list[Path]
    freesound_api_key_env_var: str
    freesound_client_id_env_var: str = Field("FREESOUND_CLIENT_ID")
    freesound_client_secret_env_var: str = Field("FREESOUND_CLIENT_SECRET")  # noqa: S106
    freesound_refresh_token_env_var: str = Field("FREESOUND_REFRESH_TOKEN")  # noqa: S106
    freesound_sort: str = Field("rating_desc")
    freesound_search_query: str
    freesound_filters: str
    freesound_max_results: int
    freesound_max_search_duration_sec: int = Field(9999)
    freesound_api_timeout_sec: int = Field(FREESOUND_DEFAULT_SEARCH_TIMEOUT_SEC)
    freesound_download_timeout_sec: int = Field(FREESOUND_DEFAULT_DOWNLOAD_TIMEOUT_SEC)
    freesound_token_expiry_sec: int = Field(FREESOUND_TOKEN_EXPIRY_SEC)
    freesound_token_refresh_buffer_sec: int = Field(FREESOUND_TOKEN_REFRESH_BUFFER_SEC)
    freesound_download_chunk_size: int = Field(FREESOUND_DOWNLOAD_CHUNK_SIZE)
    output_audio_codec: str = Field("aac")
    output_audio_bitrate: str = Field("192k")
    music_fade_in_duration: float = Field(2.0)
    music_fade_out_duration: float = Field(3.0)


class GoogleCloudVoiceCriteria(BaseModel):
    language_code: str
    ssml_gender: str | None = Field(None)
    name_contains: str | None = Field(None)


class GoogleCloudTTSSettings(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    language_code: str
    voice_selection_criteria: list[GoogleCloudVoiceCriteria] = Field(..., min_length=1)
    speaking_rate: float = Field(1.0)
    pitch: float = Field(0.0)
    volume_gain_db: float = Field(0.0)
    debug: bool = Field(False)
    api_timeout_sec: int = Field(60)
    api_max_retries: int = Field(2)
    api_retry_delay_sec: int = Field(5)
    last_word_buffer_sec: float = Field(0.3)

    @model_validator(mode="after")
    def check_audio_config_ranges(self) -> "GoogleCloudTTSSettings":
        if not (TTS_SPEAKING_RATE_MIN <= self.speaking_rate <= TTS_SPEAKING_RATE_MAX):
            logger.warning(f"Google TTS rate {self.speaking_rate} outside range.")
        if not (TTS_PITCH_MIN <= self.pitch <= TTS_PITCH_MAX):
            logger.warning(f"Google TTS pitch {self.pitch} outside range.")
        return self


class CoquiTTSSettings(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    speaker_name: str | None = Field(None)


class TTSConfig(BaseModel):
    provider_order: list[str] = Field(..., min_length=1)
    google_cloud: GoogleCloudTTSSettings | None = Field(None)
    coqui: CoquiTTSSettings | None = Field(None)

    @model_validator(mode="after")
    def check_provider_settings_exist(self) -> "TTSConfig":
        valid_providers = []
        try:
            from src.video.tts import COQUI_AVAILABLE, GOOGLE_CLOUD_AVAILABLE
        except ImportError:
            GOOGLE_CLOUD_AVAILABLE, COQUI_AVAILABLE = False, False

        for name in self.provider_order:
            if (
                name == "google_cloud" and self.google_cloud and GOOGLE_CLOUD_AVAILABLE
            ) or (name == "coqui" and self.coqui and COQUI_AVAILABLE):
                valid_providers.append(name)
            else:
                logger.warning(
                    f"TTS provider '{name}' skipped (unavailable or config missing)."
                )
        if not valid_providers:
            # In test environments or when no providers are available,
            # allow empty provider list but log warning
            logger.warning(
                "No usable TTS providers configured/available. "
                "This configuration can only be used for testing or components "
                "that don't require TTS."
            )
            self.provider_order = []
        else:
            self.provider_order = valid_providers
        return self


class GoogleCloudSTTSettings(BaseModel):
    enabled: bool = Field(True)
    language_code: str = Field("en-US")
    encoding: str = Field("LINEAR16")
    sample_rate_hertz: int = Field(24000)
    use_enhanced: bool = Field(True)
    api_timeout_sec: int = Field(120)
    api_max_retries: int = Field(2)
    api_retry_delay_sec: int = Field(10)
    use_speech_adaptation_if_script_provided: bool = Field(True)
    adaptation_boost_value: float = Field(15.0, gt=0, le=20)


class AudioProcessingSettings(BaseModel):
    """Configuration for audio processing and TTS settings."""

    coqui_gpu_enabled: bool = Field(False)
    google_tts_audio_encoding: str = Field("LINEAR16")
    min_audio_file_size_bytes: int = Field(100)
    audio_validation_timeout_sec: int = Field(30)

    # Silence removal settings for TTS voiceover trimming
    # These settings ensure Whisper STT timestamps align with actual audio
    # Whisper normalizes timestamps to start at first detected speech
    silence_removal_enabled: bool = Field(
        True,
        description="Enable silence trimming from TTS-generated voiceover. "
        "CRITICAL: Whisper STT normalizes timestamps to start at 0, removing "
        "leading silence offset. If disabled, subtitles will desync with audio.",
    )
    silence_threshold_db: int = Field(
        -40,
        description="dB threshold for silence detection. Lower = more sensitive. "
        "-40dB catches most background silence while preserving speech. "
        "Range: -60 (very sensitive) to -20 (only very quiet silence).",
    )
    silence_min_duration_sec: float = Field(
        0.1,
        description="Minimum duration in seconds to consider as silence. "
        "Prevents removing very brief pauses between words. "
        "0.1s = 100ms is optimal for natural speech cadence.",
    )
