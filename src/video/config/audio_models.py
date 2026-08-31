# src/video/config/audio_models.py
"""Audio-related configuration models for TTS, STT, and audio processing."""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.video.config.constants import (
    DEFAULT_WHISPER_MODEL_DIR,
    FREESOUND_DOWNLOAD_CHUNK_SIZE,
    FREESOUND_TOKEN_EXPIRY_SEC,
    FREESOUND_TOKEN_REFRESH_BUFFER_SEC,
    TTS_PITCH_MAX,
    TTS_PITCH_MIN,
    TTS_SPEAKING_RATE_MAX,
    TTS_SPEAKING_RATE_MIN,
)

logger = logging.getLogger(__name__)


class AudioProviderConfig(BaseModel):
    """Config for a single audio provider instance."""

    name: str
    enabled: bool = True
    settings: dict[str, Any] = Field(default_factory=dict)


class AudioSettings(BaseModel):
    music_volume_db: float
    voiceover_volume_db: float
    audio_mix_duration: str = Field("longest")
    background_music_paths: list[Path]
    audio_providers: list[AudioProviderConfig] = Field(default_factory=list)
    freesound_api_key_env_var: str
    freesound_client_id_env_var: str = Field("FREESOUND_CLIENT_ID")
    freesound_client_secret_env_var: str = Field("FREESOUND_CLIENT_SECRET")  # noqa: S106
    freesound_refresh_token_env_var: str = Field("FREESOUND_REFRESH_TOKEN")  # noqa: S106
    freesound_sort: str = Field("rating_desc")
    freesound_search_query: str
    freesound_filters: str
    freesound_max_results: int
    freesound_max_search_duration_sec: int = Field(9999)
    freesound_api_timeout_sec: int = Field(15)
    freesound_download_timeout_sec: int = Field(60)
    freesound_token_expiry_sec: int = Field(FREESOUND_TOKEN_EXPIRY_SEC)
    freesound_token_refresh_buffer_sec: int = Field(FREESOUND_TOKEN_REFRESH_BUFFER_SEC)
    freesound_download_chunk_size: int = Field(FREESOUND_DOWNLOAD_CHUNK_SIZE)
    output_audio_codec: str = Field("aac")
    output_audio_bitrate: str = Field("192k")
    music_fade_in_duration: float = Field(2.0)
    music_fade_out_duration: float = Field(3.0)

    # Voice-keyed ducking. `sidechaincompress` attenuates the music while
    # narration plays and lets it back up in the gaps, instead of holding one
    # level for the whole clip. Off by default: it changes the sound of every
    # render, and `music_volume_db` alone is a working mix.
    music_ducking_enabled: bool = Field(False)
    # Floor is `sidechaincompress`'s own lower bound. `gt=0.0` let a value
    # below it pass config load and abort the render at final assembly,
    # after the script, voiceover and subtitles had all been paid for.
    music_ducking_threshold: float = Field(0.1, ge=0.000977, le=1.0)
    music_ducking_ratio: float = Field(4.0, ge=1.0, le=20.0)
    music_ducking_attack_ms: float = Field(20.0, ge=0.01, le=2000.0)
    music_ducking_release_ms: float = Field(300.0, ge=0.01, le=9000.0)

    # Final loudness normalization (EBU R128). Platforms normalize on
    # playback, so mastering near the target keeps the video level with the
    # feed around it rather than being pushed up or down.
    loudness_normalization_enabled: bool = Field(True)
    loudness_target_lufs: float = Field(-14.0, ge=-70.0, le=-5.0)
    loudness_true_peak_db: float = Field(-1.0, ge=-9.0, le=0.0)
    loudness_range_lu: float = Field(7.0, ge=1.0, le=50.0)
    # Applied whether or not the loudness pass runs: this names a property of
    # the output, not a `loudnorm` side effect. Coupling it to normalization
    # is the bug it shipped with -- switching normalization off then dropped
    # the rate control silently. (`loudnorm` emitting at 192 kHz whatever it
    # was given is what made a resample necessary at all.)
    output_audio_sample_rate: int = Field(48000, ge=8000, le=192000)


class GoogleCloudVoiceCriteria(BaseModel):
    language_code: str
    ssml_gender: str | None = Field(None)
    name_contains: str | None = Field(None)


class GoogleCloudTTSSettings(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str = Field("")
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


class TextMarkupRule(BaseModel):
    """Rule for inserting inline markup into TTS text."""

    pattern: str  # regex pattern to match
    insert_before: str = Field("")
    insert_after: str = Field("")


class VoiceProfileConfig(BaseModel):
    """A named voice profile with style, markup, and voice preferences."""

    model_config = {"protected_namespaces": ()}

    provider: str = Field("google_cloud")
    style_prompt: str | None = Field(None)
    gemini_model_name: str = Field("gemini-2.5-flash-tts")
    voice_criteria: list[GoogleCloudVoiceCriteria] | None = Field(None)
    speaking_rate: float | None = Field(None)
    pitch: float | None = Field(None)
    markup_rules: list[TextMarkupRule] = Field(default_factory=list)


class TTSConfig(BaseModel):
    provider_order: list[str] = Field(..., min_length=1)
    google_cloud: GoogleCloudTTSSettings | None = Field(None)
    coqui: CoquiTTSSettings | None = Field(None)
    voice_profiles_enabled: bool = Field(True)
    voice_profiles: dict[str, VoiceProfileConfig] = Field(default_factory=dict)
    voice_profile_pool: list[str] = Field(default_factory=list)
    default_voice_profile: str | None = Field(None)

    @model_validator(mode="after")
    def check_provider_settings_exist(self) -> "TTSConfig":
        valid_providers = []
        try:
            from src.video.tts import COQUI_AVAILABLE, GOOGLE_CLOUD_AVAILABLE
        except ImportError:
            GOOGLE_CLOUD_AVAILABLE, COQUI_AVAILABLE = False, False

        for name in self.provider_order:
            if (
                name in ("google_cloud", "gemini")
                and self.google_cloud
                and GOOGLE_CLOUD_AVAILABLE
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
        description="ffmpeg silenceremove start_duration: the continuous "
        "non-silence window the filter must detect before it stops trimming. "
        "Audio during this window is DISCARDED, not kept, so larger values "
        "trim MORE aggressively. Keep at or below 0.1s so short trailing "
        "words (e.g. 'tips', 'tech') aren't eaten by the confirmation window.",
    )
