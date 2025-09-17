"""Text-to-Speech (TTS) Module

This module provides a unified interface for generating speech from text using multiple
TTS providers. It implements a multi-provider architecture with automatic fallback
mechanisms to ensure reliable speech generation even when a primary provider fails.

Supported TTS providers:
- Google Cloud Text-to-Speech: High-quality cloud-based TTS with many voices
- Coqui TTS: Local open-source TTS for offline generation

Features:
- Asynchronous generation for improved performance
- Automatic provider fallback on failure
- Voice selection based on language, gender, and quality preferences
- Configurable speech parameters (pitch, rate, volume)
- Debug output for troubleshooting

The module dynamically checks for available providers at runtime and gracefully
disables those that aren't available, with appropriate logging.
"""

import asyncio
import logging
import os
import random
import threading
from pathlib import Path
from typing import Any

from src.utils import ensure_dirs_exist
from src.video.video_config import (
    CoquiTTSSettings,
    GoogleCloudTTSSettings,
    GoogleCloudVoiceCriteria,
    TTSConfig,
)

# Configure module logger
logger = logging.getLogger(__name__)

# Provider availability flags - set dynamically during import
GOOGLE_CLOUD_AVAILABLE = False
COQUI_AVAILABLE = False
AIOFILES_AVAILABLE = False

# Global variables for conditional imports
Voice: Any = None
GoogleAPIError: Any = None
FailedPreconditionError: Any = None
DeadlineExceededError: Any = None
DefaultCredentialsError: Any = None
texttospeech: Any = None

# Try to import Google Cloud TTS dependencies
try:
    import google.cloud.texttospeech_v1 as texttospeech
    from google.api_core.exceptions import (
        DeadlineExceeded,
        FailedPrecondition,
        GoogleAPIError,
    )
    from google.auth.exceptions import DefaultCredentialsError

    GOOGLE_CLOUD_AVAILABLE = True
    Voice = texttospeech.types.Voice
    # Map to the actual exception classes
    DeadlineExceededError = DeadlineExceeded
    FailedPreconditionError = FailedPrecondition
except (ImportError, AttributeError) as e:
    logger.warning(
        f"Google Cloud Text-to-Speech import failed: {e}. "
        f"This provider will be disabled."
    )

    # Dummy classes for when Google Cloud TTS is not available
    # These allow the rest of the code to function without excessive conditionals
    class DummyVoice:
        """Dummy Voice class that mimics the structure of
        google.cloud.texttospeech_v1.types.Voice
        """

        name: str = "Dummy Voice"
        language_codes: list[str] = []
        ssml_gender: Any | None = None

    class DummyGoogleAPIError(Exception):
        """Dummy exception for Google API errors"""

        pass

    class DummyFailedPreconditionError(Exception):
        """Dummy exception for precondition failures"""

        pass

    class DummyDeadlineExceededError(Exception):
        """Dummy exception for deadline exceeded errors"""

        pass

    class DummyDefaultCredentialsError(Exception):
        """Dummy exception for credential errors"""

        pass

    # Use dummy classes directly to avoid type conflicts
    Voice = DummyVoice
    GoogleAPIError = DummyGoogleAPIError
    FailedPreconditionError = DummyFailedPreconditionError
    DeadlineExceededError = DummyDeadlineExceededError
    DefaultCredentialsError = DummyDefaultCredentialsError


# Try to import Coqui TTS dependencies
try:
    from TTS.api import TTS  # type: ignore[import-untyped]

    COQUI_AVAILABLE = True
except (ImportError, OSError) as e:
    logger.warning(
        f"Coqui TTS library not available: {e}. This provider will be disabled."
    )
    TTS = None
    COQUI_AVAILABLE = False

try:
    import aiofiles  # type: ignore[import-untyped]

    AIOFILES_AVAILABLE = True
except ImportError:
    logger.critical(
        "The 'aiofiles' library is required for async file operations. "
        "Please install it."
    )
    AIOFILES_AVAILABLE = False

    class Aiofiles:
        @staticmethod
        async def open(*_args: Any, **_kwargs: Any) -> Any:
            raise NotImplementedError("aiofiles library not installed.")


coqui_tts_lock = threading.Lock()
_global_coqui_tts_model: Any | None = None
_global_google_cloud_client: Any | None = None
_cached_google_cloud_voices: list[Voice] | None = None


def _initialize_coqui_tts_model(settings: CoquiTTSSettings) -> Any | None:
    global _global_coqui_tts_model
    with coqui_tts_lock:
        if not COQUI_AVAILABLE or not TTS:
            return None
        if _global_coqui_tts_model is not None:
            return _global_coqui_tts_model
        try:
            logger.info(f"Loading Coqui TTS model: {settings.model_name}")
            # Use configurable GPU setting
            from src.video.video_config import config

            use_gpu = (
                config.audio_processing.coqui_gpu_enabled
                if hasattr(config, "audio_processing") and config.audio_processing
                else os.getenv("COQUI_TTS_GPU", "false").lower() == "true"
            )
            _global_coqui_tts_model = TTS(
                model_name=settings.model_name, progress_bar=False, gpu=use_gpu
            )
            logger.info(f"Coqui TTS model loaded: {settings.model_name}")
        except Exception as e:
            logger.error(
                f"Failed to load Coqui TTS model '{settings.model_name}': {e}",
                exc_info=True,
            )
            _global_coqui_tts_model = None
        return _global_coqui_tts_model


def _generate_coqui_speech_sync(
    text: str, file_path: str, model: Any, settings: CoquiTTSSettings
):
    with coqui_tts_lock:
        kwargs: dict[str, Any] = {"text": text, "file_path": file_path}
        if settings.speaker_name:
            kwargs["speaker"] = settings.speaker_name
        try:
            model.tts_to_file(**kwargs)
        except Exception as e:
            logger.error(f"Error during Coqui TTS call: {e}", exc_info=True)
            raise


async def _initialize_google_cloud_client():
    global _global_google_cloud_client
    if not GOOGLE_CLOUD_AVAILABLE or not texttospeech:
        return
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path or not os.path.exists(credentials_path):
        logger.error("GOOGLE_APPLICATION_CREDENTIALS not set or file not found.")
        return
    if _global_google_cloud_client is None:
        try:
            logger.info("Initializing Google Cloud TTS client...")
            _global_google_cloud_client = (
                texttospeech.TextToSpeechAsyncClient.from_service_account_file(
                    filename=credentials_path
                )
            )
            logger.info("Google Cloud TTS client initialized.")
        except Exception as e:
            logger.error(
                f"Failed to initialize Google Cloud TTS client: {e}", exc_info=True
            )
            _global_google_cloud_client = None


async def _fetch_available_voices() -> list[Voice] | None:
    global _cached_google_cloud_voices
    if _cached_google_cloud_voices is not None:
        return _cached_google_cloud_voices
    if _global_google_cloud_client is None:
        await _initialize_google_cloud_client()
        if _global_google_cloud_client is None:
            return None
    try:
        logger.info("Fetching available Google Cloud TTS voices...")
        response = await _global_google_cloud_client.list_voices()
        _cached_google_cloud_voices = list(response.voices) if response else []
        logger.info(f"Fetched {len(_cached_google_cloud_voices)} voices.")
        return _cached_google_cloud_voices
    except Exception as e:
        logger.error(f"Failed to fetch voices: {e}", exc_info=True)
        return None


def _filter_and_select_voice(
    voices: list[Voice], criteria_list: list[GoogleCloudVoiceCriteria]
) -> Voice | None:
    if not voices or not criteria_list:
        return None
    ssml_gender_enum = (
        texttospeech.SsmlVoiceGender
        if GOOGLE_CLOUD_AVAILABLE and texttospeech
        else None
    )
    candidate_voices: list[Voice] = []
    for voice in voices:
        for criteria in criteria_list:
            match = True
            if criteria.language_code and not any(
                lc.lower().startswith(criteria.language_code.lower())
                for lc in voice.language_codes
            ):
                match = False
            if (
                match
                and criteria.name_contains
                and criteria.name_contains.lower() not in voice.name.lower()
            ):
                match = False
            if match and criteria.ssml_gender and ssml_gender_enum:
                criteria_gender_enum = getattr(
                    ssml_gender_enum, criteria.ssml_gender.upper(), None
                )
                if voice.ssml_gender != criteria_gender_enum:
                    match = False
            if match:
                candidate_voices.append(voice)
                break
    if not candidate_voices:
        logger.warning("No voices found matching criteria.")
        return None

    # Prioritize Chirp 3 HD voices if available
    chirp3_voices = [v for v in candidate_voices if "Chirp3" in v.name]
    chirp_voices = [
        v for v in candidate_voices if "Chirp" in v.name and "Chirp3" not in v.name
    ]
    neural2_voices = [v for v in candidate_voices if "Neural2" in v.name]

    logger.debug(
        f"Voice selection breakdown: {len(candidate_voices)} total candidates, "
        f"Chirp 3 HD: {len(chirp3_voices)}, Chirp: {len(chirp_voices)}, "
        f"Neural2: {len(neural2_voices)}"
    )

    # Select from highest priority group available
    if chirp3_voices:
        selected_voice = random.choice(chirp3_voices)  # noqa: S311
        logger.info(f"Selected Chirp 3 HD voice: {selected_voice.name}")
    elif chirp_voices:
        selected_voice = random.choice(chirp_voices)  # noqa: S311
        logger.info(f"Selected Chirp voice: {selected_voice.name}")
    elif neural2_voices:
        selected_voice = random.choice(neural2_voices)  # noqa: S311
        logger.info(f"Selected Neural2 voice: {selected_voice.name}")
    else:
        selected_voice = random.choice(candidate_voices)  # noqa: S311
        logger.info(f"Selected standard voice: {selected_voice.name}")

    gender_name = (
        ssml_gender_enum(selected_voice.ssml_gender).name
        if ssml_gender_enum
        else "Unknown"
    )
    logger.info(
        f"Final TTS voice selection: {selected_voice.name} (Gender: {gender_name})"
    )
    return selected_voice


async def _generate_google_cloud_speech(
    text: str, output_path: Path, settings: GoogleCloudTTSSettings
) -> Path | None:
    if not GOOGLE_CLOUD_AVAILABLE or not AIOFILES_AVAILABLE:
        return None
    if _global_google_cloud_client is None:
        await _initialize_google_cloud_client()
        if _global_google_cloud_client is None:
            return None
    available_voices = await _fetch_available_voices()
    if not available_voices:
        return None
    selected_voice = _filter_and_select_voice(
        available_voices, settings.voice_selection_criteria
    )
    if not selected_voice:
        return None

    ensure_dirs_exist(output_path)
    synthesis_input = texttospeech.SynthesisInput(text=text)
    # Create voice selection parameters
    voice_params_kwargs = {
        "language_code": selected_voice.language_codes[0],
        "name": selected_voice.name,
    }

    # Add model name if specified (only for voices that support it)
    if settings.model_name:
        # Try model_name parameter first, then model (different API versions)
        try:
            texttospeech.VoiceSelectionParams(
                language_code="en-US", name="test", model_name=settings.model_name
            )
            voice_params_kwargs["model_name"] = settings.model_name
            logger.debug(f"Using model_name parameter: {settings.model_name}")
        except (TypeError, ValueError):
            logger.warning(
                f"Model name '{settings.model_name}' specified but not supported by "
                f"current Google Cloud TTS API version. Using default model."
            )

    voice_params = texttospeech.VoiceSelectionParams(**voice_params_kwargs)
    # Use configurable audio encoding
    from src.video.video_config import config

    encoding_name = (
        config.audio_processing.google_tts_audio_encoding
        if hasattr(config, "audio_processing") and config.audio_processing
        else "LINEAR16"
    )
    audio_encoding = getattr(
        texttospeech.AudioEncoding, encoding_name, texttospeech.AudioEncoding.LINEAR16
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=audio_encoding,
        speaking_rate=settings.speaking_rate,
        pitch=settings.pitch,
        volume_gain_db=settings.volume_gain_db,
    )
    request = texttospeech.SynthesizeSpeechRequest(
        input=synthesis_input, voice=voice_params, audio_config=audio_config
    )

    logger.info(f"Calling Google Cloud TTS API for text (length: {len(text)})")
    for attempt in range(settings.api_max_retries + 1):
        try:
            response = await asyncio.wait_for(
                _global_google_cloud_client.synthesize_speech(request=request),
                timeout=settings.api_timeout_sec,
            )
            async with aiofiles.open(output_path, "wb") as out_file:
                await out_file.write(response.audio_content)
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise OSError("Generated voiceover file is empty.")
            logger.info(f"Google Cloud voiceover created: {output_path}")
            return output_path
        except (
            OSError,
            GoogleAPIError,
            DeadlineExceededError,
            FailedPreconditionError,
            DefaultCredentialsError,
        ) as e:
            logger.error(
                f"TTS API error (attempt {attempt+1}): {e}", exc_info=settings.debug
            )
            if (
                isinstance(e, DefaultCredentialsError)
                or attempt >= settings.api_max_retries
            ):
                break
            await asyncio.sleep(settings.api_retry_delay_sec)
        except TimeoutError:
            logger.error(f"TTS API call timed out (attempt {attempt+1}).")
            if attempt >= settings.api_max_retries:
                break
            await asyncio.sleep(settings.api_retry_delay_sec)
        except Exception as e:
            logger.error(
                f"Unexpected TTS error (attempt {attempt+1}): {e}", exc_info=True
            )
            if attempt >= settings.api_max_retries:
                break
            await asyncio.sleep(settings.api_retry_delay_sec)

    if output_path.exists():
        output_path.unlink(missing_ok=True)
    return None


class TTSManager:
    """Manages text-to-speech generation across multiple providers.

    This class orchestrates the process of converting text to speech using multiple
    TTS providers in a prioritized order. It handles provider selection, fallback logic,
    and error recovery to ensure reliable speech generation.

    The manager attempts each configured provider in the order specified in the config,
    falling back to the next provider if one fails. This provides resilience against
    API outages, credential issues, or other provider-specific problems.

    Attributes
    ----------
        config (TTSConfig): Configuration for all TTS providers and settings
        secrets (dict[str, str]): API keys and credentials for TTS services

    """

    def __init__(self, config: TTSConfig, secrets: dict[str, str]):
        """Initialize the TTS manager with configuration and credentials.

        Args:
        ----
            config: Configuration for all TTS providers
            secrets: Dictionary of API keys and credentials

        """
        self.config = config
        self.secrets = secrets

    async def generate_speech(self, text: str, output_path: Path) -> Path | None:
        """Generate speech from text using configured TTS providers.

        This method attempts to convert the provided text to speech using each
        configured TTS provider in order of preference. If one provider fails,
        it automatically falls back to the next provider in the list.

        Args:
        ----
            text: The text to convert to speech
            output_path: Path where the audio file should be saved

        Returns:
        -------
            Path to the generated audio file if successful, None otherwise

        Note:
        ----
            The method will try all configured providers before giving up.
            The output format is WAV for all providers for consistency.

        """
        if not text.strip():
            logger.warning("Empty text provided to TTS.")
            return None
        ensure_dirs_exist(output_path)

        for provider_name in self.config.provider_order:
            logger.info(f"Attempting TTS provider: {provider_name}")
            try:
                if provider_name == "google_cloud" and self.config.google_cloud:
                    voiceover_path = await _generate_google_cloud_speech(
                        text, output_path, self.config.google_cloud
                    )
                    if voiceover_path:
                        logger.info("Google Cloud TTS succeeded.")
                        return voiceover_path
                elif provider_name == "coqui" and self.config.coqui:
                    if not COQUI_AVAILABLE:
                        continue
                    model = await asyncio.to_thread(
                        _initialize_coqui_tts_model, self.config.coqui
                    )
                    if not model:
                        continue
                    await asyncio.to_thread(
                        _generate_coqui_speech_sync,
                        text,
                        str(output_path),
                        model,
                        self.config.coqui,
                    )
                    if output_path.exists() and output_path.stat().st_size > 0:
                        logger.info("Coqui TTS succeeded.")
                        return output_path
            except Exception as e:
                logger.error(
                    f"Error with provider '{provider_name}': {e}", exc_info=True
                )

            logger.warning(f"Provider '{provider_name}' failed.")

        logger.error("All configured TTS providers failed.")
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return None
