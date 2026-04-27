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
import hashlib
import html
import importlib.util
import logging
import os
import random
import re
import threading
from pathlib import Path
from typing import Any

from src.utils import ensure_dirs_exist
from src.utils.circuit_breaker import google_stt_circuit_breaker
from src.utils.retry import retry_network
from src.video.config import (
    CoquiTTSSettings,
    GoogleCloudTTSSettings,
    GoogleCloudVoiceCriteria,
    TextMarkupRule,
    TTSConfig,
    VoiceProfileConfig,
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


# Coqui TTS availability is checked without actually importing the package.
# The TTS package transitively loads torch, and torch's overrides module
# raises `RuntimeError("function '_has_torch_function' already has a docstring")`
# when reimported under pytest-cov instrumentation. Deferring the real import
# to first use (see _load_coqui_tts_class) keeps coverage runs that don't
# exercise Coqui TTS off that path.
COQUI_AVAILABLE = importlib.util.find_spec("TTS") is not None
if not COQUI_AVAILABLE:
    logger.warning("Coqui TTS library not available; this provider will be disabled.")
_TTS_CLASS: Any = None  # populated lazily on first use


def _load_coqui_tts_class() -> Any | None:
    """Lazy-import the TTS class on first use, cached afterwards.

    Returns None when Coqui isn't installed or when the import raises
    (e.g. torch/runtime issues). Callers should treat None the same as
    `not COQUI_AVAILABLE`.
    """
    global _TTS_CLASS
    if _TTS_CLASS is not None:
        return _TTS_CLASS
    if not COQUI_AVAILABLE:
        return None
    try:
        from TTS.api import TTS as _TTS

        _TTS_CLASS = _TTS
        return _TTS_CLASS
    except (ImportError, OSError) as e:
        logger.warning("Coqui TTS load failed at first use: %s. Provider disabled.", e)
        return None


try:
    import aiofiles

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
        tts_class = _load_coqui_tts_class()
        if tts_class is None:
            return None
        if _global_coqui_tts_model is not None:
            return _global_coqui_tts_model
        try:
            logger.info(f"Loading Coqui TTS model: {settings.model_name}")
            # Use configurable GPU setting
            from src.video.config import config

            use_gpu = (
                config.audio_processing.coqui_gpu_enabled
                if hasattr(config, "audio_processing") and config.audio_processing
                else os.getenv("COQUI_TTS_GPU", "false").lower() == "true"
            )
            _global_coqui_tts_model = tts_class(
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
    voices: list[Voice],
    criteria_list: list[GoogleCloudVoiceCriteria],
    product_id: str | None = None,
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

    # Build a deterministic RNG when product_id is provided
    if product_id:
        hash_obj = hashlib.md5(product_id.encode(), usedforsecurity=False)
        voice_seed = int(hash_obj.hexdigest()[24:32], 16)
        rng = random.Random(voice_seed)  # noqa: S311
    else:
        rng = random.Random()  # noqa: S311

    # Select from highest priority group available
    if chirp3_voices:
        selected_voice = rng.choice(chirp3_voices)
        logger.info(f"Selected Chirp 3 HD voice: {selected_voice.name}")
    elif chirp_voices:
        selected_voice = rng.choice(chirp_voices)
        logger.info(f"Selected Chirp voice: {selected_voice.name}")
    elif neural2_voices:
        selected_voice = rng.choice(neural2_voices)
        logger.info(f"Selected Neural2 voice: {selected_voice.name}")
    else:
        selected_voice = rng.choice(candidate_voices)
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


@google_stt_circuit_breaker
async def _generate_google_cloud_speech(
    text: str,
    output_path: Path,
    settings: GoogleCloudTTSSettings,
    product_id: str | None = None,
    voice_criteria_override: list[GoogleCloudVoiceCriteria] | None = None,
    speaking_rate_override: float | None = None,
    pitch_override: float | None = None,
) -> tuple[Path | None, str | None]:
    if not GOOGLE_CLOUD_AVAILABLE or not AIOFILES_AVAILABLE:
        return None, None
    if _global_google_cloud_client is None:
        await _initialize_google_cloud_client()
        if _global_google_cloud_client is None:
            return None, None
    available_voices = await _fetch_available_voices()
    if not available_voices:
        return None, None
    criteria = voice_criteria_override or settings.voice_selection_criteria
    selected_voice = _filter_and_select_voice(
        available_voices, criteria, product_id=product_id
    )
    if not selected_voice:
        return None, None

    ensure_dirs_exist(output_path)
    # Use SSML with break at the end to prevent last word truncation
    break_time_ms = int(settings.last_word_buffer_sec * 1000)
    escaped_text = html.escape(text)
    ssml_text = f"<speak>{escaped_text}<break time='{break_time_ms}ms'/></speak>"
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=selected_voice.language_codes[0],
        name=selected_voice.name,
    )
    # Use configurable audio encoding
    from src.video.config import config

    encoding_name = (
        config.audio_processing.google_tts_audio_encoding
        if hasattr(config, "audio_processing") and config.audio_processing
        else "LINEAR16"
    )
    audio_encoding = getattr(
        texttospeech.AudioEncoding, encoding_name, texttospeech.AudioEncoding.LINEAR16
    )

    effective_rate = (
        speaking_rate_override
        if speaking_rate_override is not None
        else settings.speaking_rate
    )
    effective_pitch = pitch_override if pitch_override is not None else settings.pitch

    audio_config = texttospeech.AudioConfig(
        audio_encoding=audio_encoding,
        speaking_rate=effective_rate,
        pitch=effective_pitch,
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
            return output_path, selected_voice.name
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
    return None, None


@google_stt_circuit_breaker
async def _generate_gemini_speech(
    text: str,
    output_path: Path,
    settings: GoogleCloudTTSSettings,
    profile: VoiceProfileConfig,
    product_id: str | None = None,
) -> tuple[Path | None, str | None]:
    """Generate speech using Gemini TTS (same API, with prompt field for style).

    Gemini voices use simple names (Kore, Charon, Aoede, Puck, etc.)
    and require model_name on VoiceSelectionParams. Also requires
    Vertex AI API enabled on the GCP project.
    """
    if not GOOGLE_CLOUD_AVAILABLE or not AIOFILES_AVAILABLE:
        return None, None
    if _global_google_cloud_client is None:
        await _initialize_google_cloud_client()
        if _global_google_cloud_client is None:
            return None, None
    available_voices = await _fetch_available_voices()
    if not available_voices:
        return None, None

    # Gemini voices use simple names (no "en-US-" prefix), so filter separately
    lang = settings.language_code
    criteria = profile.voice_criteria
    if not criteria:
        criteria = [
            GoogleCloudVoiceCriteria(
                language_code=lang,
                ssml_gender=None,
                name_contains=None,
            )
        ]

    # Filter for Gemini-compatible voices (simple names, no locale prefix)
    gemini_voices = [
        v
        for v in available_voices
        if "-" not in v.name  # Gemini voices: Kore, Charon, Aoede, Puck...
        and any(lc.startswith(lang) for lc in v.language_codes)
    ]
    if not gemini_voices:
        logger.warning("No Gemini voices found in voice catalog")
        return None, None

    selected_voice = _filter_and_select_voice(
        gemini_voices, criteria, product_id=product_id
    )
    if not selected_voice:
        return None, None

    ensure_dirs_exist(output_path)

    # Gemini TTS uses text + prompt (not SSML)
    input_kwargs: dict[str, str] = {"text": text}
    if profile.style_prompt:
        input_kwargs["prompt"] = profile.style_prompt
    synthesis_input = texttospeech.SynthesisInput(**input_kwargs)

    voice_params = texttospeech.VoiceSelectionParams(
        language_code=selected_voice.language_codes[0],
        name=selected_voice.name,
        model_name=profile.gemini_model_name,
    )

    from src.video.config import config

    encoding_name = (
        config.audio_processing.google_tts_audio_encoding
        if hasattr(config, "audio_processing") and config.audio_processing
        else "LINEAR16"
    )
    audio_encoding = getattr(
        texttospeech.AudioEncoding, encoding_name, texttospeech.AudioEncoding.LINEAR16
    )

    effective_rate = (
        profile.speaking_rate
        if profile.speaking_rate is not None
        else settings.speaking_rate
    )
    effective_pitch = profile.pitch if profile.pitch is not None else settings.pitch

    audio_config = texttospeech.AudioConfig(
        audio_encoding=audio_encoding,
        speaking_rate=effective_rate,
        pitch=effective_pitch,
        volume_gain_db=settings.volume_gain_db,
    )
    request = texttospeech.SynthesizeSpeechRequest(
        input=synthesis_input, voice=voice_params, audio_config=audio_config
    )

    logger.info(
        "Calling Gemini TTS API (voice: %s, style: %s)",
        selected_voice.name,
        profile.style_prompt[:60] if profile.style_prompt else "none",
    )
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
            logger.info("Gemini TTS voiceover created: %s", output_path)
            return output_path, selected_voice.name
        except (
            OSError,
            GoogleAPIError,
            DeadlineExceededError,
            FailedPreconditionError,
            DefaultCredentialsError,
        ) as e:
            logger.error(
                "Gemini TTS error (attempt %d): %s",
                attempt + 1,
                e,
                exc_info=settings.debug,
            )
            if (
                isinstance(e, DefaultCredentialsError)
                or attempt >= settings.api_max_retries
            ):
                break
            await asyncio.sleep(settings.api_retry_delay_sec)
        except TimeoutError:
            logger.error("Gemini TTS call timed out (attempt %d).", attempt + 1)
            if attempt >= settings.api_max_retries:
                break
            await asyncio.sleep(settings.api_retry_delay_sec)
        except Exception as e:
            logger.error(
                "Unexpected Gemini TTS error (attempt %d): %s",
                attempt + 1,
                e,
                exc_info=True,
            )
            if attempt >= settings.api_max_retries:
                break
            await asyncio.sleep(settings.api_retry_delay_sec)

    if output_path.exists():
        output_path.unlink(missing_ok=True)
    return None, None


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

    def __init__(
        self,
        config: TTSConfig,
        secrets: dict[str, str],
        product_id: str | None = None,
        voice_profile_override: str | None = None,
    ):
        self.config = config
        self.secrets = secrets
        self.product_id = product_id
        self.voice_profile_override = voice_profile_override
        # Populated after generate_speech() for metadata tracking
        self.selected_profile_name: str | None = None
        self.selected_voice_name: str | None = None

    def _select_voice_profile(self) -> tuple[str | None, VoiceProfileConfig | None]:
        """Select a voice profile, deterministic by product_id when available."""
        if not self.config.voice_profiles_enabled or not self.config.voice_profiles:
            return None, None

        # CLI override takes priority
        if self.voice_profile_override:
            if self.voice_profile_override in self.config.voice_profiles:
                logger.info(
                    "Using CLI voice profile override: '%s'",
                    self.voice_profile_override,
                )
                return self.voice_profile_override, self.config.voice_profiles[
                    self.voice_profile_override
                ]
            logger.warning(
                "Voice profile override '%s' not found, falling back",
                self.voice_profile_override,
            )

        pool = self.config.voice_profile_pool or list(self.config.voice_profiles.keys())
        pool = [p for p in pool if p in self.config.voice_profiles]
        if not pool:
            return None, None

        if self.product_id:
            hash_obj = hashlib.md5(self.product_id.encode(), usedforsecurity=False)
            seed = int(hash_obj.hexdigest()[16:24], 16)
            rng = random.Random(seed)  # noqa: S311
            name = rng.choice(pool)
        else:
            name = random.choice(pool)  # noqa: S311

        logger.info(
            "Selected voice profile '%s' for product '%s'", name, self.product_id
        )
        return name, self.config.voice_profiles[name]

    # Regex to strip Gemini inline markup like [short pause], [whispering], etc.
    _MARKUP_PATTERN = re.compile(r"\[(?:short |long )?pause\]|\[\w+\]\s*")

    @staticmethod
    def _apply_markup_rules(text: str, rules: list[TextMarkupRule]) -> str:
        """Insert inline markup into text based on profile rules."""
        for rule in rules:

            def _replacer(
                m: re.Match[str],
                b: str = rule.insert_before,
                a: str = rule.insert_after,
            ) -> str:
                return b + m.group(0) + a

            text = re.sub(rule.pattern, _replacer, text)
        return text

    @classmethod
    def _strip_markup(cls, text: str) -> str:
        """Remove Gemini inline markup tags so they aren't spoken literally."""
        return cls._MARKUP_PATTERN.sub("", text)

    async def generate_speech(self, text: str, output_path: Path) -> Path | None:
        """Generate speech from text, selecting voice profile and provider."""
        if not text.strip():
            logger.warning("Empty text provided to TTS.")
            return None
        ensure_dirs_exist(output_path)

        profile_name, profile = self._select_voice_profile()
        self.selected_profile_name = profile_name

        # Apply markup rules if the profile defines them
        markup_rules = profile.markup_rules if profile else []
        processed_text = text
        if markup_rules:
            processed_text = self._apply_markup_rules(text, markup_rules)
            logger.debug(
                "Applied %d markup rules from profile '%s'",
                len(markup_rules),
                profile_name,
            )

        # Try Gemini provider first if profile requests it
        if profile and profile.provider == "gemini" and self.config.google_cloud:
            try:
                path, voice_name = await _generate_gemini_speech(
                    processed_text,
                    output_path,
                    self.config.google_cloud,
                    profile,
                    product_id=self.product_id,
                )
                if path:
                    self.selected_voice_name = voice_name
                    logger.info("Gemini TTS succeeded (profile: %s).", profile_name)
                    return path
            except Exception as e:
                logger.warning("Gemini TTS failed, falling back: %s", e)

        # Strip markup before falling back to non-Gemini providers
        # (SSML and Coqui would speak "[short pause]" literally)
        fallback_text = (
            self._strip_markup(processed_text) if markup_rules else processed_text
        )

        # Standard provider fallback loop
        for provider_name in self.config.provider_order:
            logger.info("Attempting TTS provider: %s", provider_name)
            try:
                if provider_name == "google_cloud" and self.config.google_cloud:
                    # Apply profile overrides for voice/rate/pitch
                    voice_override = (
                        profile.voice_criteria
                        if profile and profile.provider == "google_cloud"
                        else None
                    )
                    rate_override = (
                        profile.speaking_rate
                        if profile and profile.provider == "google_cloud"
                        else None
                    )
                    pitch_override = (
                        profile.pitch
                        if profile and profile.provider == "google_cloud"
                        else None
                    )

                    voiceover_path, voice_name = await _generate_google_cloud_speech(
                        fallback_text,
                        output_path,
                        self.config.google_cloud,
                        product_id=self.product_id,
                        voice_criteria_override=voice_override,
                        speaking_rate_override=rate_override,
                        pitch_override=pitch_override,
                    )
                    if voiceover_path:
                        self.selected_voice_name = voice_name
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
                        fallback_text,
                        str(output_path),
                        model,
                        self.config.coqui,
                    )
                    if output_path.exists() and output_path.stat().st_size > 0:
                        self.selected_voice_name = self.config.coqui.model_name
                        logger.info("Coqui TTS succeeded.")
                        return output_path
            except Exception as e:
                logger.error(
                    "Error with provider '%s': %s", provider_name, e, exc_info=True
                )

            logger.warning("Provider '%s' failed.", provider_name)

        logger.error("All configured TTS providers failed.")
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return None
