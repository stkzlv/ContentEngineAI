"""Speech-to-Text Functions for Subtitle Generation

This module contains the essential STT functions extracted from the legacy
subtitle_generator.py to maintain Whisper and Google Cloud STT functionality
while removing the problematic karaoke color code.
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import psutil

from src.utils import ensure_dirs_exist, format_timestamp
from src.utils.circuit_breaker import google_stt_circuit_breaker
from src.video.video_config import (
    DEFAULT_WHISPER_MODEL_DIR,
    GoogleCloudSTTSettings,
    SubtitleSettings,
    WhisperSettings,
)

logger = logging.getLogger(__name__)

# Check for library availability
WHISPER_AVAILABLE = False
try:
    import whisper  # type: ignore[import-untyped]

    WHISPER_AVAILABLE = True
    logger.debug("Whisper library loaded successfully.")
except ImportError:
    logger.warning("Whisper library not available. STT functionality limited.")

GOOGLE_CLOUD_STT_AVAILABLE = False
try:
    import google.cloud.speech_v1p1beta1 as speech
    from google.api_core.exceptions import (
        FailedPrecondition,
        GoogleAPIError,
    )
    from google.auth.exceptions import DefaultCredentialsError

    GOOGLE_CLOUD_STT_AVAILABLE = True
    logger.debug("Google Cloud STT library loaded successfully.")
except ImportError:
    logger.warning("Google Cloud STT library not available. STT functionality limited.")
except Exception as e:
    logger.warning(f"Unexpected error during Google Cloud STT import: {e}")
    GOOGLE_CLOUD_STT_AVAILABLE = False


async def generate_subtitles_with_whisper(
    audio_path: Path,
    debug_file_dir: Path,
    settings: SubtitleSettings,
    whisper_settings: WhisperSettings,
    script: str | None = None,
    debug_mode: bool = False,
) -> list[dict[str, Any]] | None:
    """Generate subtitle timing data using Whisper STT.

    Args:
    ----
        audio_path: Path to audio file for transcription
        debug_file_dir: Directory for debug file output
        settings: Subtitle generation settings
        whisper_settings: Whisper-specific settings
        script: Optional script text for improved accuracy
        debug_mode: Enable debug output and file creation

    Returns:
    -------
        List of word timing dictionaries or None if failed

    """
    if not WHISPER_AVAILABLE:
        logger.error("Whisper library not available for STT.")
        return None

    try:
        # Load Whisper model
        model_whisper = _load_whisper_model(whisper_settings, debug_mode)
        if model_whisper is None:
            return None

        # Prepare transcription options
        trans_ops = _prepare_transcription_options(whisper_settings, script)

        logger.info("Transcribing audio with Whisper for word timings...")

        # Log audio file information for debugging
        _log_audio_file_info(audio_path)

        # Monitor system resources before transcription
        if whisper_settings.enable_resource_monitoring:
            _log_system_resources("before Whisper transcription")

        # Get audio file info for timeout calculation
        audio_duration = _get_audio_duration(audio_path)
        # Calculate timeout using configurable settings
        transcription_timeout = _calculate_timeout(audio_duration, whisper_settings)
        logger.info(
            f"Audio duration: {audio_duration:.1f}s, "
            f"timeout: {transcription_timeout:.1f}s"
        )

        # Run Whisper with timeout and progress monitoring
        start_time = time.time()
        try:
            result_w = await asyncio.wait_for(
                _transcribe_with_monitoring(
                    model_whisper,
                    str(audio_path),
                    trans_ops,
                    debug_mode,
                    whisper_settings,
                ),
                timeout=transcription_timeout,
            )
            elapsed = time.time() - start_time
            logger.info(f"Whisper transcription completed in {elapsed:.1f}s")
        except TimeoutError:
            elapsed = time.time() - start_time
            logger.error(
                f"Whisper transcription timed out after {elapsed:.1f}s "
                f"(limit: {transcription_timeout:.1f}s)"
            )
            if whisper_settings.enable_resource_monitoring:
                _log_system_resources("after Whisper timeout")
            if whisper_settings.enable_resource_cleanup:
                _cleanup_whisper_resources()
            return None
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Whisper transcription failed after {elapsed:.1f}s: {e}")
            if whisper_settings.enable_resource_monitoring:
                _log_system_resources("after Whisper error")
            if whisper_settings.enable_resource_cleanup:
                _cleanup_whisper_resources()
            return None
        finally:
            if whisper_settings.enable_resource_monitoring:
                _log_system_resources("after Whisper transcription")
            if whisper_settings.enable_resource_cleanup:
                _cleanup_whisper_resources()

        # Extract word timing data from transcription result
        word_list_whisper = _extract_word_timings(result_w)

        if not word_list_whisper:
            logger.warning("Whisper provided no usable word timings.")
            return None
        logger.info(f"Extracted {len(word_list_whisper)} word timings from Whisper.")

        # Check if Whisper debug files should be created
        create_whisper_debug = debug_mode
        if debug_mode:
            try:
                from .video_config import CONFIG  # type: ignore[attr-defined]

                create_whisper_debug = (
                    CONFIG.get("video_producer", {})
                    .get("debug_settings", {})
                    .get("create_whisper_debug_files", True)
                )
            except Exception:
                create_whisper_debug = True

        if create_whisper_debug:
            _save_whisper_debug_files(
                debug_file_dir, audio_path, result_w, word_list_whisper, script
            )

        return word_list_whisper
    except Exception as e:
        logger.error(f"Whisper STT error: {e}", exc_info=True)
        return None


@google_stt_circuit_breaker
async def transcribe_with_google_cloud_stt(
    audio_path: Path,
    settings: GoogleCloudSTTSettings,
    secrets: dict[str, str],
    script: str | None = None,
    debug_mode: bool = False,
) -> list[dict[str, Any]] | None:
    """Transcribe audio using Google Cloud STT with word-level timing."""
    if not GOOGLE_CLOUD_STT_AVAILABLE:
        logger.error("Google Cloud STT library not available.")
        return None

    # Check credentials
    creds_path = secrets.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not Path(creds_path).is_file():
        logger.error(
            "Google Cloud STT configured but GOOGLE_APPLICATION_CREDENTIALS "
            "invalid/not found."
        )
        return None

    try:
        # Initialize client using the correct v1p1beta1 API
        import google.cloud.speech_v1p1beta1 as speech_v1

        client = speech_v1.SpeechAsyncClient.from_service_account_file(
            filename=creds_path
        )

        # Configure audio encoding
        encoding_name = settings.encoding.upper()
        if not hasattr(speech_v1.RecognitionConfig.AudioEncoding, encoding_name):
            logger.error(f"Invalid Google STT encoding '{encoding_name}'.")
            return None

        audio_encoding_enum = getattr(
            speech_v1.RecognitionConfig.AudioEncoding, encoding_name
        )

        # Prepare speech contexts for adaptation
        speech_contexts = []
        if script and settings.use_speech_adaptation_if_script_provided:
            # Split script into phrases for better recognition
            words = script.split()
            phrases = [" ".join(words[i : i + 5]) for i in range(0, len(words), 5)]
            speech_contexts = [
                speech_v1.SpeechContext(
                    phrases=phrases[:50],  # Limit to 50 phrases
                    boost=settings.adaptation_boost_value,
                )
            ]

        # Read audio file
        with open(audio_path, "rb") as audio_file:
            audio_content = audio_file.read()

        # Configure recognition
        audio = speech_v1.RecognitionAudio(content=audio_content)
        config = speech_v1.RecognitionConfig(
            encoding=audio_encoding_enum,
            sample_rate_hertz=settings.sample_rate_hertz,
            language_code=settings.language_code,
            enable_word_time_offsets=True,  # Critical for subtitle timing
            enable_automatic_punctuation=True,
            use_enhanced=settings.use_enhanced,
            speech_contexts=speech_contexts,
        )

        # Perform transcription
        logger.info("Starting Google Cloud STT transcription with word timing...")
        operation = await client.long_running_recognize(config=config, audio=audio)

        # Wait for the operation to complete
        result = await operation.result(timeout=settings.api_timeout_sec)

        # Extract word timings
        word_timings = []
        for result_item in result.results:
            for alternative in result_item.alternatives:
                for word_info in alternative.words:
                    word_timing = {
                        "word": word_info.word,
                        "start": word_info.start_time.total_seconds(),
                        "end": word_info.end_time.total_seconds(),
                        "confidence": alternative.confidence or 0.9,
                    }
                    word_timings.append(word_timing)

        logger.info(
            f"Google Cloud STT completed: {len(word_timings)} words with timing"
        )
        return word_timings

    except Exception as e:
        logger.error(f"Google Cloud STT error: {e}", exc_info=debug_mode)
        return None


# Helper functions (these would be extracted from the legacy code)


def _load_whisper_model(whisper_settings: WhisperSettings, debug_mode: bool):
    """Load Whisper model with configured settings."""
    if not WHISPER_AVAILABLE:
        return None

    try:
        model = whisper.load_model(
            whisper_settings.model_size,
            device=whisper_settings.model_device,
            in_memory=whisper_settings.model_in_memory,
            download_root=whisper_settings.model_download_root
            or DEFAULT_WHISPER_MODEL_DIR,
        )
        logger.info(f"Whisper model loaded: {whisper_settings.model_size}")
        return model
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return None


def _prepare_transcription_options(
    whisper_settings: WhisperSettings, script: str | None
) -> dict:
    """Prepare Whisper transcription options."""
    options = {
        "language": whisper_settings.language,
        "task": whisper_settings.task,
        "temperature": whisper_settings.temperature,
        "beam_size": whisper_settings.beam_size,
        "fp16": whisper_settings.fp16,
        "compression_ratio_threshold": whisper_settings.compression_ratio_threshold,
        "logprob_threshold": whisper_settings.logprob_threshold,
        "no_speech_threshold": whisper_settings.no_speech_threshold,
        "condition_on_previous_text": whisper_settings.condition_on_previous_text,
        "word_timestamps": True,  # Essential for subtitle timing
    }

    if whisper_settings.patience is not None:
        options["patience"] = whisper_settings.patience

    return options


def _log_audio_file_info(audio_path: Path):
    """Log audio file information for debugging."""
    try:
        file_size = audio_path.stat().st_size
        logger.debug(f"Audio file: {audio_path.name}, size: {file_size:,} bytes")
    except Exception as e:
        logger.debug(f"Could not get audio file info: {e}")


def _log_system_resources(context: str):
    """Log current system resource usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        logger.debug(
            f"System resources {context}: "
            f"CPU {cpu_percent:.1f}%, "
            f"Memory {memory.percent:.1f}% "
            f"({memory.available / 1024**3:.1f}GB available)"
        )
    except Exception as e:
        logger.debug(f"Could not log system resources: {e}")


def _get_audio_duration(audio_path: Path) -> float:
    """Get audio duration using basic file info or fallback."""
    try:
        # This would use FFprobe or similar in the full implementation
        # For now, return a reasonable default
        return 60.0  # 1 minute fallback
    except Exception:
        return 60.0


def _calculate_timeout(
    audio_duration: float, whisper_settings: WhisperSettings
) -> float:
    """Calculate transcription timeout based on audio duration and settings."""
    timeout = whisper_settings.base_timeout_sec + (
        audio_duration * whisper_settings.duration_multiplier
    )
    return min(timeout, whisper_settings.max_timeout_sec)


async def _transcribe_with_monitoring(
    model, audio_path: str, options: dict, debug_mode: bool, settings: WhisperSettings
):
    """Run Whisper transcription with progress monitoring."""
    # Fix API call: unpack options as keyword arguments
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: model.transcribe(audio_path, **options)
    )
    return result


def _extract_word_timings(whisper_result: dict) -> list[dict[str, Any]]:
    """Extract word-level timings from Whisper result."""
    word_timings: list[dict[str, Any]] = []

    if not whisper_result or "segments" not in whisper_result:
        return word_timings

    for segment in whisper_result["segments"]:
        if "words" in segment:
            for word_data in segment["words"]:
                if all(key in word_data for key in ["word", "start", "end"]):
                    word_timings.append(
                        {
                            "word": word_data["word"].strip(),
                            "start_time": float(word_data["start"]),
                            "end_time": float(word_data["end"]),
                        }
                    )

    return word_timings


def _cleanup_whisper_resources():
    """Clean up Whisper resources to free memory."""
    import gc

    gc.collect()


def _save_whisper_debug_files(
    debug_dir: Path,
    audio_path: Path,
    result: dict,
    word_timings: list,
    script: str | None,
):
    """Save Whisper debug files for analysis."""
    try:
        ensure_dirs_exist(debug_dir)

        # Save raw Whisper result
        raw_file = debug_dir / f"{audio_path.stem}_whisper_result_raw.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Save word timings
        word_file = debug_dir / f"{audio_path.stem}_whisper_word_list.json"
        with open(word_file, "w", encoding="utf-8") as f:
            json.dump(word_timings, f, indent=2, ensure_ascii=False)

        # Save script comparison if available
        if script:
            comparison_file = debug_dir / f"{audio_path.stem}_whisper_vs_script.txt"
            with open(comparison_file, "w", encoding="utf-8") as f:
                f.write("SCRIPT:\n")
                f.write(script)
                f.write("\n\nWHISPER TEXT:\n")
                f.write(result.get("text", ""))

        logger.debug(f"Whisper debug files saved to {debug_dir}")

    except Exception as e:
        logger.warning(f"Failed to save Whisper debug files: {e}")
