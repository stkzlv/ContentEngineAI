"""Subtitle Generation Module

This module handles the generation of subtitles for video content using multiple
speech-to-text (STT) providers. It supports both Whisper (local) and Google Cloud STT
(remote) providers with automatic fallback mechanisms.

Key features:
- Multi-provider architecture with fallback logic
- Word-level timing extraction for precise subtitle synchronization
- Unified subtitle generation for both SRT and ASS formats
- Simplified positioning system with content-aware layout
- Script-based timing fallback when STT fails

The module dynamically checks for available STT providers at runtime and gracefully
disables those that aren't available, with appropriate logging.
"""

import asyncio
import contextlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import pysrt

from src.utils import ensure_dirs_exist, format_timestamp
from src.utils.circuit_breaker import google_stt_circuit_breaker
from src.utils.script_sanitizer import sanitize_script
from src.video.subtitle_positioning import (
    convert_legacy_config,
)
from src.video.subtitle_validation import validate_srt_file
from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator
from src.video.video_config import (
    DEFAULT_WHISPER_MODEL_DIR,
    GoogleCloudSTTSettings,
    SubtitleSettings,
    WhisperSettings,
)

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class SubtitleGenerationResult:
    """Result object that preserves error context for subtitle generation."""

    success: bool
    path: Path | None
    method_used: str
    error_chain: list[str]
    timing_count: int = 0


# Managed cache for loaded Whisper models with LRU eviction and memory monitoring
class WhisperModelCache:
    """LRU cache with memory pressure monitoring for Whisper models."""

    def __init__(self, max_models: int = 3, memory_threshold_gb: float = 8.0):
        self._cache: dict[str, Any] = {}
        self._access_order: list[str] = []
        self._max_models = max_models
        self._memory_threshold_bytes = memory_threshold_gb * 1024**3

    def get(self, key: str) -> Any | None:
        """Get model from cache, updating LRU order."""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, model: Any) -> None:
        """Store model in cache, evicting if necessary."""
        # Remove existing entry if present
        if key in self._cache:
            self._access_order.remove(key)

        # Check memory pressure before adding
        while len(self._cache) >= self._max_models or self._check_memory_pressure():
            if not self._evict_lru():
                break  # No more items to evict

        self._cache[key] = model
        self._access_order.append(key)

    def _evict_lru(self) -> bool:
        """Evict least recently used model. Returns True if evicted."""
        if not self._access_order:
            return False

        lru_key = self._access_order.pop(0)
        if lru_key in self._cache:
            del self._cache[lru_key]
            logger.debug(f"Evicted Whisper model from cache: {lru_key}")
        return True

    def _check_memory_pressure(self) -> bool:
        """Check if system memory usage is above threshold."""
        try:
            memory_info = psutil.virtual_memory()
            return memory_info.available < self._memory_threshold_bytes
        except Exception:
            return False  # Conservative: don't evict on monitoring failure

    def clear(self) -> None:
        """Clear all cached models."""
        self._cache.clear()
        self._access_order.clear()


_WHISPER_MODELS = WhisperModelCache()

# Use the model directory from config
DEFAULT_MODEL_DIR = os.path.expanduser(DEFAULT_WHISPER_MODEL_DIR)

# Check if Whisper is available and set up model directory
WHISPER_AVAILABLE = False
try:
    import whisper

    WHISPER_AVAILABLE = True
    os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
    logger.info(
        f"Using persistent Whisper model directory: {DEFAULT_MODEL_DIR} (from config)"
    )
except ImportError:
    logger.warning("Whisper module not available. Whisper-based STT will be disabled.")
except Exception as e:
    logger.warning(f"Error setting up Whisper model directory: {e}")
    WHISPER_AVAILABLE = False

# Check if Google Cloud Speech-to-Text is available
GOOGLE_CLOUD_STT_AVAILABLE = False
try:
    import google.cloud.speech_v1p1beta1 as speech
    from google.api_core.exceptions import (
        FailedPrecondition,
        GoogleAPIError,
    )
    from google.auth.exceptions import DefaultCredentialsError

    GOOGLE_CLOUD_STT_AVAILABLE = True
    logger.debug("Successfully imported google.cloud.speech and related exceptions.")
except ImportError:
    logger.warning(
        "Google Cloud Speech-to-Text import failed. Google Cloud STT will be disabled."
    )
except Exception as e:
    logger.warning(
        f"Unexpected error during Google Cloud STT import: {e}", exc_info=True
    )
    GOOGLE_CLOUD_STT_AVAILABLE = False


def generate_simple_srt(
    audio_path: Path,
    output_srt_path: Path,
    settings: SubtitleSettings,
    script: str | None = None,
    duration: float | None = None,
    debug_mode: bool = False,
) -> Path | None:
    """Generate a simple SRT subtitle file by evenly distributing text over time.

    This function creates a basic SRT file by dividing the script into segments and
    distributing them evenly across the audio duration. It's used as a fallback when
    more precise speech-to-text methods fail.

    The function attempts to load the script from various locations if not provided,
    and can estimate duration from the audio file if not specified.

    Args:
    ----
        audio_path: Path to the audio file (used for duration estimation if needed)
        output_srt_path: Path where the SRT file should be saved
        settings: Subtitle configuration settings
        script: The script text to convert to subtitles (optional)
        duration: Duration of the audio in seconds (optional)
        debug_mode: Whether to output additional debug information

    Returns:
    -------
        Path to the generated SRT file if successful, None otherwise

    """
    try:
        # Try to load script if not provided
        loaded_script = script
        if not loaded_script:
            for rel_path in settings.script_paths:
                potential_paths = [
                    output_srt_path.parent / rel_path,
                    Path.cwd() / rel_path,
                ]
                for p_path in potential_paths:
                    if p_path.exists():
                        try:
                            loaded_script = p_path.read_text(encoding="utf-8").strip()
                            logger.info(f"Loaded script from {p_path} for simple SRT.")
                            break
                        except Exception as e_read:
                            logger.warning(f"Error reading script {p_path}: {e_read}")
                if loaded_script:
                    break
        if not loaded_script:
            logger.error("No script available for simple SRT generation.")
            return None
        if not duration or duration <= 0:
            logger.warning(
                f"Invalid/missing duration ({duration}) for simple SRT, estimating."
            )
            # Use configurable character estimation rate
            from src.video.video_config import config

            chars_per_sec = (
                config.text_processing.script_chars_per_second_estimate
                if hasattr(config, "text_processing") and config.text_processing
                else 15
            )
            duration = len(loaded_script) / chars_per_sec
            if duration <= 0:
                logger.error("Estimated duration invalid.")
                return None

        sanitized_script_for_srt = sanitize_script(loaded_script, debug_mode)
        raw_segments = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", sanitized_script_for_srt)
            if s.strip()
        ]
        if not raw_segments:
            raw_segments = [
                s.strip() for s in sanitized_script_for_srt.split("\n") if s.strip()
            ]
        if not raw_segments:
            logger.error("Could not extract segments for simple SRT.")
            return None

        processed_segments: list[dict[str, Any]] = []
        total_chars = sum(len(s) for s in raw_segments)
        # Use configurable minimum duration fallback
        from src.video.video_config import config

        min_char_dur = (
            config.text_processing.script_min_duration_sec
            if hasattr(config, "text_processing") and config.text_processing
            else 0.05
        )
        char_dur = duration / total_chars if total_chars > 0 else min_char_dur
        current_time_srt = 0.0

        for raw_seg in raw_segments:
            current_line_text = ""
            words_in_seg = raw_seg.split()
            for word_srt in words_in_seg:
                potential_line = (current_line_text + " " + word_srt).strip()
                if len(potential_line) > settings.max_line_length and current_line_text:
                    seg_len = len(current_line_text)
                    seg_dur = seg_len * char_dur
                    end_time_srt = min(current_time_srt + seg_dur, duration)
                    if end_time_srt - current_time_srt < settings.min_subtitle_duration:
                        end_time_srt = current_time_srt + settings.min_subtitle_duration
                    processed_segments.append(
                        {
                            "text": current_line_text,
                            "start": current_time_srt,
                            "end": end_time_srt,
                        }
                    )
                    current_time_srt = end_time_srt
                    current_line_text = word_srt
                else:
                    current_line_text = potential_line
            if current_line_text:
                seg_len = len(current_line_text)
                seg_dur = seg_len * char_dur
                end_time_srt = min(current_time_srt + seg_dur, duration)
                if end_time_srt - current_time_srt < settings.min_subtitle_duration:
                    end_time_srt = current_time_srt + settings.min_subtitle_duration
                processed_segments.append(
                    {
                        "text": current_line_text,
                        "start": current_time_srt,
                        "end": end_time_srt,
                    }
                )
                current_time_srt = end_time_srt

        if not processed_segments:
            logger.warning("No segments processed for simple SRT.")
            return None

        ensure_dirs_exist(output_srt_path.parent)
        with open(output_srt_path, "w", encoding="utf-8") as f_srt:
            for i, segment_data in enumerate(processed_segments, 1):
                if (
                    not segment_data["text"].strip()
                    or segment_data["start"] >= segment_data["end"]
                ):
                    continue
                start_ts, end_ts = (
                    format_timestamp(segment_data["start"]),
                    format_timestamp(segment_data["end"]),
                )
                f_srt.write(f"{i}\n{start_ts} --> {end_ts}\n{segment_data['text']}\n\n")
        logger.info(f"Generated simple SRT file: {output_srt_path}")
        return output_srt_path
    except Exception as e_simple:
        logger.error(f"Failed to generate simple SRT file: {e_simple}", exc_info=True)
        return None


def _load_whisper_model(
    whisper_settings: WhisperSettings, debug_mode: bool = False
) -> Any | None:
    """Load Whisper model with caching support.

    Args:
    ----
        whisper_settings: Whisper configuration settings
        debug_mode: Whether to enable debug logging

    Returns:
    -------
        Loaded Whisper model or None if failed

    """
    model_size, model_device, model_in_memory = (
        whisper_settings.model_size,
        whisper_settings.model_device,
        whisper_settings.model_in_memory,
    )
    model_download_root, model_key = (
        whisper_settings.model_download_root or DEFAULT_MODEL_DIR,
        f"{model_size}_{model_device}",
    )

    model_whisper: Any
    if not model_in_memory:
        cached_model = _WHISPER_MODELS.get(model_key)
        if cached_model is not None:
            model_whisper = cached_model
        else:
            logger.info(
                f"Loading Whisper model: {model_size} on {model_device} "
                f"from {model_download_root}"
            )
            try:
                model_whisper = whisper.load_model(
                    model_size,
                    device=model_device,
                    download_root=model_download_root,
                    in_memory=model_in_memory,
                )
                _WHISPER_MODELS.put(model_key, model_whisper)
            except Exception as ex:
                logger.error(f"Failed to load Whisper model: {ex}", exc_info=debug_mode)
                return None
    else:
        # For in-memory models, don't cache
        logger.info(
            f"Loading Whisper model (in-memory): {model_size} on {model_device} "
            f"from {model_download_root}"
        )
        try:
            model_whisper = whisper.load_model(
                model_size,
                device=model_device,
                download_root=model_download_root,
                in_memory=model_in_memory,
            )
        except Exception as ex:
            logger.error(f"Failed to load Whisper model: {ex}", exc_info=debug_mode)
            return None

    return model_whisper


def _prepare_transcription_options(
    whisper_settings: WhisperSettings, script: str | None = None
) -> dict[str, Any]:
    """Prepare transcription options for Whisper.

    Args:
    ----
        whisper_settings: Whisper configuration settings
        script: Optional script to use as initial prompt

    Returns:
    -------
        Dictionary of transcription options

    """
    trans_ops = {
        "language": whisper_settings.language or "en",
        "task": whisper_settings.task,
        "verbose": False,
        "temperature": whisper_settings.temperature,
        "compression_ratio_threshold": whisper_settings.compression_ratio_threshold,
        "logprob_threshold": whisper_settings.logprob_threshold,
        "no_speech_threshold": whisper_settings.no_speech_threshold,
        "condition_on_previous_text": whisper_settings.condition_on_previous_text,
        "word_timestamps": True,
        "beam_size": whisper_settings.beam_size,
        "patience": whisper_settings.patience,
        "fp16": whisper_settings.fp16,
    }
    if script:
        trans_ops["initial_prompt"] = re.sub(r"\s+", " ", script.strip())
        logger.info("Using script as initial prompt for Whisper.")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Whisper opts: {trans_ops}")

    return trans_ops


def _extract_word_timings(result_w: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract word timing data from Whisper transcription result.

    Args:
    ----
        result_w: Whisper transcription result dictionary

    Returns:
    -------
        List of word timing dictionaries

    """
    word_list_whisper: list[dict[str, Any]] = []

    if "segments" in result_w and result_w["segments"]:
        for seg_w in result_w["segments"]:
            if "words" in seg_w and seg_w["words"] and isinstance(seg_w["words"], list):
                for word_info_w in seg_w["words"]:
                    if isinstance(word_info_w, dict):
                        w, s, e = (
                            word_info_w.get("word", "").strip(),
                            word_info_w.get("start"),
                            word_info_w.get("end"),
                        )
                        if w and s is not None and e is not None:
                            try:
                                word_list_whisper.append(
                                    {
                                        "word": w,
                                        "start_time": float(s),
                                        "end_time": float(e),
                                    }
                                )
                            except (TypeError, ValueError):
                                logger.warning(
                                    f"Whisper returned invalid time value: "
                                    f"{word_info_w}"
                                )

    return word_list_whisper


def _save_whisper_debug_files(
    debug_file_dir: Path,
    audio_path: Path,
    result_w: dict[str, Any],
    word_list_whisper: list[dict[str, Any]],
    script: str | None = None,
) -> None:
    """Save debug files from Whisper transcription.

    Args:
    ----
        debug_file_dir: Directory to save debug files
        audio_path: Path to the audio file
        result_w: Raw Whisper transcription result
        word_list_whisper: Extracted word timing data
        script: Optional original script for comparison

    """
    ensure_dirs_exist(debug_file_dir)
    (debug_file_dir / f"{audio_path.stem}_whisper_result_raw.json").write_text(
        json.dumps(result_w, indent=2, ensure_ascii=False)
    )
    (debug_file_dir / f"{audio_path.stem}_whisper_word_list.json").write_text(
        json.dumps(word_list_whisper, indent=2, ensure_ascii=False)
    )
    if script:
        (debug_file_dir / f"{audio_path.stem}_whisper_vs_script.txt").write_text(
            f"SCRIPT:\n{script}\n\nWHISPER TEXT:\n{result_w.get('text','')}"
        )


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
                from .video_config import CONFIG

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
    stt_settings: GoogleCloudSTTSettings,
    secrets: dict[str, str],
    script_for_adaptation: str | None = None,
    debug_mode: bool = False,
) -> list[dict[str, Any]] | None:
    if not GOOGLE_CLOUD_STT_AVAILABLE:
        return None
    creds_p = secrets.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_p or not Path(creds_p).is_file():
        logger.error(
            "Google Cloud STT configured but GOOGLE_APPLICATION_CREDENTIALS "
            "invalid/not found."
        )
        return None
    try:
        client = speech.SpeechAsyncClient.from_service_account_file(filename=creds_p)
    except Exception as e:
        logger.error(f"Failed to init Google STT client: {e}", exc_info=True)
        return None

    enc_name = stt_settings.encoding.upper()
    if not hasattr(speech.RecognitionConfig.AudioEncoding, enc_name):
        logger.error(f"Invalid Google STT encoding '{enc_name}'.")
        return None
    audio_enc_enum = getattr(speech.RecognitionConfig.AudioEncoding, enc_name)

    speech_ctxs = []
    if script_for_adaptation and stt_settings.use_speech_adaptation_if_script_provided:
        phrases = [
            s.strip()
            for s in re.split(r"[\n.!?]+", script_for_adaptation)
            if s.strip() and 2 < len(s.strip().split()) < 10
        ]
        if phrases:
            speech_ctxs.append(
                speech.SpeechContext(
                    phrases=phrases, boost=stt_settings.adaptation_boost_value
                )
            )
            logger.info(
                f"Using {len(phrases)} phrase hints for Google STT "
                f"(boost: {stt_settings.adaptation_boost_value})."
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Sample phrase hints: {phrases[:5]}")
        else:
            logger.info("No suitable phrases from script for STT adaptation.")

    rec_config = speech.RecognitionConfig(
        encoding=audio_enc_enum,
        sample_rate_hertz=stt_settings.sample_rate_hertz,
        language_code=stt_settings.language_code,
        enable_word_time_offsets=True,
        use_enhanced=stt_settings.use_enhanced,
        speech_contexts=speech_ctxs if speech_ctxs else None,
    )

    try:
        audio_content = audio_path.read_bytes()
    except Exception as e:
        logger.error(f"Failed to read audio {audio_path}: {e}")
        return None
    rec_audio = speech.RecognitionAudio(content=audio_content)

    for attempt in range(stt_settings.api_max_retries + 1):
        try:
            logger.info(
                f"Calling Google STT API (attempt {attempt+1}) for {audio_path.name}"
            )
            operation = await asyncio.wait_for(
                client.long_running_recognize(config=rec_config, audio=rec_audio),
                timeout=stt_settings.api_timeout_sec,
            )
            result = await asyncio.wait_for(
                operation.result(), timeout=stt_settings.api_timeout_sec * 2
            )
            timings_g: list[dict[str, Any]] = []
            if result and result.results and result.results[0].alternatives:
                alt = result.results[0].alternatives[0]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"STT Tx: {alt.transcript} (Conf: {alt.confidence:.2f})"
                    )
                if alt.words:
                    for word_info in alt.words:
                        timings_g.append(
                            {
                                "word": word_info.word,
                                "start_time": word_info.start_time.total_seconds(),
                                "end_time": word_info.end_time.total_seconds(),
                            }
                        )
                    logger.info(
                        f"Extracted {len(timings_g)} word timings from Google STT."
                    )
                    if debug_mode and timings_g:
                        debug_dir = audio_path.parent
                        ensure_dirs_exist(debug_dir)
                        (
                            debug_dir / f"{audio_path.stem}_google_stt_timings.json"
                        ).write_text(
                            json.dumps(timings_g, indent=2, ensure_ascii=False)
                        )
                    return timings_g
            logger.warning("Google STT returned no results/alternatives/words.")
            break
        except (GoogleAPIError, FailedPrecondition, DefaultCredentialsError) as e_api:
            logger.error(
                f"Google STT API error (attempt {attempt+1}): {e_api}",
                exc_info=debug_mode,
            )
            if attempt < stt_settings.api_max_retries and not isinstance(
                e_api, DefaultCredentialsError
            ):
                await asyncio.sleep(stt_settings.api_retry_delay_sec)
            else:
                break
        except TimeoutError:
            logger.error(f"Google STT API call timed out (attempt {attempt+1}).")
            if attempt < stt_settings.api_max_retries:
                await asyncio.sleep(stt_settings.api_retry_delay_sec)
            else:
                break
        except Exception as e_unexp:
            logger.error(
                f"Unexpected Google STT error (attempt {attempt+1}): {e_unexp}",
                exc_info=True,
            )
            if attempt < stt_settings.api_max_retries:
                await asyncio.sleep(stt_settings.api_retry_delay_sec)
            else:
                break

    logger.error("Google Cloud STT failed after all attempts.")
    return None


def generate_srt_from_timings(
    script_text: str,
    timings: list[dict[str, Any]],
    output_path: Path,
    settings: SubtitleSettings,
    voiceover_duration: float | None,
    debug_mode: bool = False,
) -> Path | None:
    if not timings:
        logger.warning("No timing data provided for SRT generation.")
        output_path.unlink(missing_ok=True)
        return None

    ensure_dirs_exist(output_path.parent)
    srt_content = []
    sequence_number = 1
    current_line_words_data: list[dict[str, Any]] = []
    current_line_text = ""
    current_line_start_time: float | None = None
    word_char_pattern = re.compile(r"^\W+|\W+$")

    cleaned_timings: list[dict[str, Any]] = []
    for entry in timings:
        word_val, start_val, end_val = (
            entry.get("word"),
            entry.get("start_time"),
            entry.get("end_time"),
        )
        if (
            not isinstance(word_val, str)
            or not word_val.strip()
            or start_val is None
            or end_val is None
        ):
            continue
        try:
            start_f, end_f = float(start_val), float(end_val)
            if start_f < 0 or end_f < start_f:
                continue
            cleaned_timings.append(
                {
                    "original_word": word_val,
                    "word_for_join": word_char_pattern.sub("", word_val),
                    "start_time": start_f,
                    "end_time": end_f,
                }
            )
        except (ValueError, TypeError):
            continue

    if not cleaned_timings:
        logger.warning("No valid word timings left after cleaning for SRT generation.")
        output_path.unlink(missing_ok=True)
        return None

    if debug_mode:
        logger.debug(
            f"Starting SRT segmentation with {len(cleaned_timings)} timed words. "
            f"MaxLineLen: {settings.max_line_length}, "
            f"MaxDur: {settings.max_subtitle_duration}, "
            f"PauseThresh: {settings.word_timestamp_pause_threshold}"
        )

    for i, word_data in enumerate(cleaned_timings):
        original_word, _word_for_join, start_time, end_time = (
            word_data["original_word"],
            word_data["word_for_join"],
            word_data["start_time"],
            word_data["end_time"],
        )

        if not current_line_words_data:
            current_line_start_time = start_time
            if debug_mode:
                logger.debug(
                    f"  New line starting with '{original_word}' at {start_time:.3f}s"
                )

        potential_new_text = (
            (current_line_text + " " + original_word).strip()
            if current_line_text
            else original_word
        )
        current_line_duration = (
            (end_time - current_line_start_time)
            if current_line_start_time is not None
            else 0
        )

        break_before_adding = False
        if current_line_words_data:
            if len(potential_new_text) > settings.max_line_length:
                if debug_mode:
                    logger.debug(
                        f"    Line break: Max length {settings.max_line_length} "
                        f"would be exceeded by '{{original_word}}'. "
                        f"Current text: '{{current_line_text}}'"
                    )
                break_before_adding = True
            elif current_line_duration > settings.max_subtitle_duration:
                if debug_mode:
                    logger.debug(
                        "    Line break: Max duration "
                        "{settings.max_subtitle_duration}s would be exceeded. "
                        "Current duration: {current_line_duration:.3f}s with "
                        "'{original_word}'"
                    )
                break_before_adding = True
            elif (
                settings.subtitle_split_on_punctuation
                and current_line_words_data[-1]["original_word"].strip()[-1]
                in settings.punctuation_marks
            ):
                if debug_mode:
                    logger.debug(
                        f"    Line break: Punctuation split after "
                        f"'{current_line_words_data[-1]['original_word']}'."
                    )
                break_before_adding = True
            elif (
                i > 0
                and (start_time - current_line_words_data[-1]["end_time"])
                >= settings.word_timestamp_pause_threshold
            ):
                if debug_mode:
                    logger.debug(
                        f"    Line break: Pause threshold "
                        f"{settings.word_timestamp_pause_threshold}s met before "
                        f"'{original_word}'. Pause: "
                        f"{start_time - current_line_words_data[-1]['end_time']:.3f}s"
                    )
                break_before_adding = True

        if break_before_adding and current_line_words_data:
            line_text_to_write = " ".join(
                w["original_word"] for w in current_line_words_data
            ).strip()
            line_start_to_write = current_line_words_data[0]["start_time"]
            line_end_to_write = current_line_words_data[-1]["end_time"]
            if line_end_to_write - line_start_to_write < settings.min_subtitle_duration:
                line_end_to_write = line_start_to_write + settings.min_subtitle_duration
            if voiceover_duration is not None:
                line_end_to_write = min(line_end_to_write, voiceover_duration)

            if line_end_to_write > line_start_to_write:
                srt_content.append(
                    f"{sequence_number}\n"
                    f"{format_timestamp(line_start_to_write)} --> "
                    f"{format_timestamp(line_end_to_write)}\n"
                    f"{line_text_to_write}\n"
                )
                if debug_mode:
                    logger.debug(
                        f"  Segment {sequence_number}: "
                        f"{format_timestamp(line_start_to_write)} --> "
                        f"{format_timestamp(line_end_to_write)} "
                        f"'{line_text_to_write}'"
                    )
                sequence_number += 1

            current_line_words_data = [word_data]
            current_line_text = original_word
            current_line_start_time = start_time
            if debug_mode:
                logger.debug(
                    f"  New line started after break, with '{original_word}' "
                    f"at {start_time:.3f}s"
                )
        else:
            current_line_words_data.append(word_data)
            current_line_text = potential_new_text
            if debug_mode and len(current_line_words_data) > 1:
                logger.debug(
                    f"    Added '{original_word}', line: '{current_line_text}'"
                )

    if current_line_words_data:
        line_text_to_write = " ".join(
            w["original_word"] for w in current_line_words_data
        ).strip()
        line_start_to_write = current_line_words_data[0]["start_time"]
        line_end_to_write = current_line_words_data[-1]["end_time"]
        if line_end_to_write - line_start_to_write < settings.min_subtitle_duration:
            line_end_to_write = line_start_to_write + settings.min_subtitle_duration
        if voiceover_duration is not None:
            line_end_to_write = min(line_end_to_write, voiceover_duration)

        if line_end_to_write > line_start_to_write and line_text_to_write:
            srt_content.append(
                f"{sequence_number}\n"
                f"{format_timestamp(line_start_to_write)} --> "
                f"{format_timestamp(line_end_to_write)}\n"
                f"{line_text_to_write}\n"
            )
            if debug_mode:
                logger.debug(
                    f"  Segment {sequence_number} (final): "
                    f"{format_timestamp(line_start_to_write)} --> "
                    f"{format_timestamp(line_end_to_write)} "
                    f"'{line_text_to_write}'"
                )
            sequence_number += 1

    if not srt_content:
        logger.warning(
            f"SRT content is empty after processing all timings for {output_path.name}."
        )
        output_path.unlink(missing_ok=True)
        return None

    output_path.write_text("\n".join(srt_content), encoding="utf-8")
    logger.info(
        f"SRT file generated from STT timings: {output_path} "
        f"({sequence_number-1} segments)"
    )
    return output_path


def generate_srt_from_timings_enhanced(
    script_text: str,
    timings: list[dict[str, Any]],
    output_path: Path,
    settings: SubtitleSettings,
    voiceover_duration: float | None,
    debug_mode: bool = False,
) -> Path | None:
    """Enhanced SRT generation using pysrt library for robust subtitle handling.

    This function uses the pysrt library to create well-formatted SRT files with
    proper validation and encoding support. It processes word timings into subtitle
    segments and creates SubRipItem objects for each subtitle.

    Args:
    ----
        script_text: The script text (for fallback/context)
        timings: List of word timing dictionaries with 'word', 'start_time', 'end_time'
        output_path: Path where the SRT file should be saved
        settings: Subtitle configuration settings
        voiceover_duration: Duration of the audio in seconds (optional)
        debug_mode: Whether to output additional debug information

    Returns:
    -------
        Path to the generated SRT file if successful, None otherwise

    """
    if not timings:
        logger.warning("No timing data provided for enhanced SRT generation.")
        output_path.unlink(missing_ok=True)
        return None

    try:
        ensure_dirs_exist(output_path.parent)

        # Clean and validate timings
        cleaned_timings = _clean_and_validate_timings(timings, debug_mode)
        if not cleaned_timings:
            logger.warning(
                "No valid word timings after cleaning for enhanced SRT generation."
            )
            output_path.unlink(missing_ok=True)
            return None

        # Process timings into subtitle segments
        subtitle_segments = _process_timings_to_segments(
            cleaned_timings, settings, voiceover_duration, debug_mode
        )

        if not subtitle_segments:
            logger.warning("No subtitle segments created from timings.")
            output_path.unlink(missing_ok=True)
            return None

        # Create SubRipFile using pysrt
        subs = pysrt.SubRipFile()

        for i, segment in enumerate(subtitle_segments, 1):
            # Create SubRipTime objects (pysrt expects milliseconds)
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)

            # Validate timing
            if start_ms >= end_ms:
                logger.warning(
                    f"Invalid timing for segment {i}: {start_ms}ms >= {end_ms}ms"
                )
                continue

            try:
                item = pysrt.SubRipItem(
                    index=i,
                    start=pysrt.SubRipTime(milliseconds=start_ms),
                    end=pysrt.SubRipTime(milliseconds=end_ms),
                    text=segment["text"].strip(),
                )
                subs.append(item)

                if debug_mode:
                    logger.debug(
                        f"Enhanced SRT segment {i}: {item.start} --> {item.end} "
                        f"'{item.text}'"
                    )

            except Exception as e:
                logger.warning(f"Failed to create SubRipItem for segment {i}: {e}")
                continue

        if not subs:
            logger.warning("No valid subtitle items created for enhanced SRT.")
            output_path.unlink(missing_ok=True)
            return None

        # Save using pysrt with proper encoding
        subs.save(str(output_path), encoding="utf-8")
        logger.info(
            f"Enhanced SRT file generated: {output_path} ({len(subs)} segments)"
        )

        # Validate the generated file
        if not validate_srt_file(output_path, debug_mode):
            logger.warning("Generated SRT file failed validation.")
            return None

        return output_path

    except Exception as e:
        logger.error(f"Failed to generate enhanced SRT file: {e}", exc_info=True)
        output_path.unlink(missing_ok=True)
        return None


def adjust_subtitle_timing(
    srt_path: Path,
    time_offset_ms: int,
    output_path: Path | None = None,
) -> Path | None:
    """Adjust subtitle timing by shifting all subtitles by a specified offset.

    Args:
    ----
        srt_path: Path to the existing SRT file
        time_offset_ms: Time offset in milliseconds (positive = delay,
            negative = advance)
        output_path: Output path (defaults to input path if None)

    Returns:
    -------
        Path to the adjusted SRT file if successful, None otherwise

    """
    if not srt_path.exists():
        logger.error(f"SRT file not found: {srt_path}")
        return None

    output_path = output_path or srt_path

    try:
        # Load existing SRT file
        subs = pysrt.open(str(srt_path), encoding="utf-8")

        # Shift timing
        subs.shift(milliseconds=time_offset_ms)

        # Save adjusted file
        ensure_dirs_exist(output_path.parent)
        subs.save(str(output_path), encoding="utf-8")

        logger.info(
            f"Subtitle timing adjusted by {time_offset_ms}ms: "
            f"{srt_path} -> {output_path}"
        )
        return output_path

    except Exception as e:
        logger.error(f"Failed to adjust subtitle timing: {e}", exc_info=True)
        return None


def slice_subtitles(
    srt_path: Path,
    start_time_ms: int,
    end_time_ms: int,
    output_path: Path,
) -> Path | None:
    """Extract a portion of subtitles between specified time ranges.

    Args:
    ----
        srt_path: Path to the existing SRT file
        start_time_ms: Start time in milliseconds
        end_time_ms: End time in milliseconds
        output_path: Output path for the sliced subtitles

    Returns:
    -------
        Path to the sliced SRT file if successful, None otherwise

    """
    if not srt_path.exists():
        logger.error(f"SRT file not found: {srt_path}")
        return None

    try:
        # Load existing SRT file
        subs = pysrt.open(str(srt_path), encoding="utf-8")

        # Create time objects for slicing
        start_time = pysrt.SubRipTime(milliseconds=start_time_ms)
        end_time = pysrt.SubRipTime(milliseconds=end_time_ms)

        # Slice the subtitles (pysrt slice parameters work differently)
        sliced_subs = subs.slice(starts_after=start_time, ends_before=end_time)

        # If no results, try a broader search
        if not sliced_subs:
            # Try getting overlapping subtitles instead
            sliced_subs = pysrt.SubRipFile()
            for sub in subs:
                sub_start_ms = sub.start.ordinal
                sub_end_ms = sub.end.ordinal
                # Include if subtitle overlaps with time range
                if sub_start_ms < end_time_ms and sub_end_ms > start_time_ms:
                    sliced_subs.append(sub)

        if not sliced_subs:
            logger.warning(
                f"No subtitles found in time range {start_time_ms}ms - {end_time_ms}ms"
            )
            return None

        # Adjust timing to start from 0
        sliced_subs.shift(milliseconds=-start_time_ms)

        # Save sliced file
        ensure_dirs_exist(output_path.parent)
        sliced_subs.save(str(output_path), encoding="utf-8")

        logger.info(
            f"Subtitles sliced ({start_time_ms}ms - {end_time_ms}ms): "
            f"{srt_path} -> {output_path} ({len(sliced_subs)} segments)"
        )
        return output_path

    except Exception as e:
        logger.error(f"Failed to slice subtitles: {e}", exc_info=True)
        return None


def _clean_and_validate_timings(
    timings: list[dict[str, Any]], debug_mode: bool = False
) -> list[dict[str, Any]]:
    """Clean and validate word timings before processing."""
    cleaned_timings = []
    word_char_pattern = re.compile(r"^\W+|\W+$")

    for entry in timings:
        word_val = entry.get("word")
        start_val = entry.get("start_time")
        end_val = entry.get("end_time")

        # Validate required fields
        if not isinstance(word_val, str) or not word_val.strip():
            continue
        if start_val is None or end_val is None:
            continue

        try:
            start_f = float(start_val)
            end_f = float(end_val)

            # Validate timing values
            if start_f < 0 or end_f < start_f:
                if debug_mode:
                    logger.debug(
                        f"Invalid timing for word '{word_val}': {start_f} -> {end_f}"
                    )
                continue

            cleaned_timings.append(
                {
                    "original_word": word_val,
                    "word_for_join": word_char_pattern.sub("", word_val),
                    "start_time": start_f,
                    "end_time": end_f,
                }
            )

        except (ValueError, TypeError) as e:
            if debug_mode:
                logger.debug(f"Invalid timing values for word '{word_val}': {e}")
            continue

    return cleaned_timings


def _process_timings_to_segments(
    cleaned_timings: list[dict[str, Any]],
    settings: SubtitleSettings,
    voiceover_duration: float | None,
    debug_mode: bool = False,
) -> list[dict[str, Any]]:
    """Process word timings into subtitle segments using existing logic."""
    current_line_words_data = []
    current_line_text = ""
    current_line_start_time = None
    subtitle_segments = []

    for i, word_data in enumerate(cleaned_timings):
        original_word = word_data["original_word"]
        start_time = word_data["start_time"]
        end_time = word_data["end_time"]

        if not current_line_words_data:
            current_line_start_time = start_time
            if debug_mode:
                logger.debug(
                    f"  New segment starting with '{original_word}' "
                    f"at {start_time:.3f}s"
                )

        potential_new_text = (
            (current_line_text + " " + original_word).strip()
            if current_line_text
            else original_word
        )
        current_line_duration = (
            (end_time - current_line_start_time)
            if current_line_start_time is not None
            else 0
        )

        # Check breaking conditions
        break_before_adding = False
        if current_line_words_data and (
            len(potential_new_text) > settings.max_line_length
            or current_line_duration > settings.max_subtitle_duration
            or (
                settings.subtitle_split_on_punctuation
                and current_line_words_data[-1]["original_word"].strip()[-1]
                in settings.punctuation_marks
            )
            or (
                i > 0
                and (start_time - current_line_words_data[-1]["end_time"])
                >= settings.word_timestamp_pause_threshold
            )
        ):
            break_before_adding = True

        if break_before_adding and current_line_words_data:
            # Create segment from current line
            segment = _create_segment_from_words(
                current_line_words_data, settings, voiceover_duration
            )
            if segment:
                subtitle_segments.append(segment)

            # Start new line
            current_line_words_data = [word_data]
            current_line_text = original_word
            current_line_start_time = start_time
        else:
            current_line_words_data.append(word_data)
            current_line_text = potential_new_text

    # Handle final segment
    if current_line_words_data:
        segment = _create_segment_from_words(
            current_line_words_data, settings, voiceover_duration
        )
        if segment:
            subtitle_segments.append(segment)

    # Optionally extend the last subtitle to match voiceover duration
    if (
        subtitle_segments
        and voiceover_duration is not None
        and settings.enable_subtitle_extension
    ):
        last_segment = subtitle_segments[-1]
        gap = voiceover_duration - last_segment["end"]

        # Check if gap meets threshold and extension limit
        if (
            gap >= settings.subtitle_extension_threshold
            and gap <= settings.max_subtitle_extension
        ):
            logger.info(
                f"Extending last subtitle by {gap:.2f}s to match voiceover duration "
                f"(threshold: {settings.subtitle_extension_threshold}s, "
                f"max: {settings.max_subtitle_extension}s)"
            )
            last_segment["end"] = voiceover_duration
        elif gap > settings.max_subtitle_extension:
            # Extend by max allowed amount instead of full gap
            extension = settings.max_subtitle_extension
            logger.info(
                f"Gap ({gap:.2f}s) exceeds max ({settings.max_subtitle_extension}s), "
                f"extending by {extension:.2f}s instead"
            )
            last_segment["end"] = last_segment["end"] + extension
        elif gap > 0:
            logger.debug(
                f"Gap ({gap:.2f}s) below threshold ({settings.subtitle_extension_threshold}s), "
                f"not extending subtitle"
            )

    return subtitle_segments


def _create_segment_from_words(
    words_data: list[dict[str, Any]],
    settings: SubtitleSettings,
    voiceover_duration: float | None,
) -> dict[str, Any] | None:
    """Create a subtitle segment from a list of word data."""
    if not words_data:
        return None

    line_text = " ".join(w["original_word"] for w in words_data).strip()
    line_start = words_data[0]["start_time"]
    line_end = words_data[-1]["end_time"]

    # Apply minimum duration
    if line_end - line_start < settings.min_subtitle_duration:
        line_end = line_start + settings.min_subtitle_duration

    # Respect voiceover duration limit
    if voiceover_duration is not None:
        line_end = min(line_end, voiceover_duration)

    # Validate segment
    if line_end <= line_start or not line_text:
        return None

    return {
        "text": line_text,
        "start": line_start,
        "end": line_end,
    }


async def create_final_subtitles_with_context(
    audio_path: Path,
    output_srt_path: Path,
    subtitle_settings: SubtitleSettings,
    whisper_stt_settings: WhisperSettings | None,
    google_stt_settings: GoogleCloudSTTSettings | None,
    secrets: dict[str, str],
    script: str | None,
    voiceover_duration: float | None,
    debug_mode: bool = False,
    video_config: Any = None,
) -> SubtitleGenerationResult:
    """Generate subtitles with comprehensive error tracking and context preservation."""
    error_chain = []

    # Determine output format and path
    if subtitle_settings.subtitle_format == "ass":
        output_path = output_srt_path.with_suffix(".ass")
        format_name = "ASS"
        logger.info(
            f"Generating ASS subtitles: {audio_path.name} -> {output_path.name}"
        )
    else:
        output_path = output_srt_path
        format_name = "SRT"
        logger.info(
            f"Generating SRT subtitles: {audio_path.name} -> {output_path.name}"
        )

    stt_timings: list[dict[str, Any]] | None = None
    ensure_dirs_exist(output_path.parent)

    # Create unified configuration from legacy settings
    unified_config = convert_legacy_config(subtitle_settings.__dict__)

    # Get frame size from video config
    frame_size = (1080, 1920)  # Default
    if video_config and hasattr(video_config, "video_settings"):
        frame_size = video_config.video_settings.resolution

    # Initialize unified generator
    generator = UnifiedSubtitleGenerator(unified_config, frame_size)

    # Try Whisper STT first
    if whisper_stt_settings and whisper_stt_settings.enabled:
        if WHISPER_AVAILABLE:
            logger.info("Using Whisper for STT and word timings.")
            try:
                stt_timings = await generate_subtitles_with_whisper(
                    audio_path,
                    output_srt_path.parent,
                    subtitle_settings,
                    whisper_stt_settings,
                    script,
                    debug_mode,
                )
                if stt_timings:
                    logger.info(
                        f"Whisper STT successful, got {len(stt_timings)} word timings."
                    )
                else:
                    error_msg = "Whisper STT did not return usable word timings"
                    error_chain.append(error_msg)
                    logger.warning(error_msg)
            except Exception as e:
                error_msg = f"Whisper STT failed: {e}"
                error_chain.append(error_msg)
                logger.error(error_msg, exc_info=debug_mode)
        else:
            error_msg = "Whisper STT configured but library not available"
            error_chain.append(error_msg)
            logger.warning(error_msg)
    else:
        logger.info("Whisper STT not configured or not enabled.")

    # Try Google Cloud STT as fallback
    if not stt_timings and google_stt_settings and google_stt_settings.enabled:
        if GOOGLE_CLOUD_STT_AVAILABLE:
            creds_path = secrets.get("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path and Path(creds_path).is_file():
                logger.info(
                    "Using Google Cloud STT for word timings (Whisper fallback/alternative)."
                )
                try:
                    stt_timings = await transcribe_with_google_cloud_stt(
                        audio_path, google_stt_settings, secrets, script, debug_mode
                    )
                    if stt_timings:
                        logger.info(
                            f"Google Cloud STT successful, got {len(stt_timings)} word timings."
                        )
                    else:
                        error_msg = (
                            "Google Cloud STT did not return usable word timings"
                        )
                        error_chain.append(error_msg)
                        logger.warning(error_msg)
                except Exception as e:
                    error_msg = f"Google Cloud STT failed: {e}"
                    error_chain.append(error_msg)
                    logger.error(error_msg, exc_info=debug_mode)
            else:
                error_msg = "Google Cloud STT configured but GOOGLE_APPLICATION_CREDENTIALS invalid/not found"
                error_chain.append(error_msg)
                logger.warning(error_msg)
        else:
            error_msg = "Google Cloud STT configured but library not available"
            error_chain.append(error_msg)
            logger.warning(error_msg)
    elif not stt_timings:
        logger.info(
            "Google Cloud STT not configured, not enabled, or Whisper already provided timings."
        )

    # Try STT-based generation if we have timings
    if stt_timings:
        logger.info(
            f"Proceeding to generate {format_name} from {len(stt_timings)} word timings."
        )

        try:
            # Use unified generator for both SRT and ASS
            result = generator.generate_from_timings(
                timings=stt_timings,
                output_path=output_path,
                format_type=subtitle_settings.subtitle_format,
                voiceover_duration=voiceover_duration,
                visual_bounds=None,  # TODO: Extract from assembler if available
                debug_mode=debug_mode,
            )

            if result.success:
                logger.info(
                    f"Successfully generated {format_name} file from STT timings: {result.path}"
                )
                return SubtitleGenerationResult(
                    success=True,
                    path=result.path,
                    method_used="unified_stt",
                    error_chain=error_chain + result.errors,
                    timing_count=len(stt_timings),
                )
            else:
                error_msg = f"Unified {format_name} generation from STT timings failed"
                error_chain.extend(result.errors)
                error_chain.append(error_msg)
                logger.warning(error_msg)

        except Exception as e:
            error_msg = f"Unified {format_name} generation from STT timings failed: {e}"
            error_chain.append(error_msg)
            logger.error(error_msg, exc_info=debug_mode)

    # Fallback generation based on format
    logger.info(
        f"Falling back to simple {format_name} generation from script text and duration."
    )

    if not script:
        error_msg = f"Cannot generate simple {format_name}: No script provided"
        error_chain.append(error_msg)
        logger.error(error_msg)
        return SubtitleGenerationResult(
            success=False, path=None, method_used="none", error_chain=error_chain
        )

    if not voiceover_duration or voiceover_duration <= 0:
        error_msg = f"Cannot generate simple {format_name}: Invalid voiceover duration ({voiceover_duration})"
        error_chain.append(error_msg)
        logger.error(error_msg)
        return SubtitleGenerationResult(
            success=False, path=None, method_used="none", error_chain=error_chain
        )

    try:
        # Use unified generator for script-based fallback
        result = generator.generate_from_script(
            script_text=script,
            duration=voiceover_duration,
            output_path=output_path,
            format_type=subtitle_settings.subtitle_format,
            visual_bounds=None,  # TODO: Extract from assembler if available
            debug_mode=debug_mode,
        )

        if result.success:
            if error_chain:
                logger.warning(
                    f"Using fallback subtitle generation after errors: {'; '.join(error_chain)}"
                )
            logger.info(
                f"Successfully generated fallback {format_name} file: {result.path}"
            )
            return SubtitleGenerationResult(
                success=True,
                path=result.path,
                method_used="unified_fallback",
                error_chain=error_chain + result.errors,
            )
        else:
            error_msg = f"Unified fallback {format_name} generation failed"
            error_chain.extend(result.errors)
            error_chain.append(error_msg)
            logger.error(error_msg)

    except Exception as e:
        error_msg = f"Unified fallback {format_name} generation failed: {e}"
        error_chain.append(error_msg)
        logger.error(error_msg, exc_info=debug_mode)

    # Complete failure - all methods failed
    logger.error(
        f"All subtitle generation methods failed. Error chain: {'; '.join(error_chain)}"
    )
    if output_path.exists():
        output_path.unlink(missing_ok=True)

    return SubtitleGenerationResult(
        success=False, path=None, method_used="none", error_chain=error_chain
    )


async def create_final_subtitles(
    audio_path: Path,
    output_srt_path: Path,
    subtitle_settings: SubtitleSettings,
    whisper_stt_settings: WhisperSettings | None,
    google_stt_settings: GoogleCloudSTTSettings | None,
    secrets: dict[str, str],
    script: str | None,
    voiceover_duration: float | None,
    debug_mode: bool = False,
    video_config: Any = None,
) -> Path | None:
    # Determine output format and path
    if subtitle_settings.subtitle_format == "ass":
        output_path = output_srt_path.with_suffix(".ass")
        logger.info(
            f"Generating ASS subtitles: {audio_path.name} -> {output_path.name}"
        )
    else:
        output_path = output_srt_path
        logger.info(
            f"Generating SRT subtitles: {audio_path.name} -> {output_path.name}"
        )

    stt_timings: list[dict[str, Any]] | None = None
    ensure_dirs_exist(output_path.parent)

    if whisper_stt_settings and whisper_stt_settings.enabled:
        if WHISPER_AVAILABLE:
            logger.info("Using Whisper for STT and word timings.")
            stt_timings = await generate_subtitles_with_whisper(
                audio_path,
                output_srt_path.parent,
                subtitle_settings,
                whisper_stt_settings,
                script,
                debug_mode,
            )
            if stt_timings:
                logger.info(
                    f"Whisper STT successful, got {len(stt_timings)} word timings."
                )
            else:
                logger.warning("Whisper STT did not return usable word timings.")
        else:
            logger.warning("Whisper STT configured but library not available.")
    else:
        logger.info("Whisper STT not configured or not enabled.")

    if not stt_timings and google_stt_settings and google_stt_settings.enabled:
        if GOOGLE_CLOUD_STT_AVAILABLE:
            creds_path = secrets.get("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path and Path(creds_path).is_file():
                logger.info(
                    "Using Google Cloud STT for word timings "
                    "(Whisper fallback/alternative)."
                )
                stt_timings = await transcribe_with_google_cloud_stt(
                    audio_path, google_stt_settings, secrets, script, debug_mode
                )
                if stt_timings:
                    logger.info(
                        f"Google Cloud STT successful, got {len(stt_timings)} "
                        f"word timings."
                    )
                else:
                    logger.warning(
                        "Google Cloud STT did not return usable word timings."
                    )
            else:
                logger.warning(
                    "Google Cloud STT configured but GOOGLE_APPLICATION_CREDENTIALS "
                    "invalid/not found."
                )
        else:
            logger.warning("Google Cloud STT configured but library not available.")
    elif not stt_timings:
        logger.info(
            "Google Cloud STT not configured, not enabled, or Whisper already "
            "provided timings."
        )

    if stt_timings:
        format_name = "ASS" if subtitle_settings.subtitle_format == "ass" else "SRT"
        logger.info(
            f"Proceeding to generate {format_name} from {len(stt_timings)} word timings."
        )

        if subtitle_settings.subtitle_format == "ass":
            # Generate ASS subtitles with animations
            generated_path = generate_ass_from_timings(
                timings=stt_timings,
                output_path=output_path,
                settings=subtitle_settings,
                voiceover_duration=voiceover_duration,
                debug_mode=debug_mode,
                video_config=video_config,
            )
        else:
            # Use enhanced SRT generation for better reliability and validation
            generated_path = generate_srt_from_timings_enhanced(
                script_text=script or "",
                timings=stt_timings,
                output_path=output_path,
                settings=subtitle_settings,
                voiceover_duration=voiceover_duration,
                debug_mode=debug_mode,
            )
            # Fallback to original implementation if enhanced version fails
            if not generated_path:
                logger.info(
                    "Enhanced SRT generation failed, trying original implementation."
                )
                generated_path = generate_srt_from_timings(
                    script_text=script or "",
                    timings=stt_timings,
                    output_path=output_path,
                    settings=subtitle_settings,
                    voiceover_duration=voiceover_duration,
                    debug_mode=debug_mode,
                )

        if (
            generated_path
            and generated_path.exists()
            and generated_path.stat().st_size > 0
        ):
            logger.info(
                f"Successfully generated {format_name} file from STT timings: "
                f"{generated_path}"
            )
            return generated_path
        else:
            logger.warning(
                f"{format_name} generation from STT timings failed or produced an empty file. "
                "Will attempt fallback."
            )

    # Fallback generation based on format
    format_name = "ASS" if subtitle_settings.subtitle_format == "ass" else "SRT"
    logger.info(
        f"Falling back to simple {format_name} generation from script text and duration."
    )

    if not script:
        logger.error(f"Cannot generate simple {format_name}: No script provided.")
        output_path.unlink(missing_ok=True)
        return None
    if not voiceover_duration or voiceover_duration <= 0:
        logger.error(
            f"Cannot generate simple {format_name}: Invalid voiceover duration "
            f"({voiceover_duration})."
        )
        output_path.unlink(missing_ok=True)
        return None

    if subtitle_settings.subtitle_format == "ass":
        fallback_path = generate_simple_ass(
            audio_path,
            output_path,
            subtitle_settings,
            script,
            voiceover_duration,
            debug_mode,
            video_config,
        )
    else:
        fallback_path = generate_simple_srt(
            audio_path,
            output_path,
            subtitle_settings,
            script,
            voiceover_duration,
            debug_mode,
        )
    if fallback_path and fallback_path.exists() and fallback_path.stat().st_size > 0:
        logger.info(
            f"Successfully generated fallback {format_name} file: {fallback_path}"
        )
        return fallback_path
    else:
        logger.error(
            "All subtitle generation methods failed (STT-based and simple fallback)."
        )
        output_path.unlink(missing_ok=True)
        return None


# Helper functions for Whisper debugging and timeout handling


def _log_system_resources(context: str) -> None:
    """Log current system resource usage for debugging."""
    try:
        # Get memory info
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)

        # Get CPU info
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Get process info
        process = psutil.Process()
        process_memory_mb = process.memory_info().rss / (1024**2)
        process_cpu = process.cpu_percent()

        logger.debug(
            f"System resources {context}: "
            f"Memory: {memory_percent:.1f}% used, "
            f"{memory_available_gb:.1f}GB available, "
            f"CPU: {cpu_percent:.1f}%, "
            f"Process: {process_memory_mb:.1f}MB RAM, {process_cpu:.1f}% CPU"
        )

        # Check for GPU if available
        try:
            import torch

            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.debug(
                    f"GPU resources {context}: "
                    f"{gpu_memory_allocated:.1f}GB allocated, "
                    f"{gpu_memory_reserved:.1f}GB reserved"
                )
        except ImportError:
            pass  # PyTorch not available

    except Exception as e:
        logger.warning(f"Failed to log system resources {context}: {e}")


def _cleanup_whisper_resources() -> None:
    """Clean up Whisper-related resources to free memory."""
    try:
        import gc

        gc.collect()

        # Clean up GPU memory if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleaned up GPU memory cache")
        except ImportError:
            pass  # PyTorch not available

        logger.debug("Performed resource cleanup after Whisper processing")
    except Exception as e:
        logger.warning(f"Failed to clean up Whisper resources: {e}")


def _log_audio_file_info(audio_path: Path) -> None:
    """Log audio file information for debugging purposes."""
    try:
        # Get basic file info
        file_size_mb = audio_path.stat().st_size / (1024**2)

        # Get audio format and duration using ffprobe
        import subprocess

        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(audio_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            import json

            probe_data = json.loads(result.stdout)
            format_info = probe_data.get("format", {})
            streams = probe_data.get("streams", [])

            duration = float(format_info.get("duration", 0))
            bit_rate = int(format_info.get("bit_rate", 0))
            format_name = format_info.get("format_name", "unknown")

            # Get audio stream info
            audio_stream = next(
                (s for s in streams if s.get("codec_type") == "audio"), {}
            )
            codec_name = audio_stream.get("codec_name", "unknown")
            sample_rate = audio_stream.get("sample_rate", "unknown")
            channels = audio_stream.get("channels", "unknown")

            logger.info(
                f"Audio file: {audio_path.name} "
                f"({file_size_mb:.1f}MB, {duration:.1f}s, {format_name}, "
                f"{codec_name}, {sample_rate}Hz, {channels}ch, {bit_rate}bps)"
            )
        else:
            logger.warning(f"Could not probe audio file {audio_path}: {result.stderr}")
            logger.info(f"Audio file: {audio_path.name} ({file_size_mb:.1f}MB)")

    except Exception as e:
        logger.warning(f"Failed to log audio file info for {audio_path}: {e}")


def _get_audio_duration(audio_path: Path) -> float:
    """Get audio file duration in seconds using ffprobe."""
    try:
        import subprocess

        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(audio_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
        else:
            logger.warning(f"ffprobe failed for {audio_path}: {result.stderr}")
            return 0.0

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.warning(f"Failed to get audio duration for {audio_path}: {e}")
        return 0.0
    except Exception as e:
        logger.warning(f"Error getting audio duration for {audio_path}: {e}")
        return 0.0


def _calculate_timeout(
    audio_duration: float, whisper_settings: WhisperSettings
) -> float:
    """Calculate timeout for Whisper transcription based on duration and settings."""
    if audio_duration <= 0:
        return whisper_settings.base_timeout_sec

    # Calculate timeout: base + (duration * multiplier)
    calculated_timeout = whisper_settings.base_timeout_sec + (
        audio_duration * whisper_settings.duration_multiplier
    )

    # Apply maximum limit
    return min(calculated_timeout, whisper_settings.max_timeout_sec)


async def _transcribe_with_monitoring(
    model_whisper: Any,
    audio_path: str,
    trans_ops: dict[str, Any],
    debug_mode: bool = False,
    whisper_settings: WhisperSettings | None = None,
) -> dict[str, Any]:
    """Run Whisper transcription with progress monitoring."""

    async def _transcribe_worker():
        """Worker function that runs in thread pool."""
        return await asyncio.to_thread(
            model_whisper.transcribe, audio_path, **trans_ops
        )

    if debug_mode and whisper_settings and whisper_settings.enable_resource_monitoring:
        # In debug mode, add progress monitoring
        monitor_interval = whisper_settings.progress_monitor_interval_sec
        monitor_task = asyncio.create_task(
            _monitor_transcription_progress(monitor_interval)
        )
        transcribe_task = asyncio.create_task(_transcribe_worker())

        try:
            # Wait for transcription to complete
            done, pending = await asyncio.wait(
                [transcribe_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel monitoring task
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task

            # Get result
            if transcribe_task.done():
                return await transcribe_task
            else:
                transcribe_task.cancel()
                raise RuntimeError("Transcription task did not complete")

        except Exception:
            # Clean up both tasks
            monitor_task.cancel()
            transcribe_task.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(
                    monitor_task, transcribe_task, return_exceptions=True
                )
            raise
    else:
        # Normal mode without monitoring
        return await _transcribe_worker()


async def _monitor_transcription_progress(interval_sec: int = 30):
    """Monitor transcription progress and log resource usage periodically."""
    try:
        while True:
            await asyncio.sleep(interval_sec)
            _log_system_resources("during Whisper transcription")
    except asyncio.CancelledError:
        pass


# ASS Subtitle Generation Functions


def generate_ass_from_timings(
    timings: list[dict[str, Any]],
    output_path: Path,
    settings: SubtitleSettings,
    voiceover_duration: float | None,
    debug_mode: bool = False,
    video_config: Any = None,
) -> Path | None:
    """Generate ASS subtitles with animations from word timings."""
    import random

    if not timings:
        logger.warning("No timing data for ASS generation.")
        return None

    # Use existing segment processing
    cleaned_timings = _clean_and_validate_timings(timings, debug_mode)
    if not cleaned_timings:
        logger.warning("No valid word timings after cleaning for ASS generation.")
        return None

    segments = _process_timings_to_segments(
        cleaned_timings, settings, voiceover_duration, debug_mode
    )
    if not segments:
        logger.warning("No segments created from timings for ASS generation.")
        return None

    # Generate ASS content
    ass_content = []

    # Get video resolution from config, no hardcoded fallback
    if video_config and hasattr(video_config, "video_settings"):
        width, height = video_config.video_settings.resolution
    else:
        # This should not happen - config validation should catch this
        raise ValueError("Video config is required for ASS subtitle generation")

    # Calculate positioning - use configurable margin from bottom
    margin_percent = getattr(settings, "ass_margin_bottom_percent", 0.25)
    margin_v = int(height * margin_percent)

    # Random color selection
    primary_color = "&H00FFFFFF"  # Default white
    outline_color = "&H00000000"  # Default black
    if settings.use_random_colors and settings.available_color_combinations:
        colors = random.choice(settings.available_color_combinations)  # noqa: S311
        primary_color, outline_color = colors[0], colors[1]

    # Script Info section
    ass_content.extend(
        [
            "[Script Info]",
            "Title: Generated Video",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "WrapStyle: 1",
            "",
        ]
    )

    # V4+ Styles section with enhanced font and size settings
    # Determine font and size
    font_size = getattr(settings, "ass_font_size", 48)
    font_name = "Arial"  # Default font

    # Apply random font if enabled
    if getattr(settings, "ass_randomize_fonts", False):
        available_fonts = getattr(
            settings, "ass_available_fonts", ["Arial", "Helvetica", "Impact"]
        )
        font_name = random.choice(available_fonts)  # noqa: S311
        logger.info(f"Randomized font selected: {font_name}")

    # Alignment: 5 = middle center for manual positioning with \pos()
    # Calculate Y position from bottom: height - margin
    pos_y = height - margin_v
    ass_content.extend(
        [
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,{font_name},{font_size},{primary_color},{primary_color},{outline_color},&H80000000,-1,0,1,2,0,5,10,10,0,1",
            "",
        ]
    )

    # Select one effect for the entire video if randomization is enabled
    selected_video_effect = None
    if getattr(settings, "ass_randomize_effects", False):
        available_effects = getattr(
            settings,
            "ass_available_effects",
            [
                "fade",
                "slide_in",
                "bounce",
                "glow",
                "scale_pulse",
                "color_wave",
                "karaoke",
            ],
        )
        selected_video_effect = random.choice(available_effects)  # noqa: S311
        logger.info(
            f"Selected single ASS effect for entire video: {selected_video_effect}"
        )

    # Events section
    ass_content.extend(
        [
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )

    # Pre-select colors once for consistent per-run coloring
    selected_colors = None
    if getattr(settings, "ass_randomize_colors", False):
        color_palettes = getattr(
            settings,
            "ass_color_palettes",
            [
                {
                    "primary": "&H00FFFFFF",
                    "secondary": "&H00FF6B35",
                    "accent": "&H00FFD700",
                }
            ],
        )
        selected_colors = random.choice(color_palettes)  # noqa: S311

    for index, segment in enumerate(segments):
        # Add index to segment for effect calculations and pass the selected video effect
        segment_with_index = {
            **segment,
            "index": index,
            "video_effect": selected_video_effect,
        }
        # Apply effects to text with pre-selected colors
        text = _apply_ass_effects(
            segment["text"], settings, segment_with_index, selected_colors
        )

        # Format timing (ASS uses H:MM:SS.CC format)
        start = _format_ass_time(segment["start"])
        end = _format_ass_time(segment["end"])

        # Add dialogue line with manual positioning
        pos_x = width // 2  # Center horizontally
        ass_content.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{{\\pos({pos_x},{pos_y})}}{text}"
        )

    # Write file
    ensure_dirs_exist(output_path.parent)
    output_path.write_text("\n".join(ass_content), encoding="utf-8")

    logger.info(f"Generated ASS file: {output_path} ({len(segments)} segments)")
    return output_path


def _apply_ass_effects(
    text: str,
    settings: SubtitleSettings,
    segment: dict,
    selected_colors: dict | None = None,
) -> str:
    """Apply ASS animation effects to text with randomization support."""
    import random

    effects = []

    # Get segment properties
    duration = segment["end"] - segment["start"]
    segment_index = segment.get("index", 0)

    # Use pre-selected colors if provided, otherwise use default colors
    if selected_colors:
        primary_color = selected_colors["primary"]
        secondary_color = selected_colors["secondary"]
        accent_color = selected_colors["accent"]
    else:
        primary_color = getattr(settings, "ass_primary_color", "&H00FFFFFF")
        secondary_color = getattr(settings, "ass_secondary_color", "&H00FF6B35")
        accent_color = "&H00FFD700"  # Default gold

    # Fade effects
    if settings.ass_enable_fade:
        fade_in = settings.ass_fade_in_ms
        fade_out = settings.ass_fade_out_ms
        effects.append(f"\\fad({fade_in},{fade_out})")

    # Color effects - use randomized colors if enabled
    if settings.ass_enable_colors:
        if segment_index % 3 == 1:
            effects.append(f"\\c{secondary_color}")
        elif segment_index % 3 == 2:
            effects.append(f"\\c{accent_color}")
        # Default color (primary) for segment_index % 3 == 0

    # Use video-level effect if randomization is enabled
    selected_effect = segment.get("video_effect")
    if selected_effect and getattr(settings, "ass_randomize_effects", False):
        if selected_effect == "slide_in" and settings.ass_enable_positioning:
            # Various slide-in directions
            slide_types = [
                "\\move(540,1680,540,1620)",  # Slide up
                "\\move(200,1620,540,1620)",  # Slide right
                "\\move(880,1620,540,1620)",  # Slide left
                "\\move(540,1500,540,1620)",  # Slide down
            ]
            effects.append(random.choice(slide_types))  # noqa: S311

        elif selected_effect == "bounce" and settings.ass_enable_transforms:
            # Bounce effect with scale
            effects.append("\\t(0,300,\\fscx110\\fscy110)")
            effects.append("\\t(300,600,\\fscx95\\fscy95)")
            effects.append("\\t(600,900,\\fscx105\\fscy105)")
            effects.append("\\t(900,1200,\\fscx100\\fscy100)")

        elif selected_effect == "glow" and settings.ass_enable_colors:
            # Glow effect with color transitions
            glow_colors = [accent_color, secondary_color, primary_color]
            for i, color in enumerate(glow_colors):
                start_time = i * 200
                end_time = (i + 1) * 200
                effects.append(f"\\t({start_time},{end_time},\\c{color})")

        elif selected_effect == "scale_pulse" and settings.ass_enable_transforms:
            # Pulsing scale effect
            pulse_factor = getattr(settings, "ass_pulse_duration_factor", 500)
            pulse_duration = int(duration * pulse_factor)
            effects.append(f"\\t(0,{pulse_duration},\\fscx115\\fscy115)")
            effects.append(
                f"\\t({pulse_duration},{pulse_duration*2},\\fscx100\\fscy100)"
            )

        elif selected_effect == "color_wave" and settings.ass_enable_colors:
            # Color wave effect cycling through palette
            wave_factor = getattr(settings, "ass_wave_duration_factor", 300)
            wave_duration = int(duration * wave_factor)
            effects.append(f"\\t(0,{wave_duration},\\c{secondary_color})")
            effects.append(f"\\t({wave_duration},{wave_duration*2},\\c{accent_color})")
            effects.append(
                f"\\t({wave_duration*2},{wave_duration*3},\\c{primary_color})"
            )

        elif selected_effect == "karaoke" and settings.ass_enable_karaoke:
            # Apply karaoke effect (handled separately below)
            pass  # Karaoke is applied to text, not effects list
    else:
        # Default non-randomized effects for consistency
        if settings.ass_enable_positioning and duration > 2.0:
            effects.append("\\move(540,1680,540,1620)")

        # Transform effects - scale animation for emphasis
        if settings.ass_enable_transforms and "!" in text:
            effects.append("\\t(\\fscx120\\fscy120)")
            effects.append("\\t(1000,2000,\\fscx100\\fscy100)")

    # Apply karaoke effects if enabled
    if settings.ass_enable_karaoke:
        text = _apply_karaoke_timing(text, segment, settings.ass_karaoke_style)

    # Combine effects
    if effects:
        effect_string = "".join(effects)
        return f"{{{effect_string}}}{text}"

    return text


def _apply_karaoke_timing(text: str, segment: dict, style: str) -> str:
    """Apply karaoke timing to words with enhanced color effects."""
    words = text.split()
    if len(words) <= 1:
        return text

    duration = segment["end"] - segment["start"]
    word_duration = duration / len(words) * 100  # Convert to centiseconds

    # Choose karaoke tag and colors
    k_tag = "kf" if style == "sweep" else "k"

    # Enhanced karaoke with color transitions
    karaoke_words = []
    for _i, word in enumerate(words):
        if style == "sweep":
            # Add color change on karaoke highlight
            karaoke_words.append(
                f"{{\\{k_tag}{word_duration:.0f}\\c&H0066FF\\c&HFFFFFF}}{word}"
            )
        else:
            # Standard karaoke timing
            karaoke_words.append(f"{{\\{k_tag}{word_duration:.0f}}}{word}")

    return " ".join(karaoke_words)


def _format_ass_time(seconds: float) -> str:
    """Format time for ASS format (H:MM:SS.CC)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centiseconds = int((secs % 1) * 100)
    secs = int(secs)

    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def generate_simple_ass(
    audio_path: Path,
    output_path: Path,
    settings: SubtitleSettings,
    script: str | None,
    duration: float | None,
    debug_mode: bool = False,
    video_config: Any = None,
) -> Path | None:
    """Generate simple ASS file from script (fallback)."""
    if not script or not duration:
        logger.error("Cannot generate simple ASS: No script or duration provided.")
        return None

    # Get video resolution from config, no hardcoded fallback
    if video_config and hasattr(video_config, "video_settings"):
        width, height = video_config.video_settings.resolution
    else:
        # This should not happen - config validation should catch this
        raise ValueError("Video config is required for ASS subtitle generation")

    # Calculate positioning - use configurable margin from bottom
    margin_percent = getattr(settings, "ass_margin_bottom_percent", 0.25)
    margin_v = int(height * margin_percent)

    # Random color selection
    primary_color = "&H00FFFFFF"  # Default white
    outline_color = "&H00000000"  # Default black
    if settings.use_random_colors and settings.available_color_combinations:
        colors = random.choice(settings.available_color_combinations)  # noqa: S311
        primary_color, outline_color = colors[0], colors[1]

    # Pre-select colors once for consistent coloring (if randomization is enabled)
    selected_colors = None
    if getattr(settings, "ass_randomize_colors", False):
        color_palettes = getattr(
            settings,
            "ass_color_palettes",
            [
                {
                    "primary": "&H00FFFFFF",
                    "secondary": "&H00FF6B35",
                    "accent": "&H00FFD700",
                }
            ],
        )
        selected_colors = random.choice(color_palettes)  # noqa: S311

    # Create single dialogue entry
    text = _apply_ass_effects(
        script, settings, {"start": 0, "end": duration}, selected_colors
    )
    start = _format_ass_time(0)
    end = _format_ass_time(duration)

    ass_content = [
        "[Script Info]",
        "Title: Generated Video",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 1",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,Arial,{getattr(settings, 'ass_font_size', 48)},{primary_color},{primary_color},{outline_color},&H80000000,-1,0,1,2,0,5,10,10,0,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        f"Dialogue: 0,{start},{end},Default,,0,0,0,,{{\\pos({width//2},{height - margin_v})}}{text}",
    ]

    ensure_dirs_exist(output_path.parent)
    output_path.write_text("\n".join(ass_content), encoding="utf-8")

    logger.info(f"Generated simple ASS file: {output_path}")
    return output_path
