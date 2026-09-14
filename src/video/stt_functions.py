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
from src.utils.pipeline_deadline import remaining_pipeline_seconds
from src.video.config import (
    DEFAULT_WHISPER_MODEL_DIR,
    GoogleCloudSTTSettings,
    WhisperSettings,
)

logger = logging.getLogger(__name__)

# Check for library availability
WHISPER_AVAILABLE = False
try:
    import whisper

    WHISPER_AVAILABLE = True
    logger.debug("Whisper library loaded successfully.")
except ImportError:
    logger.warning("Whisper library not available. STT functionality limited.")

GOOGLE_CLOUD_STT_AVAILABLE = False
try:
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
    logger.warning("Unexpected error during Google Cloud STT import: %s", e)
    GOOGLE_CLOUD_STT_AVAILABLE = False


async def generate_subtitles_with_whisper(
    audio_path: Path,
    debug_file_dir: Path,
    whisper_settings: WhisperSettings,
    script: str | None = None,
    debug_mode: bool = False,
    transcript_out_path: Path | None = None,
    timing_smoothing_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    """Generate subtitle timing data using Whisper STT.

    Args:
    ----
        audio_path: Path to audio file for transcription
        debug_file_dir: Directory for debug file output
        whisper_settings: Whisper-specific settings
        script: Optional script text for improved accuracy
        debug_mode: Enable debug output and file creation
        transcript_out_path: Optional path to save the raw Whisper result
            dict in ``whisper_json`` format. Used by the pycaps subtitle
            engine to consume word-level timings as the transcript source.
            Written unconditionally when set (not gated by ``debug_mode``).
        timing_smoothing_config: Optional dict with timing smoother params
            (``enabled``, ``min_word_sec``, ``gap_merge_sec``,
            ``hold_last_sec``, ``lead_sec``). When ``None`` or
            ``enabled=True`` (the default), smoothing runs with
            best-practice defaults.

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
        # Both the number and its reason from the one helper the loop uses.
        # Asking `_stt_ceiling` separately answers "is the budget binding
        # now", not "did the budget set this limit", so a formula-derived
        # limit was reported as budget-capped whenever the budget merely
        # happened to be below `max_timeout_sec` -- and the error path, three
        # lines down the same run, said the opposite of the same number.
        transcription_timeout, capped_by_run = _attempt_limit(
            _calculate_timeout(audio_duration, whisper_settings), whisper_settings
        )
        logger.info(
            "Audio duration: %.1fs, timeout: %.1fs%s",
            audio_duration,
            transcription_timeout,
            " (capped by the render's remaining budget)" if capped_by_run else "",
        )

        # Add model config to options for subprocess
        trans_ops["_model_size"] = whisper_settings.model_size
        trans_ops["_model_device"] = whisper_settings.model_device

        # Run Whisper with timeout and progress monitoring. A timeout is not
        # final: the limit is derived from audio duration and knows nothing
        # about machine speed, and by this point the run has already paid for
        # the LLM script and the TTS voiceover.
        result_w = None
        for scheduled in _timeout_schedule(transcription_timeout, whisper_settings):
            # Clamped here rather than when the schedule was built. The
            # schedule is computed once, before attempt 1, so its ceiling is
            # the budget as it stood then; attempt 1 spending its whole limit
            # leaves the retry promised time the render no longer has, and
            # the outer timeout then cancels the run with this step's limit
            # unreached -- the #398 misattribution, surviving on the retry
            # path. `capped_by_run` is decided in the same breath as the
            # limit, so the message below describes the limit that expired
            # rather than the budget at the moment it expired.
            limit, capped_by_run = _attempt_limit(scheduled, whisper_settings)
            if limit <= 0:
                logger.error(
                    "No time left in the render's budget for Whisper; raise "
                    "pipeline_timeout_sec in config/core.yaml."
                )
                break
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
                    timeout=limit,
                )
                elapsed = time.time() - start_time
                logger.info("Whisper transcription completed in %.1fs", elapsed)
                break
            except TimeoutError:
                elapsed = time.time() - start_time
                remedy = (
                    "the render's remaining budget capped it; raise "
                    "pipeline_timeout_sec in config/core.yaml"
                    if capped_by_run
                    else "the limit is derived from audio duration alone; raise "
                    "whisper_settings.duration_multiplier or max_timeout_sec in "
                    "config/ai_services.yaml"
                )
                logger.error(
                    "Whisper transcription timed out after %.1fs (limit: %.1fs). %s, "
                    "or run on a less loaded machine.",
                    elapsed,
                    limit,
                    remedy,
                )
                if whisper_settings.enable_resource_monitoring:
                    _log_system_resources("after Whisper timeout")
                if whisper_settings.enable_resource_cleanup:
                    _cleanup_whisper_resources()
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error("Whisper transcription failed after %.1fs: %s", elapsed, e)
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

        if result_w is None:
            return None

        # Rejoin words Whisper split at a separator, before the flat list is
        # extracted from this dict rather than after. `_extract_word_timings`
        # strips each word, and the leading space is the only thing that
        # separates a continuation from a word legitimately beginning with an
        # apostrophe -- joining afterwards protects pycaps and glues
        # `get 'em` into `get'em` on the FFmpeg engine, which is what a
        # default install renders with.
        #
        # Outside the smoothing gate below: that flag governs four cosmetic
        # timing rules, and turning it off must not restore a caption reading
        # `2 4GHz` for `2.4GHz`. Text correctness is not a timing preference.
        from src.video.subtitle_timing_smoother import join_continuations_in_result

        result_w = join_continuations_in_result(result_w)

        # Extract word timing data from transcription result
        word_list_whisper = _extract_word_timings(result_w)

        if not word_list_whisper:
            logger.warning("Whisper provided no usable word timings.")
            return None
        logger.info("Extracted %s word timings from Whisper.", len(word_list_whisper))

        # Apply timing smoothing to fix Whisper's coarse word timestamps.
        # Smooth both the flat list (for FFmpeg) and the raw dict (for pycaps).
        ts_cfg = timing_smoothing_config or {}
        if ts_cfg.get("enabled", True):
            from src.video.subtitle_timing_smoother import (
                smooth_whisper_result_dict,
                smooth_word_timings,
            )

            smoother_kwargs = {
                k: ts_cfg[k]
                for k in (
                    "min_word_sec",
                    "gap_merge_sec",
                    "hold_last_sec",
                    "lead_sec",
                    "hook_lead_sec",
                    "hook_lead_word_count",
                )
                if k in ts_cfg
            }
            word_list_whisper = smooth_word_timings(
                word_list_whisper, **smoother_kwargs
            )
            result_w = smooth_whisper_result_dict(result_w, **smoother_kwargs)

        # Whisper debug files are written whenever debug_mode is on. This
        # used to read a `create_whisper_debug_files` flag from the old
        # `src.video.video_config` module, but that module was removed in the
        # config modularization: the import raised, the except always set
        # True, and the flag was never a declared field on DebugSettings, so
        # it had been dead in both directions. Reinstate the flag as a typed
        # DebugSettings field if the knob is ever wanted again.
        if debug_mode:
            _save_whisper_debug_files(
                debug_file_dir, audio_path, result_w, word_list_whisper, script
            )

        # Save raw Whisper dict for pycaps consumption when requested. This is
        # intentionally independent of debug_mode so production runs of the
        # pycaps engine always get a transcript artifact.
        if transcript_out_path is not None:
            try:
                from src.video.pycaps_engine import save_whisper_transcript

                save_whisper_transcript(result_w, transcript_out_path)
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "Failed to save Whisper transcript for pycaps at %s: %s",
                    transcript_out_path,
                    e,
                    exc_info=debug_mode,
                )

        return word_list_whisper
    except Exception as e:
        logger.exception("Whisper STT error: %s", e)
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
            logger.error("Invalid Google STT encoding '%s'.", encoding_name)
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
            "Google Cloud STT completed: %s words with timing", len(word_timings)
        )
        return word_timings

    except Exception as e:
        logger.error("Google Cloud STT error: %s", e, exc_info=debug_mode)
        return None


# Helper functions (these would be extracted from the legacy code)


def _load_whisper_model(whisper_settings: WhisperSettings, debug_mode: bool):
    """Load Whisper model with configured settings."""
    if not WHISPER_AVAILABLE:
        return None

    try:
        # Fix multiprocessing issue: set PyTorch to single-threaded mode
        # This prevents "Broken pipe" errors when using asyncio executors
        import torch

        torch.set_num_threads(1)

        model = whisper.load_model(
            whisper_settings.model_size,
            device=whisper_settings.model_device,
            in_memory=whisper_settings.model_in_memory,
            download_root=whisper_settings.model_download_root
            or os.path.expanduser(DEFAULT_WHISPER_MODEL_DIR),
        )
        logger.info("Whisper model loaded: %s", whisper_settings.model_size)
        return model
    except Exception as e:
        logger.error("Failed to load Whisper model: %s", e)
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
        logger.debug(
            "Audio file: %s, size: %s bytes", audio_path.name, f"{file_size:,}"
        )
    except Exception as e:
        logger.debug("Could not get audio file info: %s", e)


def _log_system_resources(context: str):
    """Log current system resource usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        logger.debug(
            "System resources %s: CPU %.1f%%, Memory %.1f%% (%.1fGB available)",
            context,
            cpu_percent,
            memory.percent,
            memory.available / 1024**3,
        )
    except Exception as e:
        logger.debug("Could not log system resources: %s", e)


def _get_audio_duration(audio_path: Path) -> float:
    """Get audio duration using FFprobe."""
    import subprocess

    from src.video.config import config

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            # `config/core.yaml` has carried this since it shipped; it was
            # read by a dotted string until this became a declared field.
            timeout=config.system_timeouts.ffprobe_timeout,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError) as e:
        logger.warning("Failed to get audio duration: %s, using 60s fallback", e)
        return 60.0


def _stt_ceiling(whisper_settings: WhisperSettings) -> tuple[float, bool]:
    """The most this transcription may be allowed, and whether the run capped it.

    `max_timeout_sec` alone is an inner limit derived from audio length, and
    with the shipped settings it can exceed the whole render's budget: a
    59-second voiceover earned 1008s inside a 900s pipeline (#398). Whisper
    then finished inside its own limit having spent most of the run's, the
    pipeline timeout fired during assembly, and the log blamed the pipeline
    rather than the step that spent the time.

    Bounded by whatever remains of the render's budget, so a timeout here is
    reported by the step that ran out and the retry schedule cannot widen a
    limit past what exists. No deadline set means no outer bound, which is
    the case for a caller that applies no pipeline timeout at all.
    """
    ceiling = float(whisper_settings.max_timeout_sec)
    remaining = remaining_pipeline_seconds()
    if remaining is None or remaining >= ceiling:
        return ceiling, False
    return remaining, True


def _calculate_timeout(
    audio_duration: float, whisper_settings: WhisperSettings
) -> float:
    """Calculate transcription timeout based on audio duration and settings."""
    timeout = whisper_settings.base_timeout_sec + (
        audio_duration * whisper_settings.duration_multiplier
    )
    ceiling, _ = _stt_ceiling(whisper_settings)
    return min(timeout, ceiling)


def _attempt_limit(
    scheduled: float, whisper_settings: WhisperSettings
) -> tuple[float, bool]:
    """The limit for one attempt, and whether the run's budget set it.

    Read at the moment the attempt starts, not when the schedule was built.
    The schedule is computed once, so its ceiling is the budget as it stood
    before attempt 1; after attempt 1 spends its whole limit the retry would
    otherwise be handed more time than the render has left, and the outer
    timeout cancels the run with this step's limit unreached -- which is the
    #398 misattribution, surviving on the retry path.

    The flag is returned with the limit rather than recomputed when the
    attempt fails, because the budget only shrinks: recomputing it later can
    only turn a False into a True, and then the error names
    `pipeline_timeout_sec` for a limit the formula set, sending the operator
    to a knob that changes nothing.
    """
    ceiling, capped_by_run = _stt_ceiling(whisper_settings)
    if scheduled <= ceiling:
        return scheduled, False
    return ceiling, capped_by_run


def _timeout_schedule(
    first_limit: float, whisper_settings: WhisperSettings
) -> list[float]:
    """Limits to try, widening after each timeout.

    A retry that gets the same limit cannot do better than the attempt that
    just failed, so the schedule stops as soon as widening is capped. That
    also makes the empty-retry case explicit rather than a loop that silently
    repeats itself.

    The outer budget caps the widening here too, but only as it stands when
    the schedule is built. Each attempt is clamped again against the budget
    left at the moment it starts, which is the check that actually holds:
    this one cannot see what attempt 1 will spend.
    """
    limits = [first_limit]
    ceiling, _ = _stt_ceiling(whisper_settings)
    multiplier = whisper_settings.timeout_retry_multiplier

    for _ in range(max(0, whisper_settings.timeout_retry_attempts)):
        widened = min(limits[-1] * multiplier, ceiling)
        if widened <= limits[-1]:
            break
        limits.append(widened)

    return limits


async def _transcribe_with_monitoring(
    model, audio_path: str, options: dict, debug_mode: bool, settings: WhisperSettings
):
    """Run Whisper transcription with progress monitoring."""
    import concurrent.futures

    # Use ProcessPoolExecutor for CPU-intensive Whisper to avoid asyncio conflicts
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        result = await loop.run_in_executor(
            executor, _transcribe_helper, audio_path, options
        )
    return result


def _transcribe_helper(audio_path: str, options: dict):
    """Helper function for process pool execution."""
    import torch
    import whisper

    # Set single-threaded mode in subprocess
    torch.set_num_threads(1)

    # Load model in subprocess (can't pass model object across processes)
    model_size = options.pop("_model_size", "small")
    model_device = options.pop("_model_device", "cpu")
    model = whisper.load_model(model_size, device=model_device)

    # Transcribe
    return model.transcribe(audio_path, **options)


def _extract_word_timings(whisper_result: dict) -> list[dict[str, Any]]:
    """Extract word-level timings from Whisper result.

    Note: Whisper provides absolute timestamps relative to the audio file,
    including any leading silence. No offset adjustment is needed.
    """
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

        logger.debug("Whisper debug files saved to %s", debug_dir)

    except Exception as e:
        logger.warning("Failed to save Whisper debug files: %s", e)
