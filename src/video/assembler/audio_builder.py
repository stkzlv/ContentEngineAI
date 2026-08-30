"""Audio filter chain construction.

This module provides utilities for building FFmpeg audio filter chains with
support for voiceover and background music. Source video audio is not carried
into the render. Tracks are combined with fixed-level mixing (FFmpeg amix,
normalize off), not sidechain ducking.
"""

import logging
from pathlib import Path

from src.video.config import VideoConfig

logger = logging.getLogger(__name__)


class AudioFilterBuilder:
    """Build FFmpeg audio filter chains with fixed-level mixing.

    This class handles construction of complex audio filter graphs for FFmpeg,
    covering voiceover processing and background music with fades.
    """

    def __init__(self, config: VideoConfig):
        """Initialize AudioFilterBuilder.

        Args:
        ----
            config: VideoConfig containing audio settings

        """
        self.config = config

    def prepare_audio_inputs(
        self,
        input_cmd_parts: list[str],
        voiceover_audio_path: Path | None,
        music_track_path: Path | None,
        visual_input_count: int,
    ) -> tuple[int | None, int | None]:
        """Add audio inputs to FFmpeg command and return their indices.

        Args:
        ----
            input_cmd_parts: List of FFmpeg input command parts to extend
            voiceover_audio_path: Path to voiceover audio file
            music_track_path: Path to background music file
            visual_input_count: Number of visual inputs (for index calculation)

        Returns:
        -------
            Tuple of (voiceover_input_idx, music_input_idx)

        """
        audio_input_idx_start = visual_input_count
        voiceover_input_idx, music_input_idx = None, None

        if voiceover_audio_path:
            input_cmd_parts.extend(["-i", str(voiceover_audio_path)])
            voiceover_input_idx = audio_input_idx_start
            audio_input_idx_start += 1

        if music_track_path:
            input_cmd_parts.extend(["-i", str(music_track_path)])
            music_input_idx = audio_input_idx_start

        return voiceover_input_idx, music_input_idx

    def build_audio_filters(
        self,
        voiceover_input_idx: int | None,
        music_input_idx: int | None,
        total_video_duration: float,
    ) -> tuple[list[str], str]:
        """Build audio processing filters for FFmpeg.

        Args:
        ----
            voiceover_input_idx: Index of voiceover input in FFmpeg command
            music_input_idx: Index of music input in FFmpeg command
            total_video_duration: Target video duration for fade calculations

        Returns:
        -------
            Tuple of (audio_filters, final_audio_label)

        """
        audio_settings = self.config.audio_settings
        audio_filters = []
        audio_to_mix = []

        if voiceover_input_idx is not None:
            proc_label = "[a_voice_proc]"
            audio_filters.append(
                f"[{voiceover_input_idx}:a]volume={audio_settings.voiceover_volume_db}dB{proc_label}"
            )
            audio_to_mix.append(proc_label)

        if music_input_idx is not None:
            music_label, proc_label = f"[{music_input_idx}:a]", "[a_music_proc]"
            fade_out_start = max(
                0, total_video_duration - audio_settings.music_fade_out_duration
            )
            audio_filters.append(
                f"{music_label}volume={audio_settings.music_volume_db}dB,"
                f"afade=t=in:st=0:d={audio_settings.music_fade_in_duration},"
                f"afade=t=out:st={fade_out_start:.3f}:d={audio_settings.music_fade_out_duration}"
                f"{proc_label}"
            )
            audio_to_mix.append(proc_label)

        final_audio_label = ""
        if len(audio_to_mix) > 1:
            mixed_label = "[a_mixed]"
            audio_filters.append(
                f"{''.join(audio_to_mix)}amix=inputs={len(audio_to_mix)}:"
                f"duration={audio_settings.audio_mix_duration}:normalize=0{mixed_label}"
            )
            # Add apad to extend audio to match video duration and prevent truncation
            final_audio_label = "[a_final]"
            audio_filters.append(
                f"{mixed_label}apad=whole_dur={total_video_duration}{final_audio_label}"
            )
        elif len(audio_to_mix) == 1:
            # Single audio stream - still need to pad to match video duration
            padded_label = "[a_final]"
            audio_filters.append(
                f"{audio_to_mix[0]}apad=whole_dur={total_video_duration}{padded_label}"
            )
            final_audio_label = padded_label

        return audio_filters, final_audio_label
