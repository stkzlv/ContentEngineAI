"""Audio filter chain construction.

This module provides utilities for building FFmpeg audio filter chains with
support for voiceover and background music. Source video audio is not carried
into the render.

Tracks are combined with fixed-level mixing (FFmpeg amix, normalize off).
Voice-keyed ducking is available behind `music_ducking_enabled` and is off by
default, so the fixed-level mix is what a stock config produces.

The mix is then mastered to a loudness target with `loudnorm` (EBU R128),
which is on by default: platforms normalize on playback, so a render that
sits below the target is pushed up relative to the feed around it.
"""

import logging
from pathlib import Path
from typing import Any

from src.video.config import VideoConfig

logger = logging.getLogger(__name__)


class AudioFilterBuilder:
    """Build FFmpeg audio filter chains for the voiceover and music mix.

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

    def sting_path(self) -> Path | None:
        """The configured sting file, or None when unset or missing.

        A missing file warns and mixes nothing: an identity mark is not worth
        losing a render over.
        """
        sting = self.config.audio_settings.signature_sting
        if sting is None:
            return None
        if not sting.path.is_file():
            logger.warning("Signature sting %s not found; mixing none", sting.path)
            return None
        return sting.path

    def prepare_sting_input(
        self, input_cmd_parts: list[str], sting_path: Path | None
    ) -> int | None:
        """Add the sting as the next input and return its index."""
        if sting_path is None:
            return None
        index = input_cmd_parts.count("-i")
        input_cmd_parts.extend(["-i", str(sting_path)])
        return index

    def sting_delay_sec(self, sting_duration: float, voice_end: float) -> float:
        """When the sting starts, by its configured position.

        Placed against the end of the narration, not of the video: the mix
        lasts as long as its first input (`audio_mix_duration: "first"`, the
        voiceover), and the video runs an outro past it, so a sting placed
        against the video's end played in the outro and was cut to silence.
        """
        sting = self.config.audio_settings.signature_sting
        if sting is None:
            return 0.0
        latest = max(0.0, voice_end - sting_duration)
        if sting.position == "end":
            return max(0.0, latest - sting.offset_sec)
        return min(sting.offset_sec, latest)

    async def build_mix(
        self,
        input_cmd_parts: list[str],
        voiceover_audio_path: Path | None,
        music_track_path: Path | None,
        total_video_duration: float,
        media_inspector: Any,
    ) -> tuple[list[str], str]:
        """Add the audio inputs and build the whole mix, the sting included.

        The one entry point the assembler calls, so the sting's input index,
        its placement and the filter that mixes it cannot be wired in one
        place and dropped in another.
        """
        voiceover_idx, music_idx = self.prepare_audio_inputs(
            input_cmd_parts,
            voiceover_audio_path,
            music_track_path,
            input_cmd_parts.count("-i"),
        )
        sting_path = self.sting_path()
        sting_idx = self.prepare_sting_input(input_cmd_parts, sting_path)
        delay = 0.0
        if sting_path is not None:
            voice_end = (
                await media_inspector.get_media_duration(voiceover_audio_path)
                if voiceover_audio_path
                else total_video_duration
            ) or total_video_duration
            delay = self.sting_delay_sec(
                await media_inspector.get_media_duration(sting_path), voice_end
            )
        return self.build_audio_filters(
            voiceover_idx,
            music_idx,
            total_video_duration,
            sting_input_idx=sting_idx,
            sting_delay_sec=delay,
        )

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
        sting_input_idx: int | None = None,
        sting_delay_sec: float = 0.0,
    ) -> tuple[list[str], str]:
        """Build audio processing filters for FFmpeg.

        Args:
        ----
            voiceover_input_idx: Index of voiceover input in FFmpeg command
            music_input_idx: Index of music input in FFmpeg command
            total_video_duration: Target video duration for fade calculations
            sting_input_idx: Index of the signature sting input, if any
            sting_delay_sec: When the sting starts, from `sting_delay_sec()`

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

        # Duck the music under the narration, when both are present and the
        # duck is enabled. `sidechaincompress` takes two inputs and emits one:
        # the music, attenuated whenever the key input is loud. The voice has
        # to be split first, because it is both the key and a track in the
        # mix, and a stream cannot be consumed twice.
        if (
            audio_settings.music_ducking_enabled
            and voiceover_input_idx is not None
            and music_input_idx is not None
        ):
            audio_filters.append("[a_voice_proc]asplit=2[a_voice_mix][a_voice_key]")
            audio_filters.append(
                f"[a_music_proc][a_voice_key]sidechaincompress="
                f"threshold={audio_settings.music_ducking_threshold}:"
                f"ratio={audio_settings.music_ducking_ratio}:"
                f"attack={audio_settings.music_ducking_attack_ms}:"
                f"release={audio_settings.music_ducking_release_ms}"
                f"[a_music_ducked]"
            )
            audio_to_mix = ["[a_voice_mix]", "[a_music_ducked]"]

        # The sting joins the mix after the duck, so the duck never keys on
        # it, and before `loudnorm`, so it is mastered with the programme.
        sting = audio_settings.signature_sting
        if sting_input_idx is not None and sting is not None:
            delay_ms = int(round(sting_delay_sec * 1000))
            audio_filters.append(
                f"[{sting_input_idx}:a]volume={sting.volume_db}dB,"
                f"adelay={delay_ms}:all=1[a_sting]"
            )
            audio_to_mix.append("[a_sting]")

        if not audio_to_mix:
            return audio_filters, ""

        if len(audio_to_mix) > 1:
            mixed_label = "[a_mixed]"
            audio_filters.append(
                f"{''.join(audio_to_mix)}amix=inputs={len(audio_to_mix)}:"
                f"duration={audio_settings.audio_mix_duration}:normalize=0{mixed_label}"
            )
        else:
            mixed_label = audio_to_mix[0]

        # Master to the platform loudness target, before the pad rather than
        # after it: `apad` appends silence to reach the video duration, and
        # normalising is a statement about the programme, not about the
        # padding.
        #
        # This lands about 1 LU short of the target on real narration, and
        # the reason is the true-peak ceiling rather than the pass running
        # once. Mixed narration arrives above 0 dBTP, so the gain that would
        # reach the target linearly would breach `TP`; `loudnorm` refuses
        # linear normalisation, falls back to dynamic mode, and reports the
        # gap it is leaving as `target_offset`. Feeding a second pass only the
        # four `measured_*` values reports the same offset and produces a
        # byte-identical file -- but the loudnorm author's two-pass also feeds
        # back `offset=<target_offset>`, and that does move it: measured
        # -15.2 -> -14.6 LUFS with the true peak still at -1.0.
        #
        # So two-pass would help, by about half the shortfall. It is
        # unavailable here for a different reason: it needs the mixed audio
        # as a file to measure, and that only exists inside this filtergraph.
        # Taking it would mean rendering the audio separately first.
        #
        if audio_settings.loudness_normalization_enabled:
            normalized_label = "[a_norm]"
            audio_filters.append(
                f"{mixed_label}loudnorm="
                f"I={audio_settings.loudness_target_lufs}:"
                f"TP={audio_settings.loudness_true_peak_db}:"
                f"LRA={audio_settings.loudness_range_lu}"
                f"{normalized_label}"
            )
            mixed_label = normalized_label

        # Resample unconditionally, not as a tail of the loudnorm string.
        # `loudnorm` emits at 192 kHz whatever it was handed, so the resample
        # started life as a fix for that -- but `output_audio_sample_rate`
        # names an output property, and hanging it off the normalisation
        # branch meant switching normalisation off silently dropped the rate
        # control with it, leaving the render at whatever the mix negotiated
        # to. That was 24 kHz, the TTS rate, on the renders measured -- but
        # it is the negotiator's choice, not the voiceover's property: with
        # stereo music ffmpeg resamples the music down to the voice, and
        # with mono music it resamples the voice up to the music instead.
        # The delivered file can differ again: a pycaps burn re-mixes after
        # this and re-rates the output.
        rate_label = "[a_rate]"
        audio_filters.append(
            f"{mixed_label}aresample={audio_settings.output_audio_sample_rate}"
            f"{rate_label}"
        )
        mixed_label = rate_label

        # Pad to the video duration so the audio is not truncated.
        final_audio_label = "[a_final]"
        audio_filters.append(
            f"{mixed_label}apad=whole_dur={total_video_duration}{final_audio_label}"
        )

        return audio_filters, final_audio_label
