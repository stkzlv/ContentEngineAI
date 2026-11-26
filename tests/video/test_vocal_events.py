"""The caption filter drops words heard from a performed take's vocal events
and never a word the script says (REQ-CNT-183).
"""

from __future__ import annotations

import io
import json
import shutil
import wave
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video import gemini_api_tts
from src.video.config.audio_models import TTSConfig, VoiceProfileConfig
from src.video.config.core_models import WhisperSettings
from src.video.delivery_plan import DeliveryEvent, DeliveryPlan, DeliverySegment
from src.video.gemini_api_tts import join_wavs, synthesize_segments
from src.video.tts import TTSManager
from src.video.vocal_events import (
    event_windows,
    filter_result,
    filter_timings,
    leading_trim_sec,
    read_windows,
    shift_windows,
    words_to_drop,
    write_windows,
)

pytestmark = pytest.mark.req("REQ-CNT-183")

SIGH = [{"tag": "<sigh>", "start": 2.0, "end": 4.0}]


def _wav(frames: int, rate: int = 24000, value: int = 1, lead: int = 0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * lead + value.to_bytes(2, "little") * frames)
    return buf.getvalue()


def _segments() -> list[DeliverySegment]:
    return [
        DeliverySegment(
            text="The clocks stop.",
            events=[DeliveryEvent(tag="<sigh>", before_word=0)],
            gap_after_ms=500,
        ),
        DeliverySegment(text="On purpose."),
        DeliverySegment(
            text="Here is the twist.",
            events=[DeliveryEvent(tag="<laugh>", after_last=True)],
        ),
    ]


class TestTheWindows:
    def test_each_event_spans_its_segment_and_gap(self):
        windows = event_windows(_segments(), [1.0, 2.0, 1.5])
        assert windows == [
            {"tag": "<sigh>", "start": 0.0, "end": 1.5},
            {"tag": "<laugh>", "start": 3.5, "end": 5.0},
        ]

    def test_without_every_segment_length_there_are_no_windows(self):
        assert event_windows(_segments(), [1.0]) == []

    def test_the_join_reports_each_segment_length(self, tmp_path):
        assert join_wavs([_wav(24000), _wav(12000)], [500, 0], tmp_path / "v.wav") == [
            1.0,
            0.5,
        ]

    @pytest.mark.asyncio
    async def test_the_segment_lengths_reach_the_caller(self, tmp_path):
        async def fake(*args):
            return _wav(24000)

        durations: list[float] = []
        with patch.object(gemini_api_tts, "_voice_one", side_effect=fake):
            await synthesize_segments(
                _segments(),
                tmp_path / "v.wav",
                api_key="k",
                model="m",
                voice=None,
                endpoint="https://example.invalid",
                durations_out=durations,
            )
        assert durations == [1.0, 1.0, 1.0]

    def test_a_trimmed_start_moves_the_windows_earlier(self):
        assert shift_windows(SIGH, 0.5) == [{"tag": "<sigh>", "start": 1.5, "end": 3.5}]

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
    def test_the_trim_is_measured_by_running_its_start_stage(self, tmp_path):
        path = tmp_path / "v.wav"
        path.write_bytes(_wav(24000, value=20000, lead=12000))
        stage = "silenceremove=start_periods=1:start_threshold=-50dB:start_duration=0"
        assert leading_trim_sec("ffmpeg", path, stage) == pytest.approx(0.5, abs=0.03)

    def test_a_failed_measure_is_no_shift(self, tmp_path):
        path = tmp_path / "v.wav"
        path.write_bytes(_wav(10))
        assert leading_trim_sec(str(tmp_path / "no-ffmpeg"), path, "anull") == 0.0

    def test_the_file_round_trips_and_a_bad_one_reads_as_none(self, tmp_path):
        path = tmp_path / "vocal_events.json"
        write_windows(path, SIGH)
        assert read_windows(path) == SIGH
        path.write_text("{not json")
        assert read_windows(path) == []
        assert read_windows(tmp_path / "missing.json") == []


class TestTheFilter:
    def test_a_word_heard_at_a_sigh_is_dropped(self):
        words = [("Ugh,", 2.1, 2.3), ("the", 2.4, 2.5), ("clocks", 2.5, 2.9)]
        assert words_to_drop(words, SIGH, "The clocks stop.") == {0}

    def test_the_same_word_outside_the_window_stays(self):
        words = [("ugh", 6.0, 6.2), ("the", 6.3, 6.4)]
        assert words_to_drop(words, SIGH, "The clocks stop.") == set()

    def test_a_word_the_script_says_is_never_dropped(self):
        words = [("but", 2.1, 2.3), ("it", 2.4, 2.5), ("works", 2.5, 2.9)]
        assert words_to_drop(words, SIGH, "But it works.") == set()

    def test_only_the_surplus_copy_is_dropped(self):
        words = [("but", 0.5, 0.7), ("no", 0.8, 0.9), ("but", 2.5, 2.7)]
        assert words_to_drop(words, SIGH, "But no.") == {2}

    def test_two_copies_in_the_window_keep_the_one_the_script_says(self):
        words = [("but", 2.1, 2.2), ("no", 2.5, 2.6), ("but", 2.8, 2.9)]
        assert words_to_drop(words, SIGH, "But no.") == {2}

    def test_the_spoken_copy_stays_when_the_sigh_follows_it(self):
        words = [
            ("It's", 2.0, 2.1),
            ("slow,", 2.1, 2.3),
            ("but", 2.3, 2.4),
            ("it", 2.4, 2.5),
            ("works.", 2.5, 2.8),
            ("But", 3.0, 3.2),
        ]
        assert words_to_drop(words, SIGH, "It's slow, but it works.") == {5}

    def test_a_word_glued_to_punctuation_still_counts_as_scripted(self):
        words = [("slow", 2.1, 2.2), ("but", 2.3, 2.4), ("works", 2.5, 2.6)]
        assert words_to_drop(words, SIGH, "Slow\u2014but works.") == set()
        assert (
            words_to_drop([("Ugh", 2.1, 2.2), ("the", 2.3, 2.4)], SIGH, "Ugh...the")
            == set()
        )

    def test_a_stray_copy_elsewhere_does_not_cost_the_spoken_one(self):
        words = [
            ("but", 0.1, 0.2),
            ("fine", 0.3, 0.4),
            ("but", 2.3, 2.4),
            ("ok", 2.5, 2.6),
        ]
        assert words_to_drop(words, SIGH, "Fine, but ok.") == set()

    def test_a_tag_with_no_recorded_word_drops_nothing(self):
        laugh = [{"tag": "<laugh>", "start": 2.0, "end": 4.0}]
        assert words_to_drop([("ugh", 2.1, 2.3)], laugh, "Fine.") == set()

    def test_the_window_reaches_a_little_past_its_ends(self):
        assert words_to_drop([("ugh", 4.4, 4.5)], SIGH, "Fine.") == {0}
        assert words_to_drop([("ugh", 4.6, 4.7)], SIGH, "Fine.") == set()

    def test_the_flat_timings_lose_the_word(self):
        timings = [
            {"word": "ugh", "start": 2.1, "end": 2.3},
            {"word": "fine", "start": 2.4, "end": 2.6},
        ]
        assert filter_timings(timings, SIGH, "Fine.") == timings[1:]

    def test_the_transcript_loses_the_word_and_its_text(self):
        result: dict[str, Any] = {
            "segments": [
                {
                    "text": " Ugh, the clocks.",
                    "words": [
                        {"word": " Ugh,", "start": 2.1, "end": 2.3},
                        {"word": " the", "start": 2.4, "end": 2.5},
                        {"word": " clocks.", "start": 2.5, "end": 2.9},
                    ],
                }
            ]
        }
        filter_result(result, SIGH, "The clocks.")
        segment = result["segments"][0]
        assert [w["word"] for w in segment["words"]] == [" the", " clocks."]
        assert segment["text"] == " the clocks."


@pytest.mark.asyncio
async def test_whisper_hands_on_the_filtered_words(tmp_path):
    import src.video.stt_functions as stt

    audio = tmp_path / "voiceover.wav"
    audio.write_bytes(b"")
    transcript = tmp_path / "whisper_transcript.json"
    result = {
        "text": " Ugh, the clocks stop.",
        "segments": [
            {
                "text": " Ugh, the clocks stop.",
                "start": 2.0,
                "end": 3.5,
                "words": [
                    {"word": " Ugh,", "start": 2.1, "end": 2.3},
                    {"word": " the", "start": 2.4, "end": 2.6},
                    {"word": " clocks", "start": 2.6, "end": 3.0},
                    {"word": " stop.", "start": 3.0, "end": 3.4},
                ],
            }
        ],
    }
    settings = WhisperSettings(
        enable_resource_monitoring=False, enable_resource_cleanup=False
    )
    with (
        patch.object(stt, "WHISPER_AVAILABLE", True),
        patch.object(stt, "_load_whisper_model", return_value=object()),
        patch.object(stt, "_prepare_transcription_options", return_value={}),
        patch.object(stt, "_log_audio_file_info", return_value=None),
        patch.object(stt, "_get_audio_duration", return_value=10.0),
        patch.object(
            stt, "_transcribe_with_monitoring", AsyncMock(return_value=result)
        ),
    ):
        timings = await stt.generate_subtitles_with_whisper(
            audio,
            tmp_path,
            settings,
            "The clocks stop.",
            transcript_out_path=transcript,
            timing_smoothing_config={"enabled": False},
            vocal_events=SIGH,
        )
    assert [t["word"] for t in timings or []] == ["the", "clocks", "stop."]
    saved = json.loads(transcript.read_text())
    assert "Ugh" not in json.dumps(saved)


class TestTheTake:
    PROFILE = VoiceProfileConfig(
        provider="gemini_api", gemini_model_name="m", voice_name="Orus"
    )

    def _manager(self) -> TTSManager:
        config = TTSConfig(
            provider_order=["google_cloud"],
            voice_profiles={"p": self.PROFILE},
            default_voice_profile="p",
        )
        return TTSManager(config, {"GEMINI_API_KEY": "k"})

    @pytest.mark.asyncio
    async def test_a_plan_take_records_its_windows_and_the_next_take_clears_them(
        self, tmp_path
    ):
        async def fake(*args):
            return _wav(24000)

        manager = self._manager()
        with patch.object(gemini_api_tts, "_voice_one", side_effect=fake):
            await manager.generate_speech(
                "The clocks stop. On purpose. Here is the twist.",
                tmp_path / "v.wav",
                DeliveryPlan(segments=_segments()),
            )
        assert [w["tag"] for w in manager.vocal_event_windows] == [
            "<sigh>",
            "<laugh>",
        ]
        with patch(
            "src.video.tts.synthesize_segments",
            AsyncMock(side_effect=gemini_api_tts.GeminiApiTTSError("down")),
        ):
            await manager.generate_speech("Fine.", tmp_path / "w.wav", None)
        assert manager.vocal_event_windows == []


@pytest.fixture
def ctx(tmp_path):
    from src.video.config import config
    from src.video.producer.context import PipelineContext
    from src.video.producer.state import get_video_run_paths

    cfg = config.model_copy(deep=True)
    cfg.global_output_root_path = tmp_path
    paths = get_video_run_paths(cfg, "B0TEST0183", "slideshow_images1")
    paths["script_file"].parent.mkdir(parents=True, exist_ok=True)
    context = PipelineContext(
        product=MagicMock(asin="B0TEST0183", topic=None, title="A mount"),
        profile=cfg.video_profiles["slideshow_images1"],
        profile_name="slideshow_images1",
        config=cfg,
        secrets={},
        session=MagicMock(),
        run_paths=paths,
        debug_mode=False,
    )
    context.script = "The clocks stop."
    return context


@pytest.mark.asyncio
class TestTheVoiceoverStep:
    async def _run(self, ctx, windows):
        from src.video.producer import steps

        async def voice(self, text, path, plan):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(_wav(48000, value=20000, lead=12000))
            self.vocal_event_windows = windows
            return Path(path)

        with (
            patch.object(steps.TTSManager, "generate_speech", voice),
            patch.object(steps, "_get_video_duration", AsyncMock(return_value=2.0)),
            patch.object(steps.subprocess, "run"),
            patch.object(steps, "leading_trim_sec", return_value=0.5),
        ):
            await steps.step_create_voiceover(ctx)

    async def test_the_windows_are_written_shifted_by_the_trimmed_start(self, ctx):
        ctx.config.audio_processing.silence_removal_enabled = True
        ctx.config.audio_processing.silence_threshold_db = -50.0
        with patch("pathlib.Path.replace"):
            await self._run(ctx, SIGH)
        assert read_windows(Path(ctx.run_paths["vocal_events_file"])) == [
            {"tag": "<sigh>", "start": 1.5, "end": 3.5}
        ]

    async def test_a_take_without_events_leaves_no_old_windows(self, ctx):
        events = Path(ctx.run_paths["vocal_events_file"])
        write_windows(events, SIGH)
        await self._run(ctx, [])
        assert not events.exists()


def test_every_caption_transcription_gets_the_windows():
    """Each call that transcribes for captions passes them, or that branch
    silently keeps the event words.
    """
    import ast

    for path in (
        "src/video/producer/steps.py",
        "src/video/producer/two_part_subtitles.py",
    ):
        tree = ast.parse(Path(path).read_text())
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "create_unified_subtitles"
        ]
        assert calls, path
        for call in calls:
            assert "vocal_events" in {k.arg for k in call.keywords}, path
