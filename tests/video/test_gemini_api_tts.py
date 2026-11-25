"""The Gemini API speech provider voices a delivery plan (REQ-CNT-181) and
its tags reach only those requests (REQ-CNT-184).
"""

from __future__ import annotations

import base64
import io
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.video import gemini_api_tts
from src.video.config.audio_models import (
    GoogleCloudTTSSettings,
    GoogleCloudVoiceCriteria,
    TTSConfig,
    VoiceProfileConfig,
)
from src.video.delivery_plan import (
    DeliveryEvent,
    DeliveryPlan,
    DeliverySegment,
    text_with_tags,
)
from src.video.gemini_api_tts import (
    GeminiApiTTSError,
    audio_of,
    join_wavs,
    request_body,
    synthesize_segments,
)
from src.video.tts import TTSManager

SCRIPT = "The clocks stop. On purpose. Here is the twist."


def _wav(frames: int, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * frames)
    return buf.getvalue()


def _plan() -> DeliveryPlan:
    return DeliveryPlan(
        segments=[
            DeliverySegment(
                text="The clocks stop. On purpose.",
                style="dry",
                events=[DeliveryEvent(tag="<sigh>", before_word=3)],
                gap_after_ms=500,
            ),
            DeliverySegment(text="Here is the twist.", style="leaning in"),
        ]
    )


@pytest.mark.req("REQ-CNT-181")
class TestTheRequests:
    def test_tags_go_before_their_word_or_at_the_end(self):
        seg = DeliverySegment(
            text="one two three",
            events=[
                DeliveryEvent(tag="<breath>", before_word=1),
                DeliveryEvent(tag="<laugh>", after_last=True),
            ],
        )
        assert text_with_tags(seg) == "one <breath> two three <laugh>"

    def test_the_body_carries_the_style_annotation_model_and_voice(self):
        body = request_body("Hi.", "dry", "Orus", "gemini-3.8-flash-tts")
        content = body["input"][0]["content"][0]
        assert content["text"] == "Hi."
        assert content["annotations"] == [{"type": "speech_metadata", "style": "dry"}]
        assert body["model"] == "gemini-3.8-flash-tts"
        assert body["generation_config"] == {"speech_config": [{"voice": "Orus"}]}
        assert body["response_format"] == {"type": "audio"}

    def test_no_style_sends_no_annotation(self):
        content = request_body("Hi.", "", None, "m")["input"][0]["content"][0]
        assert "annotations" not in content

    def test_the_audio_is_read_from_the_reply_and_a_missing_one_raises(self):
        data = base64.b64encode(b"RIFFxx").decode()
        assert audio_of({"steps": [{"content": [{"data": data}]}]}) == b"RIFFxx"
        with pytest.raises(GeminiApiTTSError):
            audio_of({"status": "failed", "steps": []})


@pytest.mark.req("REQ-CNT-181")
class TestTheJoin:
    def test_segments_join_with_their_planned_gap(self, tmp_path):
        out = tmp_path / "v.wav"
        join_wavs([_wav(100), _wav(100)], [500, 0], out)
        with wave.open(str(out)) as w:
            assert w.getnframes() == 100 + 24000 // 2 + 100

    def test_a_format_mismatch_fails_the_voice(self, tmp_path):
        with pytest.raises(GeminiApiTTSError):
            join_wavs([_wav(10), _wav(10, rate=16000)], [0, 0], tmp_path / "v.wav")

    @pytest.mark.asyncio
    async def test_one_request_per_segment_with_tags_and_style(self, tmp_path):
        bodies = []

        async def fake(session, endpoint, key, body, retries, delay):
            bodies.append(body)
            return _wav(10)

        with patch.object(gemini_api_tts, "_voice_one", side_effect=fake):
            await synthesize_segments(
                _plan().segments,
                tmp_path / "v.wav",
                api_key="k",
                model="m",
                voice="Orus",
                endpoint="https://example.invalid",
                prepare=str.upper,
            )
        texts = [b["input"][0]["content"][0]["text"] for b in bodies]
        styles = [
            b["input"][0]["content"][0]["annotations"][0]["style"] for b in bodies
        ]
        assert texts == ["THE CLOCKS STOP. <SIGH> ON PURPOSE.", "HERE IS THE TWIST."]
        assert styles == ["dry", "leaning in"]


def _manager(profile: VoiceProfileConfig, secrets=None, **extra) -> TTSManager:
    config = TTSConfig(
        **extra,
        provider_order=["google_cloud"],
        google_cloud=GoogleCloudTTSSettings(
            language_code="en-US",
            voice_selection_criteria=[GoogleCloudVoiceCriteria(language_code="en-US")],
        ),
        voice_profiles={"p": profile},
        default_voice_profile="p",
    )
    return TTSManager(config, {"GEMINI_API_KEY": "k"} if secrets is None else secrets)


PROFILE = VoiceProfileConfig(
    provider="gemini_api",
    gemini_model_name="gemini-3.8-flash-tts",
    voice_name="Orus",
    style_prompt="wry",
)


@pytest.mark.req("REQ-CNT-181")
class TestTheManager:
    @pytest.mark.asyncio
    async def test_a_plan_is_voiced_segment_by_segment(self, tmp_path):
        out = tmp_path / "v.wav"
        with patch("src.video.tts.synthesize_segments", new_callable=AsyncMock) as s:
            s.return_value = out
            assert await _manager(PROFILE).generate_speech(SCRIPT, out, _plan()) == out
        assert s.call_args.args[0] == _plan().segments
        assert s.call_args.kwargs["voice"] == "Orus"

    @pytest.mark.asyncio
    async def test_without_a_plan_the_script_is_one_segment_in_the_profile_style(
        self, tmp_path
    ):
        out = tmp_path / "v.wav"
        with patch("src.video.tts.synthesize_segments", new_callable=AsyncMock) as s:
            s.return_value = out
            await _manager(PROFILE).generate_speech(SCRIPT, out)
        assert s.call_args.args[0] == [DeliverySegment(text=SCRIPT, style="wry")]

    @pytest.mark.asyncio
    async def test_no_key_skips_to_the_fallback(self, tmp_path):
        out = tmp_path / "v.wav"
        with (
            patch("src.video.tts.synthesize_segments", new_callable=AsyncMock) as s,
            patch(
                "src.video.tts._generate_google_cloud_speech", new_callable=AsyncMock
            ) as cloud,
        ):
            cloud.return_value = (out, "en-US-x")
            await _manager(PROFILE, secrets={}).generate_speech(SCRIPT, out)
        s.assert_not_awaited()
        cloud.assert_awaited_once()


@pytest.mark.req("REQ-CNT-184")
class TestTagsStayInTheRequests:
    @pytest.mark.asyncio
    async def test_a_failed_voice_falls_back_with_the_script_and_no_tags(
        self, tmp_path
    ):
        out = tmp_path / "v.wav"
        with (
            patch.object(
                gemini_api_tts, "_voice_one", side_effect=GeminiApiTTSError("503")
            ),
            patch(
                "src.video.tts._generate_google_cloud_speech", new_callable=AsyncMock
            ) as cloud,
        ):
            cloud.return_value = (out, "en-US-x")
            await _manager(PROFILE).generate_speech(SCRIPT, out, _plan())
        sent = cloud.call_args.args[0]
        assert "<" not in sent and sent == SCRIPT

    def test_the_plan_itself_never_holds_a_tag_in_its_text(self):
        plan = _plan()
        text_with_tags(plan.segments[0])
        assert " ".join(s.text for s in plan.segments) == SCRIPT


@pytest.mark.req("REQ-CNT-181")
def test_the_key_is_collected_only_for_a_gemini_api_profile():
    from src.video.producer.utils import collect_producer_secrets

    def cfg(provider):
        tts = SimpleNamespace(
            voice_profiles={"p": SimpleNamespace(provider=provider)},
            gemini_api=SimpleNamespace(api_key_env_var="MY_GEMINI_KEY"),
        )
        empty = SimpleNamespace(audio_providers=[])
        return SimpleNamespace(
            llm_settings=SimpleNamespace(api_key_env_var=None, fallback_provider=None),
            stock_media_settings=SimpleNamespace(pexels_api_key_env_var=None),
            audio_settings=SimpleNamespace(
                freesound_api_key_env_var=None,
                freesound_client_id_env_var=None,
                freesound_client_secret_env_var=None,
                freesound_refresh_token_env_var=None,
                audio_providers=empty.audio_providers,
            ),
            tts_config=tts,
        )

    with patch.dict("os.environ", {"MY_GEMINI_KEY": "x"}):
        assert "MY_GEMINI_KEY" in collect_producer_secrets(cfg("gemini_api"))
        assert "MY_GEMINI_KEY" not in collect_producer_secrets(cfg("gemini"))


@pytest.mark.req("REQ-CNT-181")
@pytest.mark.asyncio
@pytest.mark.parametrize("plan", [None, "plan"])
async def test_each_request_is_normalised_once(tmp_path, plan):
    from src.video.config.audio_models import TTSNormalisationSettings

    bodies = []

    async def fake(session, endpoint, key, body, retries, delay):
        bodies.append(body["input"][0]["content"][0]["text"])
        return _wav(10)

    norm = TTSNormalisationSettings(enabled=True, lexicon={"twist": "twist twist"})
    manager = _manager(PROFILE, tts_normalisation=norm)
    with patch.object(gemini_api_tts, "_voice_one", side_effect=fake):
        await manager.generate_speech(
            SCRIPT, tmp_path / "v.wav", _plan() if plan else None
        )
    assert bodies[-1].endswith("the twist twist.")


class _Resp:
    def __init__(self, status, payload=None):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return "error"

    async def json(self):
        return self._payload


class _Session:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        status = self.statuses.pop(0)
        data = base64.b64encode(_wav(5)).decode()
        return _Resp(status, {"steps": [{"content": [{"data": data}]}]})


@pytest.mark.req("REQ-CNT-181")
class TestRetries:
    @pytest.mark.asyncio
    async def test_a_server_error_is_retried(self):
        session = _Session([500, 200])
        audio = await gemini_api_tts._voice_one(session, "u", "k", {}, 1, 0)
        assert audio.startswith(b"RIFF") and session.calls == 2

    @pytest.mark.asyncio
    async def test_with_no_retries_one_failure_raises(self):
        session = _Session([500, 200])
        with pytest.raises(GeminiApiTTSError):
            await gemini_api_tts._voice_one(session, "u", "k", {}, 0, 0)
        assert session.calls == 1

    @pytest.mark.asyncio
    async def test_a_client_error_is_not_retried(self):
        session = _Session([403, 200])
        with pytest.raises(GeminiApiTTSError, match="403"):
            await gemini_api_tts._voice_one(session, "u", "k", {}, 3, 0)
        assert session.calls == 1

    @pytest.mark.asyncio
    async def test_the_session_carries_the_timeout(self, tmp_path):
        seen = {}

        class FakeSession:
            def __init__(self, timeout=None):
                seen["timeout"] = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        async def fake(*args):
            return _wav(5)

        with (
            patch.object(gemini_api_tts.aiohttp, "ClientSession", FakeSession),
            patch.object(gemini_api_tts, "_voice_one", side_effect=fake),
        ):
            await synthesize_segments(
                [DeliverySegment(text="Hi.")],
                tmp_path / "v.wav",
                api_key="k",
                model="m",
                voice=None,
                endpoint="u",
                timeout_sec=7,
            )
        assert seen["timeout"].total == 7

    @pytest.mark.parametrize(
        "reply",
        [{"steps": {"a": 1}}, {"steps": ["x"]}, [], {"steps": [{"content": 3}]}],
    )
    def test_an_unexpected_reply_is_a_voice_error(self, reply):
        with pytest.raises(GeminiApiTTSError):
            audio_of(reply)


@pytest.mark.req("REQ-CNT-181")
class TestTheRecordAndText:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("voice", "expected"),
        [("Orus", "Orus"), (None, "gemini-3.8-flash-tts default")],
    )
    async def test_the_voice_is_recorded(self, tmp_path, voice, expected):
        profile = PROFILE.model_copy(update={"voice_name": voice})
        manager = _manager(profile)
        with patch("src.video.tts.synthesize_segments", new_callable=AsyncMock) as s:
            s.return_value = tmp_path / "v.wav"
            await manager.generate_speech(SCRIPT, tmp_path / "v.wav")
        assert manager.selected_voice_name == expected

    def test_paragraph_breaks_are_kept(self):
        seg = DeliverySegment(
            text="One two.\n\nThree four.",
            events=[DeliveryEvent(tag="<sigh>", before_word=2)],
        )
        assert text_with_tags(seg) == "One two.\n\n<sigh> Three four."

    @pytest.mark.asyncio
    async def test_a_pause_plan_is_not_applied_or_logged(self, tmp_path, caplog):
        from src.video.config.audio_models import PausePlan

        profile = PROFILE.model_copy(update={"pause_plan": PausePlan(jitter=0)})
        with patch("src.video.tts.synthesize_segments", new_callable=AsyncMock) as s:
            s.return_value = tmp_path / "v.wav"
            with caplog.at_level("INFO"):
                await _manager(profile).generate_speech(SCRIPT, tmp_path / "v.wav")
        assert s.call_args.args[0][0].text == SCRIPT
        assert "pause plan" not in caplog.text

    def test_an_8_bit_gap_is_unsigned_silence(self, tmp_path):
        def wav8():
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(1)
                w.setframerate(1000)
                w.writeframes(b"\x90" * 4)
            return buf.getvalue()

        out = tmp_path / "v.wav"
        join_wavs([wav8(), wav8()], [10, 0], out)
        with wave.open(str(out)) as w:
            assert w.readframes(w.getnframes())[4:14] == b"\x80" * 10
