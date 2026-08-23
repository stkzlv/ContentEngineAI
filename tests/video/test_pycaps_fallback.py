"""Tests that a default install renders when pycaps is unavailable.

`config/subtitles.yaml` selects the pycaps engine and `fallback_policy:
fallback_ffmpeg`, but the optional group is not part of `poetry install`. The
fallback used to set a local variable that routed one branch and nothing else:
every dict handed to `create_unified_subtitles` was built from config and still
said "pycaps", so the generator wrote a transcript and no subtitle file, and
the burn step recomputed the engine from config and imported the missing
module. The run ended with no captions from either engine.

The engine is now passed explicitly, so these tests assert on the argument the
generator actually acts on rather than on the caller's branch.
"""

import warnings
from unittest.mock import MagicMock

import pytest

from src.video.pycaps_engine import is_pycaps_available


async def _run_generate_step(
    monkeypatch,
    tmp_path,
    *,
    available: bool,
    profile: str = "slideshow_images1",
    policy: str | None = None,
):
    """Drive the real `step_generate_subtitles` with pycaps reported missing.

    Returns the context and whatever the generator was handed. Captures the
    `engine` argument specifically: that is the value the generator acts on,
    and a test reading the caller's branch instead would pass against every
    version of this bug.
    """
    from src.video.config import load_video_config_modular
    from src.video.producer import steps as steps_mod

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        config = load_video_config_modular()

    monkeypatch.setattr(
        "src.video.pycaps_engine.is_pycaps_available", lambda: available
    )

    captured: dict = {}

    async def _fake_create(voiceover, out_path, settings, *a, **kw):
        captured["settings"] = settings
        captured["engine"] = kw.get("engine")
        captured["out_path"] = out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
        return out_path

    monkeypatch.setattr(steps_mod, "create_unified_subtitles", _fake_create)
    monkeypatch.setattr(
        "src.video.subtitle_utils.create_unified_subtitles", _fake_create
    )

    voiceover = tmp_path / "voiceover.wav"
    voiceover.write_bytes(b"RIFF")
    ctx = MagicMock()
    ctx.config = config
    ctx.profile_name = profile
    ctx.cli_overrides = (
        {"subtitle_settings.pycaps.fallback_policy": policy} if policy else None
    )
    ctx.state = {}
    ctx.voiceover_duration = 10.0
    ctx.debug_mode = False
    ctx.product.asin = "B0TEST0001"
    ctx.product.title = "A product"
    ctx.run_paths = {
        "voiceover_file": voiceover,
        "subtitle_file": tmp_path / "subtitles.ass",
        "whisper_transcript_file": tmp_path / "whisper_transcript.json",
        "run_root": tmp_path,
        "script_file": tmp_path / "script.txt",
        "voiceover_duration_file": tmp_path / "voiceover_duration.txt",
        "temp_dir": tmp_path,
    }
    (tmp_path / "script.txt").write_text("A script.", encoding="utf-8")
    (tmp_path / "voiceover_duration.txt").write_text("10.0", encoding="utf-8")
    await steps_mod.step_generate_subtitles(ctx)
    return ctx, captured


@pytest.mark.unit
class TestFallbackDrivesTheRealStep:
    """Drives `step_generate_subtitles` with pycaps reported unavailable."""

    async def test_the_generator_is_told_ffmpeg_when_pycaps_is_missing(
        self, monkeypatch, tmp_path
    ):
        """The bug: the generator still saw pycaps, so it wrote no subtitles."""
        _, captured = await _run_generate_step(monkeypatch, tmp_path, available=False)
        assert captured["engine"] == "ffmpeg"

    async def test_the_resolved_engine_is_recorded_for_the_burn_step(
        self, monkeypatch, tmp_path
    ):
        ctx, _ = await _run_generate_step(monkeypatch, tmp_path, available=False)
        assert ctx.state["subtitle_engine_resolved"] == "ffmpeg"

    async def test_pycaps_is_kept_when_it_is_available(self, monkeypatch, tmp_path):
        """The fallback must not fire on an install that has the library."""
        ctx, captured = await _run_generate_step(monkeypatch, tmp_path, available=True)
        assert captured["engine"] == "pycaps"
        assert ctx.state["subtitle_engine_resolved"] == "pycaps"

    @pytest.mark.parametrize(
        "profile",
        [
            "slideshow_images3",
            "slideshow_images4",
            "product_video_mixed",
            "product_video_primary",
            "product_video_sequential",
            "product_video_single",
        ],
    )
    async def test_two_part_profiles_also_reach_the_generator_as_ffmpeg(
        self, monkeypatch, tmp_path, profile
    ):
        """The two-part handler builds its own settings dict.

        Six of the eleven bundled profiles enable two-part subtitles, and that
        branch runs only when the engine is *not* pycaps -- so it is exactly
        the branch a fallback run takes. Its dict was built from config and
        still said pycaps, so the generator wrote a transcript, the existence
        check accepted it, and the run reported success with no captions.
        """
        _, captured = await _run_generate_step(
            monkeypatch, tmp_path, available=False, profile=profile
        )
        assert captured["engine"] == "ffmpeg"


@pytest.mark.unit
class TestAvailabilityProbe:
    def test_the_probe_answers_for_this_environment(self):
        """Whichever way it answers, it must not raise.

        `find_spec` does not execute the module, so a broken install can report
        available; this uses a real import for that reason.
        """
        assert isinstance(is_pycaps_available(), bool)


@pytest.mark.unit
class TestFallbackPolicyWiring:
    """The three policies are distinct outcomes, not shades of one.

    `fallback_ffmpeg` must hand the generator ffmpeg; `raise` must abort;
    `warn_and_skip` must not call the generator at all. Conflating the first
    two is what shipped a caption-less render reported as success.
    """

    async def test_fallback_ffmpeg_switches_the_engine(self, monkeypatch, tmp_path):
        ctx, captured = await _run_generate_step(
            monkeypatch, tmp_path, available=False, policy="fallback_ffmpeg"
        )
        assert captured["engine"] == "ffmpeg"
        assert ctx.state["subtitle_engine_resolved"] == "ffmpeg"

    async def test_raise_aborts_the_run(self, monkeypatch, tmp_path):
        from src.video.producer.context import PipelineError

        with pytest.raises(PipelineError):
            await _run_generate_step(
                monkeypatch, tmp_path, available=False, policy="raise"
            )

    async def test_warn_and_skip_never_reaches_the_generator(
        self, monkeypatch, tmp_path
    ):
        """It ships without subtitles, so asking for any is the bug."""
        _, captured = await _run_generate_step(
            monkeypatch, tmp_path, available=False, policy="warn_and_skip"
        )
        assert captured == {}

    def test_the_bundled_policy_is_the_forgiving_one(self):
        """A fork running `poetry install` gets no pycaps, so the shipped
        default decides whether the project works out of the box.
        """
        import yaml

        with open("config/subtitles.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        pycaps = cfg["subtitle_settings"].get("pycaps") or {}
        assert pycaps.get("fallback_policy") == "fallback_ffmpeg"

    def test_the_bundled_engine_is_the_optional_one(self):
        """Which is why the fallback has to work: the default config asks for
        a library the default install does not provide.
        """
        import yaml

        with open("config/subtitles.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["subtitle_settings"]["subtitle_engine"] == "pycaps"


@pytest.mark.unit
class TestBurnStepHonoursTheFallback:
    """The burn step recomputed the engine from config.

    On a fallback run that meant importing a library the run had already
    established was missing, killing a render whose captions FFmpeg had
    already burned. It has to read the run's decision, not the configured one.
    """

    async def _run_burn(
        self,
        monkeypatch,
        tmp_path,
        state: dict,
        *,
        available: bool = True,
        policy: str | None = None,
    ):
        from src.video.config import load_video_config_modular
        from src.video.producer import steps as steps_mod

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = load_video_config_modular()

        monkeypatch.setattr(
            "src.video.pycaps_engine.is_pycaps_available", lambda: available
        )

        # Anything that reaches the renderer is a failure for these cases.
        import src.video.pycaps_engine as pycaps_engine

        def _explode(*a, **kw):
            raise AssertionError("burn step reached the pycaps renderer")

        monkeypatch.setattr(pycaps_engine, "PycapsRenderer", _explode)

        ctx = MagicMock()
        ctx.config = config
        ctx.profile_name = "slideshow_stock"
        ctx.cli_overrides = (
            {"subtitle_settings.pycaps.fallback_policy": policy} if policy else None
        )
        ctx.state = state
        ctx.debug_mode = False
        ctx.run_paths = {
            "whisper_transcript_file": tmp_path / "absent.json",
            "final_video_output": tmp_path / "absent.mp4",
        }
        await steps_mod.step_burn_pycaps_subtitles(ctx)

    async def test_it_skips_when_the_run_fell_back_to_ffmpeg(
        self, monkeypatch, tmp_path
    ):
        """Config says pycaps; the run resolved ffmpeg. The run wins."""
        await self._run_burn(
            monkeypatch, tmp_path, {"subtitle_engine_resolved": "ffmpeg"}
        )

    async def test_a_lost_decision_under_warn_and_skip_still_skips(
        self, monkeypatch, tmp_path
    ):
        """The resolver's third answer is None, meaning ship without subtitles.

        It reaches the burn step only through the re-derive path, so nothing
        else in the suite covers it. A guard tightened to `if engine and
        engine != "pycaps"` -- which reads like a sensible None-check -- would
        drop into the burn body instead of skipping.
        """
        from src.video.producer import steps as steps_mod

        # Entering the burn body under warn_and_skip is invisible from the
        # outside: the missing transcript routes to _handle_pycaps_burn_failure,
        # which swallows it and returns the caption-less video. So assert the
        # step never got there, not merely that it did not raise.
        def _reached(*a, **kw):
            raise AssertionError("burn step entered its body instead of skipping")

        monkeypatch.setattr(steps_mod, "_handle_pycaps_burn_failure", _reached)

        await self._run_burn(
            monkeypatch, tmp_path, {}, available=False, policy="warn_and_skip"
        )

    async def test_a_lost_decision_is_re_derived_not_taken_from_config(
        self, monkeypatch, tmp_path
    ):
        """Resuming a run truncates the state, dropping the recorded engine.

        Trusting config at that point re-opens the bug on exactly the install
        the fallback exists for: pycaps is still missing, so the step would
        import it and fail a render whose FFmpeg captions are already burned.
        Re-deriving reaches the same answer the original run did.
        """
        await self._run_burn(monkeypatch, tmp_path, {}, available=False)

    async def test_an_unrecorded_engine_still_runs_where_pycaps_exists(
        self, monkeypatch, tmp_path
    ):
        """Re-deriving must not become a blanket skip.

        With the library present, config asking for pycaps is honoured, so the
        step proceeds far enough to find the transcript missing and apply
        `fallback_policy` rather than skipping silently.
        """
        from src.video.producer.context import PipelineError

        with pytest.raises(PipelineError):
            await self._run_burn(monkeypatch, tmp_path, {}, available=True)


@pytest.mark.unit
class TestExplicitEngineOverridesTheDict:
    """The generator must act on the passed engine, not the dict's copy.

    The dict is built from config, so on a default install it says "pycaps"
    however the run resolved. If the generator preferred it, the fallback
    would keep writing a transcript and no subtitle file -- the original bug,
    reachable again through any call site that builds its own dict.
    """

    async def test_a_pycaps_dict_with_an_ffmpeg_engine_writes_subtitles(
        self, monkeypatch, tmp_path
    ):
        from src.video import subtitle_utils

        async def _fake_stt(*a, **kw):
            return [
                {"word": "hello", "start_time": 0.0, "end_time": 0.4},
                {"word": "there", "start_time": 0.4, "end_time": 0.9},
            ]

        monkeypatch.setattr(
            subtitle_utils, "generate_subtitles_with_whisper", _fake_stt
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from src.video.config import load_video_config_modular

            config = load_video_config_modular()
        settings = config.get_profile_merged_settings(
            "slideshow_images1", None
        ).subtitle_settings.model_dump()
        settings["subtitle_engine"] = "pycaps"

        audio = tmp_path / "voiceover.wav"
        audio.write_bytes(b"RIFF")
        out = tmp_path / "subtitles.ass"

        result = await subtitle_utils.create_unified_subtitles(
            audio,
            out,
            settings,
            config.whisper_settings,
            config.google_cloud_stt_settings,
            {},
            "hello there",
            1.0,
            False,
            config,
            tmp_path,
            "B0TEST0001",
            engine="ffmpeg",
        )

        assert result is not None
        assert result.suffix in {".ass", ".srt"}, (
            f"expected a subtitle file, got {result.name} -- the dict's "
            "pycaps value won over the explicit engine"
        )
        assert result.exists()
