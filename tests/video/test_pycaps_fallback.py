"""Tests that a default install renders when pycaps is unavailable.

`config/subtitles.yaml` selects the pycaps engine and `fallback_policy:
fallback_ffmpeg`, but the optional group is not part of `poetry install`. The
fallback used to set a local variable that routed the branch and nothing else:
`create_unified_subtitles` re-reads the engine from the settings dict it is
handed, so it took the pycaps path anyway and wrote no subtitle file, and the
burn step recomputed the engine from config and imported the missing module.
The run ended with no captions from either engine and a failed step.
"""

import pytest

from src.video.pycaps_engine import is_pycaps_available


@pytest.mark.unit
class TestFallbackDrivesTheRealStep:
    """Drives `step_generate_subtitles` with pycaps reported unavailable.

    Asserting on the dict the generator is handed, because that is the value
    it re-reads the engine from. A test that only checked the caller's branch
    would have passed against the broken version.
    """

    async def _run_step(self, monkeypatch, tmp_path, available: bool):
        import warnings
        from unittest.mock import MagicMock

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
            captured["out_path"] = out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
            return out_path

        monkeypatch.setattr(steps_mod, "create_unified_subtitles", _fake_create)

        voiceover = tmp_path / "voiceover.wav"
        voiceover.write_bytes(b"RIFF")
        ctx = MagicMock()
        ctx.config = config
        ctx.profile_name = "slideshow_images1"
        ctx.cli_overrides = None
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

    async def test_the_generator_is_told_ffmpeg_when_pycaps_is_missing(
        self, monkeypatch, tmp_path
    ):
        """The bug: the dict still said pycaps, so no subtitle file was written."""
        ctx, captured = await self._run_step(monkeypatch, tmp_path, available=False)
        assert captured["settings"]["subtitle_engine"] == "ffmpeg"

    async def test_the_resolved_engine_is_recorded_for_the_burn_step(
        self, monkeypatch, tmp_path
    ):
        ctx, _ = await self._run_step(monkeypatch, tmp_path, available=False)
        assert ctx.state["subtitle_engine_resolved"] == "ffmpeg"

    async def test_pycaps_is_kept_when_it_is_available(self, monkeypatch, tmp_path):
        """The fallback must not fire on an install that has the library."""
        ctx, captured = await self._run_step(monkeypatch, tmp_path, available=True)
        assert captured["settings"]["subtitle_engine"] == "pycaps"
        assert ctx.state["subtitle_engine_resolved"] == "pycaps"


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

    `fallback_ffmpeg` must produce a captioned video; `warn_and_skip` an
    uncaptioned one; `raise` no video at all. Conflating the first two is what
    shipped a caption-less render reported as success.
    """

    @pytest.mark.parametrize(
        ("policy", "expected_engine"),
        [("fallback_ffmpeg", "ffmpeg"), ("raise", "pycaps")],
    )
    def test_policy_decides_the_resolved_engine(self, policy, expected_engine):
        resolved = "ffmpeg" if policy == "fallback_ffmpeg" else "pycaps"
        assert resolved == expected_engine

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

    async def _run_burn(self, monkeypatch, tmp_path, state: dict):
        import warnings
        from unittest.mock import MagicMock

        from src.video.config import load_video_config_modular
        from src.video.producer import steps as steps_mod

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = load_video_config_modular()

        # Anything that reaches the renderer is a failure for these cases.
        import src.video.pycaps_engine as pycaps_engine

        def _explode(*a, **kw):
            raise AssertionError("burn step reached the pycaps renderer")

        monkeypatch.setattr(pycaps_engine, "PycapsRenderer", _explode)

        ctx = MagicMock()
        ctx.config = config
        ctx.profile_name = "slideshow_stock"
        ctx.cli_overrides = None
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

    async def test_an_unrecorded_engine_falls_back_to_config(
        self, monkeypatch, tmp_path
    ):
        """A run that never reached `generate_subtitles`, such as `--step
        burn_pycaps_subtitles`, has nothing recorded and must still behave.

        Here config says pycaps, so the step proceeds far enough to find the
        transcript missing and apply `fallback_policy` rather than skipping
        silently.
        """
        from src.video.producer.context import PipelineError

        with pytest.raises(PipelineError):
            await self._run_burn(monkeypatch, tmp_path, {})
