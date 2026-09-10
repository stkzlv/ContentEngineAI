"""`subtitle_upper_file` is a registered run path (#413).

It used to be inserted into ``ctx.run_paths`` only inside the subtitle step,
so a resume that skipped that step read ``None`` in ``assemble_video`` and
dropped the upper line -- while the recorded artifact said the file existed.
Verification also fell back to existence-only checking for it, because
``_artifact_invalid_reason`` found no expected path to compare against, and
that comparison exists because ``pipeline_state.json`` is product-level
while these artifacts are profile-level.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.video.producer.state import _artifact_invalid_reason, get_video_run_paths


@pytest.fixture(scope="module")
def cfg():
    from src.video.config import load_video_config_modular

    return load_video_config_modular()


class TestTheKeyIsRegistered:
    def test_a_fresh_mapping_declares_it(self, cfg):
        """What the resume reads before the subtitle step has run."""
        rp = get_video_run_paths(cfg, "topic-test-413", "slideshow_images1", {})
        upper = rp.get("subtitle_upper_file")
        assert upper is not None, (
            "subtitle_upper_file must be declared without the subtitle step "
            "having inserted it"
        )
        assert upper.name.startswith("subtitle_upper")

    def test_the_suffix_follows_the_merged_format(self, cfg):
        """One source for the format: the same merge subtitle_file uses."""
        rp = get_video_run_paths(cfg, "topic-test-413", "slideshow_images1", {})
        assert rp["subtitle_upper_file"].suffix == rp["subtitle_file"].suffix

        overridden = get_video_run_paths(
            cfg,
            "topic-test-413",
            "slideshow_images1",
            {"subtitle_settings.subtitle_format": "srt"},
        )
        assert overridden["subtitle_upper_file"].suffix == ".srt"

    def test_it_matches_what_the_handler_derives(self, cfg):
        """The step's writer and the registration must name one file.

        The handler prefers the registered path now, but its fallback
        derivation must agree, or a context built without registered paths
        writes a file the resume then cannot find.
        """
        rp = get_video_run_paths(cfg, "topic-test-413", "slideshow_images1", {})
        fmt = rp["subtitle_file"].suffix.lstrip(".")
        derived = rp["subtitle_file"].with_name(f"subtitle_upper.{fmt}")
        assert rp["subtitle_upper_file"] == derived


class TestVerificationComparesThePath:
    def test_another_runs_upper_file_is_rejected(self, cfg, tmp_path):
        """The profile-scoped comparison the registration restores."""
        rp = get_video_run_paths(cfg, "topic-test-413", "slideshow_images1", {})
        stale = tmp_path / "subtitle_upper.ass"
        stale.write_text("[Script Info]")

        ctx = SimpleNamespace(run_paths=rp)
        reason = _artifact_invalid_reason(ctx, "subtitle_upper_file", str(stale))
        assert reason is not None
        assert "another run" in reason

    def test_this_runs_upper_file_is_accepted(self, cfg, tmp_path, monkeypatch):
        rp = get_video_run_paths(cfg, "topic-test-413", "slideshow_images1", {})
        # Point the registered path at a file we can actually create.
        rp["subtitle_upper_file"] = tmp_path / "subtitle_upper.ass"
        rp["subtitle_upper_file"].write_text("[Script Info]")

        ctx = SimpleNamespace(run_paths=rp)
        reason = _artifact_invalid_reason(
            ctx, "subtitle_upper_file", str(rp["subtitle_upper_file"])
        )
        assert reason is None


class TestTheHandlerPrefersTheRegisteredPath:
    def test_the_writer_reads_the_run_path(self):
        """The generate site must consume the registered key.

        Read structurally: driving the handler needs a voiceover and a
        renderer, and the property is exactly that the writer and the
        registration cannot disagree.
        """
        source = Path("src/video/producer/two_part_subtitles.py").read_text()
        idx = source.index("subtitle_upper.{subtitle_format}")
        window = source[max(0, idx - 500) : idx]
        assert (
            'run_paths.get(\n            "subtitle_upper_file"\n        )' in (window)
            or 'run_paths.get("subtitle_upper_file")' in window
        ), (
            "the upper-line writer must prefer the registered run path; the "
            "inline derivation is only the fallback for contexts without one"
        )
