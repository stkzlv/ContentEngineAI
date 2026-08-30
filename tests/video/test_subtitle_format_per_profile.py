"""A profile can set `subtitle_format`, and the path follows it.

It could not before, and the reason was a disagreement rather than a missing
field. The generator wrote whichever format the profile-merged settings named,
while `_get_subtitle_filename` derived the file's extension from the *global*
block. The moment a profile overrode the format the two parted company, and
both directions failed:

- global `ass`, profile `srt`: SRT text lands in `subtitles.ass`, the assembler
  picks its filter from the suffix and hands it to FFmpeg's `ass` filter, which
  aborts the render.
- global `srt`, profile `ass`: the generator writes `subtitles.ass`, the step
  looks for `subtitles.srt`, finds nothing, sets `subtitle_path = None`, and
  ships a caption-less video with no error at all.

The key was rejected at profile level to keep both unreachable. That was a
stopgap: the field is settable everywhere else, so its absence from profiles
was a limitation, not a decision.
"""

from __future__ import annotations

import ast

import pytest

from src.video.config.visual_models import VideoProfile


@pytest.fixture
def video_config():
    from src.video.config import config

    return config


def profile_with(video_config, base_name: str, **overrides) -> VideoProfile:
    """A real bundled profile with a field replaced.

    Built from a real one because `VideoProfile` has required fields, and
    because the question is what an operator editing a shipped profile gets.
    """
    data = video_config.video_profiles[base_name].model_dump(exclude_none=True)
    data.pop("subtitle_settings", None)
    data.update(overrides)
    return VideoProfile(**data)


class TestTheKeyIsAccepted:
    def test_the_nested_spelling_loads(self, video_config):
        profile = profile_with(
            video_config,
            "slideshow_images1",
            subtitle_settings={"subtitle_format": "srt"},
        )

        assert profile.subtitle_settings is not None
        assert profile.subtitle_settings.subtitle_format == "srt"

    def test_the_flat_spelling_is_refused_like_its_siblings(self, video_config):
        """The key is settable per profile; the flat spelling is not a spelling.

        It was deliberately absent from the legacy map for the same reason the
        nested form was rejected. It is in the map now, which under the current
        rule means the error names where to move it rather than that it is
        accepted -- no subtitle key has a flat spelling any more.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as excinfo:
            profile_with(video_config, "slideshow_images1", subtitle_format="srt")

        assert "subtitle_format -> subtitle_settings.subtitle_format" in str(
            excinfo.value
        )

    def test_an_invalid_format_is_still_rejected(self, video_config):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            profile_with(
                video_config,
                "slideshow_images1",
                subtitle_settings={"subtitle_format": "vtt"},
            )


class TestThePathFollowsTheProfile:
    """The disagreement that made the key unsafe to accept."""

    def test_a_profile_asking_for_srt_gets_an_srt_path(self, video_config, monkeypatch):
        """Global is `ass` in the bundled config; this is the aborting case."""
        assert video_config.subtitle_settings.get("subtitle_format") == "ass"

        srt_profile = profile_with(
            video_config,
            "slideshow_images1",
            subtitle_settings={"subtitle_format": "srt"},
        )
        monkeypatch.setitem(video_config.video_profiles, "srt_profile", srt_profile)

        path = video_config.get_product_paths("B0TEST001", "srt_profile")["subtitles"]

        assert path.suffix == ".srt", (
            "the path still comes from the global format, so SRT text would "
            "be written into a file the assembler feeds to FFmpeg's ass filter"
        )

    def test_a_profile_asking_for_ass_gets_an_ass_path(self, video_config, monkeypatch):
        """The mirror case, which fails silently rather than loudly."""
        ass_profile = profile_with(
            video_config,
            "slideshow_images1",
            subtitle_settings={"subtitle_format": "ass"},
        )
        monkeypatch.setitem(video_config.video_profiles, "ass_profile", ass_profile)
        monkeypatch.setitem(video_config.subtitle_settings, "subtitle_format", "srt")

        path = video_config.get_product_paths("B0TEST001", "ass_profile")["subtitles"]

        assert path.suffix == ".ass", (
            "the path follows the global `srt`, so the step looks for a file "
            "the generator never wrote and ships a caption-less video"
        )

    def test_the_path_and_the_merged_format_agree_for_every_profile(self, video_config):
        """The invariant, rather than the two cases.

        Whatever a profile resolves to, the file it is written to must carry
        that extension -- that agreement is the whole fix.
        """
        for name in video_config.video_profiles:
            if name == "base":
                continue
            merged = video_config.get_profile_merged_settings(name)
            path = video_config.get_product_paths("B0TEST001", name)["subtitles"]

            assert path.suffix == f".{merged.subtitle_settings.subtitle_format}", (
                f"{name} resolves to "
                f"{merged.subtitle_settings.subtitle_format} but writes to "
                f"{path.name}"
            )

    def test_no_profile_still_uses_the_global_value(self, video_config):
        """Callers with no profile in hand must keep working."""
        assert video_config._get_subtitle_filename("subtitles.srt") == "subtitles.ass"

    def test_an_unknown_profile_falls_back_rather_than_raising(self, video_config):
        """Reporting a bad profile name is not this function's job."""
        assert (
            video_config._get_subtitle_filename("subtitles.srt", "no_such_profile")
            == "subtitles.ass"
        )


class TestTheCliOverrideReachesThePath:
    """`--subtitle-format` resolves the same way for the path and the writer.

    This half is older than the profile key and reachable on a stock install:
    every runtime consumer merges *with* `ctx.cli_overrides` while the path did
    not, so `--subtitle-format srt` against the bundled `ass` global wrote SRT
    text into `subtitles.ass` and FFmpeg's `ass` filter aborted the render.
    Threading the profile in neither caused nor closed it. The producer's
    `--subtitle-format` is the only way to reach it -- `global_batch` has no
    such flag.
    """

    OVERRIDE = {"subtitle_settings.subtitle_format": "ass"}

    def test_the_flag_moves_the_path(self, video_config, monkeypatch):
        """The profile says `srt`; the flag says `ass` and must win."""
        profile = profile_with(
            video_config,
            "slideshow_images1",
            subtitle_settings={"subtitle_format": "srt"},
        )
        monkeypatch.setitem(video_config.video_profiles, "srt_profile", profile)

        path = video_config.get_product_paths(
            "B0TEST001", "srt_profile", self.OVERRIDE
        )["subtitles"]

        assert path.suffix == ".ass", (
            "the path ignores `--subtitle-format ass` while the generator "
            "honours it, so the step finds nothing at the recorded path and "
            "ships a caption-less video"
        )

    def test_it_agrees_with_what_the_generator_is_told(self, video_config, monkeypatch):
        """The invariant, against the same overrides the step merges with."""
        profile = profile_with(
            video_config,
            "slideshow_images1",
            subtitle_settings={"subtitle_format": "srt"},
        )
        monkeypatch.setitem(video_config.video_profiles, "srt_profile", profile)

        for override in ({"subtitle_settings.subtitle_format": "srt"}, self.OVERRIDE):
            merged = video_config.get_profile_merged_settings("srt_profile", override)
            path = video_config.get_product_paths("B0TEST001", "srt_profile", override)[
                "subtitles"
            ]

            assert path.suffix == f".{merged.subtitle_settings.subtitle_format}"

    def test_the_run_paths_carry_it(self, video_config, monkeypatch):
        """`get_video_run_paths` is what the orchestrator actually calls."""
        from src.video.producer.state import get_video_run_paths

        profile = profile_with(
            video_config,
            "slideshow_images1",
            subtitle_settings={"subtitle_format": "srt"},
        )
        monkeypatch.setitem(video_config.video_profiles, "srt_profile", profile)

        paths = get_video_run_paths(
            video_config, "B0TEST001", "srt_profile", self.OVERRIDE
        )

        assert paths["subtitle_file"].suffix == ".ass"

    def test_no_overrides_still_follows_the_profile(self, video_config, monkeypatch):
        """The plumbing must not disturb the case with no flag passed."""
        profile = profile_with(
            video_config,
            "slideshow_images1",
            subtitle_settings={"subtitle_format": "srt"},
        )
        monkeypatch.setitem(video_config.video_profiles, "srt_profile", profile)

        assert (
            video_config.get_product_paths("B0TEST001", "srt_profile")[
                "subtitles"
            ].suffix
            == ".srt"
        )


class TestTheProductionCallSitePassesThem:
    """The one wiring that makes the CLI half real, read from the source.

    `orchestration.py` is the only place `ctx.run_paths` is built, and dropping
    the fourth argument there reinstates both failure modes while the whole
    suite stays green -- the tests above call `get_video_run_paths` directly,
    so they cover the helper and not the call. Asserted the way the publisher's
    `create_publisher` kwargs are: by walking the AST of the call site, because
    the defect is a call that passes nothing, which no shared helper can guard.
    """

    def call(self):
        import ast
        from pathlib import Path

        source = Path("src/video/producer/orchestration.py").read_text(encoding="utf-8")
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_video_run_paths"
        ]
        assert len(calls) == 1, f"expected one call site, found {len(calls)}"
        return calls[0]

    def test_it_passes_the_cli_overrides(self):
        names = [a.id for a in self.call().args if isinstance(a, ast.Name)] + [
            kw.arg for kw in self.call().keywords
        ]

        assert "cli_overrides" in names, (
            "the run paths are built without the CLI overrides, so "
            "`--subtitle-format` moves the generator and not the path"
        )

    def test_it_passes_the_profile_too(self):
        names = [a.id for a in self.call().args if isinstance(a, ast.Name)] + [
            kw.arg for kw in self.call().keywords
        ]

        assert "profile_name" in names
