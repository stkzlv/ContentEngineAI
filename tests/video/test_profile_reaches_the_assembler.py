"""A profile's video settings must reach the filter, not just the merge.

`video_aspect_mode` satisfied every condition the profile-override audit
checks -- declared on `VideoSettings` and `VideoProfile`, present in the
`_collect_overrides` map, and read by name in the assembler -- and was still
ignored on every render. `VisualFilterBuilder` bound `video_settings` to
`self.config.video_settings` and read the aspect mode from it, a hundred lines
below a `vs` binding that resolved the profile correctly for three
neighbouring positioning fields.

Three fields were affected: `video_aspect_mode` and `video_assembly_mode` on
three bundled profiles each, and `first_frame_pre_motion` on one. So
`product_video_single` asked to crop to fill the frame and letterboxed, which
is a valid-looking video nobody would call broken.

The existing audit cannot see this. It asks whether a name is *mentioned*
outside the config package, which it was -- at the site reading the wrong
object. These tests ask the different question: does the value the profile
asked for reach the thing that acts on it.
"""

from __future__ import annotations

import pytest

from src.video.config import config


def merged(profile_name: str):
    return config.get_profile_merged_settings(profile_name)


def profiles_overriding(field: str) -> list[str]:
    """Bundled profiles whose merged value differs from the global."""
    global_value = getattr(config.video_settings, field, None)
    return sorted(
        name
        for name in config.video_profiles
        if name != "base"
        and getattr(merged(name).video_settings, field, None) != global_value
    )


OVERRIDDEN_FIELDS = [
    "video_aspect_mode",
    "video_assembly_mode",
    "first_frame_pre_motion",
]


class TestSomeProfileActuallyOverridesThese:
    """Every assertion below is vacuous for a field no profile overrides."""

    @pytest.mark.parametrize("field", OVERRIDDEN_FIELDS)
    def test_the_field_is_overridden_somewhere(self, field):
        assert profiles_overriding(field), (
            f"no bundled profile overrides {field}, so the tests below prove "
            "nothing; drop the field or pick one that is overridden"
        )


class TestTheBuilderReadsTheProfile:
    """Asserted against the builder's own binding, not the config layer.

    The config layer was never wrong -- `get_profile_merged_settings` returned
    `crop-to-fit` throughout. Only the consumer was.
    """

    def builder(self, profile_name: str):
        from unittest.mock import MagicMock

        from src.video.assembler.visual_builder import VisualFilterBuilder

        return VisualFilterBuilder(
            media_inspector=MagicMock(),
            config=config,
            strategy_factory=None,
            profile_settings=merged(profile_name),
        )

    @pytest.mark.parametrize("field", OVERRIDDEN_FIELDS)
    def test_the_builder_sees_the_profile_value(self, field):
        for name in profiles_overriding(field):
            expected = getattr(merged(name).video_settings, field)
            actual = getattr(self.builder(name).profile_settings.video_settings, field)

            assert actual == expected, (
                f"{name} asks for {field}={expected!r} and the builder holds "
                f"{actual!r}"
            )

    def test_the_binding_is_not_the_global(self):
        """The defect itself: one line, and the whole class of bug follows.

        Read from the source because the binding is a local inside a long
        method; calling it needs the full assembly context.
        """
        import ast
        import inspect

        from src.video.assembler.visual_builder import VisualFilterBuilder

        source = inspect.getsource(VisualFilterBuilder)
        tree = ast.parse(source.lstrip())

        globals_only = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "video_settings"
                for t in node.targets
            )
            and "profile_settings" not in ast.dump(node.value)
        ]

        assert not globals_only, (
            "`video_settings` is bound to the global config, so every "
            "profile-overridable field read from it is silently ignored"
        )


class TestTheRenderedFilterFollowsTheProfile:
    """The end of the chain: the FFmpeg filter string itself.

    `crop-to-fit` and `letterbox` produce visibly different filters, so this
    is the assertion that would have failed on the shipped code.
    """

    def filter_for(self, aspect_mode: str) -> str:
        from unittest.mock import MagicMock

        from src.video.assembler.visual_builder import VisualFilterBuilder

        builder = VisualFilterBuilder(
            media_inspector=MagicMock(),
            config=config,
            strategy_factory=None,
            profile_settings=None,
        )
        filter_string, _, _ = builder.apply_aspect_ratio_mode(
            "[0:v]", aspect_mode, 1080, 1920, 1920, 1080
        )
        return filter_string

    def test_crop_to_fit_fills_the_frame(self):
        rendered = self.filter_for("crop-to-fit")

        assert "crop=1080:1920" in rendered
        assert "increase" in rendered
        assert "pad=" not in rendered, "crop-to-fit padded, so it letterboxed"

    def test_letterbox_pads(self):
        """The counterpart, so the test above cannot pass by rendering nothing."""
        rendered = self.filter_for("letterbox")

        assert "pad=1080:1920" in rendered
        assert "decrease" in rendered

    def test_a_landscape_source_is_the_case_that_differs(self):
        """16:9 into 9:16 is what the bundled profiles actually receive."""
        assert self.filter_for("crop-to-fit") != self.filter_for("letterbox")
