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


class TestTheAssemblerEmitsWhatTheProfileAsked:
    """Drives `build_visual_chain`, the method that held the defect.

    An earlier version of this file asserted two weaker things and claimed
    they would have caught the bug. Neither would: one constructed the builder
    with `profile_settings=None` and passed the aspect mode in as a literal,
    so no profile value flowed through it; the other asserted that `__init__`
    stored the object it was handed. Nine of ten tests here passed against the
    unfixed code.

    This one renders a real landscape input under a profile that asks to crop,
    and reads the emitted filter.
    """

    async def chain_for(self, profile_name: str, tmp_path):
        from unittest.mock import AsyncMock, MagicMock

        from src.video.assembler.visual_builder import VisualFilterBuilder

        source = tmp_path / "clip.mp4"
        source.write_bytes(b"")

        inspector = MagicMock()
        inspector.is_video.return_value = True
        # 1920x1080 into a 1080x1920 frame: the case the modes differ on.
        inspector.get_media_dimensions = AsyncMock(return_value=(1920, 1080))
        inspector.get_video_dimensions = AsyncMock(return_value=(1920, 1080))
        inspector.get_media_duration = AsyncMock(return_value=20.0)

        # The assembly strategy only decides which clips play for how long;
        # the aspect handling under test happens after it. Stubbed to one
        # full-length segment so the filter chain is the only variable.
        strategy = MagicMock()
        strategy.assemble = AsyncMock(return_value=([(source, 20.0, True)], "stubbed"))
        strategy_factory = MagicMock()
        strategy_factory.get_strategy.return_value = strategy

        settings = merged(profile_name)
        builder = VisualFilterBuilder(
            media_inspector=inspector,
            config=config,
            strategy_factory=strategy_factory,
            profile_settings=settings,
        )
        filter_parts, *_ = await builder.build_visual_chain(
            visual_inputs=[source],
            total_video_duration=20.0,
            is_relative_mode=False,
            video_settings_dict=settings.video_settings.model_dump(),
        )
        return "\n".join(filter_parts)

    @pytest.mark.asyncio
    async def test_a_cropping_profile_emits_a_crop(self, tmp_path):
        """`product_video_single` sets crop-to-fit and was letterboxing."""
        chain = await self.chain_for("product_video_single", tmp_path)

        assert "crop=" in chain, (
            "the profile asks to crop and the assembler padded, so its "
            "video_aspect_mode is being ignored"
        )
        assert "pad=" not in chain

    @pytest.mark.asyncio
    async def test_a_letterboxing_profile_still_pads(self, tmp_path):
        """The counterpart, so the test above cannot pass by emitting nothing."""
        chain = await self.chain_for("product_video_mixed", tmp_path)

        assert "pad=" in chain


class TestNoOverridableFieldIsReadFromTheGlobal:
    """Derived from the override map, scoped to the method that had the bug.

    The first version of this check asserted that no assignment to the *name*
    `video_settings` lacked `profile_settings` in its value. That pinned a
    spelling: renaming the local reintroduced the defect with the test green,
    an annotated assignment skipped the walk entirely, and a legitimate global
    binding in a different method failed it with a message that was false.

    This asks the question the map already answers -- which fields a profile
    can override -- and checks that none of them is reached through
    `self.config.video_settings` inside `build_visual_chain`.
    """

    METHOD = "build_visual_chain"

    def overridable_targets(self) -> set[str]:
        from tests.video.test_profile_override_coverage import override_map

        return set(override_map().values())

    def method_node(self):
        import ast
        import inspect
        import textwrap

        from src.video.assembler.visual_builder import VisualFilterBuilder

        source = textwrap.dedent(inspect.getsource(VisualFilterBuilder))
        tree = ast.parse(source)
        return next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
            and node.name == self.METHOD
        )

    def test_the_method_is_found(self):
        """Vacuous if the walk finds nothing."""
        assert self.method_node().body
        assert len(self.overridable_targets()) > 5

    def test_no_global_read_of_an_overridable_field(self):
        import ast

        def is_global_chain(node) -> bool:
            """The literal `self.config.video_settings` attribute chain."""
            return (
                isinstance(node, ast.Attribute)
                and node.attr == "video_settings"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "config"
            )

        method_for_aliases = self.method_node()
        # Locals bound to the global object. Following these is the whole
        # point: the first version of this check matched the attribute chain
        # only, so renaming the local reintroduced the defect with the test
        # green -- which is exactly how the bug shipped in the first place.
        aliases = {
            target.id
            for node in ast.walk(method_for_aliases)
            if isinstance(node, ast.Assign) and is_global_chain(node.value)
            for target in node.targets
            if isinstance(target, ast.Name)
        } | {
            node.target.id
            for node in ast.walk(method_for_aliases)
            if isinstance(node, ast.AnnAssign)
            and node.value is not None
            and is_global_chain(node.value)
            and isinstance(node.target, ast.Name)
        }

        def is_global_settings(node) -> bool:
            """The chain, or any local aliased to it."""
            return is_global_chain(node) or (
                isinstance(node, ast.Name) and node.id in aliases
            )

        method = self.method_node()

        # A global read that is an operand of a comparison is deliberate: the
        # `image_positioning_overridden` branch asks whether the profile
        # differs from the global, which requires reading both. A global read
        # used as a value is the defect.
        compared = {
            id(inner)
            for node in ast.walk(method)
            if isinstance(node, ast.Compare)
            for operand in [node.left, *node.comparators]
            for inner in ast.walk(operand)
        }

        targets = self.overridable_targets()
        offenders = {
            node.attr
            for node in ast.walk(method)
            if isinstance(node, ast.Attribute)
            and node.attr in targets
            and is_global_settings(node.value)
            and id(node) not in compared
        }

        assert not offenders, (
            f"{sorted(offenders)} are read from the global config inside "
            f"{self.METHOD}, so a profile overriding them is ignored"
        )
