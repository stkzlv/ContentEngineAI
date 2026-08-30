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

from pathlib import Path

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

        def bound_from(predicate) -> set[str]:
            """Local names assigned from a value matching `predicate`."""
            names = set()
            for node in ast.walk(method_for_aliases):
                if isinstance(node, ast.Assign) and predicate(node.value):
                    names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
                elif (
                    isinstance(node, ast.AnnAssign)
                    and node.value is not None
                    and predicate(node.value)
                    and isinstance(node.target, ast.Name)
                ):
                    names.add(node.target.id)
            return names

        def is_self_config(node) -> bool:
            return (
                isinstance(node, ast.Attribute)
                and node.attr == "config"
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            )

        # Two hops, not one. `cfg = self.config` then `cfg.video_settings.x`
        # evades a check that only follows names bound to the settings object
        # itself, and that shape passed the first version of this guard.
        config_aliases = bound_from(is_self_config)

        def is_global_object(node) -> bool:
            """`self.config.video_settings`, or any alias reaching it."""
            return is_global_chain(node) or (
                isinstance(node, ast.Attribute)
                and node.attr == "video_settings"
                and isinstance(node.value, ast.Name)
                and node.value.id in config_aliases
            )

        settings_aliases = bound_from(is_global_object)

        def is_global_settings(node) -> bool:
            return is_global_object(node) or (
                isinstance(node, ast.Name) and node.id in settings_aliases
            )

        method = self.method_node()

        # The profile side is aliased too -- the real comparison reads
        # `vs_model.x`, where `vs_model = self.profile_settings.video_settings`.
        profile_aliases = bound_from(
            lambda value: "profile_settings" in ast.dump(value)
        )

        def is_profile_object(node) -> bool:
            return "profile_settings" in ast.dump(node) or (
                isinstance(node, ast.Name) and node.id in profile_aliases
            )

        def reads_profile(node, field: str) -> bool:
            """Any read of `field` off the profile, directly or via an alias."""
            return any(
                isinstance(inner, ast.Attribute)
                and inner.attr == field
                and is_profile_object(inner.value)
                for inner in ast.walk(node)
            )

        def exempt_comparisons(method_node) -> set[int]:
            """Global reads paired against the same field off the profile.

            The `image_positioning_overridden` branch legitimately reads both
            sides to ask whether the profile differs. Exempting every operand
            of every comparison is far too broad -- it lets a global read hide
            inside any `if a != b`, which is a defect the first version of
            this exemption swallowed.
            """
            exempt = set()
            for node in ast.walk(method_node):
                if not isinstance(node, ast.Compare):
                    continue
                operands = [node.left, *node.comparators]
                for operand in operands:
                    for inner in ast.walk(operand):
                        if not (
                            isinstance(inner, ast.Attribute)
                            and inner.attr in self.overridable_targets()
                        ):
                            continue
                        others = [o for o in operands if o is not operand]
                        if any(reads_profile(o, inner.attr) for o in others):
                            exempt.add(id(inner))
            return exempt

        compared = exempt_comparisons(method)

        targets = self.overridable_targets()
        offenders = {
            node.attr
            for node in ast.walk(method)
            if isinstance(node, ast.Attribute)
            and node.attr in targets
            and is_global_settings(node.value)
            and id(node) not in compared
        }

        # `getattr(self.config.video_settings, "field")` is an attribute read
        # that no Attribute node describes, so it slipped through untouched.
        offenders |= {
            str(node.args[1].value)
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and is_global_settings(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in targets
        }

        assert not offenders, (
            f"{sorted(offenders)} are read from the global config inside "
            f"{self.METHOD}, so a profile overriding them is ignored"
        )


class TestBothTransitionConsumersAgree:
    """The caption boundaries and the crossfade offsets share one value.

    `visual_builder` lays out `xfade` offsets from `transition_duration_sec`
    and `subtitle_builder._calculate_segment_times` computes the boundaries
    the content-aware captions are placed against from the same field. When
    the assembler moved to the profile-merged value and the subtitle builder
    did not, the two drifted by `i x dT` at segment `i` -- and every
    content-aware caption is placed against those boundaries, the main
    narration line included, not only the upper URL line.

    Asserted as the invariant rather than as two separate reads, because the
    defect is precisely that the two disagree. Reverting the subtitle-builder
    hunk leaves the rest of the suite green.
    """

    TRANSITION = 1.25  # deliberately unlike the 0.5 global

    def profile_with_transition(self):
        """A merged settings object whose transition differs from the global."""
        base = merged("product_video_single")
        video = base.video_settings.model_copy(
            update={"transition_duration_sec": self.TRANSITION}
        )
        return base.model_copy(update={"video_settings": video})

    def subtitle_builder(self, profile_settings):
        from src.video.assembler.subtitle_builder import SubtitleGraphBuilder

        return SubtitleGraphBuilder(
            config=config,
            profile_settings=profile_settings,
            product_id="B0TEST001",
        )

    def test_the_global_is_not_the_profile_value(self):
        """Vacuous if the two happen to coincide."""
        assert config.video_settings.transition_duration_sec != self.TRANSITION

    def test_segment_times_use_the_profile_transition(self):
        settings = self.profile_with_transition()
        builder = self.subtitle_builder(settings)
        timed = [(Path("a.mp4"), 10.0, True), (Path("b.mp4"), 10.0, True)]

        boundaries = builder._calculate_segment_times(timed)

        # Segment 0 is unaffected; segment 1 loses one transition.
        assert boundaries[0] == pytest.approx(10.0)
        assert boundaries[1] == pytest.approx(20.0 - self.TRANSITION), (
            "the caption boundaries use the global transition while the "
            "assembler lays out crossfades from the profile value, so every "
            "content-aware reposition after the first visual drifts"
        )

    def test_it_falls_back_to_the_global_without_a_profile(self):
        """The fallback must not become the only path that works."""
        builder = self.subtitle_builder(None)
        timed = [(Path("a.mp4"), 10.0, True), (Path("b.mp4"), 10.0, True)]

        boundaries = builder._calculate_segment_times(timed)

        assert boundaries[1] == pytest.approx(
            20.0 - config.video_settings.transition_duration_sec
        )

    def test_both_builders_resolve_the_same_value(self):
        """The invariant itself, read from each consumer's own resolution."""
        from unittest.mock import MagicMock

        from src.video.assembler.visual_builder import VisualFilterBuilder

        settings = self.profile_with_transition()
        assembler = VisualFilterBuilder(
            media_inspector=MagicMock(),
            config=config,
            strategy_factory=None,
            profile_settings=settings,
        )
        captions = self.subtitle_builder(settings)

        assert assembler.profile_settings is not None
        assembler_value = (
            assembler.profile_settings.video_settings.transition_duration_sec
        )
        timed = [(Path("a.mp4"), 10.0, True), (Path("b.mp4"), 10.0, True)]
        caption_value = 20.0 - captions._calculate_segment_times(timed)[1]

        assert caption_value == pytest.approx(assembler_value)
