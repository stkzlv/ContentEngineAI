"""A profile override that reaches nothing is invisible.

The audit that produced this file was asked for after a field existed on the
global `VideoSettings` model, was missing from `VideoProfile` and from the
`_collect_overrides` map, and so had its YAML value silently swallowed.

Two of the three original conditions are now enforced elsewhere:
`VideoProfile` sets `extra="forbid"`, so a profile naming a field the model
does not declare fails at config load rather than being dropped. What stays
silent is the map, and one thing the original framing missed entirely -- a
field can satisfy all three conditions and *still* go nowhere, if no consumer
reads the target.

That is what `video_transition_duration` was: declared on both models, present
in the map, set by four bundled profiles, and read by nothing. The assembler
reads `transition_duration_sec`. Both default to 0.5, so the values agreeing
is what hid it; changing a profile's crossfade did nothing at all.

These tests derive the answer from the models rather than listing fields, so
the audit does not go stale the next time one is added.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.video.config.visual_models import VideoProfile, VideoSettings

CORE_MODELS = Path("src/video/config/core_models.py")


def override_map() -> dict[str, str]:
    """The profile-field -> target-field map, read from the source.

    From the AST rather than by calling `get_profile_merged_settings`, because
    the map is a literal inside it and the point is to see the mapping itself.
    """
    tree = ast.parse(CORE_MODELS.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_collect_overrides"
        ):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Dict):
                for key, value in zip(arg.keys, arg.values, strict=False):
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(value, ast.Constant)
                        and isinstance(key.value, str)
                        and isinstance(value.value, str)
                    ):
                        mapping[key.value] = value.value
    return mapping


def test_the_map_is_found_at_all():
    """Every assertion below is vacuous if the AST walk finds nothing."""
    assert len(override_map()) > 5


class TestNoDeclaredOverrideIsDropped:
    def test_every_field_on_both_models_is_mapped(self):
        """The original silent-drop shape.

        A field declared on `VideoProfile` and on `VideoSettings` but absent
        from the map loads from YAML without complaint and never reaches the
        assembler.
        """
        shared = set(VideoSettings.model_fields) & set(VideoProfile.model_fields)
        unmapped = sorted(shared - set(override_map()))

        assert not unmapped, (
            f"{unmapped} are declared on both models but not in the override "
            "map, so a profile setting them is silently ignored"
        )


class TestEveryMappedTargetIsReal:
    @pytest.mark.parametrize("profile_field,target", sorted(override_map().items()))
    def test_the_source_field_exists_on_the_profile(self, profile_field, target):
        """A map key that is not a profile field can never fire.

        `_collect_overrides` reads it with `getattr(profile, key, None)`, so a
        typo is not an error -- it is an override that never happens.
        """
        assert profile_field in VideoProfile.model_fields

    @pytest.mark.parametrize("profile_field,target", sorted(override_map().items()))
    def test_the_target_field_exists_on_video_settings(self, profile_field, target):
        assert target in VideoSettings.model_fields


class TestEveryMappedTargetHasAReader:
    """The condition the original audit did not name.

    All three of its conditions can hold and the override still go nowhere,
    because nothing consumes the field it lands on.
    """

    SOURCE_DIRS = (Path("src/video"), Path("src/pipeline"))

    def _readers(self, field: str) -> int:
        count = 0
        for directory in self.SOURCE_DIRS:
            for path in directory.rglob("*.py"):
                if "config" in path.parts and path.parent.name == "config":
                    continue  # the models and the merge, not consumers
                if field in path.read_text(encoding="utf-8"):
                    count += 1
        return count

    @pytest.mark.parametrize("target", sorted(set(override_map().values())))
    def test_something_outside_the_config_package_reads_it(self, target):
        assert self._readers(target) > 0, (
            f"nothing outside the config package reads `{target}`, so every "
            "profile that overrides it is configuring a value with no effect"
        )


class TestTheTransitionOverrideLands:
    """The finding this audit produced, pinned.

    `video_transition_duration` is the profile spelling; the assembler reads
    `transition_duration_sec`. Mapping the first onto itself meant four
    bundled profiles set a crossfade duration that went nowhere.
    """

    def test_the_profile_field_maps_onto_the_field_that_is_read(self):
        assert override_map()["video_transition_duration"] == (
            "transition_duration_sec"
        )

    def test_a_profile_value_reaches_the_merged_settings(self):
        from src.video.config import config

        overriding = [
            name
            for name, profile in config.video_profiles.items()
            if getattr(profile, "video_transition_duration", None) is not None
        ]
        assert overriding, "no bundled profile overrides the transition"

        for name in overriding:
            profile = config.video_profiles[name]
            merged = config.get_profile_merged_settings(name)
            assert (
                merged.video_settings.transition_duration_sec
                == profile.video_transition_duration
            ), f"{name}'s transition override does not reach the assembler"
