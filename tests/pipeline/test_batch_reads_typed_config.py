"""The batch reads its publisher settings from one typed object.

`global_batch.py` used to open `config/publisher.yaml` in four places with a
relative path and pull raw keys out of it. Two things followed, both silent.

A run from anywhere but the repository root took the `if config_path.exists()`
false branch and got hardcoded fallbacks, and one of those fallbacks was wrong:
`immediate_publish` defaulted to `True` here against a shipped `false`, so a
batch launched from another directory would have published immediately rather
than scheduling.

The second is the shape that already shipped a defect. Each section was
re-parsed at this call site, so a setting could reach `single` and not the
batch: `tiktok_settings` was absent from the batch's parse for several
releases, and the privacy level, comment permissions and AI label silently
differed between the two paths. Nothing logged the difference.

These tests therefore assert the property rather than the parse: no inline
read remains, and a value set in YAML arrives at the publish phase through the
typed config. Asserting that a shared loader exists would not have caught the
original defect, because the defect was a call site that called no loader.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "src" / "pipeline" / "global_batch.py"


@pytest.mark.unit
class TestNoInlineConfigReadRemains:
    def test_the_batch_does_not_parse_yaml_itself(self) -> None:
        source = BATCH.read_text()

        assert "yaml.safe_load" not in source, (
            "the batch parses a config file itself again; load it through the "
            "typed loader so the path is absolute and the defaults are shared"
        )

    def test_the_batch_names_no_relative_config_path(self) -> None:
        source = BATCH.read_text()

        for literal in ('"config/publisher.yaml"', '"config/pipeline.yaml"'):
            assert literal not in source, (
                f"{literal} is relative to the working directory, so a run "
                "from elsewhere silently takes the fallbacks"
            )


@pytest.mark.unit
class TestTheLoadersResolveFromAnyDirectory:
    """The defect one level down: both loaders defaulted to a relative path."""

    def test_the_publisher_default_is_absolute(self) -> None:
        from src.publisher.config import DEFAULT_PUBLISHER_CONFIG_PATH

        assert DEFAULT_PUBLISHER_CONFIG_PATH.is_absolute()
        assert DEFAULT_PUBLISHER_CONFIG_PATH.exists()

    def test_the_pipeline_default_is_absolute(self) -> None:
        from src.pipeline.config import DEFAULT_PIPELINE_CONFIG_PATH

        assert DEFAULT_PIPELINE_CONFIG_PATH.is_absolute()
        assert DEFAULT_PIPELINE_CONFIG_PATH.exists()

    def test_the_shipped_values_load_from_another_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case the relative default got wrong."""
        from src.publisher.config import load_publisher_config

        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)
        monkeypatch.chdir(tmp_path)

        config = load_publisher_config()

        assert config.default_platforms, "fell back to the empty default"
        # The shipped file says false; the batch's old inline default said true.
        assert config.immediate_publish is False


@pytest.mark.unit
class TestDefaultPlatformsFlowFromYaml:
    """YAML -> typed config -> the value the publish phase would target."""

    @staticmethod
    def _write(tmp_path: Path, body: str) -> Path:
        path = tmp_path / "publisher.yaml"
        path.write_text("late:\n  api_key_env_var: LATE_API_KEY\n" + body)
        return path

    def test_a_changed_value_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.publisher.config import load_publisher_config

        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)
        path = self._write(tmp_path, "default_platforms:\n  - youtube\n")

        config = load_publisher_config(path)

        assert [p.value for p in config.default_platforms] == ["youtube"]

    def test_yaml_platforms_arrive_as_enums(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The annotation said Platform while YAML delivered strings.

        Nothing converted them, so `to_dict` and every caller reading `.value`
        raised on any config loaded from the shipped file, and mypy could not
        see it because it trusts the annotation.
        """
        from src.publisher.config import load_publisher_config
        from src.publisher.models import Platform

        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)
        path = self._write(
            tmp_path,
            "default_platforms:\n  - youtube\nprivacy_settings:\n  youtube: public\n",
        )

        config = load_publisher_config(path)

        assert config.default_platforms == [Platform.YOUTUBE]
        assert set(config.privacy_settings) == {Platform.YOUTUBE}
        assert config.to_dict()["default_platforms"] == ["youtube"]

    def test_an_unknown_platform_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dropping it would publish to fewer platforms than configured."""
        from src.publisher.config import load_publisher_config

        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)
        path = self._write(tmp_path, "default_platforms:\n  - facebok\n")

        with pytest.raises(ValueError, match="facebok"):
            load_publisher_config(path)

    def test_an_absent_key_falls_back_to_the_documented_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.publisher.config import load_publisher_config
        from src.publisher.models import PublisherConfig

        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)
        path = self._write(tmp_path, "")

        config = load_publisher_config(path)

        assert (
            config.default_platforms
            == PublisherConfig(provider="late", api_key="x").default_platforms
        )


@pytest.mark.unit
class TestThePublishPhaseUsesTheTypedObject:
    """A shared loader existing is not the guard; the call site is."""

    @staticmethod
    def _function(name: str) -> ast.AST:
        tree = ast.parse(BATCH.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
                and node.name == name
            ):
                return node
        raise AssertionError(f"{name} not found in {BATCH.name}")

    @pytest.mark.parametrize(
        "function",
        [
            "_execute_publishing_phase",
            "display_execution_plan",
            "_default_platforms",
            "_publisher_profiles",
        ],
    )
    def test_it_reads_the_typed_settings(self, function: str) -> None:
        node = self._function(function)

        called = {
            n.func.id
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "_publisher_settings" in called, (
            f"{function} does not read the typed publisher config, so it can "
            "drift from the CLI path the way tiktok_settings did"
        )

    def test_the_phase_no_longer_re_parses_the_sections(self) -> None:
        """Each of these was its own parse, and one was simply missing."""
        source = ast.unparse(self._function("_execute_publishing_phase"))

        for section in (
            "first_comment",
            "tiktok_settings",
            "blob_retention",
            "delivery_sweep",
            "link_in_bio",
            "affiliate_disclosure",
        ):
            assert f"'{section}'" not in source, (
                f"{section} is parsed here again instead of read off the "
                "typed config, which is how a section goes missing"
            )


@pytest.mark.unit
class TestTheWebhookComesFromTheLoadedConfig:
    def test_main_does_not_reopen_the_pipeline_file(self) -> None:
        source = BATCH.read_text()

        assert (
            "webhook_yaml" in source
        ), "the webhook slice is not carried on the config"
        assert 'open("config/pipeline.yaml")' not in source

    def test_the_loader_carries_the_slice(self) -> None:
        import argparse

        from src.pipeline.config import GlobalBatchConfig

        assert "webhook_yaml" in GlobalBatchConfig.__dataclass_fields__
        assert isinstance(argparse.Namespace(), argparse.Namespace)
