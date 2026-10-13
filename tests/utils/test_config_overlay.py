"""Private config overlays: `config/<name>.private.yaml` over the shipped file."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.utils import config_overlay
from src.utils.config_overlay import (
    ConfigOverlayError,
    deep_merge,
    read_config_yaml,
)
from src.utils.outputs_paths import get_project_root

ROOT = get_project_root()


@pytest.fixture(autouse=True)
def _overlays_on(monkeypatch):
    """The suite turns overlays off; these tests are about them."""
    monkeypatch.delenv("CONFIG_OVERLAYS", raising=False)
    monkeypatch.setattr(config_overlay, "_logged", set())
    monkeypatch.setattr(config_overlay, "_pending", [])


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A copy of the shipped config, so an overlay never lands in the tree."""
    target = tmp_path / "config"
    shutil.copytree(
        ROOT / "config", target, ignore=shutil.ignore_patterns("*.private.yaml")
    )
    return target


def _overlay(config_dir: Path, name: str, data: object) -> None:
    (config_dir / f"{name}.private.yaml").write_text(
        yaml.safe_dump(data), encoding="utf-8"
    )


class TestMerge:
    @pytest.mark.req("REQ-OPS-116")
    def test_mappings_merge_and_everything_else_replaces(self) -> None:
        base = {"a": {"b": 1, "c": [1, 2], "d": "x"}, "e": 2}
        over = {"a": {"c": [3], "d": None, "f": True}, "g": 4}
        assert deep_merge(base, over) == {
            "a": {"b": 1, "c": [3], "d": None, "f": True},
            "e": 2,
            "g": 4,
        }
        assert base == {"a": {"b": 1, "c": [1, 2], "d": "x"}, "e": 2}

    @pytest.mark.req("REQ-OPS-116")
    def test_a_file_without_overlay_reads_as_shipped(self, tmp_path: Path) -> None:
        path = tmp_path / "x.yaml"
        path.write_text("a: 1\n", encoding="utf-8")
        assert read_config_yaml(path) == {"a": 1}

    @pytest.mark.req("REQ-OPS-116")
    def test_an_overlay_file_is_not_overlaid_again(self, tmp_path: Path) -> None:
        path = tmp_path / "x.private.yaml"
        path.write_text("a: 1\n", encoding="utf-8")
        (tmp_path / "x.private.private.yaml").write_text("a: 2\n", encoding="utf-8")
        assert read_config_yaml(path) == {"a": 1}


class TestEveryLoaderReadsTheOverlay:
    """Each loader's consumer sees the overlaid value."""

    @pytest.mark.req("REQ-OPS-116")
    def test_video_config(self, config_dir: Path) -> None:
        from src.video.config import VideoConfig
        from src.video.config_adapter import ModularConfigAdapter

        def ending() -> str:
            merged = ModularConfigAdapter(str(config_dir)).get_merged_config_dict()
            return str(VideoConfig(**merged).video_settings.ending)

        shipped = ending()
        other = "loop" if shipped != "loop" else "outro"
        _overlay(config_dir, "video_production", {"video_settings": {"ending": other}})
        assert ending() == other

    @pytest.mark.req("REQ-OPS-116")
    def test_publisher_config(self, config_dir: Path, monkeypatch) -> None:
        from src.publisher.config import load_publisher_config

        monkeypatch.setenv("LATE_API_KEY", "sk_test_overlay")
        path = config_dir / "publisher.yaml"
        assert load_publisher_config(config_path=path).covers_config.enabled is False
        _overlay(config_dir, "publisher", {"covers": {"enabled": True}})
        assert load_publisher_config(config_path=path).covers_config.enabled is True

    @pytest.mark.req("REQ-OPS-116")
    def test_batch_config(self, config_dir: Path) -> None:
        from src.pipeline.config import load_global_batch_config

        path = config_dir / "pipeline.yaml"
        shipped = load_global_batch_config(argparse.Namespace(), path).max_products
        _overlay(
            config_dir, "pipeline", {"global_batch": {"max_products": shipped + 7}}
        )
        overlaid = load_global_batch_config(argparse.Namespace(), path)
        assert overlaid.max_products == shipped + 7

    @pytest.mark.req("REQ-OPS-116")
    def test_scraper_config(self, config_dir: Path) -> None:
        from src.scraper.config_adapter import ScraperConfigAdapter

        def debug_mode() -> bool:
            settings = ScraperConfigAdapter(str(config_dir)).get_settings()
            return bool(settings.global_settings.debug_mode)

        shipped = debug_mode()
        _overlay(
            config_dir, "scraper", {"global_settings": {"debug_mode": not shipped}}
        )
        assert debug_mode() is (not shipped)

    @pytest.mark.req("REQ-OPS-116")
    def test_research_config(self, config_dir: Path) -> None:
        from src.research.config import load_research_config

        path = config_dir / "research.yaml"
        _overlay(config_dir, "research", {"request_pause_sec": 0.5})
        assert load_research_config(path).request_pause_sec == 0.5

    @pytest.mark.req("REQ-OPS-116")
    def test_url_shortener_config(self, config_dir: Path) -> None:
        from src.utils.url_shortener.config import load_url_shortener_settings

        path = config_dir / "url_shortener.yaml"
        _overlay(config_dir, "url_shortener", {"url_shortener": {"provider": "picsee"}})
        assert load_url_shortener_settings(path).provider == "picsee"

    @pytest.mark.req("REQ-OPS-116")
    def test_no_config_reader_bypasses_the_overlay(self) -> None:
        """Only the helper parses config files, so no reader can miss one.

        The other parsers read a dumped mapping or a topics file named on the
        command line, never a file under `config/`.
        """
        allowed = {
            "src/utils/config_overlay.py": 2,
            "src/video/config_adapter.py": 1,
            "src/video/producer/topic_input.py": 1,
        }
        parse = re.compile(
            r"\byaml\.(?:safe_|full_|unsafe_)?load\(|(?<![.\w])(?:safe_|full_|unsafe_)load\("
        )
        found: dict[str, int] = {}
        for folder in ("src", "tools", "scripts"):
            for path in sorted((ROOT / folder).rglob("*.py")):
                text = path.read_text(encoding="utf-8")
                count = len(parse.findall(text))
                if count:
                    found[path.relative_to(ROOT).as_posix()] = count
        assert found == allowed


class TestABrokenOverlayStopsTheLoad:
    @pytest.mark.req("REQ-OPS-117")
    def test_an_unknown_key_fails_validation(self, config_dir: Path) -> None:
        from pydantic import ValidationError

        from src.video.config import VideoConfig
        from src.video.config_adapter import ModularConfigAdapter

        _overlay(
            config_dir, "video_production", {"video_settings": {"endingg": "loop"}}
        )
        merged = ModularConfigAdapter(str(config_dir)).get_merged_config_dict()
        with pytest.raises(ValidationError, match="endingg"):
            VideoConfig(**merged)

    @pytest.mark.req("REQ-OPS-117")
    @pytest.mark.parametrize("text", ["a: [1\n", "- a\n- b\n", "just text\n"])
    def test_an_unparseable_or_non_mapping_overlay_is_named(
        self, tmp_path: Path, text: str
    ) -> None:
        path = tmp_path / "x.yaml"
        path.write_text("a: 1\n", encoding="utf-8")
        (tmp_path / "x.private.yaml").write_text(text, encoding="utf-8")
        with pytest.raises(ConfigOverlayError, match=r"x\.private\.yaml"):
            read_config_yaml(path)

    @pytest.mark.req("REQ-OPS-117")
    def test_the_publisher_reports_it_as_unreadable_config(
        self, config_dir: Path, monkeypatch
    ) -> None:
        from src.publisher.config import ConfigUnreadableError, load_publisher_config

        monkeypatch.setenv("LATE_API_KEY", "sk_test_overlay")
        (config_dir / "publisher.private.yaml").write_text("- a\n", encoding="utf-8")
        with pytest.raises(ConfigUnreadableError, match="publisher.private.yaml"):
            load_publisher_config(config_path=config_dir / "publisher.yaml")

    @pytest.mark.req("REQ-OPS-117")
    def test_an_empty_overlay_changes_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "x.yaml"
        path.write_text("a: 1\n", encoding="utf-8")
        (tmp_path / "x.private.yaml").write_text("", encoding="utf-8")
        assert read_config_yaml(path) == {"a": 1}


@pytest.mark.req("REQ-OPS-118")
def test_each_overlay_is_logged_once(tmp_path: Path, caplog) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("a: 1\n", encoding="utf-8")
    (tmp_path / "x.private.yaml").write_text("a: 2\n", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="src.utils.config_overlay"):
        assert read_config_yaml(path) == {"a": 2}
        assert read_config_yaml(path) == {"a": 2}
    lines = [r.getMessage() for r in caplog.records]
    assert lines == [f"Config overlay applied: {tmp_path / 'x.private.yaml'}"]


@pytest.mark.req("REQ-OPS-119")
def test_overlays_off_reads_the_shipped_file(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("a: 1\n", encoding="utf-8")
    (tmp_path / "x.private.yaml").write_text("a: 2\n", encoding="utf-8")
    assert read_config_yaml(path) == {"a": 2}
    monkeypatch.setenv("CONFIG_OVERLAYS", "off")
    assert read_config_yaml(path) == {"a": 1}


@pytest.mark.req("REQ-OPS-119")
def test_the_suite_runs_with_overlays_off(request) -> None:
    """The root conftest turns them off before any test module imports config.

    Module-level reads (the video config singleton, the circuit breakers)
    cache what they read at collection, before any fixture runs.
    """
    assert "_no_config_overlays" in request.fixturenames
    text = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    switch = text.index('os.environ["CONFIG_OVERLAYS"] = "off"')
    assert switch < text.index("\nfrom src")


@pytest.mark.req("REQ-OPS-117")
def test_a_broken_shipped_file_is_named(tmp_path: Path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("a: [1\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError, match=r"x\.yaml"):
        read_config_yaml(path)


@pytest.mark.req("REQ-OPS-117")
def test_the_circuit_breakers_stop_on_a_broken_overlay(
    tmp_path: Path, monkeypatch
) -> None:
    from src.utils import circuit_breaker, outputs_paths

    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "performance.yaml").write_text(
        "circuit_breaker:\n  llm:\n    failure_threshold: 2\n", encoding="utf-8"
    )
    monkeypatch.setattr(outputs_paths, "get_project_root", lambda: tmp_path)
    assert circuit_breaker._load_circuit_breaker_config() == {
        "llm": {"failure_threshold": 2}
    }
    (tmp_path / "config" / "performance.private.yaml").write_text(
        "- a\n", encoding="utf-8"
    )
    with pytest.raises(ConfigOverlayError, match="performance.private.yaml"):
        circuit_breaker._load_circuit_breaker_config()


@pytest.mark.req("REQ-OPS-118")
def test_a_read_before_logging_is_set_up_is_logged_later(
    tmp_path: Path, caplog, monkeypatch
) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("a: 1\n", encoding="utf-8")
    (tmp_path / "x.private.yaml").write_text("a: 2\n", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="src.utils.config_overlay"):
        with monkeypatch.context() as early:
            early.setattr(config_overlay.logger, "hasHandlers", lambda: False)
            read_config_yaml(path)
        caplog.clear()
        read_config_yaml(path)
        read_config_yaml(path)
    assert [r.getMessage() for r in caplog.records] == [
        f"Config overlay applied: {tmp_path / 'x.private.yaml'}"
    ]


@pytest.mark.req("REQ-OPS-120")
def test_overlays_are_ignored_and_an_example_ships() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "config/publisher.private.yaml"],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0
    for name in ("publisher", "ai_services"):
        example = ROOT / "config" / f"{name}.private.yaml.example"
        data = yaml.safe_load(example.read_text(encoding="utf-8"))
        shipped = yaml.safe_load((ROOT / "config" / f"{name}.yaml").read_text())
        assert isinstance(data, dict) and data
        assert set(data) <= set(shipped)


@pytest.mark.req("REQ-OPS-118")
def test_an_import_time_read_is_logged_by_the_logging_setup(
    tmp_path: Path, monkeypatch
) -> None:
    """The only read of a file can come before any handler exists."""
    path = tmp_path / "x.yaml"
    path.write_text("a: 1\n", encoding="utf-8")
    (tmp_path / "x.private.yaml").write_text("a: 2\n", encoding="utf-8")
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", logging.INFO)
    read_config_yaml(path)
    seen: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    collect = _Collect()
    root.addHandler(collect)
    line = f"Config overlay applied: {tmp_path / 'x.private.yaml'}"
    config_overlay.log_pending()
    assert seen == [line]
    config_overlay.log_pending()
    read_config_yaml(path)
    assert seen == [line]


@pytest.mark.req("REQ-OPS-118")
def test_the_logging_setup_flushes_held_overlays() -> None:
    """setup_debug_logging calls log_pending once its handlers are attached.

    Read from the file: the suite replaces the function to keep logs off disk.
    """
    text = (ROOT / "src" / "utils" / "logging_setup.py").read_text(encoding="utf-8")
    start = text.index("\ndef setup_debug_logging(")
    end = text.find("\ndef ", start + 1)
    source = text[start : end if end != -1 else len(text)]
    assert source.index("log_pending()") > source.index(
        "root_logger.addHandler(file_handler)"
    )


@pytest.mark.req("REQ-OPS-117")
def test_the_single_file_video_loader_names_a_broken_overlay(tmp_path: Path) -> None:
    from src.video.config.core_models import load_video_config

    path = tmp_path / "video.yaml"
    path.write_text("a: 1\n", encoding="utf-8")
    (tmp_path / "video.private.yaml").write_text("- a\n", encoding="utf-8")
    with pytest.raises(ConfigOverlayError, match=r"video\.private\.yaml"):
        load_video_config(path)


@pytest.mark.req("REQ-OPS-120")
def test_the_example_overlays_load_as_written(config_dir: Path, monkeypatch) -> None:
    """Copied as they ship, both examples validate and take effect."""
    from src.publisher.config import load_publisher_config
    from src.publisher.first_comment import _CTA_MARKERS
    from src.video.config import VideoConfig
    from src.video.config_adapter import ModularConfigAdapter

    for name in ("publisher", "ai_services"):
        shutil.copy(
            ROOT / "config" / f"{name}.private.yaml.example",
            config_dir / f"{name}.private.yaml",
        )
    monkeypatch.setenv("LATE_API_KEY", "sk_test_overlay")
    publisher = load_publisher_config(config_path=config_dir / "publisher.yaml")
    slots = publisher.schedule_config.slots
    assert slots and {s.timezone for s in slots} == {"America/New_York"}
    merged = ModularConfigAdapter(str(config_dir)).get_merged_config_dict()
    templates = VideoConfig(**merged).llm_settings.script_templates
    assert templates.signature.enabled is True
    for line in templates.cta_options:
        assert line.lower().startswith(_CTA_MARKERS), line
