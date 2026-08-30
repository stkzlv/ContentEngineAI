"""`src.video.config` must be importable from either end of the cycle.

The loader lives in `src.video.config_adapter`, which imports `VideoConfig`
back out of `src.video.config`. The package `__init__` used to call that loader
at import time, so importing the adapter *first* hit a partially-initialised
module and died:

    ImportError: cannot import name 'load_video_config_modular' from
    partially initialized module 'src.video.config_adapter'

`tools/cleanup_outputs.py` imports the adapter as its first project import, so
`make clean-outputs` had never run -- which is why 19 MB of coverage HTML had
accumulated in `outputs/` with nothing to remove it.

The eager call had a second cost. It read five cwd-relative YAML files at
import time, so merely importing any submodule of this package could fail on a
machine with an unrelated video-config error, or from any directory but the
repo root. That is a real failure this project has already hit: loading an
unrelated typed config from here broke scraper construction outright.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# What `load_video_config_modular` merges. Importing the package used to read
# all of them before any caller had asked for a config.
VIDEO_CONFIG_FILES = (
    "video_production.yaml",
    "subtitles.yaml",
    "ai_services.yaml",
)


def run_python(code: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter.

    A subprocess, not an import, because the cycle is an import-time property
    and `sys.modules` in this process already holds both ends resolved.
    """
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(cwd or REPO_ROOT),
    )


class TestEitherEndOfTheCycleImports:
    def test_the_adapter_can_be_imported_first(self):
        """The exact order `tools/cleanup_outputs.py` uses."""
        result = run_python(
            "from src.video.config_adapter import load_video_config_modular\n"
            "print('ok')"
        )

        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_the_package_can_be_imported_first(self):
        result = run_python("import src.video.config\nprint('ok')")

        assert result.returncode == 0, result.stderr

    def test_a_submodule_can_be_imported_first(self):
        """Importing a submodule runs the package `__init__` too."""
        result = run_python(
            "from src.video.config.core_models import VideoConfig\nprint('ok')"
        )

        assert result.returncode == 0, result.stderr


class TestTheSingletonIsLazy:
    def test_importing_the_package_reads_no_video_config(self):
        """The eager load is what made an unrelated config error fatal here.

        Scoped to the files the singleton loads. Two other modules in the
        import chain read a YAML of their own at import time
        (`performance.yaml`, `scraper.yaml`); those are separate and untouched
        by this change, so asserting on them would be asserting on something
        else's behaviour.
        """
        result = run_python(
            "import builtins\n"
            "opened = []\n"
            "real = builtins.open\n"
            "def spy(file, *a, **k):\n"
            "    opened.append(str(file))\n"
            "    return real(file, *a, **k)\n"
            "builtins.open = spy\n"
            "import src.video.config  # noqa\n"
            "builtins.open = real\n"
            "print('OPENED:', opened)\n"
        )

        assert result.returncode == 0, result.stderr
        for name in VIDEO_CONFIG_FILES:
            assert name not in result.stdout, (
                f"importing the package still reads {name}, so an unrelated "
                "video-config error is fatal to every importer -- including "
                f"the scraper: {result.stdout}"
            )

    def test_the_singleton_still_resolves_on_access(self):
        """Lazy must not mean absent: the spelling every consumer uses."""
        result = run_python(
            "from src.video.config import config\n"
            "print('PROFILES:', len(config.video_profiles) > 0)"
        )

        assert result.returncode == 0, result.stderr
        assert "PROFILES: True" in result.stdout

    def test_the_loader_is_still_re_exported(self):
        """It was re-exported as a side effect of the import this replaces.

        Dropping it broke twenty tests that spell it
        `from src.video.config import load_video_config_modular`.
        """
        result = run_python(
            "from src.video.config import load_video_config_modular\n"
            "print('CALLABLE:', callable(load_video_config_modular))"
        )

        assert result.returncode == 0, result.stderr
        assert "CALLABLE: True" in result.stdout

    def test_an_unknown_attribute_still_raises(self):
        """A module `__getattr__` that returns something for every name turns
        a typo'd import into a mystery at first use.
        """
        result = run_python(
            "import src.video.config as m\n"
            "try:\n"
            "    m.no_such_thing\n"
            "except AttributeError as e:\n"
            "    print('RAISED:', 'no_such_thing' in str(e))\n"
        )

        assert "RAISED: True" in result.stdout, result.stderr


class TestTheToolRuns:
    def test_cleanup_outputs_imports_and_parses_args(self):
        """`make clean-outputs` runs this; it had never got past the import."""
        result = subprocess.run(
            [sys.executable, "tools/cleanup_outputs.py", "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )

        assert result.returncode == 0, result.stderr
        assert "--dry-run" in result.stdout

    @pytest.mark.parametrize("tool", ["cleanup_outputs.py", "performance_report.py"])
    def test_every_tool_still_imports(self, tool):
        """The tools are outside the test suite's import graph.

        Nothing else would notice one of them acquiring a broken import, which
        is how this one stayed broken.
        """
        path = REPO_ROOT / "tools" / tool
        if not path.exists():
            pytest.skip(f"{tool} not present")

        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )

        assert result.returncode == 0, f"{tool} --help failed:\n{result.stderr}"
