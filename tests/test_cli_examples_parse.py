"""Every example in a CLI's help epilog parses with that CLI's parser.

An example is the first thing a reader copies, and nothing ran them: the
publisher's showed a global option after its subcommand, which argparse
refuses, and three CLIs named profiles that don't exist.
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Callable

import pytest

from src.pipeline.cli import create_argument_parser as pipeline_parser
from src.publisher.late.cli import build_argument_parser as publisher_parser
from src.scraper.amazon.cli import build_argument_parser as scraper_parser
from src.video.producer.cli import create_argument_parser as producer_parser

PARSERS: dict[str, Callable[[], argparse.ArgumentParser]] = {
    "src.pipeline": pipeline_parser,
    "src.publisher.late": publisher_parser,
    "src.scraper.amazon.scraper": scraper_parser,
    "src.video.producer": producer_parser,
}


def examples(module: str) -> list[list[str]]:
    """The argument lists of each `python -m <module>` line in the epilog."""
    epilog = PARSERS[module]().epilog or ""
    joined = epilog.replace("\\\n", " ")
    found = []
    for line in joined.splitlines():
        words = shlex.split(line.strip(), comments=True)
        if module in words and "-m" in words:
            found.append(words[words.index(module) + 1 :])
    return found


@pytest.mark.parametrize("module", sorted(PARSERS))
def test_every_epilog_example_parses(module: str) -> None:
    parser = PARSERS[module]()
    for argv in examples(module):
        try:
            parser.parse_args(argv)
        except SystemExit:
            pytest.fail(f"{module} help example does not parse: {shlex.join(argv)}")


@pytest.mark.parametrize("module", ["src.pipeline", "src.publisher.late"])
def test_the_examples_are_found(module: str) -> None:
    assert examples(module)


def test_profile_examples_name_real_profiles() -> None:
    """Help text and config comments name profiles that exist."""
    import re
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parent.parent
    profiles = set(
        yaml.safe_load((repo / "config" / "video_production.yaml").read_text())[
            "video_profiles"
        ]
    )
    named = set()
    for path in [
        repo / "src" / "pipeline" / "cli.py",
        repo / "src" / "video" / "producer" / "cli.py",
        repo / "config" / "pipeline.yaml",
        repo / "config" / "video_production.yaml",
    ]:
        for line in path.read_text().splitlines():
            if "profile" not in line.lower() and "Example" not in line:
                continue
            # Quoted names, and the words after a profile flag.
            words = re.findall(r'"([a-z][a-z0-9_]*_[a-z0-9_]+)"', line)
            for flag in re.findall(r"--profile(?:-pool)? ((?:[a-z][\w]* ?)+)", line):
                words += flag.split()
            named |= {w for w in words if "video" in w or "slideshow" in w}
    assert named
    assert not named - profiles, f"unknown profiles named: {named - profiles}"
