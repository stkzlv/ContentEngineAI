"""The reference pages list what the code accepts, and nothing it doesn't.

A flag added to a parser with no row on its reference page is invisible to a
reader, and a row for a flag that was renamed or removed sends them to an error.
Review misses both, because the parser and the page sit in different files.
The flags are read from the parser source with the same parse the docs check
uses, so no CLI module is imported.

Config keys are checked one way only: every key in `config/*.yaml` is named
somewhere in `docs/reference/`. The keys not yet documented are listed in
`UNDOCUMENTED_KEYS`; the list may only shrink.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tools.check_docs import CLI_PAGES, cli_flags
from tools.requirements_coverage import REPO

REFERENCE = REPO / "docs" / "reference"
# A flag in a table row: `--name`, the way every reference table writes one.
TABLE_FLAG = re.compile(r"`(--[a-z][\w-]*)")


def pages() -> dict[str, list[str]]:
    """Each reference page and the CLI sources whose flags it lists."""
    grouped: dict[str, list[str]] = {}
    for cli, page in CLI_PAGES.items():
        grouped.setdefault(page, []).append(cli)
    return grouped


def flags_of(sources: list[str]) -> set[str]:
    found: set[str] = set()
    for source in sources:
        found |= {
            f
            for f in cli_flags((REPO / source).read_text(encoding="utf-8"))
            if f.startswith("--")
        }
    return found


def every_flag() -> set[str]:
    return flags_of(list(CLI_PAGES))


def table_flags(text: str) -> set[str]:
    rows = (line for line in text.splitlines() if line.startswith("|"))
    return {flag for row in rows for flag in TABLE_FLAG.findall(row)}


@pytest.mark.parametrize("page", sorted(pages()))
def test_every_flag_has_a_row(page: str) -> None:
    text = (REPO / page).read_text(encoding="utf-8")
    missing = sorted(f for f in flags_of(pages()[page]) if f not in text)
    assert not missing, f"{page} doesn't mention {missing}"


@pytest.mark.parametrize("page", sorted(pages()))
def test_every_row_names_a_real_flag(page: str) -> None:
    """A `--no-x` row stands for a `--x` declared as a boolean option."""
    known = every_flag()
    text = (REPO / page).read_text(encoding="utf-8")
    unknown = sorted(
        f
        for f in table_flags(text)
        if f not in known and f.replace("--no-", "--", 1) not in known
    )
    assert not unknown, f"{page} lists flags no parser declares: {unknown}"


def test_the_flag_parse_sees_the_parsers() -> None:
    assert (
        len(every_flag()) > 90
    ), "the flag parse found too few to be reading the parsers"


def yaml_keys(data: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
    """(dotted path, key) for every mapping key, at any depth."""
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}{key}"
            yield path, str(key)
            yield from yaml_keys(value, f"{path}.")


def undocumented_keys() -> set[str]:
    docs = "".join(p.read_text(encoding="utf-8") for p in REFERENCE.glob("*.md"))
    missing = set()
    for path in sorted((REPO / "config").glob("*.yaml")):
        if ".private." in path.name:
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for dotted, key in yaml_keys(data):
            if key not in docs:
                missing.add(f"{path.name}:{dotted}")
    return missing


def test_every_config_key_is_documented() -> None:
    missing = sorted(undocumented_keys())
    assert not missing, (
        f"config keys with no mention in docs/reference/: {missing}. Document "
        "them in the reference page for their file."
    )
