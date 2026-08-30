"""No shipped prose tells a reader to write a key that aborts the config load.

Four review rounds each found another copy of the same stale instruction, in a
different file: the producer reference, the pycaps reference, the configuration
worked example, and finally `config/subtitles.yaml` itself -- the shipped
config, telling operators to override pycaps per profile with `pycaps_template`.

Each one costs more than a wrong sentence normally does. `load_video_config_modular()`
raises on the first offending profile, so following any of them aborts every
render, not only the one the reader was editing.

The sweep is derived from the legacy maps rather than from a list of files, so
it cannot go stale when a key is added or renamed. It covers every name in
them except `subtitle_engine` and `subtitle_format`, which are the only two
that are also real global keys and so say nothing about whether the sentence
around them is about a profile. Everything else is refused at profile level
and has no global spelling either -- globally they are `margin`,
`randomize_fonts`, `pycaps.template_name`, `safe_zone.min_x` -- so an
occurrence is wrong wherever it appears.

Excluding the whole flat family instead would have left three of the five
sites above uncovered, since only `config/subtitles.yaml`'s two were in the
`pycaps_*` and `safe_zone_*` families.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.video.config.visual_models import (
    _LEGACY_FLAT_TO_NESTED,
    _LEGACY_PYCAPS_FIELDS,
    _LEGACY_SAFE_ZONE_FIELDS,
)

SEARCHED = ("docs", "config")
ALSO = (Path("CLAUDE.md"), Path("GEMINI.md"), Path("README.md"))

# The only two legacy names that are also a real global key, so a hit on one
# says nothing about whether the sentence is about a profile. Every other
# name in the three maps is refused at profile level and does not exist
# globally either -- globally they are `margin`, `randomize_fonts`,
# `pycaps.template_name`, `safe_zone.min_x` -- so an occurrence is wrong
# wherever it appears.
GLOBAL_TOO = {"subtitle_engine", "subtitle_format"}

# A line may name a refused key when it is saying that the key is refused, and
# `pycaps_template` is additionally the md5 salt for template selection
# (`pycaps_engine/renderer.py`), which is a literal in the source, not a key.
#
# Deliberately not `legacy`: that word exempts the lines most likely to be
# wrong. `docs/requirements.md` used to read "Legacy flat per-profile keys
# (...) still load with a deprecation warning", which was false the moment
# this branch landed and which a bare `legacy` alternative waves through.
# The exemption has to key on the refusal, not on the topic.
ALLOWED = re.compile(
    r"refused|rejected|no longer|not accepted|:pycaps_template", re.IGNORECASE
)


def files() -> list[Path]:
    found = [p for p in ALSO if p.exists()]
    for d in SEARCHED:
        for pattern in ("*.md", "*.yaml"):
            found.extend(Path(d).rglob(pattern))
    return sorted(found)


def offenders() -> list[str]:
    keys = sorted(
        (
            set(_LEGACY_SAFE_ZONE_FIELDS)
            | set(_LEGACY_PYCAPS_FIELDS)
            | set(_LEGACY_FLAT_TO_NESTED)
        )
        - GLOBAL_TOO
    )
    # The glob spellings too: `subtitle_safe_zone_* fields` named the family
    # without naming a member, and is wrong for exactly the same reason.
    globs = [r"subtitle_safe_zone_\*", r"pycaps_\*"]
    pattern = re.compile(r"(\b(" + "|".join(keys) + r")\b|" + "|".join(globs) + r")")
    hits = []
    for path in files():
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if pattern.search(line) and not ALLOWED.search(line):
                hits.append(f"{path}:{number}: {line.strip()}")
    return hits


def test_the_sweep_looks_at_something():
    """Every assertion below is vacuous if the file walk finds nothing."""
    assert len(files()) > 10
    assert (
        Path("config/subtitles.yaml") in files()
    ), "no YAML is being swept, so the guard would go green if config/ moved"
    assert set(_LEGACY_SAFE_ZONE_FIELDS) and set(_LEGACY_PYCAPS_FIELDS)


def test_no_shipped_prose_names_a_key_that_aborts_the_load():
    found = offenders()

    assert not found, (
        "these lines name a per-profile key that is refused at config load, so "
        "a reader following them aborts every render:\n" + "\n".join(found)
    )


@pytest.mark.parametrize(
    "key", ["pycaps_renderer", "subtitle_safe_zone_max_y", "subtitle_margin"]
)
def test_the_sweep_would_catch_a_new_one(key, tmp_path, monkeypatch):
    """The guard is only worth having if it fires."""
    doc = tmp_path / "docs" / "made-up.md"
    doc.parent.mkdir()
    doc.write_text(f"Set `{key}: 0.5` on the profile.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert offenders(), f"a doc naming {key} passes the sweep"
