"""No shipped prose tells a reader to write a key that aborts the config load.

Four review rounds each found another copy of the same stale instruction, in a
different file: the producer reference, the pycaps reference, the configuration
worked example, and finally `config/subtitles.yaml` itself -- the shipped
config, telling operators to override pycaps per profile with `pycaps_template`.

Each one costs more than a wrong sentence normally does. `load_video_config_modular()`
raises on the first offending profile, so following any of them aborts every
render, not only the one the reader was editing.

The sweep is derived from the legacy maps rather than from a list of files, so
it cannot go stale when a key is added or renamed. It covers the two families
that have no legal spelling anywhere -- globally they are `pycaps.<field>` and
`safe_zone.<field>`, so an occurrence is wrong wherever it appears. The flat
`subtitle_*` family is deliberately excluded: those names ARE the global
spelling (`subtitle_settings.subtitle_engine`), so a hit there says nothing
about whether the surrounding sentence is about a profile. That is a real gap
-- two of the four rounds' findings were in that family (`subtitle_margin`,
`subtitle_randomize_fonts`) and this would not have caught either. Closing it
needs the sentence's subject, not its tokens; the two families here are the
part that can be decided from the token alone.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.video.config.visual_models import (
    _LEGACY_PYCAPS_FIELDS,
    _LEGACY_SAFE_ZONE_FIELDS,
)

SEARCHED = ("docs", "config")
ALSO = (Path("CLAUDE.md"), Path("README.md"))

# A line may name a refused key when it is saying that the key is refused, and
# `pycaps_template` is additionally the md5 salt for template selection
# (`pycaps_engine/renderer.py`), which is a literal in the source, not a key.
ALLOWED = re.compile(r"refused|rejected|legacy|:pycaps_template", re.IGNORECASE)


def files() -> list[Path]:
    found = [p for p in ALSO if p.exists()]
    for d in SEARCHED:
        for pattern in ("*.md", "*.yaml"):
            found.extend(Path(d).rglob(pattern))
    return sorted(found)


def offenders() -> list[str]:
    keys = sorted(set(_LEGACY_SAFE_ZONE_FIELDS) | set(_LEGACY_PYCAPS_FIELDS))
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
    assert set(_LEGACY_SAFE_ZONE_FIELDS) and set(_LEGACY_PYCAPS_FIELDS)


def test_no_shipped_prose_names_a_key_that_aborts_the_load():
    found = offenders()

    assert not found, (
        "these lines name a per-profile key that is refused at config load, so "
        "a reader following them aborts every render:\n" + "\n".join(found)
    )


@pytest.mark.parametrize("key", ["pycaps_renderer", "subtitle_safe_zone_max_y"])
def test_the_sweep_would_catch_a_new_one(key, tmp_path, monkeypatch):
    """The guard is only worth having if it fires."""
    doc = tmp_path / "docs" / "made-up.md"
    doc.parent.mkdir()
    doc.write_text(f"Set `{key}: 0.5` on the profile.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert offenders(), f"a doc naming {key} passes the sweep"
