"""No shipped prose tells a reader to write a key that aborts the config load.

Four review rounds each found another copy of the same stale instruction, in a
different file: the producer reference, the pycaps reference, the configuration
worked example, and finally `config/subtitles.yaml` itself -- the shipped
config, telling operators to override pycaps per profile with `pycaps_template`.

Each one costs more than a wrong sentence normally does. `load_video_config_modular()`
raises on the first offending profile, so following any of them aborts every
render, not only the one the reader was editing.

The sweep is derived from the legacy maps rather than from a list of files, so
it cannot go stale when a key is added or renamed. So is the exemption, from
the models that back the swept YAML, because a hardcoded list of exceptions is
the thing that goes stale -- the first version of this guard hardcoded two
names, missed three, and would have failed CI on a legal `config/pipeline.yaml`.

Three tiers, and the middle one is the reason this is not a flat allowlist:

- `subtitle_engine` and `subtitle_format` are exempt everywhere: they are
  global `subtitle_settings` keys, so a hit says nothing about whether the
  sentence is about a profile.
- `pycaps_template`, `pycaps_template_pool` and `pycaps_renderer` are exempt
  only in batch scope -- `config/pipeline.yaml` or a line naming the
  `global_batch` section -- where they are `GlobalBatchConfig` keys. Anywhere
  else they are swept, and that is what catches two of the six sites below.
  Exempting them everywhere would be simpler and would give those two up.
- Everything else is judged wherever it appears. Globally those names are
  `margin`, `randomize_fonts`, `safe_zone.min_x`, so the flat form is wrong
  in any context.

`two_part_subtitles` is refused at profile level by a fourth branch of the
validator, is in none of the three maps, and is the legal name of the nested
block -- so it is unjudgeable from the token, like `subtitle_engine`. One of
the sites fixed alongside this test was exactly that.

Two of the sites the review rounds found by hand name `subtitle_engine` and
are therefore review-only surface, which this would not have caught.

Measured against the tree before those fixes, the sweep flags six lines.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from src.video.config.visual_models import (
    _LEGACY_FLAT_TO_NESTED,
    _LEGACY_PYCAPS_FIELDS,
    _LEGACY_SAFE_ZONE_FIELDS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The `global_batch:` YAML section or a dotted reference to it -- not
# `global_batch.py` or `src.pipeline.global_batch`, which are the dominant
# spellings of the token in this repo and are module references, not config.
BATCH_SCOPE = re.compile(r"global_batch[:.](?!py\b)")

SEARCHED = ("docs", "config")
ALSO = (Path("CLAUDE.md"), Path("GEMINI.md"), Path("README.md"))

# A line may name a refused key when it is saying that the key is refused, and
# `pycaps_template` is additionally the md5 salt for template selection
# (`pycaps_engine/renderer.py`), which is a literal in the source, not a key.
#
# Deliberately not a bare `legacy`: that word exempts the lines most likely to
# be wrong. `docs/requirements.md` used to read "Legacy flat per-profile keys
# (...) still load with a deprecation warning", which was false the moment
# this branch landed and which a `legacy` alternative waves through.
#
# `deprecated`, `renamed` and `removed` were tried and dropped. Each exempts
# that canary shape one inflection away -- "are deprecated and still load with
# a warning", "was renamed to X and still works" -- and measured against every
# commit on this branch, none of the three changed the hit count. A word that
# only widens the hole is not worth the reach it buys.
#
# `no longer` is the loose one that stays: it also exempts "Set
# `subtitle_margin` on the profile; the global value is no longer used", which
# is wrong. That hole is inherent to deciding this from words on a line, and
# is why the sweep is a backstop for review rather than a replacement for it.
ALLOWED = re.compile(
    r"refused|rejected|raises?|aborts?"
    r"|no longer|not accepted|not supported|:pycaps_template",
    re.IGNORECASE,
)


def files() -> list[Path]:
    """The swept set, relative to the cwd -- CI runs pytest from the root."""
    found = [p for p in ALSO if p.exists()]
    for d in SEARCHED:
        for pattern in ("*.md", "*.yaml"):
            found.extend(Path(d).rglob(pattern))
    return sorted(found)


def legal_anywhere() -> set[str]:
    """Legacy names that are also a global `subtitle_settings` key.

    A hit on one of these says nothing about whether the sentence is about a
    profile, so they cannot be swept at all. `subtitle_engine` and
    `subtitle_format` are the two.
    """
    from src.video.config.subtitle_models import SubtitleSettings

    return set(SubtitleSettings.model_fields)


def legal_in_batch_config() -> set[str]:
    """Legacy names that are also `global_batch` keys in `config/pipeline.yaml`.

    `pycaps_template`, `pycaps_template_pool` and `pycaps_renderer` are
    declared on `GlobalBatchConfig` and read out of that section, so they are
    correct there and refused on a profile. Exempting them everywhere would
    give up two of the sweep's real catches, so the exemption is scoped to
    where they are legal: that file, or a line that names `global_batch`.

    Derived rather than listed. The first version of this guard hardcoded two
    exempt names, missed these three, and would have failed CI on a legal
    `config/pipeline.yaml`.

    Read from the source with `ast` rather than imported, because importing
    `src.pipeline.config` pulls in the whole video config -- which makes a
    doc-text guard fail for reasons that have nothing to do with docs.
    """
    source = REPO_ROOT / "src" / "pipeline" / "config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GlobalBatchConfig":
            return {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
    raise AssertionError("GlobalBatchConfig not found; the sweep would over-report")


def offenders() -> list[str]:
    """Lines naming a legacy key in a place where that key is refused."""
    judgeable = (
        set(_LEGACY_SAFE_ZONE_FIELDS)
        | set(_LEGACY_PYCAPS_FIELDS)
        | set(_LEGACY_FLAT_TO_NESTED)
    ) - legal_anywhere()
    batch_keys = legal_in_batch_config() & judgeable

    hits = []
    for path in files():
        in_batch_file = path.as_posix() == "config/pipeline.yaml"
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            named = {m for m in judgeable if re.search(rf"\b{m}\b", line)}
            # The glob spellings too: `subtitle_safe_zone_* fields` named the
            # family without naming a member, and is wrong for the same reason.
            if re.search(r"subtitle_safe_zone_\*|pycaps_\*", line):
                named.add("<glob>")
            if in_batch_file or BATCH_SCOPE.search(line):
                named -= batch_keys  # correct there; refused on a profile
            if named and not ALLOWED.search(line):
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


class TestTheBatchExemptionIsScoped:
    """`pycaps_template` is correct under `global_batch`, refused on a profile.

    Exempting the three names everywhere would be simpler and would give up
    two of the sweep's real catches, both of which name `pycaps_template` in
    a profile context.
    """

    def test_the_batch_keys_are_found(self):
        assert legal_in_batch_config() >= {
            "pycaps_template",
            "pycaps_template_pool",
            "pycaps_renderer",
        }

    def _sweep(self, tmp_path, monkeypatch, relative, text):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        return offenders()

    def test_the_batch_config_file_is_exempt(self, tmp_path, monkeypatch):
        found = self._sweep(
            tmp_path,
            monkeypatch,
            "config/pipeline.yaml",
            'global_batch:\n  pycaps_template: "word-focus"\n',
        )

        assert not found, f"legal global_batch config reported as refused: {found}"

    def test_a_line_naming_global_batch_is_exempt(self, tmp_path, monkeypatch):
        found = self._sweep(
            tmp_path,
            monkeypatch,
            "docs/x.md",
            "`global_batch.pycaps_template` pins the template for a batch run.\n",
        )

        assert not found

    def test_the_same_key_in_a_profile_context_is_not(self, tmp_path, monkeypatch):
        found = self._sweep(
            tmp_path,
            monkeypatch,
            "docs/x.md",
            "Set `pycaps_template: hype` on the profile.\n",
        )

        assert found, "the scoped exemption leaks to profile-context lines"
