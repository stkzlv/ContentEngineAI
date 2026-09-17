"""The publisher does not import the video config package (#448).

`src/video/config/` had become the project-wide constants drawer: six
publisher modules read their provider upload limits, retry delay, webhook
history bound and slot-search limits from it, so changing a publishing
constant edited the video package -- and every one of those importers sat one
attribute access away from the config singleton, which loads five YAML files.
The constants live in `src/publisher/constants.py` now.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.utils.outputs_paths import get_project_root

REPO = get_project_root()


def _imports_of(path) -> set[str]:
    """Every module this file imports, at any scope, as an absolute name.

    Relative imports are resolved rather than skipped: `from ...video.config
    import X` inside `src/publisher/late/` reaches the same module by another
    spelling, and a guard that only reads absolute ones would say the boundary
    holds.
    """
    package = list(path.relative_to(REPO).parent.parts)
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                resolved = node.module or ""
            else:
                base = package[: len(package) - (node.level - 1)]
                resolved = ".".join([*base, node.module] if node.module else base)
            names.add(resolved)
            # Each imported name may itself be a submodule: `from src import
            # video` records only `src` otherwise, and reaches the package the
            # guard is about.
            names.update(f"{resolved}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


OLD_PACKAGE = "src.video.config"


def _old_path_reads(source: str) -> list[int]:
    """Lines that reach `LLMSettings` through the old package, by any spelling.

    Aliases are resolved per file: `import src.video.config as cfg` binds
    `cfg`, `from src.video import config as vc` binds `vc`, a plain
    `import src.video.config` binds the dotted name itself, and the same
    holds for the package's submodules (`core_models` imports the model too)
    and for `from src import video`, which binds `video.config`. An attribute
    read of `LLMSettings` off any of those is a hit.

    Absolute spellings only. `src/video` uses no relative imports and the
    sweep runs on source text, so `from .config import LLMSettings` inside
    that package would pass; the sibling `_imports_of` resolves relative
    imports for the publisher boundary, where they do occur.
    """
    tree = ast.parse(source)
    aliases = {OLD_PACKAGE}
    lines: list[int] = []

    def in_old_package(module: str) -> bool:
        return module == OLD_PACKAGE or module.startswith(OLD_PACKAGE + ".")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if in_old_package(alias.name):
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "src":
                aliases.update(
                    f"{alias.asname or alias.name}.config"
                    for alias in node.names
                    if alias.name == "video"
                )
            if module == "src.video":
                aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "config"
                )
            if in_old_package(module):
                # A submodule bound by name (`from src.video.config import
                # core_models`) carries the attribute too; a class bound this
                # way has no `LLMSettings` attribute, so it adds no false hit.
                aliases.update(alias.asname or alias.name for alias in node.names)
                if any(alias.name == "LLMSettings" for alias in node.names):
                    lines.append(node.lineno)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "LLMSettings"
            and ast.unparse(node.value) in aliases
        ):
            lines.append(node.lineno)
    return sorted(set(lines))


class TestThePublisherOwnsItsConstants:
    def test_no_publisher_module_imports_the_video_config(self):
        offenders = []
        for path in sorted((REPO / "src/publisher").rglob("*.py")):
            video = {name for name in _imports_of(path) if name.startswith("src.video")}
            if video:
                offenders.append(f"{path.relative_to(REPO)}: {sorted(video)}")
        assert not offenders, (
            "publisher modules import from the video package: "
            + "; ".join(offenders)
            + ". A publishing constant belongs in src/publisher/constants.py"
        )

    def test_the_constants_kept_their_values(self):
        """A move must not become an edit."""
        from src.publisher.constants import (
            LATE_API_KEY_MIN_LENGTH,
            LATE_DEFAULT_RETRY_AFTER_SEC,
            LATE_DIRECT_UPLOAD_MAX_BYTES,
            LATE_MAX_UPLOAD_SIZE_BYTES,
            SCHEDULE_ALTERNATIVE_SEARCH_MULTIPLIER,
            SCHEDULE_MAX_SLOT_SEARCH_ATTEMPTS,
            WEBHOOK_EVENT_HISTORY_LIMIT,
        )

        assert LATE_DIRECT_UPLOAD_MAX_BYTES == 4 * 1024 * 1024
        assert LATE_MAX_UPLOAD_SIZE_BYTES == 500 * 1024 * 1024
        assert LATE_DEFAULT_RETRY_AFTER_SEC == 60
        assert LATE_API_KEY_MIN_LENGTH == 10
        assert WEBHOOK_EVENT_HISTORY_LIMIT == 1000
        assert SCHEDULE_MAX_SLOT_SEARCH_ATTEMPTS == 100
        assert SCHEDULE_ALTERNATIVE_SEARCH_MULTIPLIER == 10


class TestLLMSettingsLivesBesideItsReaders:
    def test_it_is_defined_under_src_ai(self):
        from src.ai.llm_settings import LLMSettings

        assert LLMSettings.__module__ == "src.ai.llm_settings"

    def test_it_does_not_reach_back_into_the_video_config(self):
        """Otherwise the move buys nothing: importing it would still run that
        package's `__init__` and everything it reaches.
        """
        video = {
            name
            for name in _imports_of(REPO / "src/ai/llm_settings.py")
            if name.startswith("src.video")
        }
        assert not video, f"src/ai/llm_settings.py imports {sorted(video)}"

    def test_the_old_import_path_is_gone(self):
        """The shim carried it for one release, and that release has passed.

        While it existed, four modules kept importing from the old path,
        which the shim made invisible: the issue that removed it said there
        were no consumers left, and a grep for the one-line form found three
        of the four. Only the AST sweep below saw the multi-name import.
        """
        import src.video.config as video_config

        assert not hasattr(video_config, "LLMSettings")
        assert "LLMSettings" not in video_config.__all__

    def test_nothing_spells_the_old_import_path(self):
        """An import that resolves through the shim looks like any other.

        The only way to see one is to look at the source, so this walks every
        module for the name reached through the old package, in every form
        that resolves at runtime: `from src.video.config import LLMSettings`,
        the multi-name block, the submodule path
        (`src.video.config.core_models` imports the model too), and an
        attribute read off the package under whatever alias the file bound
        it to. mypy catches none of these: the package's module `__getattr__`
        types an unknown attribute as `Any`.
        """
        offenders = []
        for root in ("src", "tests", "tools"):
            for path in (REPO / root).rglob("*.py"):
                if path == Path(__file__).resolve():
                    continue
                offenders += [
                    f"{path.relative_to(REPO)}:{line}"
                    for line in _old_path_reads(path.read_text(encoding="utf-8"))
                ]

        assert (
            not offenders
        ), f"still reach LLMSettings through src.video.config: {offenders}"

    @pytest.mark.parametrize(
        "source",
        [
            "from src.video.config import LLMSettings\n",
            "from src.video.config import (\n    AudioSettings,\n    LLMSettings,\n)\n",
            "from src.video.config.core_models import LLMSettings\n",
            "import src.video.config as cfg\nx = cfg.LLMSettings\n",
            "import src.video.config\nx = src.video.config.LLMSettings\n",
            "from src.video import config\nx = config.LLMSettings\n",
            "from src.video import config as vc\nx = vc.LLMSettings\n",
            "import src.video.config.core_models as cm\nx = cm.LLMSettings\n",
            "from src.video.config import core_models\nx = core_models.LLMSettings\n",
            "import src.video.config.core_models\n"
            "x = src.video.config.core_models.LLMSettings\n",
            "from src import video\nx = video.config.LLMSettings\n",
            "import src.video\nx = src.video.config.LLMSettings\n",
        ],
        ids=[
            "direct",
            "multi-name",
            "submodule",
            "import-as",
            "dotted",
            "from-package",
            "from-package-as",
            "submodule-as",
            "submodule-from",
            "submodule-dotted",
            "from-src",
            "import-src-video",
        ],
    )
    def test_the_sweep_sees_every_spelling(self, source: str):
        """The first version matched three of these; two review passes found
        the other nine, four per pass and one more that only the seed alias
        covers.
        """
        assert _old_path_reads(source), f"not flagged: {source!r}"

    @pytest.mark.parametrize(
        "source",
        [
            "from src.ai.llm_settings import LLMSettings\n",
            "import src.video.config as cfg\nx = cfg.VideoConfig\n",
            "settings = object()\nx = settings.LLMSettings\n",
        ],
        ids=["new-home", "other-name-off-package", "unrelated-attribute"],
    )
    def test_the_sweep_leaves_the_rest_alone(self, source: str):
        assert not _old_path_reads(source)
