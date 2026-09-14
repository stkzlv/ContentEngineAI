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
                if node.module:
                    names.add(node.module)
                continue
            base = package[: len(package) - (node.level - 1)]
            names.add(".".join([*base, node.module] if node.module else base))
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


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

    def test_the_old_import_path_still_works(self):
        """The shim carries it for one release."""
        import src.video.config as video_config
        from src.ai.llm_settings import LLMSettings

        assert video_config.LLMSettings is LLMSettings
