"""No method in the scheduler is big enough to hide a defect in (#449).

`auto_schedule` was 744 lines with no nested definitions: occupancy, dedupe,
slot conflict, both publish branches, tracking, cleanup and all three
post-publish hooks, separated only by comments. The caption-clamping defect
(#403/#408) survived four review rounds inside those branches, and the fix
that finally held had to be a structural AST test, because reading a function
this size kept failing.

The bound counts code, not the docstring: `auto_schedule` documents eleven
public arguments and an example, which is not the complexity this is about.
"""

from __future__ import annotations

import ast

from src.utils.outputs_paths import get_project_root

REPO = get_project_root()
SCHEDULE = REPO / "src/publisher/schedule.py"
MAX_CODE_LINES = 150


def _code_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    total = (node.end_lineno or node.lineno) - node.lineno + 1
    docstring = node.body[0] if node.body else None
    if (
        isinstance(docstring, ast.Expr)
        and isinstance(docstring.value, ast.Constant)
        and isinstance(docstring.value.value, str)
    ):
        total -= (docstring.end_lineno or docstring.lineno) - docstring.lineno + 1
    return total


class TestTheSchedulerIsReadable:
    def test_no_method_is_oversized(self):
        tree = ast.parse(SCHEDULE.read_text(encoding="utf-8"))
        oversized = {
            node.name: _code_lines(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and _code_lines(node) > MAX_CODE_LINES
        }
        assert not oversized, (
            f"oversized methods in the scheduler: {oversized}. "
            "Extract, rather than adding another comment-separated section"
        )

    def test_a_resolved_conflict_counts_even_if_the_publish_fails(self):
        """The one number the split nearly changed.

        The conflict counter used to be incremented where the conflict was
        resolved, which is before the post is attempted; carrying the outcome
        as a value made it easy to tally only on the scheduled branch, and a
        product whose slot was resolved and whose upload then failed stopped
        reporting the resolution. Nothing else reads the number, which is
        exactly why a refactor could move it unnoticed.
        """
        import asyncio
        import tempfile
        from datetime import UTC, datetime, timedelta
        from pathlib import Path
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.publisher.base import PublishError
        from src.publisher.models import (
            CleanupConfig,
            ConflictResolution,
            Platform,
            RecurringSlot,
            ScheduleConfig,
        )
        from src.publisher.schedule import ScheduleManager

        async def run(tmp: Path) -> dict[str, int]:
            config = ScheduleConfig(
                enabled=True,
                slots=[
                    RecurringSlot(day_of_week="monday", time="10:00:00", timezone="UTC")
                ],
            )
            manager = ScheduleManager(
                schedule_path=tmp / "schedule.json", config=config
            )

            publisher = MagicMock()
            publisher.list_posts = AsyncMock(return_value=[])
            publisher.get_accounts = AsyncMock(
                return_value=[{"platform": "youtube", "account_id": "a1"}]
            )
            publisher.upload_media = AsyncMock(
                side_effect=PublishError("upload failed")
            )
            publisher.first_comment_config = None

            video = tmp / "B0TEST00001" / "video.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"x")

            with (
                patch("src.publisher.schedule.ScheduleValidator") as validator,
                patch.object(manager, "resolve_conflict") as resolver,
                patch(
                    "src.publisher.schedule.is_already_published", return_value=False
                ),
            ):
                validator.return_value.validate.return_value = (False, "slot taken")
                resolver.return_value = ConflictResolution(
                    original_time=datetime.now(UTC),
                    conflict_reason="taken",
                    alternatives=[],
                    auto_resolved=True,
                    resolved_time=datetime.now(UTC) + timedelta(days=1),
                )
                return await manager.auto_schedule(
                    videos=[video],
                    platforms=[Platform.YOUTUBE],
                    publisher=publisher,
                    auto_resolve=True,
                    outputs_dir=None,
                    cleanup_config=CleanupConfig(enabled=False),
                )

        with tempfile.TemporaryDirectory() as directory:
            summary = asyncio.run(run(Path(directory)))

        assert summary["failed"] == 1
        assert (
            summary["conflicts_resolved"] == 1
        ), "a conflict resolved before a failed publish stopped being counted"

    def test_the_publish_branches_are_their_own_methods(self):
        """The two that the caption defect hid between.

        Named rather than merely counted: the point is that each posting mode
        is a thing a reviewer can read start to finish, with its own target
        list in scope.
        """
        tree = ast.parse(SCHEDULE.read_text(encoding="utf-8"))
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        assert {"_post_unified", "_post_per_platform"} <= names
