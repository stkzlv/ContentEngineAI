"""The by-hand topic pool check (REQ-VID-151)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.ai.step_list import Step, StepList
from tools import check_topic_pool


def _list(failures: list[str]) -> StepList:
    step = Step("Tap Off", "Settings > Off", "it is off", "https://a/")
    return StepList("Settings", "iOS", False, [step], topic_failures=failures)


@pytest.mark.req("REQ-VID-151")
def test_each_topic_is_reported_and_a_drop_fails_the_run(
    tmp_path: Path, capsys
) -> None:
    pool = tmp_path / "topics.yaml"
    pool.write_text('- title: "Turn off refresh on iPhone"\n- title: "Fix any phone"\n')
    answers = AsyncMock(side_effect=[_list([]), _list(["not specific"])])

    with patch("src.ai.step_list.build_step_list", answers):
        code = check_topic_pool.main([str(pool)])

    out = capsys.readouterr().out.splitlines()
    assert out == [
        "ok    Turn off refresh on iPhone",
        "drop  Fix any phone: fails the topic filter: not specific",
    ]
    assert code == 1


def test_a_clean_pool_exits_zero(tmp_path: Path) -> None:
    pool = tmp_path / "topics.yaml"
    pool.write_text('- title: "Turn off refresh on iPhone"\n')

    with patch("src.ai.step_list.build_step_list", AsyncMock(return_value=_list([]))):
        assert check_topic_pool.main([str(pool)]) == 0


def test_without_a_file_the_batch_pool_is_read() -> None:
    with patch("src.pipeline.config._configured_topics", return_value=[]) as configured:
        assert check_topic_pool.configured_pool(None) == []

    configured.assert_called_once()
