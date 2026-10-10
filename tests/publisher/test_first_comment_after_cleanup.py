"""The first comment survives the render's cleanup (REQ-PUB-153).

A successful render deletes `temp/`, script included, and the publisher
runs after it, so every batch publish skipped its YouTube first comment.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from src.publisher.first_comment import build_first_comment
from src.publisher.models import FirstCommentConfig
from src.utils import cleanup_temp_dirs
from src.utils.script_signoff import (
    SCRIPT_RECORD,
    STATE_FILE,
    keep_script_record,
    read_script,
)

SCRIPT = (
    "It arrived today. USB-C or Lightning, which annoys you more? "
    "That's my quick find. Link in bio if you want one."
)


def _config() -> FirstCommentConfig:
    return FirstCommentConfig(enabled=True, platforms={"youtube": "{closing_line}"})


def _render(tmp_path: Path, product_id: str = "B0KEEP") -> Path:
    product = tmp_path / product_id
    temp = product / "temp"
    temp.mkdir(parents=True)
    (temp / "script.txt").write_text(SCRIPT)
    (temp / STATE_FILE).write_text(json.dumps({"signoff": "That's my quick find."}))
    return product


@pytest.mark.unit
@pytest.mark.req("REQ-PUB-153")
class TestTheRecord:
    def test_the_first_comment_is_built_after_temp_is_removed(
        self, tmp_path: Path
    ) -> None:
        product = _render(tmp_path)

        keep_script_record(product, product / "temp")
        cleanup_temp_dirs(product / "temp")

        assert not (product / "temp").exists()
        assert (
            build_first_comment(_config(), "youtube", "B0KEEP", tmp_path)
            == "USB-C or Lightning, which annoys you more?"
        )

    def test_the_record_keeps_the_signoff(self, tmp_path: Path) -> None:
        product = _render(tmp_path)

        keep_script_record(product, product / "temp")
        cleanup_temp_dirs(product / "temp")

        assert read_script(product) == (SCRIPT, "That's my quick find.")

    def test_temp_wins_over_the_record(self, tmp_path: Path) -> None:
        product = _render(tmp_path)
        (product / SCRIPT_RECORD).write_text(
            json.dumps({"script": "Old. Stale question?", "signoff": None})
        )

        assert read_script(product) == (SCRIPT, "That's my quick find.")

    @pytest.mark.parametrize("record", ["{not json", '["a list"]', '{"script": 3}'])
    def test_a_broken_record_skips_the_comment(
        self, tmp_path: Path, record: str
    ) -> None:
        product = tmp_path / "B0BROKEN"
        product.mkdir()
        (product / SCRIPT_RECORD).write_text(record)

        assert read_script(product) is None
        assert build_first_comment(_config(), "youtube", "B0BROKEN", tmp_path) is None

    def test_no_script_writes_no_record(self, tmp_path: Path) -> None:
        product = tmp_path / "B0EMPTY"
        (product / "temp").mkdir(parents=True)

        keep_script_record(product, product / "temp")

        assert not (product / SCRIPT_RECORD).exists()


@pytest.mark.unit
@pytest.mark.req("REQ-PUB-153")
def test_the_producer_keeps_the_record_before_cleaning_up() -> None:
    from src.video.producer import orchestration

    source = inspect.getsource(orchestration)
    keep = source.index("keep_script_record(")
    clean = source.index('cleanup_temp_dirs(run_paths["intermediate_base"])')

    assert keep < clean
    assert 'run_paths["run_root"], run_paths["intermediate_base"]' in source[keep:clean]
