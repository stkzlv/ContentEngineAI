"""A chunked `--product-ids` run must scrape only the products it was given.

`load_batch_config` reads `cli_keywords=None` as "the CLI named no keywords"
and falls back to the configured list. The chunk loop passed `None` for every
chunk after the first, so a run with `--batch-size` small enough to produce
two chunks searched all 54 configured keywords from chunk 2 on -- on top of
the requested ASINs, and silently, because the log reads like a normal
keyword run.

Same class as the pillar-map defect fixed earlier: a "not supplied" sentinel
colliding with "supplied as empty".
"""

from __future__ import annotations

import ast
import contextlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

from src.scraper.amazon.config import load_batch_config


class TestTheLoaderDistinguishesEmptyFromAbsent:
    """The fix depends on this, so it is asserted rather than assumed."""

    def test_an_explicit_empty_list_searches_nothing(self):
        config = load_batch_config(cli_product_ids=["B0AAAAAAAA"], cli_keywords=[])

        assert config.keywords == []

    def test_absent_falls_back_to_the_configured_list(self):
        config = load_batch_config(cli_product_ids=["B0AAAAAAAA"], cli_keywords=None)

        assert config.keywords, "the YAML fallback is what `None` is for"

    def test_a_named_list_is_used_as_given(self):
        config = load_batch_config(
            cli_product_ids=["B0AAAAAAAA"], cli_keywords=["wireless earbuds"]
        )

        assert config.keywords == ["wireless earbuds"]


def test_later_chunks_are_given_an_empty_list_not_none():
    """Reads the call site, as a second line of defence.

    The outcome is pinned by the run test below; this pins the shape, so a
    reviewer reading the loop sees why the argument is `[]`. It is deliberately
    strict about the form it recognises: skipping an unfamiliar one would make
    this test pass on a call site it never inspected.
    """
    tree = ast.parse(Path("src/scraper/amazon/scraper.py").read_text())

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_batch_config"
        and any(kw.arg == "cli_keywords" for kw in node.keywords)
    ]
    assert calls, "no load_batch_config call passing cli_keywords"

    for call in calls:
        value = next(kw.value for kw in call.keywords if kw.arg == "cli_keywords")
        assert isinstance(value, ast.IfExp), (
            "the argument is no longer a chunk-index ternary, so this check "
            "cannot see what later chunks are passed; assert the new shape "
            "here rather than leaving it unread"
        )
        # The else-branch is what every chunk after the first receives.
        assert not (
            isinstance(value.orelse, ast.Constant) and value.orelse.value is None
        ), (
            "later chunks are passed None, which the loader reads as 'no CLI "
            "keywords' and answers with the whole configured list"
        )
        assert (
            isinstance(value.orelse, ast.List) and not value.orelse.elts
        ), "later chunks should be passed an explicit empty list"


def test_a_chunked_run_searches_no_keywords_after_the_first_chunk(
    monkeypatch, tmp_path
):
    """The outcome, driven through the real CLI with no browser.

    Two product ids at `--batch-size 1` make two chunks. The loader runs for
    real; only the scraper and the batch controller are stubbed, so what is
    measured is the keyword list each chunk actually resolved to.
    """
    import dotenv

    from src.scraper.amazon import batch_controller as batch_mod
    from src.scraper.amazon import config as scraper_config
    from src.scraper.amazon import scraper as scraper_mod

    resolved: list[list[str]] = []
    real_loader = scraper_config.load_batch_config

    def recording_loader(*args, **kwargs):
        config = real_loader(*args, **kwargs)
        resolved.append(list(config.keywords or []))
        return config

    class _Scraper:
        def __init__(self, *a, **kw):
            pass

        def __getattr__(self, name):
            return MagicMock()

    class _Controller:
        def __init__(self, *a, **kw):
            pass

        def run_batch(self):
            return batch_mod.BatchSummary(
                total_attempted=0,
                product_ids_attempted=0,
                keywords_attempted=0,
                successful=0,
                failed=0,
                successful_products=[],
                failed_products=[],
                media_stats={},
                duration_sec=0.0,
            )

    monkeypatch.setattr(scraper_config, "load_batch_config", recording_loader)
    monkeypatch.setattr(scraper_mod, "BotasaurusAmazonScraper", _Scraper)
    # Both are imported inside `main`, so patch them at their source module.
    monkeypatch.setattr(batch_mod, "BatchController", _Controller)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scraper",
            "--product-ids",
            "B0AAAAAAAA",
            "B0BBBBBBBB",
            "--batch-size",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
    )

    with contextlib.suppress(SystemExit):
        scraper_mod.main()

    assert len(resolved) == 2, f"expected two chunks, got {len(resolved)}"
    assert resolved[1] == [], (
        f"chunk 2 searched {len(resolved[1])} configured keywords on a run "
        "that named only product ids"
    )
