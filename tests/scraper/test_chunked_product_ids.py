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
from pathlib import Path

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
    """Reads the call site: the loader's behaviour is right either way.

    A behavioural test would need a full chunked scrape against a live
    browser. What can be checked cheaply is the sentinel the loop passes, and
    that is where the defect was -- `None` on chunk 2 is the whole bug.
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
        if not isinstance(value, ast.IfExp):
            continue
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
