"""The standalone scraper takes the outputs root as every module does (#740).

With no `--output-dir` it wrote under the repository's `outputs/`, ignoring
`OUTPUTS_DIR` and `CONTENT_ENGINE_OUTPUT`, which every other module honours.
"""

from __future__ import annotations

import pytest

_BUILT: dict = {}


class _BuiltError(Exception):
    """Raised by the fake scraper to stop `main` once it has the arguments."""


@pytest.mark.unit
@pytest.mark.req("REQ-OPS-115")
class TestTheStandaloneScraperTakesTheOutputsRoot:
    """The scraper CLI ignored OUTPUTS_DIR while every other module honoured it."""

    @staticmethod
    def _built_with(monkeypatch, argv):
        from src.scraper.amazon import cli as scraper_cli

        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            raise _BuiltError

        monkeypatch.setattr(scraper_cli, "BotasaurusAmazonScraper", fake)
        monkeypatch.setattr(scraper_cli, "_start_logging", lambda: None)
        monkeypatch.setattr(scraper_cli, "_resolve_debug", lambda *_a: None)
        # `main` loads `.env`, which would put back a variable a test removed.
        monkeypatch.setattr("dotenv.load_dotenv", lambda *_a, **_k: None)
        cleaned: list = []
        monkeypatch.setattr(scraper_cli, "_clean_outputs", cleaned.append)
        seen["cleaned"] = cleaned
        monkeypatch.setattr("sys.argv", ["scraper", "--keywords", "x", *argv])
        with pytest.raises(_BuiltError):
            scraper_cli.main()
        _BUILT.clear()
        _BUILT.update(seen)
        return seen["output_dir"]

    def test_the_environment_moves_the_root(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CONTENT_ENGINE_OUTPUT", raising=False)
        monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "env-root"))

        assert self._built_with(monkeypatch, []) == str(tmp_path / "env-root")

    def test_the_flag_wins_over_the_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "env-root"))

        got = self._built_with(monkeypatch, ["--output-dir", str(tmp_path / "flag")])

        assert got == str(tmp_path / "flag")

    def test_without_either_the_shared_resolver_decides(self, monkeypatch, tmp_path):
        """No flag: the CLI asks `resolve_outputs_dir`, whose default is tested
        with the other modules. Stubbed, since the CLI creates the root and a
        fresh checkout has no `outputs/` for this test to create.
        """
        from src.scraper.amazon import cli as scraper_cli

        asked = []

        def resolver(value):
            asked.append(value)
            return tmp_path / "outputs"

        monkeypatch.setattr(scraper_cli, "resolve_outputs_dir", resolver)

        assert self._built_with(monkeypatch, []) == str(tmp_path / "outputs")
        assert asked == [None]

    def test_clean_cleans_the_root_the_scrape_uses(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "env-root"))

        self._built_with(monkeypatch, ["--clean"])

        assert _BUILT["cleaned"] == [tmp_path / "env-root"]

    def test_a_root_whose_parents_are_missing_gets_them(self, monkeypatch, tmp_path):
        root = tmp_path / "a" / "b" / "outputs"
        monkeypatch.setenv("OUTPUTS_DIR", str(root))

        self._built_with(monkeypatch, [])

        assert root.parent.is_dir()


@pytest.mark.unit
@pytest.mark.req("REQ-OPS-115")
def test_clean_removes_products_under_the_given_root_only(tmp_path):
    from src.scraper.amazon import cli as scraper_cli

    root, other = tmp_path / "root", tmp_path / "other"
    (root / "B0AAAAAAAA").mkdir(parents=True)
    (root / "reports").mkdir()
    (other / "B0BBBBBBBB").mkdir(parents=True)

    scraper_cli._clean_outputs(root)

    assert sorted(p.name for p in root.iterdir()) == ["reports"]
    assert (other / "B0BBBBBBBB").is_dir()
