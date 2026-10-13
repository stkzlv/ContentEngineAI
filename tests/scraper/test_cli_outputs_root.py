"""The standalone scraper takes the outputs root as every module does (#740).

With no `--output-dir` it wrote under the repository's `outputs/`, ignoring
`OUTPUTS_DIR` and `CONTENT_ENGINE_OUTPUT`, which every other module honours.
"""

from __future__ import annotations

import pytest


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
        monkeypatch.setattr("sys.argv", ["scraper", "--keywords", "x", *argv])
        with pytest.raises(_BuiltError):
            scraper_cli.main()
        return seen["output_dir"]

    def test_the_environment_moves_the_root(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CONTENT_ENGINE_OUTPUT", raising=False)
        monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "env-root"))

        assert self._built_with(monkeypatch, []) == str(tmp_path / "env-root")

    def test_the_flag_wins_over_the_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "env-root"))

        got = self._built_with(monkeypatch, ["--output-dir", str(tmp_path / "flag")])

        assert got == str(tmp_path / "flag")

    def test_without_either_it_is_the_repository_outputs(self, monkeypatch):
        from src.utils.outputs_paths import get_project_root

        monkeypatch.delenv("OUTPUTS_DIR", raising=False)
        monkeypatch.delenv("CONTENT_ENGINE_OUTPUT", raising=False)

        assert self._built_with(monkeypatch, []) == str(get_project_root() / "outputs")
