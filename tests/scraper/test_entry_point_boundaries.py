"""The scraper does not reach into the video package, and its CLI stays small.

One call did: `main()` resolved a video profile name to a single boolean by
loading the video config -- another package's five YAML files, read for one
field, on every scrape that passed `--profile`. It also swallowed the failure,
so a broken video config degraded a scrape silently. The CLI takes the boolean
now, and the batch, where profiles live, computes it itself.

The size half is the other reason that call was hard to see: it sat in the
middle of a 580-line `main()`.
"""

from __future__ import annotations

import ast

from src.utils.outputs_paths import get_project_root

REPO = get_project_root()
SCRAPER_PACKAGE = REPO / "src/scraper"
CLI = SCRAPER_PACKAGE / "amazon/cli.py"

# A function longer than this is doing more than one job. The entry point had
# one of 580 lines; the parser is the only one near the bound, and it is a
# flat list of `add_argument` calls rather than logic.
MAX_FUNCTION_LINES = 150


def _imports_of(path) -> set[str]:
    """Every module imported, at any scope, resolved to an absolute name."""
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
            names.update(f"{resolved}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


class TestTheScraperOwnsItsInputs:
    def test_no_scraper_module_imports_the_video_package(self):
        offenders = []
        for path in sorted(SCRAPER_PACKAGE.rglob("*.py")):
            video = {name for name in _imports_of(path) if name.startswith("src.video")}
            if video:
                offenders.append(f"{path.relative_to(REPO)}: {sorted(video)}")
        assert not offenders, (
            "scraper modules import from the video package: "
            + "; ".join(offenders)
            + ". Pass the value in; the batch already computes it"
        )

    def test_the_cli_takes_the_boolean_rather_than_a_profile_name(self):
        from src.scraper.amazon.cli import build_argument_parser

        parser = build_argument_parser()

        assert parser.parse_args([]).profile_uses_videos is None
        assert parser.parse_args(["--profile-uses-videos"]).profile_uses_videos is True
        assert (
            parser.parse_args(["--no-profile-uses-videos"]).profile_uses_videos is False
        )

    def test_the_flag_reaches_the_scraper(self):
        """The parser accepting it proves nothing on its own."""
        source = CLI.read_text(encoding="utf-8")
        tree = ast.parse(source)
        construction = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "BotasaurusAmazonScraper"
        )
        passed = {
            keyword.arg: ast.unparse(keyword.value) for keyword in construction.keywords
        }
        assert passed.get("profile_uses_videos") == "args.profile_uses_videos"


class TestTheTestsPatchWhereTheCliReads:
    """A patch on the wrong module intercepts nothing, and says nothing.

    `main()` moved out of `scraper.py`, so the four tests that drive it and
    patched `BotasaurusAmazonScraper` there kept passing while constructing
    the real class ten times a session -- resetting the websocket logger and
    leaving its filters attached. Nothing failed, which is why this is
    pinned rather than left to the next mover to notice.
    """

    def test_every_main_driving_test_patches_the_cli_module(self):
        drivers = [
            path
            for path in sorted((REPO / "tests").rglob("test_*.py"))
            # This file carries the patterns it searches for, so it matches
            # itself; the strings below are why it has to be skipped.
            if path.name != "test_entry_point_boundaries.py"
            and (
                "scraper_module.main()" in path.read_text(encoding="utf-8")
                or "scraper_mod.main()" in path.read_text(encoding="utf-8")
            )
        ]
        assert drivers, "no test drives the scraper's main(); this guard is blind"

        wrong = []
        for path in drivers:
            source = path.read_text(encoding="utf-8")
            if 'scraper_mod, "BotasaurusAmazonScraper"' in source or (
                'scraper_module, "BotasaurusAmazonScraper"' in source
            ):
                wrong.append(str(path.relative_to(REPO)))
        assert not wrong, (
            "these tests patch the scraper class on the module that no longer "
            f"constructs it, so the real one is built: {wrong}"
        )


class TestTheEntryPointStaysReadable:
    def test_no_function_in_the_cli_is_oversized(self):
        tree = ast.parse(CLI.read_text(encoding="utf-8"))
        lengths = {
            node.name: (node.end_lineno or node.lineno) - node.lineno + 1
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        oversized = {
            name: length
            for name, length in lengths.items()
            if length > MAX_FUNCTION_LINES
        }
        assert not oversized, f"oversized functions in the scraper CLI: {oversized}"

    def test_the_documented_invocation_still_resolves(self):
        """`python -m src.scraper.amazon.scraper` is what every runbook says."""
        from src.scraper.amazon import cli, scraper

        assert callable(scraper.main)
        assert callable(cli.main)
