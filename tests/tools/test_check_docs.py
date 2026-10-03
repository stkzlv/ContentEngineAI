"""The docs check that holds CONTRIBUTING's definition of done.

A design doc left `Accepted` after its feature shipped, a pull request citing a
requirement id that doesn't exist, and a new CLI flag with no line on its
reference page all read fine in review. These pin what the check refuses and
that the repository's own docs pass it.
"""

from __future__ import annotations

from pathlib import Path

import tools.check_docs as check_docs
from tools.check_docs import (
    Design,
    cited_findings,
    design_findings,
    docs_only,
    expand_ids,
    flag_changes,
    surface_findings,
)


def design(status: str, *ids: str) -> Design:
    return Design(Path("0099-x.md"), status, ids)


def test_the_repository_passes() -> None:
    assert design_findings() == []


def test_a_range_names_every_id_in_it() -> None:
    ids = expand_ids("REQ-PUB-072 to REQ-PUB-074, REQ-VID-001")
    assert sorted(ids) == ["REQ-PUB-072", "REQ-PUB-073", "REQ-PUB-074", "REQ-VID-001"]
    assert len(ids) == len(set(ids))


def test_an_accepted_design_whose_requirements_all_shipped_is_refused() -> None:
    found = design_findings([design("Accepted", "REQ-A")], {"REQ-A": "shipped"})
    assert found and "set the design to Implemented" in found[0]


def test_an_implemented_design_with_open_requirements_is_refused() -> None:
    found = design_findings(
        [design("Implemented", "REQ-A", "REQ-B")],
        {"REQ-A": "shipped", "REQ-B": "planned #1"},
    )
    assert found and "REQ-B" in found[0]


def test_agreeing_designs_pass() -> None:
    statuses = {"REQ-A": "shipped", "REQ-B": "planned #1", "REQ-C": "deprecated"}
    docs = [
        design("Accepted", "REQ-A", "REQ-B"),
        design("Implemented", "REQ-A", "REQ-C"),
        design("Draft"),
    ]
    assert design_findings(docs, statuses) == []


def test_a_design_naming_a_missing_requirement_is_refused() -> None:
    found = design_findings([design("Accepted", "REQ-Z")], {})
    assert found and "REQ-Z" in found[0]


def test_an_accepted_design_must_name_a_requirement() -> None:
    assert design_findings([design("Accepted")], {})


def test_docs_and_a_version_bump_are_docs_only() -> None:
    paths = ["docs/guides/x.md", "AGENTS.md", "CHANGELOG.md", "pyproject.toml"]
    assert docs_only(paths, ['version = "1.0.0"', 'version = "1.0.1"'])


def test_the_doc_tests_data_is_docs() -> None:
    assert docs_only(["tests/docs/undocumented_config_keys.txt"], [])
    assert not docs_only(["tests/docs/test_reference_matches_code.py"], [])


def test_a_prompt_template_is_code() -> None:
    assert not docs_only(["src/ai/prompts/video_script.md"], [])


def test_a_dependency_change_is_code() -> None:
    assert not docs_only(["pyproject.toml"], ['requests = "^2"'])


def test_a_workflow_change_is_code() -> None:
    assert not docs_only(["docs/x.md", ".github/workflows/ci.yml"], [])


def test_an_empty_diff_is_not_docs_only() -> None:
    assert not docs_only([], [])


CLI_BEFORE = """
parser.add_argument(
    "--keywords",
    nargs="+",
    help="Search terms",
)
"""


def test_a_flag_added_in_the_formatter_s_layout_is_seen() -> None:
    after = (
        CLI_BEFORE
        + 'parser.add_argument(\n    "--new-flag",\n    action="store_true",\n)\n'
    )
    assert flag_changes(CLI_BEFORE, after) == ["--new-flag"]


def test_a_removed_or_redefined_flag_is_seen() -> None:
    assert flag_changes(CLI_BEFORE, "") == ["--keywords"]
    assert flag_changes(CLI_BEFORE, CLI_BEFORE.replace("Search", "Find")) == [
        "--keywords"
    ]


def test_a_change_to_one_subcommand_s_copy_of_a_flag_is_seen() -> None:
    twice = CLI_BEFORE + CLI_BEFORE
    assert flag_changes(twice, CLI_BEFORE + CLI_BEFORE.replace("Search", "Find"))


def test_a_cli_edit_that_touches_no_flag_passes() -> None:
    assert flag_changes(CLI_BEFORE, CLI_BEFORE + "logger.info('x')\n") == []


def test_a_changed_flag_without_its_page_is_refused() -> None:
    cli = "src/scraper/amazon/cli.py"
    flags = {cli: ["--new-flag"]}
    found = surface_findings([cli], {}, flags)
    assert found and "docs/reference/scraper.md" in found[0]
    assert surface_findings([cli, "docs/reference/scraper.md"], {}, flags) == []


def test_a_move_lists_both_sides(monkeypatch) -> None:
    """A file moved from src/ into docs/ must not read as docs-only."""
    calls = []

    def fake_git(*args: str) -> str:
        calls.append(args)
        return "docs/moved.md\nsrc/ai/prompts/moved.md\n"

    monkeypatch.setattr(check_docs, "git", fake_git)
    paths = check_docs.changed_paths("origin/main")
    assert "--no-renames" in calls[0]
    assert not docs_only(paths, [])


def test_a_new_config_key_without_the_reference_is_refused() -> None:
    path = "config/core.yaml"
    found = surface_findings([path], {path: ["  new_key: 3"]})
    assert found and "docs/reference/configuration.md" in found[0]


def test_a_config_comment_edit_passes() -> None:
    path = "config/core.yaml"
    assert surface_findings([path], {path: ["  # new_key: 3"]}) == []


def test_a_new_model_field_without_the_reference_is_refused() -> None:
    path = "src/video/config/core_models.py"
    found = surface_findings([path], {path: ["    new_field: int = 3"]})
    assert found


def test_an_unknown_requirement_id_is_refused() -> None:
    assert cited_findings("Implements REQ-PUB-999", {"REQ-PUB-001"})
    assert cited_findings("Implements REQ-PUB-001", {"REQ-PUB-001"}) == []


def test_the_base_side_of_a_flag_is_read_at_the_fork_point(monkeypatch) -> None:
    """Main moving on after the fork must not change what the branch changed."""
    cli = "src/publisher/late/cli.py"
    replies = {
        "diff": f"{cli}\n",
        "log": "",
        "merge-base": "forksha\n",
    }
    monkeypatch.setattr(check_docs, "git", lambda *a: replies[a[0]])
    monkeypatch.setattr(check_docs, "changed_lines", lambda base, path: [])
    read = []

    def fake_file_at(ref: str, path: str) -> str:
        read.append(ref)
        return CLI_BEFORE

    monkeypatch.setattr(check_docs, "file_at", fake_file_at)
    assert check_docs.pr_findings("origin/main", "") == []
    assert read == ["forksha", "HEAD"]
