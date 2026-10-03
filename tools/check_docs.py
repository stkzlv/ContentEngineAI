"""Check that the docs a change must touch move with it.

CONTRIBUTING.md's "Definition of done" names the docs each kind of change
updates. Prose alone did not hold that rule for releases, so the parts a
script can see are checked here:

- A design doc's status agrees with its requirements: `Implemented` means
  every requirement it names is `shipped` (or `deprecated`), and `Accepted`
  means at least one is not yet.
- Every requirement id a pull request cites, in its description or its
  commits, exists.
- A pull request that adds, changes or removes a CLI flag or a config key also
  changes the page that documents it, or says `Docs: none` and why.

It also answers whether a diff touches only documentation, which CI uses to
skip the full test suite for such a pull request.

It reads files and git, and imports nothing from the pipeline, so it runs
without the project's dependencies.

Usage:
  python -m tools.check_docs                              # repository state
  python -m tools.check_docs --base origin/main           # what CI runs on a PR
  python -m tools.check_docs --base origin/main --docs-only   # print true/false
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tools.requirements_coverage import REPO, requirements

DESIGN = REPO / "docs" / "design"
REQ_ID = re.compile(r"\bREQ-[A-Z]{3}-\d{3}\b")
REQ_RANGE = re.compile(r"\b(REQ-([A-Z]{3})-(\d{3})) to REQ-\2-(\d{3})\b")
OPT_OUT = re.compile(r"^\s*Docs:\s*none\b", re.IGNORECASE | re.MULTILINE)

DONE_STATUSES = ("shipped", "deprecated")

# Paths whose change can't alter what the code does. `*.md` under src/ is a
# prompt template, so it is code.
DOC_PATTERNS = (
    "docs/*",
    "*.md",
    ".github/ISSUE_TEMPLATE/*",
    ".github/PULL_REQUEST_TEMPLATE.md",
    "LICENSE",
)
CODE_ROOTS = ("src/", "tests/", "config/", "tools/", "deploy/")

# Each CLI module and the page that lists its flags.
CLI_PAGES = {
    "src/scraper/amazon/cli.py": "docs/reference/scraper.md",
    "src/video/producer/cli.py": "docs/reference/video-producer.md",
    "src/video/producer/shared_cli.py": "docs/reference/video-producer.md",
    "src/publisher/late/cli.py": "docs/reference/publisher.md",
    "src/pipeline/cli.py": "docs/guides/batch-processing.md",
}

CONFIG_PAGE = "docs/reference/configuration.md"
CONFIG_YAML = "config/*.yaml"
CONFIG_MODELS = ("src/video/config/*_models.py", "src/scraper/config_models.py")
YAML_KEY = re.compile(r"^\s*[A-Za-z_][\w-]*:(\s|$)")
MODEL_FIELD = re.compile(r"^\s+[a-z_]\w*:\s*[^=]+(=.*)?$")


@dataclass(frozen=True)
class Design:
    path: Path
    status: str
    req_ids: tuple[str, ...]


def expand_ids(text: str) -> list[str]:
    """Ids named in text, with `REQ-PUB-072 to REQ-PUB-084` spelled out."""
    ids = REQ_ID.findall(text)
    for _, area, start, end in REQ_RANGE.findall(text):
        ids += [f"REQ-{area}-{n:03d}" for n in range(int(start), int(end) + 1)]
    return list(dict.fromkeys(ids))


def designs() -> list[Design]:
    found = []
    for path in sorted(DESIGN.glob("[0-9][0-9][0-9][0-9]-*.md")):
        if path.name.startswith("0000-"):
            continue
        text = path.read_text(encoding="utf-8")
        status = re.search(r"^- \*\*Status:\*\* (.+)$", text, re.MULTILINE)
        reqs = re.search(r"^- \*\*Requirements:\*\* (.+)$", text, re.MULTILINE)
        found.append(
            Design(
                path,
                status.group(1).strip() if status else "",
                tuple(expand_ids(reqs.group(1))) if reqs else (),
            )
        )
    return found


def design_findings(
    docs: list[Design] | None = None, statuses: dict[str, str] | None = None
) -> list[str]:
    docs = designs() if docs is None else docs
    if statuses is None:
        statuses = {r.req_id: r.status for r in requirements()}
    problems = []
    for doc in docs:
        name = doc.path.name
        if doc.status in ("Accepted", "Implemented") and not doc.req_ids:
            problems.append(f"{name}: an {doc.status} design names no requirement")
        missing = [i for i in doc.req_ids if i not in statuses]
        if missing:
            problems.append(f"{name}: names requirements that don't exist: {missing}")
        known = [i for i in doc.req_ids if i in statuses]
        open_ids = [i for i in known if statuses[i] not in DONE_STATUSES]
        if doc.status == "Implemented" and open_ids:
            problems.append(
                f"{name}: Implemented, but {open_ids} are not shipped; ship them "
                "or set the design back to Accepted"
            )
        if doc.status == "Accepted" and known and not open_ids:
            problems.append(
                f"{name}: every requirement is shipped; set the design to Implemented"
            )
    return problems


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def changed_paths(base: str) -> list[str]:
    """Every path the diff touches. `--no-renames` lists both sides of a move,
    so a file moved out of `src/` into `docs/` still counts as a code change.
    """
    out = git("diff", "--name-only", "--no-renames", f"{base}...HEAD")
    return [p for p in out.splitlines() if p]


def file_at(ref: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=REPO, capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else ""


def cli_flags(source: str) -> dict[str, list[str]]:
    """Each flag an `add_argument` call declares, mapped to the call's text.

    Parsed rather than matched line by line, because the formatter puts the
    flag on the line after `add_argument(`. The call's dump stands for its
    definition, so a changed default or help text counts as a change. Subcommands
    repeat flags, so a name maps to every call that declares it.
    """
    if not source:
        return {}
    flags: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        names = [
            a.value
            for a in node.args
            if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        for name in names:
            if name.startswith("-"):
                flags.setdefault(name, []).append(ast.dump(node))
    return {name: sorted(dumps) for name, dumps in flags.items()}


def flag_changes(old: str, new: str) -> list[str]:
    """Flags added, removed or redefined between two versions of a CLI file."""
    before, after = cli_flags(old), cli_flags(new)
    return sorted(
        name
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    )


def changed_lines(base: str, path: str) -> list[str]:
    """The added and removed lines of one file, without the diff markers."""
    out = git("diff", "-U0", f"{base}...HEAD", "--", path)
    return [
        line[1:]
        for line in out.splitlines()
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]


def is_doc_path(path: str) -> bool:
    if path.startswith(CODE_ROOTS):
        return False
    return any(fnmatch.fnmatch(path, pattern) for pattern in DOC_PATTERNS)


def only_version_bump(lines: list[str]) -> bool:
    return all(line.startswith("version = ") for line in lines if line.strip())


def docs_only(paths: list[str], pyproject_lines: list[str]) -> bool:
    """True when nothing in the diff can change what the code does.

    `pyproject.toml` rides along with every pull request for the version bump,
    so it counts as documentation when the bump is all that changed.
    """
    for path in paths:
        if path == "pyproject.toml" and only_version_bump(pyproject_lines):
            continue
        if not is_doc_path(path):
            return False
    return bool(paths)


def surface_findings(
    paths: list[str],
    diff: dict[str, list[str]],
    flags: dict[str, list[str]] | None = None,
) -> list[str]:
    """CLI flags and config keys that changed without their page."""
    touched = set(paths)
    problems = []
    for cli, changed in (flags or {}).items():
        page = CLI_PAGES[cli]
        if changed and page not in touched:
            problems.append(f"{cli} changes {', '.join(changed)}; update {page}")
    config_changed = []
    for path, lines in diff.items():
        if path.endswith(".example"):
            continue
        if fnmatch.fnmatch(path, CONFIG_YAML):
            pattern = YAML_KEY
        elif any(fnmatch.fnmatch(path, glob) for glob in CONFIG_MODELS):
            pattern = MODEL_FIELD
        else:
            continue
        keys = [line for line in lines if pattern.match(line) and not _comment(line)]
        if keys:
            config_changed.append(path)
    if config_changed and CONFIG_PAGE not in touched:
        problems.append(
            f"{', '.join(config_changed)} add, change or remove config keys; "
            f"update {CONFIG_PAGE}"
        )
    return problems


def _comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def cited_findings(text: str, known: set[str]) -> list[str]:
    missing = sorted(set(REQ_ID.findall(text)) - known)
    if not missing:
        return []
    return [f"the pull request cites requirements that don't exist: {missing}"]


def pr_findings(base: str, body: str) -> list[str]:
    paths = changed_paths(base)
    messages = git("log", "--format=%B", f"{base}..HEAD")
    text = f"{body}\n{messages}"
    known = {r.req_id for r in requirements()}
    problems = cited_findings(text, known)
    watched = [
        p
        for p in paths
        if fnmatch.fnmatch(p, CONFIG_YAML)
        or any(fnmatch.fnmatch(p, glob) for glob in CONFIG_MODELS)
    ]
    diff = {p: changed_lines(base, p) for p in watched}
    # The fork point, as `base...HEAD` uses: a flag main added since must not
    # count against this branch.
    fork = git("merge-base", base, "HEAD").strip()
    flags = {
        cli: flag_changes(file_at(fork, cli), file_at("HEAD", cli))
        for cli in CLI_PAGES
        if cli in paths
    }
    surface = surface_findings(paths, diff, flags)
    if surface and not OPT_OUT.search(text):
        problems += surface
        problems.append(
            "if the change needs no doc update, say `Docs: none` and why in the "
            "pull request description"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="git ref the pull request merges into")
    parser.add_argument(
        "--docs-only",
        action="store_true",
        help="print whether the diff against --base touches only docs",
    )
    parser.add_argument(
        "--body-file", help="pull request description (default: $PR_BODY)"
    )
    args = parser.parse_args(argv)

    if args.docs_only:
        if not args.base:
            parser.error("--docs-only needs --base")
        paths = changed_paths(args.base)
        print(
            "true"
            if docs_only(paths, changed_lines(args.base, "pyproject.toml"))
            else "false"
        )
        return 0

    problems = design_findings()
    if args.base:
        body = (
            Path(args.body_file).read_text(encoding="utf-8")
            if args.body_file
            else os.environ.get("PR_BODY", "")
        )
        problems += pr_findings(args.base, body)

    for problem in problems:
        print(f"docs check: {problem}", file=sys.stderr)
    if problems:
        print(
            "docs check: the table of what each change updates is in "
            "CONTRIBUTING.md, Definition of done",
            file=sys.stderr,
        )
        return 1
    print("docs check: in order")
    return 0


if __name__ == "__main__":
    sys.exit(main())
