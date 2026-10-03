# 0006. Research lives with what it justifies; contributor docs stay at the top of docs/

- **Status:** Accepted
- **Date:** 2026-10-03
- **Amends:** [0001](0001-documentation-structure.md)

## Context and problem

Decision 0001 sketched a `research/` folder for evidence pages and a `contributing/` folder for contributor docs, and planned to split the user guides only as each was next edited. Carrying it out showed three of those parts to be wrong for this repository:

- A standalone research page goes stale unseen: nothing re-reads it when the design it justifies changes, and its dated platform claims outlive their truth.
- Release tooling and contributor workflows look for `docs/versioning.md` and `docs/development.md` at those exact paths.
- Splitting the guides in passes left the docs half in one shape and half in another for longer than doing it at once.

## Options considered

- **Keep `research/`** as sketched. Evidence stays in one place, but away from the decisions it supports.
- **Fold each finding into what it justifies**: the design doc's Context, an explanation page, or a decision record's Background.
- **Move contributor docs to `contributing/`** and update the tooling that looks for them.
- **Leave contributor docs at the top of `docs/`.**

## Decision

Fold research into the docs that use it: a finding that justifies a design goes in that design doc's Context with its evidence grade and source; craft findings the pipeline implements go in the matching explanation page; the documentation-practices research is decision 0001's Background. Findings with no home in the public repo are kept outside it.

Contributor docs (`development.md`, `linting.md`, `testing.md`, `versioning.md`) stay at the top of `docs/`.

The module guides were split in one pass rather than as each was next edited.

## Consequences

- There is no `research/` folder; the documentation map lists no such layer.
- Changing a design re-reads its evidence, because the evidence is in the same file.
- `docs/promotional-video-best-practices.md` stays as a forwarding page while the script prompts that cite it are frozen for the reach test.
