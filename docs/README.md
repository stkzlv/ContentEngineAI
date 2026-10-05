# Documentation map

Each folder holds one kind of document, with its own reader and its own update rule. Read this page to find where something belongs before you add or move a file.

## Layers

| Folder | Question it answers | Update rule |
|---|---|---|
| `roadmap.md` | Where is the project going? | Themes and Now / Next / Later horizons, a "done when" per item, and links to issues and design docs. Specification detail goes in a design doc, not here. |
| `requirements/` | What must be true? | One file per area. Each requirement has a stable id, one testable statement, acceptance criteria and a status. Kept current with the code. |
| `design/` | How will a feature work, and why this way? | One numbered file per feature, with a status header. Frozen once the feature ships; a later change gets a new design doc that supersedes it. |
| `decisions/` | What was chosen, and what was rejected? | One numbered record per decision. Append-only: a later record supersedes an earlier one, and neither is deleted. |
| `architecture.md` | How does the system fit together? | Kept current with the code. |
| `notes/` | What broke in a module, and what catches it now? | One file per module, an entry per defect. Read the file before you change the module. |
| `guides/` | How do I do a task? | How-to pages for users and operators. |
| `reference/` | What exactly does this flag or key do? | Configuration, CLI and API reference. |
| `explanation/` | Why does the system behave this way? | Background on concepts and defaults that a guide or reference page would only state, with the evidence and sources behind them. |

The contributor docs stay at the top of `docs/`: `development.md`, `linting.md`, `testing.md` and `versioning.md`. Release tooling and contributor workflows look for `docs/versioning.md` and `docs/development.md` at those paths.

Each module has a guide (`guides/publishing.md`, `producing-videos.md`, `scraping.md`), a reference page (`reference/publisher.md`, `video-producer.md`, `scraper.md`, and `global-batch.md` for the batch pipeline) and an explanation page (`explanation/publishing.md`, `video-pipeline.md`, `scraping.md`). `promotional-video-best-practices.md` is a forwarding page kept for the script prompts that cite it.

## From idea to shipped feature

1. **Open an issue.** Most work is specified in the issue alone.
2. **Write a design doc** only when the work takes more than a few days, adds an external dependency, or changes config or a data schema.
3. **Record a decision** when a choice between real alternatives would otherwise be argued again. Use `decisions/0000-template.md`.
4. **Add or change the requirement** in the same pull request as the code, and cite its id in the pull request.
5. **When the feature ships**, the same pull request sets its design doc to `Implemented` and its requirements to `shipped`.

The maintainer accepts design docs and decision records. Proposals are discussed in their issue or pull request, in public.

## Statuses

Requirements use `shipped`, `partial`, `planned #N`, `planned (decision NNNN)`, `held` or `deprecated`; [the requirements index](requirements/README.md) defines each.

Design docs use `Draft`, `Accepted`, `Held` (built, switched off), `Implemented` or `Superseded by NNNN`. Decision records use `Accepted`, `Amended by NNNN` (still in force, with a later record changing part of it) or `Superseded by NNNN`.

A feature that changes rendered output ships off by default ([decision 0002](decisions/0002-output-changes-ship-off-by-default.md)). Its design doc's rollout section says what turns it on and when the switch can be removed.

## Private overlays

A gitignored `<name>.private.md` next to a public page carries the motivation the public page leaves out. Keep the two aligned item by item; [AGENTS.md](../AGENTS.md) has the rules.
