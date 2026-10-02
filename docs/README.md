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
| `contributing/` | How do I work on this repository? | Development setup, linting, testing and versioning. |

A folder is created when its first file moves in. Pages that still sit at the top level of `docs/` belong in these folders:

| Page | Belongs in |
|---|---|
| `installation.md`, `batch-processing.md`, `troubleshooting.md` | `guides/` |
| `configuration.md`, `zernio-client.md`, `lnkbio-api.md` | `reference/` |
| `platform-safe-zones.md`, `pycaps-subtitles.md`, `tts-voice-profiles.md`, `compliance.md` | `explanation/` |
| `publisher.md`, `video-producer.md`, `scraper.md` | split across `guides/`, `reference/` and `explanation/` |
| `development.md`, `linting.md`, `testing.md`, `versioning.md` | `contributing/` |

## From idea to shipped feature

1. **Open an issue.** Most work is specified in the issue alone.
2. **Write a design doc** only when the work takes more than a few days, adds an external dependency, or changes config or a data schema.
3. **Record a decision** when a choice between real alternatives would otherwise be argued again. Use `decisions/0000-template.md`.
4. **Add or change the requirement** in the same pull request as the code, and cite its id in the pull request.

## Statuses

Requirements use `shipped`, `partial`, `planned #N`, `planned (decision NNNN)`, `held` or `deprecated`; [the requirements index](requirements/README.md) defines each.

Design docs use `Draft`, `Accepted`, `Implemented` or `Superseded by NNNN`. Decision records use `Accepted` or `Superseded by NNNN`.

A feature that changes rendered output ships off by default ([decision 0002](decisions/0002-output-changes-ship-off-by-default.md)). Its design doc's rollout section says what turns it on and when the switch can be removed.

## Private overlays

A gitignored `<name>.private.md` next to a public page carries the motivation the public page leaves out. Keep the two aligned item by item; `CLAUDE.md` has the rules.
