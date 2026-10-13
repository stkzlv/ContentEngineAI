# 0016. Scraper screenshots stay where the browser driver writes them

- **Status:** Accepted
- **Date:** 2026-10-13

## Context and problem

With `--outputs-dir`, `--output-dir` or `OUTPUTS_DIR`, scraped products, renders and run state go to the named outputs root (REQ-OPS-031, REQ-OPS-115). The scraper's working files did not: the media-download records and the debug image analysis were written under the repository's `outputs/` by two helpers that took no root (#742). Those now take the root the scraper was given. One kind remains: the bot-detection and debug-step screenshots the browser functions save without a path, which the Botasaurus driver puts under `output/screenshots/` in the process's working directory. The debug image-analysis screenshot is saved with a path and follows the root.

## Options considered

- **Pass a path to every screenshot call.** Each of the four call sites builds a filename under the root read from `_effective_dir()`, as the debug analysis does; a call added later without a path still falls back to the driver's default.
- **Leave the screenshots where the driver writes them, and say so.** No change to the browser functions; the requirement names the exception.

## Decision

Leave them. They are written only in a debug run, read by a person and by nothing in the pipeline, so moving them changes nothing any later phase or check depends on.

## Consequences

- REQ-OPS-031 names the screenshots as its exception.
- `--clean` does not remove them: it clears `screenshots/` under the outputs root, which the driver does not write.
- The `output/` directory they land in is gitignored.
