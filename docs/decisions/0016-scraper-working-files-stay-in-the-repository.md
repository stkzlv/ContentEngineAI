# 0016. The scraper's working files stay in the repository's outputs

- **Status:** Accepted
- **Date:** 2026-10-13

## Context and problem

With `--outputs-dir`, `--output-dir` or `OUTPUTS_DIR`, scraped products, renders and run state go to the named outputs root (REQ-OPS-031, REQ-OPS-115). The scraper's own working files do not: the Botasaurus browser output, cache and temp files and the debug analysis files resolve from `global_settings.output_config.base_directory` or the default outputs root, and the browser configuration that names one of them is built once, at import, before any root is known (#742). In use they hold tens of kilobytes of browser cache.

## Options considered

- **Move every working directory onto the run's root.** Needs the browser configuration rebuilt per run and every helper that resolves these paths taught the root, for files no later phase reads.
- **Keep them in the repository's `outputs/` and say so.** No code change; the requirement names the exception.

## Decision

Keep them. They are scratch for the scraper itself, nothing downstream reads them, and the size is negligible, so moving them buys nothing that justifies rebuilding the import-time browser configuration.

## Consequences

- REQ-OPS-031 names the scraper's working files as the exception, and `--clean` clears `cache/`, `temp/` and `screenshots/` only under the resolved root, which is the repository's `outputs/` unless a root is set.
- An install that puts its outputs root on another disk still gets a small cache in the repository tree.
- If a later change makes these files large or read by another phase, this is revisited.
