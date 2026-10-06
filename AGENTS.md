# Agent instructions: ContentEngineAI

Instructions for any coding agent working in this repository. Humans start at [CONTRIBUTING.md](CONTRIBUTING.md); the rules are the same.

ContentEngineAI is a video production pipeline: it scrapes product data, writes a script, voices it, renders a vertical video with captions, and publishes it. Where each kind of document lives is in [the documentation map](docs/README.md); read it before adding or moving a doc.

## Before you start

- **Python environment.** `pyenv version && python3 --version` should report the `ContentEngineAI` virtualenv and Python 3.12; `.python-version` activates it. A foreign `VIRTUAL_ENV` from another project hijacks `PATH` and Poetry (`poetry.toml` sets `virtualenvs.create=false`), and `poetry run` then fails. Deactivate it, or run the gates through `~/.pyenv/versions/ContentEngineAI/bin/<tool>`.
- **Private overlays.** Files matching `*.private.md` and anything under `.business/` are gitignored contributor overlays. Read the ones relevant to the task (`find . -name '*.private.md' -not -path './.venv/*' -not -path './outputs/*'`): they explain why a public item exists. A pair stays aligned item by item; the public side describes the capability, the private side the motivation. Nothing from a private file reaches the public tree: no quotes, account handles, figures, persona names, and no mention of the file in commits, PRs, issues or code. Public YAML ships generic defaults.
- **Module notes.** Read the notes file for a module before changing it; most entries exist because the obvious change was already tried and broke something.

| Module | Notes |
|---|---|
| Render pipeline (producer, assembler, prompts, timeouts, state) | [video.md](docs/notes/video.md) |
| Subtitles (the pycaps engine, its fallbacks and templates) | [subtitles.md](docs/notes/subtitles.md) |
| Publishing (provider SDK, disclosures, scheduling, tracking, analytics) | [publisher.md](docs/notes/publisher.md) |
| Scraping (config, throttling, extraction, the browser stack) | [scraper.md](docs/notes/scraper.md) |
| Audio providers | [audio.md](docs/notes/audio.md) |
| Link-in-bio | [link-in-bio.md](docs/notes/link-in-bio.md) |
| CI, gates and dependency pinning | [ci-and-dependencies.md](docs/notes/ci-and-dependencies.md) |
| Where the batch and the standalone modules have drifted | [batch-alignment.md](docs/notes/batch-alignment.md) |
| The worked end-to-end cases | [end-to-end-checks.md](docs/notes/end-to-end-checks.md) |

## Running things

Scraping and rendering are heavy: a render peaks around 2-2.5 GB and runs for minutes, and running one bare while the machine is in use gets unrelated apps killed by the out-of-memory guard.

- Full scrapes, renders and batch runs go through the low-priority targets: `make scrape-lowpri`, `make produce-lowpri`, `make batch-lowpri`, `make publish-lowpri`, with arguments in `ARGS="..."`. They run under a memory cap (`MEM_LIMIT`, default 6G; don't lower it for a render) with `MemorySwapMax=0` and `nice`, make the run the out-of-memory victim rather than another app, and refuse to run without `systemd-run` unless `ALLOW_UNCAPPED=1` is set. Each render first waits for free memory (`memory_guard`); `ALLOW_LOW_MEMORY=1` skips that. An ad-hoc heavy command uses the same scope: `make -s print-lowpri-scope` prints it.
- The full test suite goes through `make test-lowpri`; on a busy machine bound the workers, `make test-lowpri ARGS="-n 4"`.
- Bare `python -m ...` runs are for targeted tests, `--step` debug runs, `--dry-run`, and help or config-loading calls.
- The lowpri recipes deliberately don't use `poetry run`: inside `systemd-run --scope` it resolves an interpreter without the project's dependencies. Don't "simplify" them back.
- How-to commands for each module are in [docs/guides/](docs/guides/); flags and keys in [docs/reference/](docs/reference/).

**Logs** are in `outputs/logs/`, one file per component per day (`global_pipeline-`, `scraper-`, `producer-`, `publisher-YYYY-MM-DD.log`), kept 45 days. Every line carries the run id and the product id, so `grep ' B0ASIN ' outputs/logs/producer-*.log` returns one product's story. A new per-product loop binds the id with `log_context(product_id=...)`.

## Code standards

- snake_case functions, PascalCase classes, UPPER_CASE constants; modern typing (`dict[str, Any]`, `X | None`); 88-character lines.
- Specific exceptions, never bare `except Exception`.
- **Logging:** lazy format (`logger.debug("msg: %s", val)`), never f-strings (ruff's `G`/`LOG` groups enforce it). A literal `%` is `%%` only in a message that takes arguments; `tests/utils/test_lazy_logging.py` counts placeholders against arguments. No emojis, no separator-only lines, no message opening with a newline. Third-party loggers stay quiet through `QUIET_LOGGERS`.
- **Configuration** lives in YAML under `config/`, validated by the Pydantic models in `src/video/config/` and `src/scraper/config_models.py`.
- **Secrets** for a render are collected once, by `collect_producer_secrets`, for both the producer CLI and the global batch.
- **Module/batch alignment.** The standalone CLIs (scraper, producer, publisher) and the global batch re-implement the same logic (scheduling, validation, retries, cleanup). When you fix or add behaviour in one, check the other for the same gap. Known drift: [batch-alignment.md](docs/notes/batch-alignment.md).
- **Duplicates from normal operation are tolerated**: a `--force` republish, a knowing re-render, an ASIN with two link-in-bio links ([decision 0005](docs/decisions/0005-duplicates-are-tolerated.md)). Take the action asked for. The guards stay on, though, and a guard that silently stops working is a defect.

## Making a change

The process is in [CONTRIBUTING.md](CONTRIBUTING.md); re-read it before each pull request rather than recalling it. The parts agents most often get wrong:

- **Branch first**, never commit to `main`: `feature/`, `bugfix/`, `hotfix/` or `docs/`.
- **Every pull request is a release**: bump `pyproject.toml` and move the CHANGELOG entries under a dated heading (`date -u +%F`). `make release-check` checks it, and the required `version-check` CI job refuses a PR without it. CI tags and publishes the release after the merge; don't tag by hand. Rules: [docs/versioning.md](docs/versioning.md).
- **Definition of done**: the docs a change must touch are in CONTRIBUTING's "Definition of done" table (requirements, design docs, reference pages, module notes, decisions). Cite the requirement ids in the PR.
- **Output changes ship off.** A change to rendered output, prompts or published captions ships behind a switch that defaults to today's behaviour until the reach test reads out ([decision 0002](docs/decisions/0002-output-changes-ship-off-by-default.md)). The prompt templates under `src/ai/prompts/` stay byte-identical until then.
- **Verify the real path.** After a change to runtime behaviour, run the path it touches and inspect the artifact (the file, the video with `ffprobe` and a frame, the log line, the published post), not only the tests. The change-type table is in [docs/testing.md](docs/testing.md) ("Verifying a change end-to-end"). When the logic exists in both a module CLI and the global batch, verify both.
- **Commit and PR text**: imperative mood, short, plain. Never mention AI tools or assistants and never add `Co-Authored-By` trailers. Track follow-up work as GitHub issues with the `follow-up` label, not as docs files.
- **Gates** before pushing: `ruff check .`, `ruff format --check .`, `mypy .`, the tests, `make test-docs`, `make release-check`, `make check-docs`. Start any line that pipes a gate with `set -o pipefail;`.
