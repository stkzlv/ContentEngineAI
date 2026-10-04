# tools

One-off and operational utilities. Everything here is invoked explicitly; nothing is imported by the pipeline.

| Tool | Wired from | Purpose |
|---|---|---|
| `lint.py` | `make lint` | Aggregated lint runner |
| `cleanup_outputs.py` | `make clean-outputs` | Outputs-directory cleanup (dry run by default; `CONFIRM=1` for real) |
| `performance_report.py` | documented commands | Performance monitoring reports |
| `enumerate_topics.py` | `scripts/render-topics-batch.sh` | Topic list expansion for the per-step batch renderer |
| `vulture_whitelist.py` | `tools/lint.py` (vulture invocation under `make lint`) | Dead-code scanner whitelist |
| `check_docs.py` | `make check-docs`, CI `docs-check` job; `--docs-only` in the CI `test` job | Checks the docs a change must touch moved with it (CONTRIBUTING, Definition of done) and whether a diff touches only docs |
| `merge_was_tested.py` | CI `test` job, on a push to `main` | Says whether the merged tree is the one its pull request's test job already passed, so the suite is not run twice |
| `release_check.py` | `make release-check`, CI `version-check` and `release` jobs | Checks a branch bumps the version and dates its CHANGELOG heading; prints the version for tagging |
| `requirements_coverage.py` | run by hand | Lists the requirement ids in `docs/requirements/` that no test cites |
| `freesound_oauth2_setup.py` | run by hand, once | Interactive OAuth2 bootstrap for the Freesound provider; writes tokens to `.env` |
