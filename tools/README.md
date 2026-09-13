# tools

One-off and operational utilities. Everything here is invoked explicitly; nothing is imported by the pipeline.

| Tool | Wired from | Purpose |
|---|---|---|
| `lint.py` | `make lint` | Aggregated lint runner |
| `cleanup_outputs.py` | `make clean-outputs` | Outputs-directory cleanup (dry run by default; `CONFIRM=1` for real) |
| `performance_report.py` | documented commands | Performance monitoring reports |
| `enumerate_topics.py` | `scripts/render-topics-batch.sh` | Topic list expansion for the per-step batch renderer |
| `vulture_whitelist.py` | vulture config | Dead-code scanner whitelist |
| `freesound_oauth2_setup.py` | run by hand, once | Interactive OAuth2 bootstrap for the Freesound provider; writes tokens to `.env` |
