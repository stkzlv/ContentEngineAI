# ContentEngineAI: Claude Code instructions

The shared instructions for every coding agent are in `AGENTS.md`, imported here:

@AGENTS.md

## Claude Code only

- **After a compaction**, re-read `CONTRIBUTING.md` and `docs/versioning.md` before the next branch, commit, PR, merge or release, and check CI on any open PR: the summary does not carry the repository's process rules.
- `gh pr edit` fails on this repository with a `projectCards` GraphQL error: update a PR body through the REST API (`gh api -X PATCH repos/<owner>/<repo>/pulls/<n> -F body=@file`).
