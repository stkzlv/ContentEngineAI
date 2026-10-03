# ContentEngineAI: Claude Code instructions

The shared instructions for every coding agent are in `AGENTS.md`, imported here:

@AGENTS.md

## Claude Code only

- **After a compaction**, invoke the github-workflow skill before the next branch, commit, PR, merge or release, and check CI on any open PR: the summary does not carry the repository's process rules, so re-read them rather than recall them.
- **MCP servers**: Context7 for current library docs (resolve the library id first); the GitHub server for issues and PRs, with the `gh` CLI as the fallback. `gh pr edit` fails on this repository with a `projectCards` GraphQL error: update a PR body through the API or the MCP tool instead.
