Review all changes in the current branch compared to main.

## Process
1. Run formatting: `poetry run ruff format . && poetry run ruff check --fix .`
2. Commit any formatting changes if present
3. Run `git diff main...HEAD` to analyze ALL branch changes (not just recent commits)
4. Run `git log main..HEAD --oneline` to see commit history
5. Check for existing PR using `mcp__github__list_pull_requests`
6. Create PR with `mcp__github__create_pull_request` or update with `mcp__github__update_pull_request`

## PR Description Format
```markdown
## Summary
Brief description of what changed and why (2-3 sentences max)

## Changes
- Key change 1
- Key change 2

## Testing
How to verify these changes work
```

## Rules
- **CRITICAL: NEVER mention authors, Claude Code, AI tools, assistants, or include `Co-Authored-By` in any form**
- Keep description short and simple
- Explain what and why, not how
- Reference issues if applicable (e.g., "Fixes #123")
- Use conventional commit prefix in title (`feat:`, `fix:`, `docs:`, etc.)

## PR Title Format
`<type>: <short summary>` (under 50 chars)

Examples:
- `feat: Add retry utilities for network resilience`
- `fix: Resolve subtitle positioning issue`
- `docs: Update configuration guide`
