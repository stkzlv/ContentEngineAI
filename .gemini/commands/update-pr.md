# Update PR Playbook

Review all changes in the current branch compared to main and prepare a Pull Request.

## Process
1. Run `git diff main...HEAD` to analyze ALL branch changes.
2. Run `git log main..HEAD --oneline` to see commit history.
3. Check for existing PR using `list_pull_requests`.
4. Create PR with `create_pull_request` or update manually if needed.

## PR Description Format
```markdown
## Summary
Brief description of what changed and why (2-3 sentences max).

## Changes
- Key change 1
- Key change 2

## Testing
How to verify these changes work.
```

## Rules
- **Never mention Gemini, Claude, or AI tools.**
- Keep description short and simple.
- Explain what and why, not how.
- Reference issues if applicable (e.g., "Fixes #123").
- Use conventional commit prefix in title (`feat:`, `fix:`, `docs:`, etc.).

## PR Title Format
`<type>: <short summary>` (under 50 chars)

Examples:
- `feat: Add retry utilities for network resilience`
- `fix: Resolve subtitle positioning issue`
- `docs: Update configuration guide`
