Review all uncommitted changes in the current branch.

## Process
1. Run `git status` and `git diff --stat` to review changes
2. Stage appropriate files with `git add`
3. Create commit following conventions below
4. Push to remote with `git push`

## Commit Message Format
Use conventional commits: `<type>: <description>`

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `ci`, `build`

## Rules
- Use imperative mood ("Add feature" not "Added feature")
- Subject line under 50 characters
- No period at end of subject
- Never mention authors or AI tools
- Keep it short and simple
- Describe what and why, not how

## Multi-line Format (for larger changes)
```
<type>: <short summary>

- Bullet point details
- Another change
```
