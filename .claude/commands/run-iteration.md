Start a new development iteration with a feature branch.

## Arguments
- `$ARGUMENTS` - Main objective for the iteration

## Process
1. Read `CONTRIBUTING.md` and `docs/versioning.md` for guidelines
2. Pull latest main: `git checkout main && git pull origin main`
3. Create feature branch with descriptive name
4. Push branch: `git push -u origin <branch-name>`
5. Confirm ready to begin implementation

## Branch Naming Conventions
Based on `CONTRIBUTING.md`:
- `feature/` - New features (e.g., `feature/retry-utilities`)
- `bugfix/` - Bug fixes (e.g., `bugfix/subtitle-positioning`)
- `hotfix/` - Critical fixes (e.g., `hotfix/api-key-leak`)
- `docs/` - Documentation updates (e.g., `docs/configuration-guide`)

## Branch Name Guidelines
- Use lowercase with hyphens
- Keep names short but descriptive
- Derive from objective (e.g., "Add user auth" → `feature/user-authentication`)

## Example
```
Objective: "Add retry logic for network calls"
Branch: feature/retry-network-logic

git checkout main
git pull origin main
git checkout -b feature/retry-network-logic
git push -u origin feature/retry-network-logic
```

## After Branch Creation
- Summarize the objective
- Confirm branch is ready
- Ask if user wants to proceed with implementation
