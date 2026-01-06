Run linters and fix all issues. Don't stop until all issues are fixed.

## Process
1. Read `docs/linting.md` for project linting guidelines
2. Run quick check first: `make quick-check`
3. Fix auto-fixable issues: `make lint-fix`
4. Manually fix remaining issues
5. Verify all passes: `make lint`

## Commands
```bash
make quick-check   # Fast: Ruff + MyPy only
make lint          # Full: All linting tools
make lint-fix      # Auto-fix what's possible
make format        # Format code with Ruff
make security      # Security scans (Bandit, Safety)
make full-check    # Everything: lint + security + tests
```

## Auto-Fix Workflow
```bash
poetry run ruff format .       # Fix formatting
poetry run ruff check --fix .  # Fix linting violations
make lint                      # Verify all passes
```

## Guidelines
- Fix issues in order: formatting → linting → type errors → security
- Use `# noqa: RULE` sparingly and document why
- For type errors, prefer fixing over `# type: ignore`
- Read `docs/linting.md` for tool-specific configurations

## Checklist
- [ ] `make quick-check` passes (Ruff + MyPy)
- [ ] `make lint` passes (all tools)
- [ ] `make security` passes (Bandit, Safety)
- [ ] No new `# noqa` or `# type: ignore` without justification
