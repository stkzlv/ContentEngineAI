Review and update project tests for quality and coverage.

## Process
1. Read `docs/testing.md` for testing guidelines
2. Run tests with coverage: `make test-cov`
3. Review coverage report for gaps
4. Identify and remove outdated tests
5. Verify tests against current codebase
6. Add tests for uncovered critical paths

## Commands
```bash
make test              # Run all tests
make test-cov          # Run with coverage report
make test-parallel     # Run tests in parallel (faster)
poetry run pytest tests/path/to/test.py  # Run specific test
poetry run pytest -k "test_name"         # Run by name pattern
```

## Coverage Report
- HTML report: `outputs/coverage/index.html`
- Terminal: Shows missing lines with `--cov-report=term-missing`
- Read `docs/testing.md` for current coverage targets

## Discover Test Structure
```bash
# View test directory layout
tree tests/ -L 2 -d

# List all test files
find tests/ -name "test_*.py" | head -20

# Count tests per directory
find tests/ -name "test_*.py" -exec dirname {} \; | sort | uniq -c
```

## Best Practices
- Follow AAA pattern: Arrange, Act, Assert
- Keep tests isolated and independent
- Use fixtures for shared setup
- Mock external dependencies (APIs, databases)
- Mirror source structure in tests/
- Name tests descriptively: `test_<function>_<scenario>_<expected>`

## Review Checklist
- [ ] Remove tests for deleted code
- [ ] Update tests for modified functions
- [ ] Add tests for new functionality
- [ ] Check for flaky tests (random failures)
- [ ] Verify mocks match current interfaces
