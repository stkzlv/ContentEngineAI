# Linting and Code Quality Configuration

This document describes the linting and code quality tools configured for the ContentEngineAI project.

## Overview

The project uses a comprehensive set of linting tools to ensure code quality, security, and maintainability:

- **Ruff**: Fast Python linter and formatter (replaces flake8, black, isort)
- **MyPy**: Static type checker with third-party library compatibility
- **Bandit**: Security linter
- **Vulture**: Dead code detector
- **Safety**: Dependency vulnerability checker
- **Pre-commit**: Git hooks for automated checks

## Quick Start

### Installation

```bash
# Install all development dependencies
make install-dev

# Install pre-commit hooks
make pre-commit
```

### Running Linting

#### Basic Commands:

```bash
# Run all linting checks (with parallel execution)
make lint

# Run linting with automatic fixes
make lint-fix

# Format code
make format

# Run type checking
make type-check

# Run security checks
make security

# Run tests with coverage
make test-cov
```

#### Advanced Linting Options:

```bash
# Show detailed linting output with timing
make lint-verbose

# Disable parallel execution for debugging
make lint-no-parallel

# Run specific linting tool
make lint-tool TOOL=ruff
make lint-tool TOOL=mypy

# List all available linting tools
make lint-list

# Generate detailed JSON report
make lint-report
```

## Tool Configuration

### Ruff Configuration

Ruff is configured in `pyproject.toml` with the following rules:

- **E**: pycodestyle errors
- **F**: pyflakes (unused imports, undefined variables)
- **W**: pycodestyle warnings
- **I**: isort (import sorting)
- **UP**: pyupgrade (modern Python syntax)
- **B**: flake8-bugbear (common bugs)
- **C4**: flake8-comprehensions (better comprehensions)
- **SIM**: flake8-simplify (code simplification)
- **S**: bandit (security issues)
- **D**: pydocstyle (docstring style) - development-friendly settings
- **N**: pep8-naming (naming conventions) - pragmatic settings

**Disabled Rules:**
- `F401`: Unused imports (handled by isort)
- Most docstring rules (D100-D415) for development efficiency
- Naming convention rules (N802, N803, N806)
- Security rules S101, S603, S607 for common development patterns

### MyPy Configuration

MyPy is configured for development efficiency while maintaining type safety:

**Enabled:**
- `warn_return_any`: Warn about returning Any
- `warn_unused_configs`: Warn about unused config options
- `check_untyped_defs`: Check untyped function definitions
- `warn_unused_ignores`: Warn about unused type ignore comments
- `warn_no_return`: Warn about functions that don't return
- `warn_unreachable`: Warn about unreachable code

**Disabled for Development Efficiency:**
- `disallow_untyped_defs`: Allow functions without type hints
- `disallow_incomplete_defs`: Allow parameters without type hints
- `disallow_untyped_decorators`: Allow decorators without type hints
- `no_implicit_optional`: Allow implicit optional parameters
- `warn_redundant_casts`: Don't warn about redundant type casts
- `strict_equality`: Don't enforce strict equality checks

**Library Overrides:**
- Third-party libraries (botasaurus, coqui_tts, etc.) have error checking disabled
- Specific modules with complex external dependencies ignore type errors
- Test modules have relaxed type checking for mock objects

### Security Tools

**Bandit:**
- Scans for common security issues
- Excludes test directories
- Skips assert_used (B101) and paramiko_calls (B601)

**Safety:**
- Checks dependencies for known vulnerabilities
- Runs automatically in pre-commit hooks

### Dead Code Detection

**Vulture:**
- Detects unused code with 80% confidence threshold
- Excludes common directories and files

## Pre-commit Hooks

The following hooks run automatically on commit:

1. **Ruff**: Linting and formatting
2. **MyPy**: Type checking
3. **Bandit**: Security scanning
4. **Safety**: Dependency vulnerability checking
5. **YAML validation**: Checks configuration files
6. **File hygiene**: End-of-file, trailing whitespace, etc.

## Custom Linting Script

The project includes an **optimized custom linting script** (`tools/lint.py`) with advanced features:

### Key Features:
- **Parallel execution** for independent tools (reduces runtime by ~60%)
- **Smart tool ordering** (fixes run first, tests run last)
- **Proper fix command support** for ruff and ruff-format
- **Enhanced error handling** with timeouts and specific guidance
- **Performance monitoring** with execution timing
- **Environment validation** and dependency checking
- **Detailed progress indicators** and actionable feedback

### Basic Usage:

```bash
# Run all tools with parallel execution
python tools/lint.py

# Run with automatic fixes
python tools/lint.py --fix

# Run specific tool
python tools/lint.py --tool ruff

# Verbose output with timing
python tools/lint.py --verbose

# Save detailed results to JSON
python tools/lint.py --output results.json
```

### Advanced Options:

```bash
# List available tools
python tools/lint.py --list-tools

# Disable parallel execution (for debugging)
python tools/lint.py --no-parallel

# Run specific tool with fixes
python tools/lint.py --tool ruff --fix

# Skip environment validation (for CI)
python tools/lint.py --skip-env-check

# Generate comprehensive report
python tools/lint.py --output reports/lint.json --verbose
```

## IDE Integration

### VS Code

Add to `.vscode/settings.json`:

```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff",
    "python.linting.mypyEnabled": true,
    "python.linting.banditEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### PyCharm

1. Install the Ruff plugin
2. Configure MyPy integration
3. Enable format on save

## Troubleshooting

### Common Issues

1. **MyPy errors with external libraries:**
   - Check the `[tool.mypy.overrides]` section in `pyproject.toml`
   - Add problematic modules to the ignore list

2. **Ruff formatting conflicts:**
   - Run `make format` to fix formatting issues
   - Check that your editor uses the same Ruff version

3. **Pre-commit hook failures:**
   - Run `pre-commit run --all-files` to see all issues
   - Use `pre-commit run --all-files --hook-stage manual` to run manually

### Ignoring Rules

To ignore specific rules in code:

```python
# Ignore specific Ruff rule
some_code  # noqa: E501

# Ignore MyPy error
some_code  # type: ignore

# Ignore Bandit warning
some_code  # nosec

# Ignore security warning for non-cryptographic random usage
delay = random.uniform(min_delay, max_delay)  # noqa: S311
```

## Performance Improvements

The optimized linting system provides significant performance benefits:

- **~60% faster execution** through parallel tool execution
- **Smart scheduling** prevents tool conflicts
- **Early failure detection** stops on critical errors
- **Incremental feedback** shows progress in real-time
- **Timeout protection** prevents hanging processes

### Execution Flow:

1. **Sequential Phase**: Ruff linting → Ruff formatting (if fixing)
2. **Parallel Phase**: MyPy, Bandit, Vulture, Safety (concurrent)
3. **Final Phase**: Pytest (after all linting passes)

## Best Practices

1. **Run linting before committing:**
   ```bash
   make lint-fix
   ```

2. **Use verbose mode for debugging issues:**
   ```bash
   make lint-verbose
   ```

3. **Generate reports for CI/analysis:**
   ```bash
   make lint-report
   ```

4. **Test specific tools during development:**
   ```bash
   make lint-tool TOOL=ruff
   ```

5. **Use type hints when possible:**
   ```python
   def process_data(data: List[str]) -> Dict[str, int]:
       # ...
   ```

6. **Keep functions small and focused:**
   - Aim for functions under 20 lines
   - Single responsibility principle

7. **Write descriptive docstrings:**
   ```python
   def calculate_total(items: List[float]) -> float:
       """Calculate the total sum of all items.

       Args:
           items: List of numeric values

       Returns:
           The sum of all items
       """
   ```

8. **Use meaningful variable names:**
   ```python
   # Good
   user_count = len(users)

   # Bad
   c = len(u)
   ```

## Configuration Files

- `pyproject.toml`: Main configuration for all tools
- `.pre-commit-config.yaml`: Pre-commit hooks configuration
- `pytest.ini`: Test configuration
- `Makefile`: Convenient commands for development
- `tools/lint.py`: **Optimized custom linting script** with parallel execution
- `reports/`: Directory for generated linting reports (created automatically)

## Contributing

When contributing to the project:

1. **Quick validation:** `make lint` (uses parallel execution)
2. **Fix issues automatically:** `make lint-fix`
3. **Run comprehensive checks:** `make full-check`
4. **Generate report for review:** `make lint-report`
5. **Run tests:** `make test`
6. **Update documentation if needed**

### CI/CD Integration:

```bash
# For CI environments
python tools/lint.py --skip-env-check --output ci-report.json

# For detailed analysis
python tools/lint.py --verbose --no-parallel --output detailed-report.json
```

The optimized linting system balances **speed** and **thoroughness**, making it efficient for both local development and CI/CD pipelines while maintaining high code quality standards.
