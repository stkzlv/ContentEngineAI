# Makefile for ContentEngineAI project
# Provides convenient commands for development tasks

# Configuration
PYTHON_VERSION := 3.12
POETRY_VERSION := 1.7.0
PARALLEL_JOBS := $(shell nproc 2>/dev/null || echo 4)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.PHONY: help install install-dev lint lint-fix lint-verbose lint-no-parallel lint-tool lint-list lint-report format type-check security test test-cov clean \
	validate-env dev-setup quick-check full-check ruff ruff-fix bandit vulture safety \
	build package docs release-prep update-deps clean-all clean-outputs docker-build docker-run perf-trends perf-detailed perf-compare \
	install-botasaurus validate-migration rollback-migration \
	scrape-test scrape-advanced produce-video migration-status \
	batch batch-lowpri scrape-lowpri scrape-watch produce-lowpri publish publish-lowpri analytics \
	test-parallel test-lowpri \
	print-python install-analytics-timer uninstall-analytics-timer analytics-timer-status

# Default target
help:
	@echo "$(BLUE)ContentEngineAI Development Commands$(NC)"
	@echo "$(BLUE)====================================$(NC)"
	@echo ""
	@echo "$(GREEN)Environment Setup:$(NC)"
	@echo "  validate-env  - Validate development environment"
	@echo "  install       - Install production dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  dev-setup     - Complete development environment setup"
	@echo ""
	@echo "$(GREEN)Linting and Formatting:$(NC)"
	@echo "  lint          - Run all linting checks (with parallel execution)"
	@echo "  lint-fix      - Run linting with automatic fixes"
	@echo "  lint-verbose  - Run linting with detailed output"
	@echo "  lint-no-parallel - Run linting sequentially (for debugging)"
	@echo "  lint-tool     - Run specific linting tool (use TOOL=name)"
	@echo "  lint-list     - List available linting tools"
	@echo "  lint-report   - Generate detailed linting report (JSON)"
	@echo "  format        - Format code with Ruff"
	@echo "  type-check    - Run MyPy type checking"
	@echo "  security      - Run security checks (Bandit + Safety)"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  test          - Run tests"
	@echo "  test-cov      - Run tests with coverage report"
	@echo "  test-parallel - Run tests in parallel (PYTEST_WORKERS=N to bound)"
	@echo "  test-lowpri   - Run tests under a memory cap and low priority"
	@echo ""
	@echo "$(GREEN)Build and Package:$(NC)"
	@echo "  build         - Build the package"
	@echo "  package       - Create distribution packages"
	@echo "  docs          - Generate documentation"
	@echo ""
	@echo "$(GREEN)Development Workflow:$(NC)"
	@echo "  quick-check   - Run essential checks (ruff + type-check)"
	@echo "  full-check    - Run all checks (lint + security + test-cov)"
	@echo "  update-deps   - Update dependencies"
	@echo "  release-prep  - Prepare for release"
	@echo ""
	@echo "$(GREEN)Utilities:$(NC)"
	@echo "  clean         - Clean up cache and temporary files"
	@echo "  clean-all     - Deep clean (including dependencies)"
	@echo "  clean-outputs - Clean up unexpected files in outputs directory"
	@echo "  perf-report   - Generate performance monitoring report"
	@echo "  perf-trends   - Generate performance trends report"
	@echo "  perf-detailed - Generate detailed performance report"
	@echo "  perf-compare  - Compare performance across profiles"
	@echo "  pre-commit    - Install pre-commit hooks"
	@echo ""
	@echo "$(GREEN)Individual Tools:$(NC)"
	@echo "  ruff          - Run Ruff linter"
	@echo "  ruff-fix      - Run Ruff with fixes"
	@echo "  bandit        - Run security scanner"
	@echo "  vulture       - Run dead code detector"
	@echo "  safety        - Check dependency vulnerabilities"
	@echo ""
	@echo "$(YELLOW)Botasaurus Migration:$(NC)"
	@echo "  install-botasaurus - Install Botasaurus dependencies"
	@echo "  validate-migration - Validate migration is working"
	@echo "  rollback-migration - Emergency rollback to pre-migration"
	@echo "  migration-status   - Show migration status"
	@echo ""
	@echo "$(GREEN)Batch Pipeline:$(NC)"
	@echo "  batch         - Run global batch pipeline (ARGS=\"--keywords foo\")"
	@echo "  batch-lowpri  - Same but with reduced CPU/IO/memory priority"
	@echo "                  Defaults MEM_LIMIT=6G NICE_LEVEL=15; tighten with MEM_LIMIT=4G NICE_LEVEL=19"
	@echo ""
	@echo "$(YELLOW)Scraping Commands:$(NC)"
	@echo "  scrape-test        - Run test scrape with Botasaurus"
	@echo "  scrape-advanced    - Run advanced search scrape"
	@echo "  scrape-lowpri      - Run scraper with reduced priority"
	@echo "  produce-video      - Generate video from scraped data"
	@echo "  produce-lowpri     - Run producer with reduced priority (supports --product-ids)"
	@echo "  publish            - Schedule posts for products (ARGS=\"schedule --debug\")"
	@echo "  publish-lowpri     - Same but with reduced priority"
	@echo "  analytics          - Capture day-N views and durability (size: analytics.limit)"
	@echo "  install-analytics-timer   - Install and start the daily analytics sweep"
	@echo "  uninstall-analytics-timer - Remove it (leaves captured figures alone)"
	@echo "  analytics-timer-status    - When it last ran, when it runs next"
	@echo ""
	@echo "$(YELLOW)Advanced Options:$(NC)"
	@echo "  lint-verbose  - Show detailed linting output"
	@echo "  lint-no-parallel - Disable parallel execution for debugging"
	@echo "  lint-tool TOOL=name - Run specific tool (e.g., TOOL=ruff)"
	@echo "  lint-list     - Show all available linting tools"
	@echo ""
	@echo "$(YELLOW)Parallel Execution:$(NC)"
	@echo "  Use 'make -j$(PARALLEL_JOBS)' for parallel make execution"
	@echo "  Example: make -j$(PARALLEL_JOBS) build test"
	@echo "  Note: Linting tools run in parallel automatically"

# Environment validation
validate-env:
	@echo "$(BLUE)Validating development environment...$(NC)"
	@command -v python3 >/dev/null 2>&1 || { echo "$(RED)Error: python3 is not installed$(NC)"; exit 1; }
	@python3 --version | grep -E "Python $(PYTHON_VERSION)\.[0-9]+" >/dev/null || { echo "$(RED)Error: Python $(PYTHON_VERSION).x is required$(NC)"; exit 1; }
	@command -v poetry >/dev/null 2>&1 || { echo "$(RED)Error: Poetry is not installed$(NC)"; exit 1; }
	@poetry --version >/dev/null 2>&1 || { echo "$(RED)Error: Poetry is not working correctly$(NC)"; exit 1; }
	@echo "$(GREEN)Environment validation passed!$(NC)"

# Installation
install: validate-env
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	@poetry install --only main || { echo "$(RED)Error: Failed to install production dependencies$(NC)"; exit 1; }
	@echo "$(GREEN)Production dependencies installed!$(NC)"

install-dev: validate-env
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	@poetry install || { echo "$(RED)Error: Failed to install development dependencies$(NC)"; exit 1; }
	@echo "$(GREEN)Development dependencies installed!$(NC)"

# Linting and formatting
lint:
	@echo "$(BLUE)Running comprehensive linting with parallel execution...$(NC)"
	@test -f tools/lint.py || { echo "$(RED)Error: tools/lint.py not found$(NC)"; exit 1; }
	@poetry run python tools/lint.py || { echo "$(RED)Linting failed$(NC)"; exit 1; }
	@echo "$(GREEN)Linting completed!$(NC)"

lint-fix:
	@echo "$(BLUE)Running linting with automatic fixes...$(NC)"
	@test -f tools/lint.py || { echo "$(RED)Error: tools/lint.py not found$(NC)"; exit 1; }
	@poetry run python tools/lint.py --fix || { echo "$(RED)Linting with fixes failed$(NC)"; exit 1; }
	@echo "$(GREEN)Linting with fixes completed!$(NC)"

lint-verbose:
	@echo "$(BLUE)Running comprehensive linting with verbose output...$(NC)"
	@test -f tools/lint.py || { echo "$(RED)Error: tools/lint.py not found$(NC)"; exit 1; }
	@poetry run python tools/lint.py --verbose || { echo "$(RED)Verbose linting failed$(NC)"; exit 1; }
	@echo "$(GREEN)Verbose linting completed!$(NC)"

lint-no-parallel:
	@echo "$(BLUE)Running linting sequentially (no parallel execution)...$(NC)"
	@test -f tools/lint.py || { echo "$(RED)Error: tools/lint.py not found$(NC)"; exit 1; }
	@poetry run python tools/lint.py --no-parallel || { echo "$(RED)Sequential linting failed$(NC)"; exit 1; }
	@echo "$(GREEN)Sequential linting completed!$(NC)"

lint-tool:
	@echo "$(BLUE)Running specific linting tool: $(TOOL)...$(NC)"
	@test -f tools/lint.py || { echo "$(RED)Error: tools/lint.py not found$(NC)"; exit 1; }
	@test -n "$(TOOL)" || { echo "$(RED)Error: TOOL variable not set. Use: make lint-tool TOOL=ruff$(NC)"; exit 1; }
	@poetry run python tools/lint.py --tool $(TOOL) || { echo "$(RED)Tool $(TOOL) failed$(NC)"; exit 1; }
	@echo "$(GREEN)Tool $(TOOL) completed!$(NC)"

lint-list:
	@echo "$(BLUE)Available linting tools:$(NC)"
	@poetry run python tools/lint.py --list-tools

lint-report:
	@echo "$(BLUE)Generating comprehensive linting report...$(NC)"
	@test -f tools/lint.py || { echo "$(RED)Error: tools/lint.py not found$(NC)"; exit 1; }
	@mkdir -p outputs/reports
	@poetry run python tools/lint.py --output outputs/reports/lint-report.json --verbose || { echo "$(RED)Report generation failed$(NC)"; exit 1; }
	@echo "$(GREEN)Linting report saved to outputs/reports/lint-report.json$(NC)"

format:
	@echo "$(BLUE)Formatting code...$(NC)"
	poetry run ruff format src/ tests/
	@echo "$(GREEN)Code formatting completed!$(NC)"

type-check:
	@echo "$(BLUE)Running type checking...$(NC)"
	@poetry run mypy src/ || { echo "$(RED)Type checking failed$(NC)"; exit 1; }
	@echo "$(GREEN)Type checking completed!$(NC)"

security:
	@echo "$(BLUE)Running security checks...$(NC)"
	poetry run bandit -r src/ -f json
	poetry run safety check --json
	@echo "$(GREEN)Security checks completed!$(NC)"

# Testing
test:
	@echo "$(BLUE)Running tests...$(NC)"
	@poetry run pytest || { echo "$(RED)Tests failed$(NC)"; exit 1; }
	@echo "$(GREEN)Tests completed!$(NC)"

test-cov:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@poetry run pytest --cov=src --cov-report=html:outputs/coverage --cov-report=term-missing || { echo "$(RED)Tests with coverage failed$(NC)"; exit 1; }
	@echo "$(GREEN)Tests with coverage completed!$(NC)"

# `auto` is one worker per core, which on a 16-core box is 16 uncapped pytest
# processes. Overridable so a developer sharing the machine can bound it:
# `make test-parallel PYTEST_WORKERS=4`.
PYTEST_WORKERS ?= auto

test-parallel:
	@echo "$(BLUE)Running tests in parallel ($(PYTEST_WORKERS) workers)...$(NC)"
	@poetry run python -c "import pytest_xdist" 2>/dev/null || { echo "$(RED)Error: pytest-xdist plugin not installed. Run 'poetry install' or use 'make test'$(NC)"; exit 1; }
	poetry run pytest -n $(PYTEST_WORKERS)
	@echo "$(GREEN)Parallel tests completed!$(NC)"

# Build and package
build: validate-env
	@echo "$(BLUE)Building package...$(NC)"
	@poetry build || { echo "$(RED)Package build failed$(NC)"; exit 1; }
	@echo "$(GREEN)Package built successfully!$(NC)"

package: build
	@echo "$(BLUE)Creating distribution packages...$(NC)"
	@echo "$(GREEN)Distribution packages created in dist/$(NC)"

docs:
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "$(YELLOW)Documentation generation not yet implemented$(NC)"
	@echo "$(GREEN)Documentation placeholder completed!$(NC)"

# Utilities
clean:
	@echo "$(BLUE)Cleaning up cache and temporary files...$(NC)"
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf outputs/coverage/
	rm -rf dist/
	rm -rf build/
	find . -type d -name __pycache__ -print0 | xargs -0 rm -rf 2>/dev/null || true
	find . -type f -name "*.pyc" -print0 | xargs -0 rm -f 2>/dev/null || true
	@echo "$(GREEN)Cleanup completed!$(NC)"

clean-all: clean
	@echo "$(BLUE)Performing deep clean...$(NC)"
	@echo "$(YELLOW)Removing virtual environments...$(NC)"
	@poetry env list --full-path 2>/dev/null | while read env; do \
		echo "Removing: $$env"; \
		poetry env remove "$$env" 2>/dev/null || true; \
	done
	rm -rf .venv/ venv/ 2>/dev/null || true
	@echo "$(GREEN)Deep clean completed!$(NC)"

clean-outputs:
	@echo "$(BLUE)Cleaning up outputs directory...$(NC)"
	@test -f tools/cleanup_outputs.py || { echo "$(RED)Error: tools/cleanup_outputs.py not found$(NC)"; exit 1; }
	@poetry run python tools/cleanup_outputs.py --dry-run
	@echo "$(YELLOW)This was a dry run. To perform actual cleanup, run:$(NC)"
	@echo "$(YELLOW)  poetry run python tools/cleanup_outputs.py$(NC)"
	@echo "$(GREEN)Outputs cleanup preview completed!$(NC)"

perf-report:
	@echo "$(BLUE)Generating performance monitoring report...$(NC)"
	@test -f tools/performance_report.py || { echo "$(RED)Error: tools/performance_report.py not found$(NC)"; exit 1; }
	@poetry run python tools/performance_report.py --report-type summary
	@echo "$(GREEN)Performance report completed!$(NC)"

perf-trends:
	@echo "$(BLUE)Generating performance trends report...$(NC)"
	@test -f tools/performance_report.py || { echo "$(RED)Error: tools/performance_report.py not found$(NC)"; exit 1; }
	@poetry run python tools/performance_report.py --report-type trends
	@echo "$(GREEN)Trends report completed!$(NC)"

perf-detailed:
	@echo "$(BLUE)Generating detailed performance report...$(NC)"
	@test -f tools/performance_report.py || { echo "$(RED)Error: tools/performance_report.py not found$(NC)"; exit 1; }
	@poetry run python tools/performance_report.py --report-type detailed
	@echo "$(GREEN)Detailed report completed!$(NC)"

perf-compare:
	@echo "$(BLUE)Generating profile comparison report...$(NC)"
	@test -f tools/performance_report.py || { echo "$(RED)Error: tools/performance_report.py not found$(NC)"; exit 1; }
	@poetry run python tools/performance_report.py --report-type comparison
	@echo "$(GREEN)Comparison report completed!$(NC)"

pre-commit:
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	poetry run pre-commit install
	@echo "$(GREEN)Pre-commit hooks installed!$(NC)"

# Individual tool commands
ruff:
	@echo "$(BLUE)Running Ruff linter...$(NC)"
	poetry run ruff check src/ tests/
	@echo "$(GREEN)Ruff check completed!$(NC)"

ruff-fix:
	@echo "$(BLUE)Running Ruff with fixes...$(NC)"
	poetry run ruff check --fix src/ tests/
	@echo "$(GREEN)Ruff fixes completed!$(NC)"

bandit:
	@echo "$(BLUE)Running security scanner...$(NC)"
	poetry run bandit -r src/
	@echo "$(GREEN)Security scan completed!$(NC)"

vulture:
	@echo "$(BLUE)Running dead code detector...$(NC)"
	poetry run vulture src/ --min-confidence 80
	@echo "$(GREEN)Dead code detection completed!$(NC)"

safety:
	@echo "$(BLUE)Checking dependency vulnerabilities...$(NC)"
	poetry run safety check
	@echo "$(GREEN)Dependency vulnerability check completed!$(NC)"

# Development workflow
dev-setup: install-dev pre-commit
	@echo "$(GREEN)Development environment setup complete!$(NC)"
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  1. Copy .env.example to .env and configure"
	@echo "  2. Run 'make quick-check' to verify setup"
	@echo "  3. Run 'make test' to run tests"

quick-check: ruff type-check
	@echo "$(GREEN)Quick checks completed!$(NC)"

full-check: lint security test-cov
	@echo "$(GREEN)Full checks completed!$(NC)"

# Dependency management
update-deps:
	@echo "$(BLUE)Updating dependencies...$(NC)"
	poetry update
	poetry run safety check
	@echo "$(GREEN)Dependencies updated!$(NC)"

# Release preparation
release-prep: clean-all install-dev lint security test-cov build
	@echo "$(GREEN)Release preparation completed!$(NC)"
	@echo "$(BLUE)Ready for release!$(NC)"

# Docker support (if needed)
docker-build:
	@echo "$(BLUE)Building Docker image...$(NC)"
	@echo "$(YELLOW)Docker build not yet implemented$(NC)"

docker-run:
	@echo "$(BLUE)Running Docker container...$(NC)"
	@echo "$(YELLOW)Docker run not yet implemented$(NC)"

# Parallel execution helpers
lint-parallel:
	@echo "$(BLUE)Running linting tools in parallel...$(NC)"
	@$(MAKE) ruff & \
	$(MAKE) bandit & \
	$(MAKE) vulture & \
	$(MAKE) safety & \
	wait
	@echo "$(GREEN)Parallel linting completed!$(NC)"

# CI/CD helpers
ci-setup: validate-env install-dev
	@echo "$(GREEN)CI environment setup completed!$(NC)"

ci-test: test-parallel
	@echo "$(GREEN)CI tests completed!$(NC)"

ci-lint: lint
	@echo "$(GREEN)CI linting completed!$(NC)"

# Development shortcuts
dev: install-dev
	@echo "$(GREEN)Development environment ready!$(NC)"

check: quick-check
	@echo "$(GREEN)Quick check completed!$(NC)"

all: full-check
	@echo "$(GREEN)All checks completed!$(NC)"

# Botasaurus Migration Commands

install-botasaurus: ## Install Botasaurus dependencies for migration
	@echo "$(YELLOW)Installing Botasaurus dependencies...$(NC)"
	poetry add botasaurus botasaurus-requests
	@echo "$(YELLOW)Removing old dependencies...$(NC)"
	poetry remove playwright playwright-stealth tenacity || true
	@echo "$(GREEN)Botasaurus dependencies installed and old dependencies removed$(NC)"

validate-migration: ## Validate Botasaurus migration is working
	@echo "$(YELLOW)Validating Botasaurus migration...$(NC)"
	python scripts/validate_botasaurus_migration.py
	@echo "$(GREEN)Migration validation completed$(NC)"

rollback-migration: ## Emergency rollback to pre-migration state  
	@echo "$(RED)Rolling back to pre-migration state...$(NC)"
	@echo "$(RED)This will reset to commit: 4c9314e7b8ca02e5333934a6934904e65fcd52e9$(NC)"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	git reset --hard 4c9314e7b8ca02e5333934a6934904e65fcd52e9
	git clean -fd
	poetry install
	@echo "$(GREEN)Rollback completed$(NC)"

migration-status: ## Show current migration status
	@echo "$(YELLOW)Botasaurus Migration Status$(NC)"
	@echo "============================"
	@if poetry show botasaurus >/dev/null 2>&1; then echo "✅ Botasaurus dependency installed"; else echo "❌ Botasaurus dependency missing - run 'make install-botasaurus'"; fi
	@if poetry show playwright >/dev/null 2>&1; then echo "⚠️  Old Playwright dependency still present"; else echo "✅ Old Playwright dependency removed"; fi
	@if [ -f "src/scraper/amazon/scraper.py" ]; then echo "✅ Scraper file exists"; else echo "❌ Scraper file missing"; fi
	@echo ""
	@echo "Next steps:"
	@echo "1. Run 'make install-botasaurus' if dependencies not installed"
	@echo "2. Replace src/scraper/amazon/scraper.py with Botasaurus implementation"  
	@echo "3. Run 'make validate-migration' to verify everything works"
	@echo "4. Use 'make rollback-migration' if rollback needed"

# Scraper-specific commands
scrape-test: ## Run scraper with test ASIN (Botasaurus)
	@echo "$(BLUE)Running Botasaurus scraper test...$(NC)"
	poetry run python -m src.scraper.amazon.scraper --keywords "B0BTYCRJSS" --debug --clean

scrape-advanced: ## Run scraper with advanced search parameters
	@echo "$(BLUE)Running advanced scraper test...$(NC)"
	poetry run python -m src.scraper.amazon.scraper \
		--keywords "wireless headphones" --min-price 20 --max-price 100 \
		--min-rating 4 --prime-only --sort price-asc-rank --debug --clean

# Batch pipeline commands
# Resource limits for low-priority mode
NICE_LEVEL := 15
IONICE_CLASS := 2
IONICE_LEVEL := 6
MEM_LIMIT := 6G

# `systemd-run --user --scope` starts the process via the user service manager,
# which does not inherit the caller's PATH / virtualenv, so `poetry run python`
# inside the scope resolves a bare interpreter missing project deps
# (ModuleNotFoundError). Resolve the real interpreter and run it directly inside
# the scope with PATH forwarded. `poetry run python` is unreliable as the probe:
# with virtualenvs.create=false it returns the base interpreter, not the venv.
# Candidates are tried in order and the first that can import a project dep wins.
# `.python-version` comes first because it is the only candidate that names THIS
# project: the other three read the ambient environment, so an unrelated venv
# active in the shell hijacks all of them at once ($$VIRTUAL_ENV and PATH both
# point at it, and poetry reports it too under virtualenvs.create=false), leaving
# no usable candidate even though the project's own interpreter is installed.
# Lazily assigned (=) so it only runs when a *-lowpri recipe needs it.
LOWPRI_PYTHON = $(shell for p in "$$HOME/.pyenv/versions/$$(cat .python-version 2>/dev/null)/bin/python" "$$VIRTUAL_ENV/bin/python" "$$(python3 -c 'import sys;print(sys.executable)' 2>/dev/null)" "$$(poetry env info -p 2>/dev/null)/bin/python"; do [ -x "$$p" ] && "$$p" -c 'import yaml' >/dev/null 2>&1 && { echo "$$p"; break; }; done)

test-lowpri: ## Run the test suite under the lowpri cgroup (ARGS="tests/publisher")
	@command -v ionice >/dev/null 2>&1 || { echo "$(RED)ionice not found (install util-linux)$(NC)"; exit 1; }
	@PY='$(LOWPRI_PYTHON)'; \
	[ -n "$$PY" ] || { echo "$(RED)No project interpreter found (tried .python-version, active venv, python3, poetry env). Run 'poetry install' first.$(NC)"; exit 1; }; \
	if command -v systemd-run >/dev/null 2>&1; then \
		echo "$(BLUE)Running tests with nice=$(NICE_LEVEL), ionice=$(IONICE_CLASS)/$(IONICE_LEVEL), memory cap=$(MEM_LIMIT)$(NC)"; \
		nice -n $(NICE_LEVEL) ionice -c $(IONICE_CLASS) -n $(IONICE_LEVEL) \
			systemd-run --user --scope -p MemoryMax=$(MEM_LIMIT) -p MemorySwapMax=0 \
			env PATH="$$(dirname "$$PY"):$$PATH" "$$PY" -m pytest $(ARGS); \
	else \
		echo "$(YELLOW)systemd-run not available, skipping memory limit$(NC)"; \
		echo "$(BLUE)Running tests with nice=$(NICE_LEVEL), ionice=$(IONICE_CLASS)/$(IONICE_LEVEL)$(NC)"; \
		nice -n $(NICE_LEVEL) ionice -c $(IONICE_CLASS) -n $(IONICE_LEVEL) \
			"$$PY" -m pytest $(ARGS); \
	fi

batch: ## Run global batch pipeline (pass ARGS="--keywords foo --debug")
	poetry run python -m src.pipeline.global_batch $(ARGS)

batch-lowpri: ## Run batch pipeline with reduced CPU/IO/memory priority
	@command -v ionice >/dev/null 2>&1 || { echo "$(RED)ionice not found (install util-linux)$(NC)"; exit 1; }
	@PY='$(LOWPRI_PYTHON)'; \
	[ -n "$$PY" ] || { echo "$(RED)No project interpreter found (tried .python-version, active venv, python3, poetry env). Run 'poetry install' first.$(NC)"; exit 1; }; \
	if command -v systemd-run >/dev/null 2>&1; then \
		echo "$(BLUE)Running with nice=$(NICE_LEVEL), ionice=$(IONICE_CLASS)/$(IONICE_LEVEL), memory cap=$(MEM_LIMIT)$(NC)"; \
		nice -n $(NICE_LEVEL) ionice -c $(IONICE_CLASS) -n $(IONICE_LEVEL) \
			systemd-run --user --scope -p MemoryMax=$(MEM_LIMIT) -p MemorySwapMax=0 \
			env PATH="$$(dirname "$$PY"):$$PATH" "$$PY" -m src.pipeline.global_batch $(ARGS); \
	else \
		echo "$(YELLOW)systemd-run not available, skipping memory limit$(NC)"; \
		echo "$(BLUE)Running with nice=$(NICE_LEVEL), ionice=$(IONICE_CLASS)/$(IONICE_LEVEL)$(NC)"; \
		nice -n $(NICE_LEVEL) ionice -c $(IONICE_CLASS) -n $(IONICE_LEVEL) \
			"$$PY" -m src.pipeline.global_batch $(ARGS); \
	fi

scrape-lowpri: ## Run scraper with reduced CPU/IO/memory priority
	@command -v ionice >/dev/null 2>&1 || { echo "$(RED)ionice not found (install util-linux)$(NC)"; exit 1; }
	@PY='$(LOWPRI_PYTHON)'; \
	[ -n "$$PY" ] || { echo "$(RED)No project interpreter found (tried .python-version, active venv, python3, poetry env). Run 'poetry install' first.$(NC)"; exit 1; }; \
	if command -v systemd-run >/dev/null 2>&1; then \
		echo "$(BLUE)Running scraper with nice=$(NICE_LEVEL), ionice=$(IONICE_CLASS)/$(IONICE_LEVEL), memory cap=$(MEM_LIMIT)$(NC)"; \
		nice -n $(NICE_LEVEL) ionice -c $(IONICE_CLASS) -n $(IONICE_LEVEL) \
			systemd-run --user --scope -p MemoryMax=$(MEM_LIMIT) -p MemorySwapMax=0 \
			env PATH="$$(dirname "$$PY"):$$PATH" "$$PY" -m src.scraper.amazon.scraper $(ARGS); \
	else \
		echo "$(YELLOW)systemd-run not available, skipping memory limit$(NC)"; \
		echo "$(BLUE)Running scraper with nice=$(NICE_LEVEL), ionice=$(IONICE_CLASS)/$(IONICE_LEVEL)$(NC)"; \
		nice -n $(NICE_LEVEL) ionice -c $(IONICE_CLASS) -n $(IONICE_LEVEL) \
			"$$PY" -m src.scraper.amazon.scraper $(ARGS); \
	fi

# Watchable debug scrape. Headful Chrome cannot be driven on a live Wayland
# session (its CDP endpoint freezes), so this runs the browser on a dedicated
# Xvfb display and exposes it over VNC. Connect a viewer to localhost:$(VNC_PORT).
VNC_DISPLAY := :99
VNC_PORT := 5900
VNC_GEOMETRY := 1920x1080x24

scrape-watch: ## Debug scrape on a dedicated Xvfb, watch over VNC (localhost:5900). Pass ARGS="..."
	@command -v Xvfb >/dev/null 2>&1 || { echo "$(RED)Xvfb not found: sudo apt install -y xvfb$(NC)"; exit 1; }
	@command -v x11vnc >/dev/null 2>&1 || { echo "$(RED)x11vnc not found: sudo apt install -y x11vnc$(NC)"; exit 1; }
	@echo "$(BLUE)Starting Xvfb on $(VNC_DISPLAY) ($(VNC_GEOMETRY))$(NC)"
	@Xvfb $(VNC_DISPLAY) -screen 0 $(VNC_GEOMETRY) >/dev/null 2>&1 & echo $$! > /tmp/ceai-xvfb.pid
	@sleep 1
	@# x11vnc refuses to start if it detects a Wayland login session, so WAYLAND_DISPLAY
	@# and XDG_SESSION_TYPE must be truly unset (empty string is not enough for getenv).
	@env -u WAYLAND_DISPLAY -u XDG_SESSION_TYPE x11vnc -display $(VNC_DISPLAY) \
		-localhost -nopw -forever -quiet -bg >/dev/null 2>&1 || true
	@echo "$(GREEN)Watch at vnc://localhost:$(VNC_PORT) (e.g. vncviewer localhost:$(VNC_PORT))$(NC)"
	@DISPLAY=$(VNC_DISPLAY) WAYLAND_DISPLAY= XAUTHORITY= \
		nice -n $(NICE_LEVEL) ionice -c $(IONICE_CLASS) -n $(IONICE_LEVEL) \
		poetry run python -m src.scraper.amazon.scraper $(ARGS) --debug; ret=$$?; \
		echo "$(BLUE)Cleaning up Xvfb/x11vnc$(NC)"; \
		kill `cat /tmp/ceai-xvfb.pid 2>/dev/null` 2>/dev/null || true; \
		pkill -x x11vnc 2>/dev/null || true; \
		rm -f /tmp/ceai-xvfb.pid; \
		exit $$ret

produce-lowpri: ## Run video producer with reduced CPU/IO/memory priority
	@command -v ionice >/dev/null 2>&1 || { echo "$(RED)ionice not found (install util-linux)$(NC)"; exit 1; }
	@PY='$(LOWPRI_PYTHON)'; \
	[ -n "$$PY" ] || { echo "$(RED)No project interpreter found (tried .python-version, active venv, python3, poetry env). Run 'poetry install' first.$(NC)"; exit 1; }; \
	if command -v systemd-run >/dev/null 2>&1; then \
		echo "$(BLUE)Running producer with nice=$(NICE_LEVEL), ionice=$(IONICE_CLASS)/$(IONICE_LEVEL), memory cap=$(MEM_LIMIT)$(NC)"; \
		nice -n $(NICE_LEVEL) ionice -c $(IONICE_CLASS) -n $(IONICE_LEVEL) \
			systemd-run --user --scope -p MemoryMax=$(MEM_LIMIT) -p MemorySwapMax=0 \
			env PATH="$$(dirname "$$PY"):$$PATH" "$$PY" -m src.video.producer $(ARGS); \
	else \
		echo "$(YELLOW)systemd-run not available, skipping memory limit$(NC)"; \
		echo "$(BLUE)Running producer with nice=$(NICE_LEVEL), ionice=$(IONICE_CLASS)/$(IONICE_LEVEL)$(NC)"; \
		nice -n $(NICE_LEVEL) ionice -c $(IONICE_CLASS) -n $(IONICE_LEVEL) \
			"$$PY" -m src.video.producer $(ARGS); \
	fi

publish: ## Schedule posts for products (ARGS="schedule --debug" or ARGS="single B0ASIN1 --debug")
	poetry run python -m src.publisher.late $(ARGS)

analytics: ## Capture per-post day-N and durability figures (size: analytics.limit)
	poetry run python -m src.publisher.late analytics $(ARGS)

# Print the project interpreter. The installed timer needs the same one the
# *-lowpri targets resolve, and a second copy of that candidate list in shell
# would drift from this one.
print-python:
	@echo '$(LOWPRI_PYTHON)'

install-analytics-timer: ## Install and start the daily analytics sweep timer
	@echo "$(BLUE)Installing the analytics timer...$(NC)"
	@./deploy/install-timer.sh || { echo "$(RED)Install failed$(NC)"; exit 1; }

uninstall-analytics-timer: ## Remove the analytics timer (keeps captured figures)
	@./deploy/install-timer.sh --uninstall || { echo "$(RED)Uninstall failed$(NC)"; exit 1; }

analytics-timer-status: ## Show when the sweep last ran and when it runs next
	@systemctl --user list-timers contentengineai-analytics.timer --no-pager || true
	@systemctl --user status contentengineai-analytics.service --no-pager -n 5 || true
	@# Resolve REPO_DIR the way the installer does: environment, then
	@# schedule.env, then the default. The failure handler writes its log under
	@# the REPO_DIR the timer was rendered with, so a bare relative path here
	@# reports "no failures" forever when either points at a second checkout.
	@# Saving the exported value across the source is what keeps the two in
	@# agreement -- sourcing alone would silently outrank the one-off override
	@# the sample documents, and this target would then answer for the wrong
	@# tree while claiming to read what the unit read.
	@set -e; \
	REPO_DIR_ENV="$${REPO_DIR-}"; \
	[ -f deploy/schedule.env ] && . ./deploy/schedule.env; \
	REPO_DIR="$${REPO_DIR_ENV:-$${REPO_DIR:-.}}"; \
	if [ -s "$$REPO_DIR/outputs/logs/analytics-failures.log" ]; then \
		echo "$(RED)Recorded failures ($$REPO_DIR/outputs/logs/analytics-failures.log):$(NC)"; \
		tail -20 "$$REPO_DIR/outputs/logs/analytics-failures.log"; \
	else \
		echo "$(GREEN)No recorded sweep failures.$(NC)"; \
	fi; \
	if [ -f "$$REPO_DIR/outputs/post_metrics.json" ]; then \
		echo "$(BLUE)Figures last written: $$(date -r "$$REPO_DIR/outputs/post_metrics.json" '+%F %T')$(NC)"; \
	else \
		echo "$(YELLOW)$$REPO_DIR/outputs/post_metrics.json does not exist yet.$(NC)"; \
	fi

publish-lowpri: ## Schedule posts with reduced CPU/IO/memory priority
	@command -v ionice >/dev/null 2>&1 || { echo "$(RED)ionice not found (install util-linux)$(NC)"; exit 1; }
	@PY='$(LOWPRI_PYTHON)'; \
	[ -n "$$PY" ] || { echo "$(RED)No project interpreter found (tried .python-version, active venv, python3, poetry env). Run 'poetry install' first.$(NC)"; exit 1; }; \
	if command -v systemd-run >/dev/null 2>&1; then \
		echo "$(BLUE)Running publisher with nice=$(NICE_LEVEL), ionice=$(IONICE_CLASS)/$(IONICE_LEVEL), memory cap=$(MEM_LIMIT)$(NC)"; \
		nice -n $(NICE_LEVEL) ionice -c $(IONICE_CLASS) -n $(IONICE_LEVEL) \
			systemd-run --user --scope -p MemoryMax=$(MEM_LIMIT) -p MemorySwapMax=0 \
			env PATH="$$(dirname "$$PY"):$$PATH" "$$PY" -m src.publisher.late $(ARGS); \
	else \
		echo "$(YELLOW)systemd-run not available, skipping memory limit$(NC)"; \
		echo "$(BLUE)Running publisher with nice=$(NICE_LEVEL), ionice=$(IONICE_CLASS)/$(IONICE_LEVEL)$(NC)"; \
		nice -n $(NICE_LEVEL) ionice -c $(IONICE_CLASS) -n $(IONICE_LEVEL) \
			"$$PY" -m src.publisher.late $(ARGS); \
	fi

# Video production commands
produce-video: ## Run video producer on scraped data
	@echo "$(BLUE)Running video producer...$(NC)"
	@if [ -d "outputs" ] && [ -n "$$(find outputs -name 'data.json' -path '*/output/data.json' 2>/dev/null)" ]; then \
		poetry run python -m src.video.producer $$(find outputs -name 'data.json' -path '*/output/data.json' | head -1) slideshow_images1 --debug; \
	else \
		echo "$(RED)No scraped data found. Run 'make scrape-test' first.$(NC)"; \
	fi
