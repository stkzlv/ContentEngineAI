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
	build package docs release-prep update-deps clean-all clean-outputs docker-build docker-run \
	install-botasaurus validate-migration rollback-migration \
	scrape-test scrape-advanced produce-video migration-status

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
	@echo "  test-parallel - Run tests in parallel"
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
	@echo "$(YELLOW)Scraping Commands:$(NC)"
	@echo "  scrape-test        - Run test scrape with Botasaurus"
	@echo "  scrape-advanced    - Run advanced search scrape"
	@echo "  produce-video      - Generate video from scraped data"
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

test-parallel:
	@echo "$(BLUE)Running tests in parallel...$(NC)"
	@poetry run python -c "import pytest_xdist" 2>/dev/null || { echo "$(RED)Error: pytest-xdist plugin not installed. Run 'poetry install' or use 'make test'$(NC)"; exit 1; }
	poetry run pytest -n auto
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

# Video production commands
produce-video: ## Run video producer on scraped data
	@echo "$(BLUE)Running video producer...$(NC)"
	@if [ -d "outputs" ] && [ -n "$$(find outputs -name 'data.json' -path '*/output/data.json' 2>/dev/null)" ]; then \
		poetry run python -m src.video.producer $$(find outputs -name 'data.json' -path '*/output/data.json' | head -1) slideshow_images1 --debug; \
	else \
		echo "$(RED)No scraped data found. Run 'make scrape-test' first.$(NC)"; \
	fi
