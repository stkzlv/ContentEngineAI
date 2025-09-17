# Contributing to ContentEngineAI

We welcome contributions! This guide will help you get started with contributing to ContentEngineAI.

## Getting Started

### Prerequisites

- Python 3.12+
- Poetry for dependency management
- FFmpeg installed and in your PATH
- Git for version control

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/ContentEngineAI.git
   cd ContentEngineAI
   ```

3. Install development dependencies:
   ```bash
   poetry install
   poetry run playwright install
   ```

4. Install pre-commit hooks:
   ```bash
   make install-dev
   make pre-commit
   ```

## Development Workflow

ContentEngineAI follows **GitHub Flow** - a simple, branch-based workflow perfect for teams and projects that deploy regularly.

### GitHub Flow Process

1. **Create a branch** from `main` for your feature or bug fix:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

   **Branch naming conventions:**
   - `feature/` - New features
   - `bugfix/` - Bug fixes
   - `hotfix/` - Critical fixes
   - `docs/` - Documentation updates

2. **Make your changes** following our code style guidelines

3. **Test thoroughly** - ensure all checks pass:
   ```bash
   make lint
   make test
   make security
   ```

4. **Commit regularly** with clear, descriptive messages:
   ```bash
   git add .
   git commit -m "Add subtitle animation controls"
   ```

5. **Push your branch** and create a Pull Request:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub targeting the `main` branch

7. **Collaborate** - respond to feedback and make additional commits as needed

8. **Merge** - once approved, your PR will be merged via squash merge for clean history

### Code Style Guidelines

We follow strict code quality standards enforced by automated tools:

#### Linting Tools
- **Ruff**: Fast Python linter and formatter (88-char line limit)
- **MyPy**: Type checking with external library compatibility
- **Bandit**: Security scanning
- **Vulture**: Dead code detection (80% confidence)
- **Safety**: Dependency vulnerability checking

#### Quick Commands
```bash
make lint          # Run all linting checks
make lint-fix      # Run linting with automatic fixes
make format        # Code formatting
make typecheck     # Type checking
make security      # Security scanning
```

**📖 Complete linting documentation**: [LINTING.md](LINTING.md)

#### Code Standards
- **Naming Conventions**:
  - Variables/functions: `snake_case`
  - Classes/exceptions: `PascalCase`
  - Constants: `UPPER_CASE`
  - Descriptive, context-rich names

- **Type Annotations**: Use modern Python typing (`dict[str, Any]`, `| None`, union `|` syntax)
- **Error Handling**: Specific exceptions, structured logging, graceful degradation
- **Async Operations**: Use `asyncio.gather()`, semaphores for rate limiting, context managers
- **Documentation**: Comprehensive docstrings for public functions, type hints for all parameters/returns

### Testing

#### Test Structure
- **Tiers**: Unit (>90% coverage), Integration (>80% coverage), End-to-end
- **Framework**: Pytest, 280 test cases across 20 files
- **Features**: Mocked APIs/filesystems, async testing, parallel execution
- **Markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`

#### Running Tests
```bash
make test          # Run all tests
make test-unit     # Run unit tests only
poetry run pytest tests/test_specific.py  # Run specific test file
```

**📖 Detailed testing guide**: [TESTING.md](TESTING.md)

#### Writing Tests
- Write tests before implementation (TDD approach)
- Use descriptive test names that explain the scenario
- Mock external dependencies (APIs, file systems)
- Test both success and failure scenarios
- Include edge cases and boundary conditions

## Submitting Changes

### Pull Request Process

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request against the main repository's `main` branch

3. Ensure your PR:
   - Has a clear title and description
   - References any related issues
   - Passes all CI checks
   - Includes tests for new functionality
   - Updates documentation if needed

### Pull Request Guidelines

- **Title**: Use conventional commit format (`feat:`, `fix:`, `docs:`, etc.)
- **Description**: Explain what changes you made and why
- **Testing**: Describe how you tested your changes
- **Documentation**: Update relevant documentation files

### Code Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged

## Project Architecture

### Key Components
- `src/video/producer.py`: Main pipeline orchestrator
- `src/video/assembler.py`: FFmpeg-based video assembly
- `src/ai/script_generator.py`: LLM-powered script generation
- `src/scraper/`: **Multi-platform e-commerce scraping architecture**
  - `src/scraper/base/`: Platform-agnostic foundation (6 modules)
  - `src/scraper/amazon/`: Amazon implementation (9 specialized modules)
  - `src/scraper/__init__.py`: ScraperFactory & platform registry
- `src/utils/`: Shared utilities and performance optimization

### Configuration
- All settings controlled via `config/video_producer.yaml`
- Pydantic models in `src/video/video_config.py` for validation
- Environment variables for sensitive information

For detailed architecture information, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Common Development Tasks

### Adding New Features

1. **New Pipeline Step**: Extend the pipeline in `producer.py`
2. **New E-Commerce Platform**: Implement `BaseScraper` interface with platform-specific modules
3. **New Configuration**: Add to `video_config.py` Pydantic models
4. **New Provider**: Implement provider interface with fallback support
5. **New Media Source**: Extend media fetching with attribution tracking

#### **Adding New E-Commerce Platform Example:**
```python
from src.scraper.base.models import BaseScraper, BaseProductData
from src.scraper.base.config import register_scraper, Platform

@register_scraper(Platform.EBAY)
class EbayScraper(BaseScraper):
    async def scrape_products(self, keywords: List[str]) -> List[BaseProductData]:
        """eBay-specific implementation"""
        pass
    
    def validate_product_id(self, product_id: str) -> bool:
        """Validate eBay item ID format"""
        return re.match(r'^[0-9]{12}$', product_id) is not None
```

**Architecture Requirements:**
- Extend `BaseScraper` abstract interface
- Auto-register with `@register_scraper` decorator
- Create modular platform-specific package structure
- Add platform configuration in `config/scrapers.yaml`

### Performance Optimization

The project includes comprehensive performance monitoring:
- Use `make perf-report` to analyze performance
- Follow async/await patterns for I/O operations
- Implement caching for expensive operations
- Use connection pooling for HTTP requests

### Debugging

- Use `--debug` flag for detailed logging
- Check `outputs/logs/producer/video_producer.log` for file logs
- Use `--step` to run specific pipeline steps
- Enable debug mode in configuration for intermediate file saving

## Getting Help

- Check existing [Issues](https://github.com/ContentEngineAI/ContentEngineAI/issues)
- Review [Troubleshooting Guide](TROUBLESHOOTING.md)
- Ask questions in issue discussions
- Check the [Development Guide](DEVELOPMENT.md) for detailed technical information

## Code of Conduct

Be respectful and inclusive in all interactions. We want to maintain a welcoming environment for all contributors.

## Recognition

Contributors will be recognized in our changelog and repository. Thank you for helping make ContentEngineAI better!