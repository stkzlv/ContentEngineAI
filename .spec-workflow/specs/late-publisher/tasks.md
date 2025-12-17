# Tasks Document

## Phase 1: Foundation - Base Interfaces and Models

- [x] 1. Create BasePublisher abstract interface in src/publisher/base.py
  - File: src/publisher/base.py
  - Create abstract base class defining publisher contract
  - Define methods: authenticate(), get_accounts(), upload_media(), publish(), get_status()
  - Purpose: Establish provider-agnostic interface for all publisher implementations
  - _Leverage: src/utils/url_shortener/base.py (provider pattern), src/ai/platform_metadata/base.py (abstract base class pattern)_
  - _Requirements: REQ-1 (Late SDK Integration), Design Component 1_
  - _Prompt: Role: Python Architect specializing in abstract base classes and provider patterns | Task: Create BasePublisher abstract interface in src/publisher/base.py with methods authenticate(), get_accounts(), upload_media(), publish(), get_status() following provider pattern from src/utils/url_shortener/base.py and ABC patterns from src/ai/platform_metadata/base.py | Restrictions: Must use ABC and abstractmethod decorators, include comprehensive docstrings with Args/Returns/Raises, do not implement concrete logic in base class | Success: BasePublisher defines complete contract for publishers, all methods are abstract with clear signatures, docstrings document expected behavior and exceptions_

- [x] 2. Create data models in src/publisher/models.py
  - File: src/publisher/models.py
  - Define PublishResult, PublishMetadata, PublisherConfig, BatchPublishSummary dataclasses
  - Add validation methods and helper functions
  - Purpose: Provide type-safe data structures for publisher operations
  - _Leverage: src/scraper/base/models.py (dataclass patterns), src/video/config/core_models.py (Pydantic models)_
  - _Requirements: Design Data Models section_
  - _Prompt: Role: Python Developer with expertise in dataclasses and type annotations | Task: Create data models in src/publisher/models.py defining PublishResult, PublishMetadata, PublisherConfig, BatchPublishSummary as dataclasses with validation, following patterns from src/scraper/base/models.py and src/video/config/core_models.py | Restrictions: Must use modern Python typing (dict[str, Any], | None), include field validation, add __post_init__ methods for complex validation, use frozen=True for immutable models | Success: All data models defined with proper types, validation logic implemented, models are immutable where appropriate, comprehensive docstrings included_

- [x] 3. Create publisher registry and factory in src/publisher/registry.py
  - File: src/publisher/registry.py
  - Implement PublisherProvider enum and @register_publisher decorator
  - Create PublisherFactory class with get_publisher() method
  - Purpose: Enable dynamic publisher registration and instantiation
  - _Leverage: src/utils/url_shortener/registry.py (registry pattern), src/scraper/__init__.py (factory pattern)_
  - _Requirements: Design Component 6 (Registry)_
  - _Prompt: Role: Python Developer specializing in design patterns and dependency injection | Task: Create publisher registry in src/publisher/registry.py with PublisherProvider enum, @register_publisher decorator, and PublisherFactory.get_publisher() following patterns from src/utils/url_shortener/registry.py and src/scraper/__init__.py | Restrictions: Must support dynamic registration, validate provider exists before instantiation, raise clear errors for unregistered providers, maintain thread-safety | Success: Registry supports decorator-based registration, factory instantiates publishers correctly, clear error messages for missing providers, follows existing registry patterns_

## Phase 2: Late.dev Client Implementation

- [x] 4. Implement LatePublisher client initialization in src/publisher/late/client.py
  - File: src/publisher/late/client.py
  - Create LatePublisher class extending BasePublisher
  - Implement __init__ with Late SDK client initialization
  - Add credential validation and session management
  - Purpose: Provide concrete Late.dev implementation of publisher interface
  - _Leverage: late-sdk package, src/publisher/base.py (BasePublisher), src/ai/description_generator.py (aiohttp session management)_
  - _Requirements: REQ-1 (Late SDK Integration), Design Component 2_
  - _Prompt: Role: Python Developer with expertise in async programming and API client libraries | Task: Create LatePublisher class in src/publisher/late/client.py extending BasePublisher, implementing __init__ to initialize Late SDK client with API key, Vercel token, timeout, and max_retries, following async patterns from src/ai/description_generator.py | Restrictions: Must validate credentials at initialization, raise clear errors for missing API keys, use async context managers, maintain single aiohttp session | Success: LatePublisher initializes Late client correctly, credentials validated at init, session management follows async best practices, clear error messages for configuration issues_

- [x] 5. Implement authentication and account management in src/publisher/late/client.py
  - File: src/publisher/late/client.py (continue from task 4)
  - Implement authenticate() method with credential validation
  - Implement get_accounts() method using client.accounts.list()
  - Add retry logic with exponential backoff
  - Purpose: Enable account discovery and authentication validation
  - _Leverage: late-sdk accounts API, src/publisher/base.py_
  - _Requirements: REQ-2 (Account Discovery and Validation), Design Component 2.2_
  - _Prompt: Role: Backend Developer with expertise in API authentication and error handling | Task: Implement authenticate() and get_accounts() methods in LatePublisher using late-sdk client.accounts.list(), adding exponential backoff retry logic for transient failures (network timeouts, 5xx errors) | Restrictions: Must retry up to 3 times with exponential backoff (2s, 4s, 8s), log authentication failures clearly, return structured account data with platform/username/id, handle 401 errors as permanent failures | Success: authenticate() validates credentials correctly, get_accounts() returns all connected accounts, retry logic handles transient failures, clear error messages for auth issues_

- [x] 6. Implement media upload for small files in src/publisher/late/client.py
  - File: src/publisher/late/client.py (continue from task 5)
  - Implement upload_media() method for files ≤4 MB using client.media.upload()
  - Add file validation (existence, readability, size check)
  - Add retry logic for upload failures
  - Purpose: Enable video upload to Late.dev for files up to 4MB
  - _Leverage: late-sdk media API, src/publisher/models.py_
  - _Requirements: REQ-3 (Media Upload Management - small files), Design Component 2.3_
  - _Prompt: Role: Python Developer with expertise in file I/O and async uploads | Task: Implement upload_media() method in LatePublisher for files ≤4MB using client.media.upload(), adding file validation (existence, readable, size ≤4MB) and retry logic with exponential backoff (2s, 4s, 8s delays) | Restrictions: Must validate file before upload attempt, return Late media ID on success, raise clear errors for missing/unreadable files or size >4MB, log upload progress for debugging | Success: Small files upload successfully, file validation prevents invalid uploads, retry logic handles network failures, returns media ID for successful uploads_

- [x] 7. Implement media upload for large files in src/publisher/late/client.py
  - File: src/publisher/late/client.py (continue from task 6)
  - Extend upload_media() to handle files >4 MB and ≤500 MB using client.media.upload_large()
  - Add Vercel token validation for large uploads
  - Implement progress tracking with callbacks
  - Purpose: Enable large video file uploads with progress reporting
  - _Leverage: late-sdk large file upload API, src/publisher/models.py_
  - _Requirements: REQ-3 (Media Upload Management - large files), Design Component 2.3_
  - _Prompt: Role: Python Developer with expertise in chunked uploads and progress tracking | Task: Extend upload_media() in LatePublisher to handle files 4-500MB using client.media.upload_large() with Vercel token, implementing progress callbacks that log every 10% completion | Restrictions: Must validate Vercel token exists for large uploads, raise error for files >500MB, stream files in 32KB chunks to minimize memory, display progress updates, handle upload resume on failure | Success: Large files upload successfully with progress tracking, Vercel token validated before upload, memory-efficient streaming implemented, progress displayed every 10%_

- [x] 8. Implement post creation and publishing in src/publisher/late/client.py
  - File: src/publisher/late/client.py (continue from task 7)
  - Implement publish() method using client.posts.create()
  - Add support for immediate vs scheduled publishing
  - Handle multi-platform posting with platform-specific account mapping
  - Purpose: Enable content publishing to social platforms via Late.dev
  - _Leverage: late-sdk posts API, src/publisher/models.py (PublishResult, PublishMetadata)_
  - _Requirements: REQ-5 (Single Video Publishing), Design Component 2.4_
  - _Prompt: Role: Backend Developer with expertise in API integrations and datetime handling | Task: Implement publish() method in LatePublisher using client.posts.create(), supporting immediate (publish_now=True) and scheduled publishing with datetime conversion to UTC, handling multi-platform posts with account ID mapping | Restrictions: Must validate scheduled time is not in past, convert all datetimes to UTC for API, create separate platform objects for each target, return PublishResult with post ID and status, log platform-specific errors | Success: Publishes successfully to multiple platforms, scheduled times validated and converted correctly, returns post IDs and published URLs, handles platform failures gracefully_

## Phase 3: Supporting Modules

- [x] 9. Create metadata loader module in src/publisher/metadata.py
  - File: src/publisher/metadata.py
  - Implement load_platform_metadata() function to load JSON files
  - Add fallback to UPLOAD_INSTRUCTIONS.txt parsing
  - Implement platform-specific character limit validation
  - Purpose: Load and validate platform-optimized metadata for publishing
  - _Leverage: src/ai/platform_metadata/models.py (PlatformMetadata), src/video/config/core_models.py_
  - _Requirements: REQ-4 (Platform-Specific Metadata Integration), Design Component 3_
  - _Prompt: Role: Python Developer with expertise in JSON parsing and file I/O | Task: Create metadata.py with load_platform_metadata(product_id, platform) function that loads metadata_<platform>.json from outputs/{product_id}/text/, falling back to UPLOAD_INSTRUCTIONS.txt if missing, validating character limits per platform | Restrictions: Must handle missing files gracefully, parse JSON safely with error handling, extract title/description/hashtags/keywords, validate against platform limits (YouTube: 100 title/5000 desc, TikTok: 150 caption, Instagram: 2200 caption), return PublishMetadata object | Success: Loads metadata correctly from JSON files, fallback extraction works from UPLOAD_INSTRUCTIONS.txt, validation catches limit violations, returns structured PublishMetadata_

- [x] 10. Create configuration module in src/publisher/config.py
  - File: src/publisher/config.py
  - Implement three-tier configuration loading (CLI > env > YAML)
  - Create load_publisher_config() function
  - Add configuration validation and defaults
  - Purpose: Provide centralized configuration management with precedence rules
  - _Leverage: src/video/video_config.py (Pydantic models), src/video/config/llm_settings.py (config patterns)_
  - _Requirements: REQ-7 (Publishing Configuration), Design Component 4_
  - _Prompt: Role: Python Developer with expertise in configuration management and Pydantic | Task: Create config.py with load_publisher_config() implementing three-tier precedence (CLI args > env vars > YAML config/publisher.yaml), using Pydantic models from src/publisher/models.py (PublisherConfig) following patterns from src/video/video_config.py | Restrictions: Must strictly follow precedence order, validate all config fields, provide sensible defaults (immediate publish, max_retries=3, timeout=30s), raise clear errors for missing required fields (API keys), support privacy settings per platform | Success: Configuration loads correctly with proper precedence, validates all fields, provides clear error messages for missing keys, defaults are sensible_

- [x] 11. Create batch publishing orchestrator in src/publisher/batch.py
  - File: src/publisher/batch.py
  - Implement BatchPublisher class with publish_batch() method
  - Add video discovery from outputs directory
  - Implement staggered delays between posts (configurable 30-60s)
  - Add progress tracking and summary reporting
  - Purpose: Enable batch publishing of multiple videos across platforms
  - _Leverage: src/pipeline/global_batch.py (batch patterns), src/publisher/metadata.py, src/publisher/late/client.py_
  - _Requirements: REQ-6 (Batch Publishing Workflow), Design Component 5_
  - _Prompt: Role: Python Developer with expertise in batch processing and async workflows | Task: Create batch.py with BatchPublisher class implementing publish_batch() that scans outputs directory for completed videos, maps to metadata files, publishes with staggered delays (30-60s configurable), tracks progress, generates summary report, continues on individual failures unless --fail-fast specified, following patterns from src/pipeline/global_batch.py | Restrictions: Must scan outputs/{product_id}/video_*.mp4 files, load corresponding metadata, apply random staggered delays between posts, handle rate limits (429) with retry-after wait, continue on failures unless fail-fast, return BatchPublishSummary with success/fail counts | Success: Discovers all videos correctly, maps metadata properly, publishes with delays, handles failures gracefully, generates comprehensive summary report_

## Phase 4: CLI and Error Handling

- [x] 12. Create CLI interface in src/publisher/late/__main__.py
  - File: src/publisher/late/__main__.py
  - Implement argument parser with commands: single, batch, list-accounts
  - Add arguments: --video, --platform, --schedule, --immediate, --debug
  - Integrate with PublisherFactory and BatchPublisher
  - Purpose: Provide user-friendly command-line interface for publishing
  - _Leverage: src/scraper/amazon/__main__.py (CLI patterns), argparse module_
  - _Requirements: REQ-10 (CLI Interface), Design Component 6_
  - _Prompt: Role: Python Developer with expertise in CLI design and argparse | Task: Create __main__.py with argument parser for commands (single, batch, list-accounts) and flags (--video, --platform youtube/tiktok/instagram, --schedule DATETIME, --immediate, --batch, --debug), integrating PublisherFactory and BatchPublisher, following CLI patterns from src/scraper/amazon/__main__.py | Restrictions: Must validate required arguments per command, provide usage help with examples, support multiple platforms via repeated --platform flag, validate datetime format for --schedule, enable debug logging with --debug, display clear error messages | Success: CLI accepts all arguments correctly, provides helpful usage messages, validates inputs properly, integrates with publisher modules seamlessly_

- [x] 13. Add comprehensive error handling and retry logic
  - File: src/publisher/late/client.py (enhance existing methods)
  - Implement error handling for all 7 error scenarios from design
  - Add exponential backoff for network timeouts and 5xx errors
  - Handle rate limits (429) with retry-after header extraction
  - Purpose: Ensure robust error recovery and clear error reporting
  - _Leverage: aiohttp exceptions, late-sdk error types_
  - _Requirements: REQ-9 (Error Handling and Resilience), Design Section 6_
  - _Prompt: Role: Senior Python Developer with expertise in error handling and resilience | Task: Enhance error handling in client.py for scenarios: network timeout, rate limit (429), auth failure (401), file validation, platform failures, API validation errors, adding exponential backoff for transient failures (2s, 4s, 8s), extracting retry-after header for rate limits, continuing multi-platform posts on single platform failure | Restrictions: Must catch specific exceptions (aiohttp.ClientTimeout, aiohttp.ClientResponseError), log full error context without exposing API keys (show first 4 chars only), retry up to max_retries for transient errors, treat 401/403 as permanent failures, continue batch on failure unless fail-fast | Success: All error scenarios handled gracefully, retry logic works for transient failures, rate limits respected with proper delays, error messages are actionable and detailed_

- [x] 14. Add status tracking and progress reporting
  - File: src/publisher/late/client.py (add get_status method), src/publisher/batch.py (enhance reporting)
  - Implement get_status() method to fetch post status from Late.dev
  - Add real-time progress updates for uploads and batch publishing
  - Implement summary report generation with success/fail counts
  - Purpose: Provide visibility into publishing status and outcomes
  - _Leverage: late-sdk posts API, src/publisher/models.py (BatchPublishSummary)_
  - _Requirements: REQ-8 (Status Tracking and Reporting), Design Section 5.2_
  - _Prompt: Role: Python Developer with expertise in progress tracking and reporting | Task: Implement get_status() in client.py to fetch post status, add progress updates displaying [N/total] format during batch, generate BatchPublishSummary with total attempted/successful/failed/skipped per platform, display upload progress for large files every 10%, log published post URLs | Restrictions: Must show progress for both upload (%) and batch ([N/total]), fetch post status from Late API after creation, include scheduled time in user timezone for scheduled posts, generate human-readable summary with counts per platform, continue on status check failures | Success: Status tracking fetches post info correctly, progress updates displayed in real-time, summary report comprehensive with all counts, user timezone conversion accurate_

## Phase 5: Testing and Documentation

- [x] 15. Create unit tests for base interfaces and models
  - File: tests/publisher/test_base.py, tests/publisher/test_models.py
  - Write tests for BasePublisher interface contract
  - Test data model validation and serialization
  - Test publisher registry and factory
  - Purpose: Ensure foundation components work correctly
  - _Leverage: pytest, tests/conftest.py (fixtures), src/publisher/base.py, src/publisher/models.py, src/publisher/registry.py_
  - _Requirements: Design Testing Strategy - Unit Tests_
  - _Prompt: Role: QA Engineer with expertise in Python unit testing and pytest | Task: Create test_base.py testing BasePublisher abstract interface cannot be instantiated directly, concrete implementations must implement all methods, and test_models.py testing PublishResult/PublishMetadata/PublisherConfig/BatchPublishSummary validation, serialization, immutability, and test_registry.py testing @register_publisher decorator, factory get_publisher(), following pytest patterns from tests/conftest.py | Restrictions: Must test abstract class enforcement, model field validation, factory registration, use pytest fixtures for setup, mock external dependencies, maintain test isolation, achieve >90% coverage | Success: All base components tested thoroughly, abstract interface enforcement verified, model validation works correctly, factory registration tested, high test coverage achieved_

- [x] 16. Create unit tests for LatePublisher client
  - File: tests/publisher/late/test_client.py
  - Write tests for LatePublisher methods with mocked late-sdk client
  - Test authentication, account listing, media upload, post creation
  - Test error handling and retry logic
  - Purpose: Ensure Late.dev client implementation is robust
  - _Leverage: pytest, pytest-mock, unittest.mock, src/publisher/late/client.py_
  - _Requirements: Design Testing Strategy - Unit Tests_
  - _Prompt: Role: QA Engineer specializing in API client testing and mocking | Task: Create test_client.py testing LatePublisher authenticate(), get_accounts(), upload_media() (both small and large), publish() with mocked late-sdk client, testing retry logic, error scenarios (network timeout, rate limit, auth failure), multi-platform posting, using pytest-mock and unittest.mock | Restrictions: Must mock all late-sdk calls, test both success and failure paths, verify retry exponential backoff timing, test rate limit handling with retry-after, validate error messages, achieve >90% code coverage, use pytest fixtures for client initialization | Success: All LatePublisher methods tested with proper mocking, error scenarios covered, retry logic verified, multi-platform logic tested, high coverage achieved_

- [x] 17. Create integration tests with Late.dev sandbox
  - File: tests/integration/test_late_publisher.py
  - Write integration tests using real Late.dev sandbox API
  - Test full workflow: authenticate → upload → publish → status check
  - Test batch publishing with multiple videos
  - Purpose: Verify integration with actual Late.dev API
  - _Leverage: pytest, pytest-asyncio, late-sdk, .env.test configuration_
  - _Requirements: Design Testing Strategy - Integration Tests_
  - _Prompt: Role: QA Engineer with expertise in integration testing and API testing | Task: Create test_late_publisher.py with integration tests using real Late.dev sandbox API (requires sandbox API key in .env.test), testing full workflow: initialize client, authenticate, fetch accounts, upload test video (small and large), create post (immediate and scheduled), check status, test batch publishing with 2-3 videos, using pytest-asyncio for async tests | Restrictions: Must skip tests if sandbox credentials missing (pytest.skip), use test video files from tests/fixtures/, verify published posts in Late dashboard after tests, clean up test posts after completion, test both immediate and scheduled posts, achieve >80% coverage | Success: Full workflow tested end-to-end, both small and large uploads work, scheduled posts verified, batch publishing tested, cleanup performed correctly_

- [x] 18. Create end-to-end tests for complete publishing workflow
  - File: tests/e2e/test_publisher_workflow.py
  - Write E2E tests for CLI commands with real video files
  - Test complete workflow: scraper → producer → metadata generation → publishing
  - Test error recovery and failure scenarios
  - Purpose: Validate entire content pipeline from scraping to publishing
  - _Leverage: pytest, subprocess module, outputs directory with test products_
  - _Requirements: Design Testing Strategy - E2E Tests_
  - _Prompt: Role: QA Automation Engineer with expertise in end-to-end testing | Task: Create test_publisher_workflow.py testing complete pipeline: run scraper for test product, generate video, generate platform metadata (v0.17.0), publish single video via CLI, verify success, test batch mode with multiple products, test error scenarios (missing metadata, invalid credentials, network failure), using subprocess to invoke CLI commands, checking exit codes and output messages | Restrictions: Must use test products (real ASINs), verify outputs at each step, check published post URLs, test failure recovery (e.g., skip video if metadata missing), verify batch summary report, run in CI with mocked Late API or skip if no sandbox credentials, cleanup test outputs after tests | Success: Complete E2E workflow tested successfully, CLI commands work correctly, error scenarios handled properly, pipeline integration verified end-to-end_

- [x] 19. Create documentation and usage examples
  - File: docs/PUBLISHER.md, README.md (add publisher section)
  - Document CLI usage with examples
  - Document configuration options and precedence
  - Add troubleshooting guide for common errors
  - Purpose: Enable users to understand and use publisher feature
  - _Leverage: existing documentation structure in docs/, README.md patterns_
  - _Requirements: REQ-10 (CLI Interface), Design Section 7_
  - _Prompt: Role: Technical Writer with expertise in developer documentation | Task: Create PUBLISHER.md documenting Late.dev publisher feature: setup (API key, Vercel token in .env), CLI usage examples (single, batch, list-accounts), configuration (publisher.yaml structure, precedence rules), platform metadata integration, error handling, troubleshooting common issues (auth failures, rate limits, missing metadata), update README.md with publisher section and quick start examples | Restrictions: Must include copy-paste command examples, document all CLI flags, explain three-tier config precedence clearly, provide troubleshooting steps for each error scenario, link to Late.dev documentation, follow existing docs style and formatting | Success: Documentation is clear and comprehensive, examples are copy-pasteable and work correctly, troubleshooting covers common issues, README.md updated appropriately_

- [x] 20. Final integration and validation
  - Files: All publisher modules
  - Run full test suite (unit, integration, E2E)
  - Verify code quality with linting (ruff, mypy, bandit)
  - Test complete workflow with real product data
  - Update CHANGELOG.md with feature details
  - Purpose: Ensure feature is production-ready and meets all requirements
  - _Leverage: make commands (make lint, make test, make security)_
  - _Requirements: All requirements, Design all sections_
  - _Prompt: Role: Senior Developer with expertise in code quality and release management | Task: Complete final validation by running full test suite (pytest), verify linting passes (make lint), test security (make security), run complete workflow with real scraped product (scraper → producer → metadata → publish), verify batch publishing with 3+ products, ensure all error scenarios handled, update CHANGELOG.md with v0.18.0 feature description, verify documentation completeness | Restrictions: Must pass all quality gates (tests >90% coverage, linting clean, security scan pass), test with real products and real Late sandbox, verify published posts in social platforms, ensure backward compatibility with existing features, follow contribution guidelines | Success: All tests pass with high coverage, linting and security checks clean, complete workflow tested successfully, documentation comprehensive, feature ready for PR and release_
