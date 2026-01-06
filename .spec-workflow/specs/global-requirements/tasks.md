# Tasks Document: Global Requirements

## Implementation Status

Most Global Requirements infrastructure is already implemented. These tasks address gaps and ensure comprehensive test coverage.

## Tasks

- [x] 1. Add secret masking utility function
  - File: src/utils/secrets.py (new)
  - Create `mask_secret()` function that shows first/last 4 characters
  - Add helper to check if a value is a secret (based on key name patterns)
  - Purpose: Prevent accidental secret exposure in logs and error messages
  - _Leverage: src/utils/logging_setup.py_
  - _Requirements: 3.5_
  - _Prompt: Role: Python Developer specializing in security utilities | Task: Create a secrets utility module with mask_secret() function that masks middle characters of secrets (showing first/last 4 chars) and a helper to detect secret keys by name patterns (API_KEY, TOKEN, SECRET, PASSWORD) | Restrictions: Must handle edge cases (short strings, None values), do not import secrets module (stdlib name collision), follow existing utils patterns | Success: mask_secret("sk-1234567890abcdef") returns "sk-1...cdef", all secret-like env vars are detected_
  - _Status: Implemented with mask_secret(), is_secret_key(), mask_secrets_in_dict()_

- [x] 2. Integrate secret masking into logging setup
  - File: src/utils/logging_setup.py (modify)
  - Create SecretMaskingFilter class extending logging.Filter
  - Apply filter to all handlers in setup_debug_logging()
  - Purpose: Automatically mask secrets in all log output
  - _Leverage: src/utils/secrets.py (from task 1)_
  - _Requirements: 3.5_
  - _Prompt: Role: Python Developer with logging expertise | Task: Create a SecretMaskingFilter that scans log records for secret patterns and masks them before output, integrate into setup_debug_logging() for all handlers | Restrictions: Must not significantly impact logging performance, handle all record attributes (message, args, exc_info), maintain thread safety | Success: Secrets are masked in console and file output, filter has minimal performance overhead (<1ms per record)_
  - _Status: SecretMaskingFilter integrated, performance verified (18.56 µs/record)_

- [x] 3. Add startup configuration validation
  - File: src/config_manager.py (modify)
  - Add `validate_required_secrets()` method to UnifiedConfigManager
  - Check for required env vars at startup with clear error messages
  - Purpose: Fail fast with actionable error when required config missing
  - _Leverage: src/utils/secrets.py_
  - _Requirements: 2.4, 3.2_
  - _Prompt: Role: Python Developer with configuration management expertise | Task: Add validate_required_secrets() method that checks OPENROUTER_API_KEY (required for LLM), GOOGLE_APPLICATION_CREDENTIALS (optional for TTS), LATE_API_KEY (optional for publishing), failing with clear message naming missing variable and providing guidance | Restrictions: Only fail for truly required secrets, provide helpful error messages with setup instructions, support optional secrets gracefully | Success: Missing required secrets cause immediate exit with message like "Set OPENROUTER_API_KEY environment variable (see .env.example)"_
  - _Status: SecretDefinition dataclass, SECRETS_REGISTRY, SecretsValidationResult, validate_required_secrets() method implemented_

- [x] 4. Verify and update .env.example completeness
  - File: .env.example (modify if needed)
  - Audit all env var references in codebase
  - Ensure all required and optional secrets documented with descriptions
  - Purpose: Provide complete template for environment setup
  - _Leverage: grep -r "os.environ" src/, grep -r "getenv" src/_
  - _Requirements: 3.4_
  - _Prompt: Role: DevOps Engineer | Task: Audit all environment variable references in src/ and ensure .env.example documents each with description, default value (if any), and whether required or optional | Restrictions: Do not include actual secret values, maintain alphabetical ordering, group by functional area | Success: .env.example documents all env vars used in codebase with clear descriptions_
  - _Status: 44 env vars documented across 10 sections with descriptions, defaults, and required/optional status_

- [x] 5. Add unit tests for secret masking
  - File: tests/test_secrets.py (new)
  - Test mask_secret() with various input lengths
  - Test secret key detection patterns
  - Purpose: Ensure secret masking works correctly in all cases
  - _Leverage: tests/conftest.py_
  - _Requirements: 3.5_
  - _Prompt: Role: QA Engineer | Task: Create comprehensive unit tests for secrets.py covering mask_secret() with edge cases (short strings, None, empty, normal length) and secret key detection (API_KEY, TOKEN, PASSWORD patterns, non-secret keys) | Restrictions: Test edge cases thoroughly, use pytest parametrize for multiple inputs, maintain test isolation | Success: 100% coverage of secrets.py, all edge cases tested_
  - _Status: 26 unit tests covering all functions, edge cases, and patterns_

- [x] 6. Add unit tests for configuration validation
  - File: tests/test_config_manager.py (new)
  - Add tests for validate_required_secrets() method
  - Test missing required vs optional secrets behavior
  - Purpose: Ensure validation catches missing config at startup
  - _Leverage: tests/conftest.py, unittest.mock for env vars_
  - _Requirements: 2.4, 3.2_
  - _Prompt: Role: QA Engineer | Task: Add tests for validate_required_secrets() covering: all secrets present (success), required secret missing (failure with message), optional secret missing (success with warning log) | Restrictions: Mock environment variables properly, test error message content, maintain test isolation | Success: Tests verify validation behavior for all secret combinations_
  - _Status: 17 unit tests covering SecretDefinition, SecretsValidationResult, validate_required_secrets(), and SECRETS_REGISTRY_

- [x] 7. Update docs/configuration.md with Global Requirements
  - File: docs/configuration.md (modify)
  - Add section on three-tier precedence with examples
  - Document all environment variables with types and defaults
  - Add troubleshooting for common config issues
  - Purpose: Provide comprehensive configuration reference
  - _Leverage: src/config_manager.py for env var list_
  - _Requirements: 1, 2, 3_
  - _Prompt: Role: Technical Writer | Task: Update configuration.md with: 1) Three-tier precedence explanation with CLI example, 2) Environment variables table (name, type, default, description), 3) Troubleshooting section for common issues | Restrictions: Use existing doc style, keep examples runnable, maintain accuracy with code | Success: Users can configure system using docs alone, all env vars documented_
  - _Status: Added three-tier precedence section with visual diagram and CLI examples, comprehensive env var tables (44+ variables across 7 categories), and expanded troubleshooting section covering 10 common issues_

- [x] 8. Add integration test for full config precedence
  - File: tests/integration/test_config_precedence.py (new)
  - Test CLI overrides env overrides YAML end-to-end
  - Test type conversion from string env vars
  - Purpose: Verify configuration system works correctly in integration
  - _Leverage: tests/conftest.py, tempfile for YAML_
  - _Requirements: 1, 2_
  - _Prompt: Role: QA Engineer | Task: Create integration test that: 1) Creates temp YAML with base values, 2) Sets env var overrides, 3) Passes CLI args, 4) Verifies final config has correct precedence (CLI > ENV > YAML) | Restrictions: Use temp files/dirs, clean up after test, test realistic config values | Success: Test proves three-tier precedence works correctly end-to-end_
  - _Status: 11 integration tests covering full precedence chain, type conversion, boolean variants, nested overrides, alternative env var names, and CLI short names. Uses standalone config manager to avoid circular import issues with root conftest.py_

### Retry Logic Tasks

- [x] 14. Add tenacity dependency for retry logic
  - File: pyproject.toml (modify)
  - Add tenacity package to dependencies
  - Purpose: Enable exponential backoff retries for transient failures
  - _Requirements: 4 (Graceful Degradation)_
  - _Prompt: Role: Python Developer | Task: Add tenacity to pyproject.toml dependencies and run poetry lock | Restrictions: Use latest stable version, no dev dependency | Success: tenacity importable, poetry lock succeeds_
  - _Status: tenacity 9.1.2 already present as transitive dependency (via safety), importable and ready for use_

- [x] 15. Create retry utilities module
  - File: src/utils/retry.py (new)
  - Create reusable retry decorators for sync and async operations
  - Configure exponential backoff with jitter (min=1s, max=30s)
  - Define retryable exceptions (network timeouts, connection errors, rate limits)
  - Purpose: Centralized retry logic for all network operations
  - _Leverage: tenacity library_
  - _Requirements: 4 (Graceful Degradation)_
  - _Prompt: Role: Python Developer with resilience patterns expertise | Task: Create retry.py with @retry_network decorator using tenacity, exponential backoff (multiplier=1, min=1, max=30), retry on requests.Timeout, requests.ConnectionError, httpx.TimeoutException, and 429/503 status codes, log retries with before_sleep callback | Restrictions: Support both sync and async, don't retry on 4xx client errors (except 429), max 3 attempts by default | Success: Decorator works for sync/async, logs retry attempts, respects exception types_
  - _Status: Created src/utils/retry.py with @retry_network decorator supporting sync/async, exponential backoff (1-30s), retries on Timeout/ConnectionError/429/503, logs via before_sleep callback, 3 attempts default_

- [x] 16. Integrate retry logic into scraper network operations
  - File: src/scraper/base/http_client.py (modify if exists) or relevant network module
  - Apply retry decorator to HTTP request methods
  - Ensure retries work with circuit breaker (retry inside, breaker outside)
  - Purpose: Make scraper resilient to transient network failures
  - _Leverage: src/utils/retry.py, src/utils/circuit_breaker.py_
  - _Requirements: 4 (Graceful Degradation)_
  - _Prompt: Role: Python Developer | Task: Apply @retry_network decorator to scraper HTTP methods, ensure circuit breaker wraps retry logic (breaker → retry → actual call), verify order prevents retry storms when circuit is open | Restrictions: Don't retry when circuit is open, maintain existing error handling | Success: Transient failures retry up to 3x, circuit breaker still trips after threshold_
  - _Status: Updated src/scraper/base/downloader.py: replaced @exponential_backoff_retry with @scraper_circuit_breaker + @retry_network() on validate_url(), download_file_sync(), and download_file_async(). Added scraper_circuit_breaker to circuit_breaker.py. Decorator order ensures circuit breaker fails fast when open (no retries)._

- [x] 17. Integrate retry logic into external API calls
  - Files: src/ai/script_generator.py, src/video/producer/tts.py, src/video/producer/stock_media.py
  - Apply retry decorator to LLM, TTS, and stock media API calls
  - Purpose: Make AI and media services resilient to transient failures
  - _Leverage: src/utils/retry.py_
  - _Requirements: 4 (Graceful Degradation)_
  - _Prompt: Role: Python Developer | Task: Add @retry_network to OpenRouter API calls in script_generator.py, Google TTS calls in tts.py, Pexels/Freesound calls in stock_media.py | Restrictions: Respect existing circuit breakers, don't retry on authentication errors | Success: API calls retry on timeout, log shows retry attempts_
  - _Status: tts.py: Added @google_stt_circuit_breaker to _generate_google_cloud_speech (already has internal retry). stock_media.py: Added @retry_network() to _search_and_select_pexels. script_generator.py and freesound_client.py already had sufficient circuit breaker + retry protection._

- [x] 18. Add unit tests for retry utilities
  - File: tests/utils/test_retry.py (new)
  - Test retry decorator with mock failures
  - Test exponential backoff timing
  - Test exception filtering (retry vs no-retry)
  - Purpose: Ensure retry logic works correctly
  - _Leverage: tests/conftest.py, unittest.mock, tenacity.wait.wait_none for fast tests_
  - _Requirements: 4 (Graceful Degradation)_
  - _Prompt: Role: QA Engineer | Task: Create tests for retry.py covering: successful call (no retry), transient failure then success (retry works), max attempts exceeded (raises), non-retryable exception (raises immediately), async variant | Restrictions: Use wait_none() in tests for speed, mock time/sleep, test both sync and async | Success: 100% coverage of retry.py, all edge cases tested_
  - _Status: 53 tests across 10 test classes covering sync/async decorators, exception filtering, status codes, RetryableHTTPError, helper functions. 93% coverage of retry.py (uncovered: httpx optional imports)._

### Documentation Tasks

- [x] 9. Verify required root documentation files exist
  - Files: README.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, CHANGELOG.md, LICENSE
  - Verify each file exists and contains required sections
  - README.md: project title, description, key features, quick start, links to detailed docs
  - CHANGELOG.md: follows Keep a Changelog format (Added, Changed, Fixed, etc.)
  - Purpose: Ensure documentation standards are met
  - _Leverage: docs/requirements.md Documentation Requirements section_
  - _Requirements: 9_
  - _Prompt: Role: Technical Writer | Task: Audit root documentation files for completeness: README.md must have title, description, features, quick start, doc links; CHANGELOG.md must follow Keep a Changelog format; all required files must exist | Restrictions: Do not modify content unless incomplete, document any gaps found | Success: All 6 required root files exist with proper sections_
  - _Status: All 6 files verified: README.md (title, features, quick start, doc links), CONTRIBUTING.md (218 lines), CODE_OF_CONDUCT.md, SECURITY.md (policy + reporting), CHANGELOG.md (Keep a Changelog format), LICENSE (MIT)_

- [x] 10. Verify docs/ directory structure and content
  - Files: docs/installation.md, docs/configuration.md, docs/development.md, docs/troubleshooting.md, docs/versioning.md
  - Verify documentation uses GFM and relative internal links
  - Check code examples are working with context and expected output
  - Purpose: Ensure organized documentation structure
  - _Leverage: docs/ directory contents_
  - _Requirements: 10_
  - _Prompt: Role: Technical Writer | Task: Audit docs/ directory structure: verify expected files exist, check GFM formatting, verify internal links use relative paths, ensure code examples have context | Restrictions: Document gaps but do not create missing files, report findings | Success: docs/ directory organized per standards with working examples
  - _Status: All 5 required files verified + 6 additional docs. GFM formatting with language specifiers, relative internal links, code examples with context. Total 344 code blocks across docs/_

### Outputs Directory Tasks

- [x] 11. Verify outputs path utilities exist and are complete
  - File: src/utils/outputs_paths.py
  - Verify centralized path management utilities
  - Check all required functions: get_outputs_root, get_product_directory, get_product_images_directory, get_product_videos_directory, get_cache_directory, get_logs_directory, get_reports_directory
  - Purpose: Ensure consistent path handling across all modules
  - _Requirements: 11, 12_
  - _Status: Implemented and verified_

- [x] 12. Verify outputs validation and cleanup utilities
  - File: src/utils/outputs_paths.py
  - Verify validate_outputs_structure() reports valid/invalid products
  - Verify cleanup_invalid_outputs() with dry-run support
  - Purpose: Enable outputs directory health checks
  - _Requirements: 11, 12_
  - _Status: Implemented and verified_

- [x] 13. Add unit tests for outputs path utilities
  - File: tests/utils/test_outputs_paths.py (new)
  - Test all path getter functions with default and custom outputs dir
  - Test validation logic for valid/invalid product directories
  - Test cleanup with dry-run mode
  - Purpose: Ensure outputs path utilities work correctly
  - _Leverage: tests/conftest.py, tempfile for isolated testing_
  - _Requirements: 11, 12_
  - _Prompt: Role: QA Engineer | Task: Create comprehensive unit tests for outputs_paths.py covering all path getters, validation logic, and cleanup utilities | Restrictions: Use temp directories for isolation, test edge cases (empty dirs, missing data.json), maintain test independence | Success: 100% coverage of outputs_paths.py_
  - _Status: 45 unit tests across 12 test classes covering all path getters (project root, outputs root, product directories, global directories), validation logic (product ID patterns, directory structure), and cleanup utilities (dry-run mode, actual cleanup, preserving valid products)_
