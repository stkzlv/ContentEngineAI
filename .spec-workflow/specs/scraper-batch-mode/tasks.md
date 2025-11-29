# Tasks Document

## Implementation Tasks

- [x] 1. Create batch configuration data models
  - Files: `src/scraper/amazon/models.py`
  - Add `BatchConfig`, `BatchSummary`, `ProductResult` dataclasses
  - Extend existing models module with batch-specific structures
  - Purpose: Define data structures for batch processing configuration and results
  - _Leverage: Existing `ProductData` and `SearchParameters` models_
  - _Requirements: 1.6, 1.7_
  - _Prompt: Implement the task for spec scraper-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Python Developer specializing in data modeling and type safety | Task: Create batch processing data models (BatchConfig, BatchSummary, ProductResult) in src/scraper/amazon/models.py following requirements 1.6 and 1.7. Use Pydantic BaseModel or dataclasses with type hints (dict[str, Any], list[str], | None). Leverage existing ProductData and SearchParameters patterns. | Restrictions: Do not modify existing model classes, maintain backward compatibility, follow modern Python typing conventions (dict[str, Any] not Dict), use snake_case for field names | _Leverage: src/scraper/amazon/models.py (ProductData, SearchParameters patterns) | _Requirements: 1.6 (Configuration), 1.7 (Summary Reporting) | Success: All three models defined with complete type hints, fields match design document exactly, models are importable and instantiable | Instructions: 1) Mark this task as in-progress in tasks.md (change [ ] to [-]), 2) Implement the models, 3) Use log-implementation tool to record implementation with artifacts (classes created with fields and locations), 4) Mark task as complete in tasks.md (change [-] to [x])_

- [ ] 2. Extend YAML configuration schema
  - Files: `config/scraper.yaml`, `src/scraper/amazon/config.py`
  - Add `batch` section to scraper.yaml with `product_ids` and `keywords` lists
  - Create `load_batch_config()` function to load batch configuration with CLI precedence
  - Purpose: Enable YAML-based batch configuration with CLI override support
  - _Leverage: Existing `CONFIG` dict and `get_default_search_parameters()` function_
  - _Requirements: 1.6_
  - _Prompt: Implement the task for spec scraper-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Configuration Engineer with expertise in YAML and Python configuration management | Task: Extend YAML configuration schema and create load_batch_config() function following requirement 1.6. Add batch section to config/scraper.yaml, implement CLI > YAML > Defaults precedence in src/scraper/amazon/config.py. | Restrictions: Do not modify existing configuration structure, maintain backward compatibility (empty batch lists if section missing), validate product_ids and keywords are lists | _Leverage: src/scraper/amazon/config.py (CONFIG global dict, existing YAML loading patterns) | _Requirements: 1.6 (Configuration and CLI Interface) | Success: batch section in YAML loads correctly, load_batch_config() properly implements 3-tier precedence, empty lists default correctly when section missing | Instructions: 1) Mark task in-progress in tasks.md, 2) Implement YAML schema and load function, 3) Log implementation with artifacts (functions created with signature and location, configuration schema added), 4) Mark complete in tasks.md_

- [ ] 3. Create BatchController class
  - Files: `src/scraper/amazon/batch_controller.py` (new file)
  - Implement `BatchController` with `__init__()`, `run_batch()`, `_process_product_ids()`, `_process_keywords()`, `_deduplicate_products()` methods
  - Add progress tracking with `[N/total]` format logging
  - Purpose: Orchestrate batch processing of product IDs and keywords
  - _Leverage: BotasaurusAmazonScraper.scrape_products() and scrape_products_unified()_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - _Prompt: Implement the task for spec scraper-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Python Developer specializing in orchestration and control flow | Task: Create BatchController class in new file src/scraper/amazon/batch_controller.py following requirements 1.1-1.5. Implement sequential processing with progress logging ([N/total] format), deduplication by ASIN, fail-fast support. Accept BotasaurusAmazonScraper instance in __init__, delegate actual scraping to existing scraper methods. | Restrictions: Do not modify BotasaurusAmazonScraper class, do not duplicate scraping logic, maintain sequential processing (no parallelization), use existing logger from scraper instance | _Leverage: src/scraper/amazon/scraper.py (BotasaurusAmazonScraper.scrape_products(), scrape_products_unified(), logger), src/scraper/amazon/utils.py (validate_asin_format) | _Requirements: 1.1 (Product ID List), 1.2 (Keyword List), 1.3 (Mixed Input), 1.4 (Progress Tracking), 1.5 (Error Handling) | Success: BatchController class created with all methods, delegates to existing scraper without modification, progress logging works with correct format, deduplication removes duplicates by ASIN, fail-fast stops on first error | Instructions: 1) Mark in-progress, 2) Create new file and implement class, 3) Log implementation with artifacts (class with all methods and their signatures, integration with scraper), 4) Mark complete_

- [ ] 4. Add CLI arguments for batch mode
  - Files: `src/scraper/amazon/scraper.py` (modify main() function)
  - Add `--product-ids` argument with `nargs="+"` for multiple ASINs
  - Add `--fail-fast` flag with `action="store_true"`
  - Modify existing `--keywords` to support `nargs="+"` (currently single value)
  - Purpose: Enable batch mode through command-line interface
  - _Leverage: Existing argparse argument parser in main()_
  - _Requirements: 1.6_
  - _Prompt: Implement the task for spec scraper-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: CLI Developer with expertise in argparse and Python command-line interfaces | Task: Add batch mode CLI arguments to src/scraper/amazon/scraper.py main() function following requirement 1.6. Add --product-ids (nargs="+"), --fail-fast (store_true), modify --keywords to nargs="+" for multiple values. | Restrictions: Do not remove or break existing CLI arguments, maintain backward compatibility (single --keywords value must still work), follow existing argument naming conventions | _Leverage: src/scraper/amazon/scraper.py (main() function, existing argparse parser) | _Requirements: 1.6 (Configuration and CLI Interface) | Success: All three arguments added correctly, --keywords accepts both single and multiple values without breaking existing usage, help text is clear and descriptive | Instructions: 1) Mark in-progress, 2) Modify main() to add arguments, 3) Log implementation with artifacts (CLI arguments added with their configurations), 4) Mark complete_

- [ ] 5. Integrate BatchController into CLI main()
  - Files: `src/scraper/amazon/scraper.py` (modify main() function)
  - Detect batch mode (product_ids or multiple keywords present)
  - Instantiate BatchController with scraper and config
  - Call `run_batch()` and log summary report
  - Purpose: Wire batch processing into CLI entry point
  - _Leverage: Existing scraper instantiation in main(), load_batch_config() from task 2_
  - _Requirements: 1.1, 1.4, 1.7_
  - _Prompt: Implement the task for spec scraper-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Integration Engineer specializing in Python application wiring | Task: Integrate BatchController into CLI main() function in src/scraper/amazon/scraper.py following requirements 1.1, 1.4, 1.7. Detect batch mode from CLI args, load batch config, instantiate BatchController with scraper instance, execute run_batch(), log final summary. | Restrictions: Do not break existing single-product mode, maintain existing scraper instantiation code, ensure backward compatibility when no batch args provided | _Leverage: src/scraper/amazon/scraper.py (main() function, existing scraper instantiation), src/scraper/amazon/config.py (load_batch_config from task 2), src/scraper/amazon/batch_controller.py (BatchController from task 3) | _Requirements: 1.1 (Product ID List), 1.4 (Progress Tracking), 1.7 (Summary Reporting) | Success: Batch mode activates when --product-ids or multiple --keywords provided, single-product mode still works unchanged, BatchController executes correctly, summary logged at completion | Instructions: 1) Mark in-progress, 2) Add batch detection and controller integration to main(), 3) Log implementation with artifacts (integration points, control flow changes), 4) Mark complete_

- [ ] 6. Create unit tests for BatchController
  - Files: `tests/scraper/test_batch_controller.py` (new file)
  - Test product ID validation and deduplication
  - Test progress tracking and summary generation
  - Test fail-fast behavior
  - Mock BotasaurusAmazonScraper for isolated testing
  - Purpose: Ensure BatchController reliability and correctness
  - _Leverage: Existing test utilities and pytest fixtures_
  - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.7_
  - _Prompt: Implement the task for spec scraper-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: QA Engineer with expertise in Python unit testing and pytest | Task: Create comprehensive unit tests for BatchController in tests/scraper/test_batch_controller.py following requirements 1.1, 1.2, 1.3, 1.5, 1.7. Test validation, deduplication, progress tracking, fail-fast, summary generation. Mock BotasaurusAmazonScraper to isolate BatchController logic. | Restrictions: Must test in isolation (mock all external dependencies including scraper), test both success and failure scenarios, do not make actual HTTP requests or scrape real products | _Leverage: tests/scraper/conftest.py (existing fixtures if any), pytest mocking utilities (pytest-mock or unittest.mock) | _Requirements: 1.1 (Product IDs), 1.2 (Keywords), 1.3 (Mixed Input), 1.5 (Error Handling), 1.7 (Summary) | Success: All BatchController methods tested, deduplication logic verified, progress tracking format verified, fail-fast behavior confirmed, summary statistics accurate, tests pass in isolation | Instructions: 1) Mark in-progress, 2) Create test file with comprehensive coverage, 3) Log implementation with artifacts (test functions created with what they test), 4) Mark complete_

- [ ] 7. Create integration tests for end-to-end batch scraping
  - Files: `tests/scraper/test_batch_integration.py` (new file)
  - Test complete batch flow with product ID list
  - Test keyword list with filters
  - Test mixed input (product IDs + keywords)
  - Test configuration precedence (CLI overrides YAML)
  - Purpose: Verify batch mode works end-to-end
  - _Leverage: Existing integration test patterns_
  - _Requirements: 1.1, 1.2, 1.3, 1.6_
  - _Prompt: Implement the task for spec scraper-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: QA Engineer with expertise in integration testing and Python | Task: Create integration tests for batch scraping in tests/scraper/test_batch_integration.py following requirements 1.1, 1.2, 1.3, 1.6. Test end-to-end flows: product ID list, keyword list with filters, mixed input, CLI vs YAML config precedence. Use test ASINs or mock responses if scraping live products is not feasible. | Restrictions: Consider using VCR.py or similar to record/replay HTTP interactions, or use test fixtures for product data, tests should be repeatable and not depend on external state changes | _Leverage: tests/scraper/ (existing integration test patterns), pytest fixtures for test data | _Requirements: 1.1 (Product IDs), 1.2 (Keywords), 1.3 (Mixed Input), 1.6 (Configuration) | Success: All batch modes tested end-to-end, configuration precedence verified, tests are repeatable and reliable, test coverage includes success and error scenarios | Instructions: 1) Mark in-progress, 2) Create integration tests, 3) Log implementation with artifacts (test scenarios covered), 4) Mark complete_

- [ ] 8. Update documentation and examples
  - Files: `README.md`, `CLAUDE.md` (if batch commands should be added to essential commands)
  - Add batch mode usage examples to README
  - Document YAML configuration schema
  - Add example batch configurations
  - Purpose: Enable users to discover and use batch mode
  - _Leverage: Existing documentation structure_
  - _Requirements: All requirements_
  - _Prompt: Implement the task for spec scraper-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Technical Writer with expertise in developer documentation | Task: Update README.md with batch mode documentation covering all requirements. Add CLI usage examples (--product-ids, --keywords, --fail-fast), YAML configuration examples, explain configuration precedence. Consider adding to CLAUDE.md essential commands if appropriate for workflow. | Restrictions: Do not remove existing documentation, maintain consistent formatting with existing docs, keep examples realistic and tested | _Leverage: README.md (existing structure and format), CLAUDE.md (essential commands section) | _Requirements: All requirements (1.1-1.7) | Success: Batch mode fully documented with clear examples, YAML schema documented, CLI usage examples provided, configuration precedence explained clearly | Instructions: 1) Mark in-progress, 2) Update documentation with examples and explanations, 3) Log implementation with artifacts (documentation sections added), 4) Mark complete_

## Testing Checklist

After implementation, verify:

- [ ] 9.1 Product ID list scraping works with multiple ASINs
- [ ] 9.2 Keyword list scraping works with multiple search terms
- [ ] 9.3 Mixed input (product IDs + keywords) processes both sources
- [ ] 9.4 CLI arguments override YAML configuration correctly
- [ ] 9.5 YAML configuration used when no CLI arguments provided
- [ ] 9.6 Invalid product IDs skipped with warning, batch continues
- [ ] 9.7 Fail-fast stops on first failure
- [ ] 9.8 Graceful continuation works (default behavior)
- [ ] 9.9 Deduplication removes duplicate ASINs across sources
- [ ] 9.10 Progress logging shows [N/total] format
- [ ] 9.11 Summary report shows accurate counts and statistics
- [ ] 9.12 Backward compatibility: single-product mode still works
- [ ] 9.13 All unit tests pass with good coverage
- [ ] 9.14 All integration tests pass reliably
