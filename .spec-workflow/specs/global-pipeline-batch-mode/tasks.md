# Tasks Document

## Implementation Tasks

- [x] 1. Create pipeline module structure
  - Files: `src/pipeline/__init__.py`, `src/pipeline/__main__.py`
  - Create new pipeline package directory
  - Set up CLI entry point in `__main__.py`
  - Purpose: Establish module structure for global batch pipeline
  - _Leverage: Existing package patterns from scraper and producer_
  - _Requirements: 1.1_
  - _Prompt: Implement the task for spec global-pipeline-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Python Developer specializing in module organization | Task: Create pipeline package structure following requirement 1.1. Create src/pipeline/ directory with __init__.py and __main__.py, set up CLI entry point that calls main() function from global_batch.py. | Restrictions: Follow existing module organization patterns from src/scraper and src/video, use standard Python package structure, ensure __main__.py enables 'python -m src.pipeline.global_batch' execution | _Leverage: src/scraper/__init__.py, src/video/producer/__main__.py (package structure patterns) | _Requirements: 1.1 (Unified Pipeline Execution) | Success: Package created with proper structure, __main__.py enables module execution, imports work correctly | Instructions: 1) Mark in-progress in tasks.md, 2) Create package structure and entry point, 3) Log implementation with artifacts (files created), 4) Mark complete_

- [x] 2. Create configuration data models
  - Files: `src/pipeline/config.py`
  - Create `GlobalBatchConfig`, `ScrapingPhaseSummary`, `ProductionPhaseSummary`, `PipelineSummary` dataclasses
  - Purpose: Define data structures for pipeline configuration and reporting
  - _Leverage: Existing configuration patterns from scraper and producer_
  - _Requirements: 2.1, 2.7, 8.1_
  - _Prompt: Implement the task for spec global-pipeline-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Python Developer specializing in data modeling | Task: Create pipeline configuration data models in src/pipeline/config.py following requirements 2.1, 2.7, 8.1. Implement GlobalBatchConfig with scraper and producer settings, ScrapingPhaseSummary, ProductionPhaseSummary, PipelineSummary with all required fields from design. Use modern Python typing (dict[str, Any], list[str], | None). | Restrictions: Do not modify existing configuration classes, use dataclasses or Pydantic BaseModel, ensure all fields have type hints, follow snake_case naming | _Leverage: src/scraper/amazon/models.py (configuration patterns), src/video/config.py (VideoConfig patterns) | _Requirements: 2.1 (Input Configuration), 2.7 (Configuration Management), 8.1 (Summary Reporting) | Success: All four data models defined with complete type hints, fields match design document exactly, models are importable and instantiable | Instructions: 1) Mark in-progress, 2) Implement data models, 3) Log implementation with artifacts (classes created with fields), 4) Mark complete_

- [x] 3. Implement configuration loading with precedence
  - Files: `src/pipeline/config.py`
  - Add `load_global_batch_config()` function implementing CLI > YAML > defaults precedence
  - Add `validate_global_batch_config()` function for validation
  - Purpose: Load and validate unified pipeline configuration
  - _Leverage: Existing configuration loading from scraper and producer_
  - _Requirements: 2.1, 2.7_
  - _Prompt: Implement the task for spec global-pipeline-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Configuration Engineer with expertise in YAML and precedence logic | Task: Implement configuration loading in src/pipeline/config.py following requirements 2.1 and 2.7. Create load_global_batch_config() with 3-tier precedence (CLI > YAML > defaults), validate_global_batch_config() to check inputs exist and profiles valid. Raise clear ValueError messages for validation failures. | Restrictions: Follow existing precedence patterns, validate before any processing starts, provide actionable error messages, don't modify existing config loaders | _Leverage: src/scraper/amazon/config.py (YAML loading), src/video/producer/cli.py (argument precedence patterns) | _Requirements: 2.1 (Input Configuration), 2.7 (Configuration Management) | Success: Configuration loads correctly with precedence, validation catches missing inputs and invalid profiles, error messages are clear and helpful | Instructions: 1) Mark in-progress, 2) Implement loading and validation, 3) Log implementation with artifacts (functions with signatures), 4) Mark complete_

- [x] 4. Create GlobalPipelineOrchestrator class
  - Files: `src/pipeline/global_batch.py`
  - Implement orchestrator with `__init__()`, `run_pipeline()`, phase execution methods
  - Add phase logging with clear headers
  - Purpose: Coordinate scraping, handoff, and production phases
  - _Leverage: Scraper and producer as black boxes_
  - _Requirements: 1.1, 3.1, 4.1, 5.1_
  - _Prompt: Implement the task for spec global-pipeline-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Python Developer specializing in orchestration and async programming | Task: Create GlobalPipelineOrchestrator class in src/pipeline/global_batch.py following requirements 1.1, 3.1, 4.1, 5.1. Implement run_pipeline() orchestrating three phases sequentially, _execute_scraping_phase(), _execute_handoff_phase(), _execute_production_phase(). Log clear phase headers ("SCRAPING PHASE", "VIDEO PRODUCTION PHASE"). | Restrictions: Do not modify scraper or producer code, treat them as black boxes, maintain sequential execution (scraping → handoff → production), use async/await for async operations | _Leverage: src/scraper/amazon/scraper.py (programmatic invocation), src/video/producer/cli.py (discover_products_for_batch, batch loop patterns) | _Requirements: 1.1 (Unified Execution), 3.1 (Scraping Phase), 4.1 (Handoff), 5.1 (Production Phase) | Success: Orchestrator coordinates all phases correctly, phase headers logged clearly, scraper and producer invoked as black boxes, sequential execution maintained | Instructions: 1) Mark in-progress, 2) Implement orchestrator class, 3) Log implementation with artifacts (class and methods created), 4) Mark complete_

- [x] 5. Implement scraping phase execution
  - Files: `src/pipeline/global_batch.py`
  - Implement `_execute_scraping_phase()` method
  - Invoke scraper programmatically with product IDs and keywords
  - Track scraping statistics and generate ScrapingPhaseSummary
  - Purpose: Execute scraping phase and collect results
  - _Leverage: BotasaurusAmazonScraper programmatic invocation_
  - _Requirements: 3.1_
  - _Prompt: Implement the task for spec global-pipeline-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Python Developer with expertise in API integration | Task: Implement scraping phase execution in _execute_scraping_phase() method following requirement 3.1. Instantiate BotasaurusAmazonScraper, call scrape_products() with product_ids and keywords from config, track success/failure counts, generate ScrapingPhaseSummary with statistics. Log progress with [N/total] format. | Restrictions: Use scraper's public API only, don't modify scraper internals, handle scraper exceptions gracefully, respect fail_fast setting | _Leverage: src/scraper/amazon/scraper.py (BotasaurusAmazonScraper.scrape_products()), src/scraper/amazon/models.py (SearchParameters) | _Requirements: 3.1 (Scraping Phase Execution) | Success: Scraper invoked correctly, all products processed, statistics tracked accurately, ScrapingPhaseSummary generated with correct counts, fail-fast behavior works | Instructions: 1) Mark in-progress, 2) Implement scraping phase, 3) Log implementation with artifacts (method implementation, scraper integration), 4) Mark complete_

- [x] 6. Implement handoff phase execution
  - Files: `src/pipeline/global_batch.py`
  - Implement `_execute_handoff_phase()` method
  - Scan outputs/ directory for products with data.json
  - Filter products by media availability based on profile requirements
  - Purpose: Identify products ready for video production
  - _Leverage: discover_products_for_batch() from producer_
  - _Requirements: 4.1_
  - _Prompt: Implement the task for spec global-pipeline-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Python Developer with expertise in file system operations | Task: Implement handoff phase in _execute_handoff_phase() method following requirement 4.1. Call discover_products_for_batch() to find products with data.json, filter by media availability using profile requirements, log transition message with count of ready products. Return list of ready products for video production. | Restrictions: Reuse existing discover_products_for_batch() function, don't duplicate product discovery logic, handle case of zero ready products gracefully | _Leverage: src/video/producer/cli.py (discover_products_for_batch function) | _Requirements: 4.1 (Handoff Phase) | Success: Products discovered correctly, media filtering works, zero products case handled gracefully, transition logged clearly | Instructions: 1) Mark in-progress, 2) Implement handoff phase, 3) Log implementation with artifacts (method implementation), 4) Mark complete_

- [x] 7. Implement production phase execution
  - Files: `src/pipeline/global_batch.py`
  - Implement `_execute_production_phase()` method
  - Process each product through video pipeline with configured profile
  - Support both fixed profile and random profile modes
  - Track production statistics and generate ProductionPhaseSummary
  - Purpose: Execute video production for all ready products
  - _Leverage: Video producer batch logic, create_video_for_product()_
  - _Requirements: 5.1_
  - _Prompt: Implement the task for spec global-pipeline-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Python Developer with expertise in async orchestration | Task: Implement production phase in _execute_production_phase() method following requirement 5.1. Iterate through products, select profile (fixed or random based on config), call create_video_for_product() for each, track success/failure/skipped counts, generate ProductionPhaseSummary with statistics. Log progress with [N/total] format. | Restrictions: Reuse existing video production logic, don't duplicate batch processing code, handle timeouts and exceptions per product, respect fail_fast setting | _Leverage: src/video/producer/orchestration.py (create_video_for_product), src/video/producer/cli.py (batch loop patterns, profile randomization if implemented) | _Requirements: 5.1 (Video Production Phase Execution) | Success: All products processed through video pipeline, profile selection works (fixed and random), statistics tracked accurately, ProductionPhaseSummary generated correctly | Instructions: 1) Mark in-progress, 2) Implement production phase, 3) Log implementation with artifacts (method implementation, producer integration), 4) Mark complete_

- [x] 8. Implement summary reporting
  - Files: `src/pipeline/global_batch.py`
  - Implement `_generate_final_summary()` method
  - Calculate end-to-end statistics from phase summaries
  - Add summary formatting for readable output
  - Purpose: Generate comprehensive pipeline statistics
  - _Leverage: Summary formatting patterns from scraper and producer_
  - _Requirements: 8.1_
  - _Prompt: Implement the task for spec global-pipeline-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Python Developer with expertise in reporting and data presentation | Task: Implement final summary generation in _generate_final_summary() method following requirement 8.1. Calculate end-to-end success (scraped AND produced), partial success (scraped only), total failures. Add format() method to PipelineSummary for readable output showing scraping stats, production stats, end-to-end stats, total duration. | Restrictions: Don't duplicate summary logic from phases, calculate derived statistics from phase summaries, maintain readable formatting | _Leverage: src/scraper/amazon/scraper.py (summary formatting), src/video/producer/cli.py (batch summary patterns) | _Requirements: 8.1 (Comprehensive Summary Reporting) | Success: All statistics calculated correctly, summary format is readable and comprehensive, includes all required sections from requirements | Instructions: 1) Mark in-progress, 2) Implement summary generation and formatting, 3) Log implementation with artifacts (summary logic), 4) Mark complete_

- [x] 9. Create CLI argument parser
  - Files: `src/pipeline/global_batch.py`
  - Create `create_argument_parser()` function
  - Add all scraper arguments (--product-ids, --keywords, filters)
  - Add all producer arguments (--profile, --random-profile, --profile-pool)
  - Add common arguments (--fail-fast, --outputs-dir, --debug)
  - Purpose: Provide unified CLI interface for global batch pipeline
  - _Leverage: Argument parser patterns from scraper and producer_
  - _Requirements: 2.1, 2.7_
  - _Prompt: Implement the task for spec global-pipeline-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: CLI Developer with expertise in argparse | Task: Create argument parser in create_argument_parser() function following requirements 2.1 and 2.7. Include all scraper arguments (--product-ids, --keywords, --max-products, price/rating filters), producer arguments (--profile, --random-profile, --profile-pool), common arguments (--fail-fast, --outputs-dir, --debug). Provide clear help text for each. | Restrictions: Follow existing argument naming conventions, maintain consistency with scraper and producer CLIs, provide comprehensive help text | _Leverage: src/scraper/amazon/scraper.py (main function argument parser), src/video/producer/cli.py (create_argument_parser patterns) | _Requirements: 2.1 (Input Configuration), 2.7 (Configuration Management) | Success: All arguments defined with proper types and help text, argument names consistent with existing CLIs, parser works correctly | Instructions: 1) Mark in-progress, 2) Create argument parser, 3) Log implementation with artifacts (CLI arguments defined), 4) Mark complete_

- [x] 10. Implement main() CLI entry point
  - Files: `src/pipeline/global_batch.py`
  - Create `main()` function as CLI entry point
  - Parse arguments, load configuration, validate, execute pipeline
  - Log final summary
  - Purpose: Wire all components together in CLI entry point
  - _Leverage: Main function patterns from scraper and producer_
  - _Requirements: All requirements_
  - _Prompt: Implement the task for spec global-pipeline-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Integration Engineer specializing in CLI applications | Task: Create main() CLI entry point in src/pipeline/global_batch.py. Parse arguments with create_argument_parser(), call load_global_batch_config(), validate config, instantiate GlobalPipelineOrchestrator, call run_pipeline(), log final summary. Handle exceptions gracefully with proper exit codes. | Restrictions: Follow existing main() patterns, set up logging before pipeline starts, handle keyboard interrupts gracefully, use appropriate exit codes (0 for success, 1 for errors) | _Leverage: src/scraper/amazon/scraper.py (main function patterns), src/video/producer/cli.py (main function patterns) | _Requirements: All requirements | Success: CLI executes complete pipeline, logging configured correctly, summary displayed at end, exceptions handled gracefully, exit codes appropriate | Instructions: 1) Mark in-progress, 2) Implement main() function, 3) Log implementation with artifacts (CLI entry point integration), 4) Mark complete_

- [x] 11. Extend YAML configuration schema
  - Files: `config/pipeline.yaml` (new file or extend existing)
  - Add `global_batch` section with scraper and video configuration
  - Document schema with inline comments
  - Purpose: Enable persistent pipeline configuration via YAML
  - _Leverage: Existing YAML configuration patterns_
  - _Requirements: 2.7_
  - _Prompt: Implement the task for spec global-pipeline-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Configuration Engineer with expertise in YAML schema design | Task: Create or extend YAML configuration following requirement 2.7. Add global_batch section with scraper settings (product_ids, keywords, max_products, filters) and video settings (profile, random_profile, profile_pool, fail_fast). Add clear comments explaining each field. | Restrictions: Don't modify existing configuration sections, follow existing YAML formatting conventions, ensure backward compatibility | _Leverage: config/scraper.yaml, config/video_production.yaml (existing YAML structure) | _Requirements: 2.7 (Configuration Management) | Success: YAML schema includes all required fields, comments are clear and helpful, YAML is valid and parseable, configuration loads correctly | Instructions: 1) Mark in-progress, 2) Create/update YAML configuration, 3) Log implementation with artifacts (configuration schema), 4) Mark complete_

- [x] 12. Create unit tests for orchestrator
  - Files: `tests/pipeline/test_global_batch_orchestrator.py` (new file)
  - Test configuration loading and validation
  - Test phase coordination (scraping → handoff → production)
  - Test fail-fast behavior between phases
  - Mock scraper and producer for isolated testing
  - Purpose: Ensure orchestrator logic is correct and reliable
  - _Leverage: Existing test utilities and pytest fixtures_
  - _Requirements: 1.1, 6.1_
  - _Prompt: Implement the task for spec global-pipeline-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: QA Engineer with expertise in Python unit testing and mocking | Task: Create comprehensive unit tests for GlobalPipelineOrchestrator in tests/pipeline/test_global_batch_orchestrator.py. Test configuration loading/validation, phase coordination (correct sequence), fail-fast between phases. Mock BotasaurusAmazonScraper and video producer to isolate orchestrator logic. | Restrictions: Test in complete isolation (mock all external dependencies), test both success and failure scenarios, verify phase sequence is correct | _Leverage: pytest fixtures, unittest.mock for mocking scraper and producer | _Requirements: 1.1 (Unified Execution), 6.1 (Error Handling) | Success: All orchestrator methods tested, phase coordination verified, fail-fast behavior confirmed, configuration validation tested, tests pass consistently | Instructions: 1) Mark in-progress, 2) Create test file with comprehensive coverage, 3) Log implementation with artifacts (test functions created), 4) Mark complete_

- [x] 13. Create integration tests for end-to-end pipeline
  - Files: `tests/pipeline/test_global_batch_integration.py` (new file)
  - Test complete pipeline with product IDs
  - Test complete pipeline with keywords
  - Test mixed input mode
  - Test fail-fast at different phases
  - Test configuration precedence
  - Purpose: Verify pipeline works end-to-end
  - _Leverage: Existing integration test patterns_
  - _Requirements: All requirements_
  - _Prompt: Implement the task for spec global-pipeline-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: QA Engineer with expertise in integration testing | Task: Create integration tests for global batch pipeline in tests/pipeline/test_global_batch_integration.py. Test complete flows: product IDs only, keywords only, mixed input, fail-fast at scraping/production phases, CLI vs YAML config precedence. Consider using test fixtures or mocked data to avoid actual scraping/video generation. | Restrictions: Tests should be repeatable and fast, consider mocking external APIs or using recorded responses, ensure cleanup of temporary files, tests should not depend on external state | _Leverage: tests/scraper/ and tests/video/producer/ (integration test patterns), pytest fixtures | _Requirements: All requirements | Success: All pipeline modes tested end-to-end, configuration precedence verified, fail-fast tested at all phases, tests are reliable and repeatable | Instructions: 1) Mark in-progress, 2) Create integration tests, 3) Log implementation with artifacts (test scenarios covered), 4) Mark complete_

- [x] 14. Update documentation with global batch examples
  - Files: `README.md`, `CLAUDE.md`
  - Add global batch pipeline documentation
  - Document CLI usage with examples
  - Document YAML configuration schema
  - Explain phase flow and handoff behavior
  - Purpose: Enable users to discover and use global batch pipeline
  - _Leverage: Existing documentation structure_
  - _Requirements: All requirements_
  - _Prompt: Implement the task for spec global-pipeline-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Technical Writer with expertise in developer documentation | Task: Update README.md and CLAUDE.md with global batch pipeline documentation. Add CLI usage examples (product IDs, keywords, mixed, random profiles), document YAML schema, explain three-phase flow (scraping → handoff → production), show example summary output. Add to CLAUDE.md essential commands if appropriate. | Restrictions: Don't remove existing documentation, maintain consistent formatting, keep examples realistic and tested, explain both CLI and YAML approaches clearly | _Leverage: README.md (existing structure), CLAUDE.md (essential commands) | _Requirements: All requirements | Success: Global batch fully documented with clear examples, CLI arguments documented, YAML schema documented, phase flow explained, summary example shown | Instructions: 1) Mark in-progress, 2) Update documentation, 3) Log implementation with artifacts (documentation sections added), 4) Mark complete_

## Testing Checklist

After implementation, verify:

- [ ] 15.1 Complete pipeline executes: scrape → handoff → produce
- [ ] 15.2 Product IDs input mode works end-to-end
- [ ] 15.3 Keywords input mode works end-to-end
- [ ] 15.4 Mixed input (product IDs + keywords) works correctly
- [ ] 15.5 CLI arguments override YAML configuration
- [ ] 15.6 YAML configuration used when no CLI override
- [ ] 15.7 Validation catches missing inputs before processing
- [ ] 15.8 Validation catches invalid profiles before processing
- [ ] 15.9 Fail-fast stops pipeline after scraping phase failure
- [ ] 15.10 Fail-fast stops pipeline after production phase failure
- [ ] 15.11 Graceful continuation works (default behavior)
- [ ] 15.12 Handoff phase filters products by media availability
- [ ] 15.13 Zero products ready case exits gracefully
- [ ] 15.14 Fixed profile mode works in production phase
- [ ] 15.15 Random profile mode works in production phase
- [ ] 15.16 Scraping summary shows correct statistics
- [ ] 15.17 Production summary shows correct statistics
- [ ] 15.18 Final summary shows end-to-end statistics
- [ ] 15.19 Profile distribution shown when randomization enabled
- [ ] 15.20 All unit tests pass with good coverage
- [ ] 15.21 All integration tests pass reliably
