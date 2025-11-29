# Tasks Document

## Implementation Tasks

- [ ] 1. Create profile selection utility functions
  - Files: `src/video/producer/utils.py`
  - Add `select_profile_for_product()` function with deterministic random selection
  - Add `load_profile_pool()` function for CLI > YAML > defaults precedence
  - Add `ProfileUsageTracker` class for statistics tracking
  - Purpose: Provide reusable utilities for profile randomization logic
  - _Leverage: Existing VideoConfig for profile validation_
  - _Requirements: 2.3, 3.3_
  - _Prompt: Implement the task for spec producer-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Python Developer specializing in utility functions and algorithms | Task: Create profile selection utilities in src/video/producer/utils.py following requirements 2.3 and 3.3. Implement select_profile_for_product() with deterministic seeding (hash product ID for seed), load_profile_pool() with 3-tier precedence, ProfileUsageTracker for statistics. | Restrictions: Do not modify existing utility functions, use hash() for deterministic seeding (not random.seed() directly in production code), ensure reproducibility (same product ID = same profile), follow modern Python typing | _Leverage: src/video/config.py (VideoConfig for profile validation) | _Requirements: 2.3 (Profile Randomization), 3.3 (Profile Randomization for Global Batch) | Success: select_profile_for_product() returns same profile for same product ID consistently, load_profile_pool() correctly implements precedence, ProfileUsageTracker accurately counts usage, all functions have proper type hints | Instructions: 1) Mark in-progress in tasks.md, 2) Implement utility functions, 3) Log implementation with artifacts (functions with signatures and purpose), 4) Mark complete_

- [ ] 2. Add CLI arguments for profile randomization
  - Files: `src/video/producer/cli.py`
  - Add `--random-profile` flag (action="store_true")
  - Add `--profile-pool` argument (nargs="+", type=str)
  - Add mutual exclusivity validation between `--batch-profile` and `--random-profile`
  - Purpose: Enable profile randomization through command-line interface
  - _Leverage: Existing argparse argument parser in create_argument_parser()_
  - _Requirements: 2.3, 2.6_
  - _Prompt: Implement the task for spec producer-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: CLI Developer with expertise in argparse and argument validation | Task: Add profile randomization CLI arguments to src/video/producer/cli.py following requirements 2.3 and 2.6. Add --random-profile flag and --profile-pool argument, implement validation that prevents using both --batch-profile and --random-profile simultaneously, ensure --random-profile requires --batch. | Restrictions: Do not remove or break existing CLI arguments, maintain backward compatibility, add validation in argument parsing section (not later in code flow), provide clear error messages | _Leverage: src/video/producer/cli.py (create_argument_parser() function, existing validation patterns) | _Requirements: 2.3 (Profile Randomization), 2.6 (Configuration and CLI Interface) | Success: Both arguments added correctly with proper help text, mutual exclusivity validation works and shows clear error, --random-profile validation requires --batch, backward compatibility maintained | Instructions: 1) Mark in-progress, 2) Add arguments and validation to parser, 3) Log implementation with artifacts (CLI arguments added), 4) Mark complete_

- [ ] 3. Extend YAML configuration schema for profile pool
  - Files: `config/video_production.yaml`
  - Add `profile_pool: []` to batch configuration section
  - Document schema with comments explaining empty list means all profiles
  - Purpose: Enable persistent profile pool configuration via YAML
  - _Leverage: Existing batch configuration structure_
  - _Requirements: 2.6_
  - _Prompt: Implement the task for spec producer-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Configuration Engineer with expertise in YAML schema design | Task: Extend YAML configuration in config/video_production.yaml following requirement 2.6. Add profile_pool list to batch section, add clear comments explaining that empty list defaults to all available profiles, maintain existing configuration structure. | Restrictions: Do not modify existing configuration fields, ensure backward compatibility (missing profile_pool should be treated as empty list), follow existing YAML formatting conventions | _Leverage: config/video_production.yaml (existing batch configuration structure) | _Requirements: 2.6 (Configuration and CLI Interface) | Success: profile_pool field added to batch section with proper indentation, comments clearly explain behavior, YAML is valid and parseable | Instructions: 1) Mark in-progress, 2) Update YAML schema with profile_pool, 3) Log implementation with artifacts (configuration schema added), 4) Mark complete_

- [ ] 4. Integrate profile randomization into batch loop
  - Files: `src/video/producer/cli.py`
  - Modify batch processing loop (lines ~571-638) to select profile per product
  - Call `select_profile_for_product()` or use fixed `batch_profile` based on mode
  - Log selected profile for each product
  - Track profile usage with ProfileUsageTracker
  - Purpose: Enable per-product profile selection in batch mode
  - _Leverage: Existing batch loop, select_profile_for_product() from task 1_
  - _Requirements: 2.2, 2.3, 2.4_
  - _Prompt: Implement the task for spec producer-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Python Developer specializing in control flow and integration | Task: Integrate profile randomization into batch processing loop in src/video/producer/cli.py following requirements 2.2, 2.3, 2.4. Modify existing batch loop to check random_profile mode, call select_profile_for_product() when randomization enabled or use batch_profile when fixed, log selected profile before create_video_for_product() call, track usage with ProfileUsageTracker. | Restrictions: Do not break existing batch mode with fixed profile, maintain sequential processing, ensure profile logged before video creation, do not modify create_video_for_product() signature | _Leverage: src/video/producer/cli.py (existing batch loop ~lines 571-638), src/video/producer/utils.py (select_profile_for_product, ProfileUsageTracker from task 1) | _Requirements: 2.2 (Batch Execution), 2.3 (Profile Randomization), 2.4 (Progress Tracking) | Success: Profile selection integrated correctly, logs show selected profile per product, ProfileUsageTracker records all usages, fixed profile mode still works unchanged | Instructions: 1) Mark in-progress, 2) Modify batch loop with profile selection logic, 3) Log implementation with artifacts (integration points, control flow changes), 4) Mark complete_

- [ ] 5. Implement profile pool loading with precedence
  - Files: `src/video/producer/cli.py`
  - Load profile pool using `load_profile_pool()` from task 1
  - Implement CLI > YAML > all profiles precedence
  - Validate all profiles in pool exist before batch starts
  - Purpose: Ensure correct profile pool configuration with validation
  - _Leverage: load_profile_pool() from task 1, VideoConfig for validation_
  - _Requirements: 2.3, 2.6_
  - _Prompt: Implement the task for spec producer-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Configuration Engineer with expertise in configuration loading and validation | Task: Implement profile pool loading in src/video/producer/cli.py following requirements 2.3 and 2.6. Call load_profile_pool() with CLI args, YAML config, and VideoConfig, validate all profiles exist before batch processing starts, raise clear error listing invalid profiles if validation fails. | Restrictions: Validation must happen before any video processing, use VideoConfig to check profile existence, provide actionable error messages listing available profiles when validation fails | _Leverage: src/video/producer/utils.py (load_profile_pool from task 1), src/video/config.py (VideoConfig for profile validation) | _Requirements: 2.3 (Profile Randomization), 2.6 (Configuration and CLI) | Success: Profile pool loaded correctly with precedence, validation catches invalid profiles before processing, error messages are clear and actionable, empty pool defaults to all profiles | Instructions: 1) Mark in-progress, 2) Add pool loading and validation logic, 3) Log implementation with artifacts (configuration loading logic), 4) Mark complete_

- [ ] 6. Extend summary reporting with profile statistics
  - Files: `src/video/producer/cli.py`
  - Add profile usage distribution to final summary (lines ~640-665)
  - Format statistics using ProfileUsageTracker.format_summary()
  - Include statistics only when randomization enabled
  - Purpose: Provide visibility into profile distribution across batch
  - _Leverage: ProfileUsageTracker from task 1, existing summary reporting_
  - _Requirements: 2.3, 2.7_
  - _Prompt: Implement the task for spec producer-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Python Developer with expertise in reporting and data presentation | Task: Extend batch summary reporting in src/video/producer/cli.py following requirements 2.3 and 2.7. Add profile usage distribution section to final summary, call ProfileUsageTracker.format_summary() to get formatted stats, only include when randomization enabled (not for fixed profile mode). | Restrictions: Do not break existing summary format, add profile stats as additional section (not replacement), only show when randomization was used, maintain readable formatting | _Leverage: src/video/producer/cli.py (existing summary reporting ~lines 640-665), src/video/producer/utils.py (ProfileUsageTracker.format_summary from task 1) | _Requirements: 2.3 (Profile Randomization), 2.7 (Summary Reporting) | Success: Profile distribution appears in summary when randomization used, statistics are accurate and readable, fixed profile mode summary unchanged, formatting is consistent with existing summary | Instructions: 1) Mark in-progress, 2) Add profile statistics to summary, 3) Log implementation with artifacts (summary reporting changes), 4) Mark complete_

- [ ] 7. Create unit tests for profile selection utilities
  - Files: `tests/video/producer/test_profile_selection.py` (new file)
  - Test `select_profile_for_product()` determinism (same ID → same profile)
  - Test `load_profile_pool()` precedence (CLI > YAML > all profiles)
  - Test `ProfileUsageTracker` counting accuracy
  - Test profile validation catches invalid profiles
  - Purpose: Ensure profile selection logic is correct and reliable
  - _Leverage: Existing test utilities and pytest fixtures_
  - _Requirements: 2.3_
  - _Prompt: Implement the task for spec producer-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: QA Engineer with expertise in Python unit testing and pytest | Task: Create comprehensive unit tests for profile selection utilities in tests/video/producer/test_profile_selection.py following requirement 2.3. Test deterministic behavior (multiple calls with same product ID return same profile), test precedence logic (CLI overrides YAML, YAML overrides defaults), test ProfileUsageTracker accuracy, verify validation catches invalid profiles. | Restrictions: Test in isolation (mock VideoConfig), test both success and failure scenarios, verify determinism with multiple iterations, do not test actual VideoConfig internals | _Leverage: pytest fixtures, unittest.mock for mocking VideoConfig | _Requirements: 2.3 (Profile Randomization) | Success: All utility functions tested thoroughly, determinism verified with statistical tests, precedence logic confirmed, ProfileUsageTracker counts verified, tests pass consistently | Instructions: 1) Mark in-progress, 2) Create test file with comprehensive coverage, 3) Log implementation with artifacts (test functions and what they verify), 4) Mark complete_

- [ ] 8. Create integration tests for batch profile randomization
  - Files: `tests/video/producer/test_batch_profile_integration.py` (new file)
  - Test end-to-end batch with random profile selection
  - Test CLI override of YAML profile pool
  - Test mutual exclusivity validation (--batch-profile vs --random-profile)
  - Test profile usage distribution in summary
  - Purpose: Verify profile randomization works end-to-end in batch mode
  - _Leverage: Existing integration test patterns_
  - _Requirements: 2.3, 2.6, 2.7_
  - _Prompt: Implement the task for spec producer-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: QA Engineer with expertise in integration testing | Task: Create integration tests for batch profile randomization in tests/video/producer/test_batch_profile_integration.py following requirements 2.3, 2.6, 2.7. Test complete batch flow with random profiles, verify CLI overrides YAML, test validation errors (mutual exclusivity, invalid profiles), verify profile distribution appears in summary. Use test fixtures or mock data to avoid actual video generation. | Restrictions: Tests should be repeatable, consider using fixtures for test products, may need to mock create_video_for_product to avoid actual video generation, ensure tests clean up temporary files | _Leverage: tests/video/producer/ (existing integration test patterns), pytest fixtures for test data | _Requirements: 2.3 (Randomization), 2.6 (Configuration), 2.7 (Summary) | Success: All batch modes tested end-to-end, configuration precedence verified, validation errors caught correctly, profile distribution in summary verified, tests are reliable and repeatable | Instructions: 1) Mark in-progress, 2) Create integration tests, 3) Log implementation with artifacts (test scenarios covered), 4) Mark complete_

- [ ] 9. Update documentation with profile randomization examples
  - Files: `README.md`, `CLAUDE.md`
  - Add profile randomization usage examples
  - Document `--random-profile` and `--profile-pool` arguments
  - Explain deterministic behavior and reproducibility
  - Add YAML configuration examples
  - Purpose: Enable users to discover and use profile randomization
  - _Leverage: Existing documentation structure_
  - _Requirements: All requirements_
  - _Prompt: Implement the task for spec producer-batch-mode, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Technical Writer with expertise in developer documentation | Task: Update README.md and CLAUDE.md with profile randomization documentation covering all requirements. Add CLI usage examples (--random-profile, --profile-pool), explain deterministic behavior (same product = same profile), document YAML configuration, show profile distribution in summary examples. Consider adding to CLAUDE.md essential commands if appropriate. | Restrictions: Do not remove existing documentation, maintain consistent formatting, keep examples realistic and tested, explain both CLI and YAML approaches | _Leverage: README.md (existing structure), CLAUDE.md (essential commands section) | _Requirements: All requirements (2.1-2.8) | Success: Profile randomization fully documented with clear examples, CLI arguments documented, YAML schema documented, deterministic behavior explained, summary statistics example shown | Instructions: 1) Mark in-progress, 2) Update documentation with examples, 3) Log implementation with artifacts (documentation sections added), 4) Mark complete_

## Testing Checklist

After implementation, verify:

- [ ] 10.1 Random profile selection assigns different profiles to different products
- [ ] 10.2 Same product ID consistently gets same profile (deterministic)
- [ ] 10.3 CLI --profile-pool overrides YAML configuration
- [ ] 10.4 YAML profile_pool used when no CLI override
- [ ] 10.5 Empty/missing pool defaults to all available profiles
- [ ] 10.6 Mutual exclusivity: cannot use --batch-profile AND --random-profile
- [ ] 10.7 --random-profile requires --batch mode
- [ ] 10.8 Invalid profiles in pool detected before processing starts
- [ ] 10.9 Profile selection logged for each product during batch
- [ ] 10.10 Profile usage distribution shown in summary
- [ ] 10.11 Profile statistics only shown when randomization enabled
- [ ] 10.12 Fixed profile mode (--batch-profile) still works unchanged
- [ ] 10.13 Products with incompatible media skip gracefully
- [ ] 10.14 All unit tests pass with good coverage
- [ ] 10.15 All integration tests pass reliably
