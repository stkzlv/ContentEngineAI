# Requirements Document: Batch Processing Module

## Introduction

This spec defines the complete requirements for the ContentEngineAI Batch Processing Module (Global Pipeline), which orchestrates end-to-end automated content production from product discovery to video publishing. The pipeline executes four sequential phases: scraping, handoff, video production, and publishing.

The implementation provides a unified CLI module `src.pipeline.global_batch` that coordinates scraper, video producer, and publisher modules in sequence with unified configuration and comprehensive reporting.

## Alignment with Product Vision

The Batch Processing Module directly supports the product principles defined in product.md:

- **Automation Over Manual Intervention**: Single command from product discovery to video delivery and publishing
- **Modular Flexibility**: Treats scraper, producer, and publisher as black boxes with clean interfaces
- **Fail Gracefully**: Phase isolation ensures failures don't cascade; individual product failures don't stop pipeline
- **Performance at Scale**: Sequential phase execution with resource pooling handles large product batches

## Requirements

### Section 1: Pipeline Execution

#### Requirement 1: Unified Pipeline Execution

**User Story:** As a content creator, I want to scrape products, generate videos, and publish them in a single command execution, so that I can automate the entire workflow without manual intervention.

##### Acceptance Criteria

1. WHEN `python -m src.pipeline.global_batch` runs THEN system SHALL execute scraping → handoff → production → publishing phases sequentially
2. WHEN each phase completes THEN system SHALL automatically transition to next phase without user intervention
3. WHEN pipeline executes THEN system SHALL log clear phase headers "SCRAPING PHASE", "VIDEO PRODUCTION PHASE", "PUBLISHING PHASE"
4. WHEN all phases complete THEN system SHALL generate comprehensive end-to-end summary
5. WHEN pipeline timeout configured THEN system SHALL apply separate timeouts per phase (not single timeout for entire pipeline)

#### Requirement 2: Input Configuration

**User Story:** As a content creator, I want to configure the pipeline with product IDs, keywords, video profiles, and publishing settings in a single unified configuration.

##### Acceptance Criteria

1. WHEN `--product-ids` specified THEN system SHALL scrape those product IDs in scraping phase
2. WHEN `--keywords` specified THEN system SHALL search for products using those keywords
3. WHEN both `--product-ids` and `--keywords` provided THEN system SHALL process product IDs first, then keyword searches
4. WHEN `--profile` or `--batch-profile` specified THEN system SHALL use that profile for all videos in production phase
5. WHEN `--random-profile` enabled THEN system SHALL randomly assign profiles in production phase
6. WHEN `--skip-publish` specified THEN system SHALL skip publishing phase entirely
7. WHEN CLI arguments provided AND YAML configuration exists THEN CLI SHALL override YAML settings
8. WHEN no CLI arguments AND YAML `global_batch` section exists THEN system SHALL use YAML configuration

### Section 2: Scraping Phase

#### Requirement 3: Scraping Phase Execution

**User Story:** As a content creator, I want the scraping phase to collect product data and media files, so that I have all necessary inputs ready for video production.

##### Acceptance Criteria

1. WHEN scraping phase starts THEN system SHALL log "SCRAPING PHASE" header with total items to process
2. WHEN processing each item THEN system SHALL display `[N/total] Scraping product: {product_id}` format
3. WHEN product scraping completes THEN system SHALL create `data.json` in `outputs/{product_id}/` directory
4. WHEN product scraping fails THEN system SHALL log error and continue to next product (unless fail-fast enabled)
5. WHEN scraping phase completes THEN system SHALL log summary: total attempted, successful, failed products
6. WHEN scraping phase completes THEN system SHALL report media collection statistics (images, videos per product)

### Section 3: Handoff Phase

#### Requirement 4: Handoff Phase

**User Story:** As a content creator, I want the pipeline to automatically identify scraped products ready for video production, so that only products with sufficient media are processed.

##### Acceptance Criteria

1. WHEN scraping phase completes THEN system SHALL scan `outputs/` directory for products with `data.json` files
2. WHEN checking media availability THEN system SHALL filter products based on video profile media requirements
3. WHEN product lacks required media THEN system SHALL skip it in production phase and log reason
4. WHEN building production list THEN system SHALL log transition message with count of ready products
5. WHEN no products ready THEN system SHALL exit gracefully with message "No products with sufficient media for video production"

### Section 4: Video Production Phase

#### Requirement 5: Video Production Phase Execution

**User Story:** As a content creator, I want all successfully scraped products to be processed through video production automatically, so that I get complete videos without additional commands.

##### Acceptance Criteria

1. WHEN production phase starts THEN system SHALL log "VIDEO PRODUCTION PHASE" header with total products
2. WHEN processing each product THEN system SHALL display `[N/total] Producing video: {product_id}` format
3. WHEN video profile specified THEN system SHALL apply that profile to all products
4. WHEN profile randomization enabled THEN system SHALL select random profile per product with deterministic seeding
5. WHEN video production completes THEN system SHALL log success with output path
6. WHEN video production fails THEN system SHALL log error and continue (unless fail-fast enabled)
7. WHEN production phase completes THEN system SHALL log summary: attempted, successful, failed, skipped

### Section 5: Publishing Phase

#### Requirement 6: Publishing Phase Execution

**User Story:** As a content creator, I want produced videos automatically published to social media platforms, so that content reaches audiences without manual uploading.

##### Acceptance Criteria

1. WHEN publishing phase starts THEN system SHALL log "PUBLISHING PHASE" header with total videos
2. WHEN processing each video THEN system SHALL display `[N/total] Publishing video for {product_id}` format
3. WHEN platforms configured THEN system SHALL publish to each platform sequentially
4. WHEN `--schedule-time` provided THEN system SHALL schedule videos for specified time
5. WHEN auto-scheduling enabled THEN system SHALL find next available recurring slot per platform
6. WHEN publishing succeeds THEN system SHALL log success per platform with post ID
7. WHEN publishing fails THEN system SHALL log error and continue (unless fail-fast-publish enabled)
8. WHEN `--skip-publish` specified THEN system SHALL skip entire publishing phase

#### Requirement 7: Auto-Scheduling

**User Story:** As a content creator, I want videos automatically scheduled to optimal time slots, so that I maintain consistent posting cadence.

##### Acceptance Criteria

1. WHEN `immediate_publish: false` in config AND recurring_schedule enabled THEN system SHALL auto-schedule
2. WHEN finding available slot THEN system SHALL check existing posts to avoid conflicts
3. WHEN slot conflicts exist THEN system SHALL find next available slot
4. WHEN no slots available within horizon THEN system SHALL publish immediately with warning
5. WHEN stagger delay configured THEN system SHALL apply delay between platform publishes

### Section 6: Error Handling

#### Requirement 8: Error Handling and Resilience

**User Story:** As a content creator, I want the pipeline to handle errors gracefully and continue processing, so that one failure doesn't stop my entire batch operation.

##### Acceptance Criteria

1. WHEN scraping fails for product AND `--fail-fast` not enabled THEN system SHALL log error, mark as failed, continue
2. WHEN scraping phase fails AND `--fail-fast` enabled THEN system SHALL stop pipeline immediately
3. WHEN video production fails AND `--fail-fast` not enabled THEN system SHALL log error, mark as failed, continue
4. WHEN video production fails AND `--fail-fast` enabled THEN system SHALL stop pipeline immediately
5. WHEN publishing fails AND `--fail-fast-publish` not enabled THEN system SHALL log error, continue
6. WHEN publishing fails AND `--fail-fast-publish` enabled THEN system SHALL stop publishing phase immediately
7. WHEN partial success occurs THEN system SHALL generate summary showing successful and failed products
8. WHEN scraping succeeds but later phases fail THEN system SHALL preserve scraped data (no rollback)

### Section 7: Configuration

#### Requirement 9: Configuration Management

**User Story:** As a content creator, I want flexible configuration through CLI and YAML, so that I can choose between runtime customization and persistent pipeline definitions.

##### Acceptance Criteria

1. WHEN YAML `global_batch.enabled: true` THEN system SHALL use YAML configuration for pipeline
2. WHEN CLI arguments provided THEN system SHALL override corresponding YAML settings
3. WHEN no input provided (no product IDs or keywords) THEN system SHALL error "No product IDs or keywords provided"
4. WHEN publishing enabled AND LATE_API_KEY missing THEN system SHALL error with clear message
5. WHEN configuration validated THEN system SHALL check all profiles exist before pipeline starts

### Section 8: Summary Reporting

#### Requirement 10: Comprehensive Summary Reporting

**User Story:** As a content creator, I want detailed summary statistics for all phases and overall pipeline, so that I can understand complete pipeline performance.

##### Acceptance Criteria

1. WHEN scraping phase completes THEN system SHALL report: total attempted, successful, failed, media statistics
2. WHEN production phase completes THEN system SHALL report: attempted, successful, failed, skipped, profile distribution
3. WHEN publishing phase completes THEN system SHALL report: attempted, successful, failed, skipped, per-platform results
4. WHEN pipeline completes THEN system SHALL report end-to-end success count (scraped AND produced AND published)
5. WHEN pipeline completes THEN system SHALL report partial success count (scraped but not fully completed)
6. WHEN pipeline completes THEN system SHALL report total failures and total pipeline duration

### Section 9: Performance

#### Requirement 11: Performance and Resource Management

**User Story:** As a content creator, I want the pipeline to efficiently manage resources across all phases, so that processing is fast and doesn't consume excessive memory.

##### Acceptance Criteria

1. WHEN pipeline executes THEN system SHALL share HTTP session pool across all phases
2. WHEN pipeline executes THEN system SHALL maintain background processor context across entire pipeline
3. WHEN phase completes THEN system SHALL clean up phase-specific resources before next phase
4. WHEN product completes THEN system SHALL clean up per-product resources while maintaining global context
5. WHEN pipeline timeout configured THEN system SHALL apply timeout per product (not entire pipeline)

## Non-Functional Requirements

### Code Architecture and Modularity

- **Single Responsibility Principle**: GlobalPipelineOrchestrator coordinates existing modules without duplicating logic
- **Modular Design**: Pipeline module invokes scraper, producer, and publisher as black boxes
- **Dependency Management**: Minimize coupling between pipeline orchestrator and phase implementations
- **Clear Interfaces**: Define clean contracts between pipeline controller and phase executors

### Performance

- **Sequential Phase Execution**: Complete each phase before starting next
- **Sequential Product Processing**: Process products one at a time within each phase
- **Resource Pooling**: Share HTTP sessions and background processor across entire pipeline
- **Memory Efficiency**: Clean up phase-specific resources after each phase completion

### Security

- **Configuration Validation**: Validate all inputs before starting any processing
- **Path Validation**: Sanitize product IDs before creating file paths
- **API Key Protection**: Never log or expose API keys in pipeline output
- **Credential Validation**: Verify LATE_API_KEY present if publishing enabled

### Reliability

- **Phase Isolation**: Scraping failures don't prevent video production of successful scrapes
- **Product Isolation**: Individual product failures don't stop pipeline for other products
- **Data Integrity**: Preserve scraped data even if video production or publishing fails
- **Error Recovery**: Provide clear error messages for troubleshooting

### Usability

- **Clear Progress**: Real-time progress updates with phase headers and `[N/total]` format
- **Comprehensive Logging**: Detailed logs with DEBUG mode for troubleshooting
- **Unified Command**: Single command with intuitive arguments
- **Helpful Errors**: Clear validation errors with actionable guidance
