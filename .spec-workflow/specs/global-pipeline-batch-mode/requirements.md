# Requirements Document

## Introduction

The global pipeline batch mode feature enables end-to-end automated content production by orchestrating both scraping and video generation in a single command execution. This feature creates a seamless workflow from product discovery to video delivery, eliminating manual handoffs and enabling fully automated content pipelines at scale.

The implementation creates a new CLI module `src.pipeline.global_batch` that coordinates the scraper and video producer modules in sequence, passing scraped products directly to video production with unified configuration and comprehensive reporting.

## Alignment with Product Vision

This feature directly supports ContentEngineAI's vision of fully automated video production by:
- **End-to-End Automation**: Single command from product discovery to video delivery
- **Workflow Integration**: Seamless handoff between scraping and video production
- **Unified Configuration**: Single configuration for entire pipeline with consistent settings
- **Scalability**: Process multiple products through complete pipeline efficiently

## Requirements

### Requirement 1: Unified Pipeline Execution

**User Story:** As a content creator, I want to scrape products and generate videos in a single command execution, so that I can automate the entire workflow without manual intervention.

#### Acceptance Criteria

1. WHEN `--global-batch` flag provided THEN system SHALL execute scraping phase followed by video production phase
2. WHEN scraping phase completes THEN system SHALL automatically transition to video production phase without user intervention
3. WHEN pipeline executes THEN system SHALL log clear phase headers "SCRAPING PHASE" and "VIDEO PRODUCTION PHASE"
4. WHEN both phases complete THEN system SHALL generate comprehensive end-to-end summary
5. WHEN pipeline timeout configured THEN system SHALL apply separate timeouts per phase (not single timeout for entire pipeline)

### Requirement 2: Input Configuration

**User Story:** As a content creator, I want to configure the pipeline with product IDs, keywords, and video settings in a single unified configuration, so that I have consistent settings across both phases.

#### Acceptance Criteria

1. WHEN `--product-ids` specified THEN system SHALL scrape those product IDs in scraping phase
2. WHEN `--keywords` specified THEN system SHALL search for products using those keywords
3. WHEN both `--product-ids` and `--keywords` provided THEN system SHALL process product IDs first, then keyword searches
4. WHEN `--profile` or `--batch-profile` specified THEN system SHALL use that profile for all videos in production phase
5. WHEN `--random-profile` enabled THEN system SHALL randomly assign profiles in production phase
6. WHEN CLI arguments provided AND YAML configuration exists THEN CLI SHALL override YAML settings
7. WHEN no CLI arguments AND YAML `global_batch` section exists THEN system SHALL use YAML configuration

### Requirement 3: Scraping Phase Execution

**User Story:** As a content creator, I want the scraping phase to collect product data and media files, so that I have all necessary inputs ready for video production.

#### Acceptance Criteria

1. WHEN scraping phase starts THEN system SHALL log "SCRAPING PHASE" header with total items to process
2. WHEN processing each item THEN system SHALL display `[N/total] Scraping product: {product_id}` format
3. WHEN product scraping completes THEN system SHALL create `data.json` in `outputs/{product_id}/` directory
4. WHEN product scraping fails THEN system SHALL log error and continue to next product (unless fail-fast enabled)
5. WHEN scraping phase completes THEN system SHALL log summary: total attempted, successful, failed products
6. WHEN scraping phase completes THEN system SHALL report media collection statistics (images, videos per product)

### Requirement 4: Handoff Phase

**User Story:** As a content creator, I want the pipeline to automatically identify scraped products ready for video production, so that only products with sufficient media are processed.

#### Acceptance Criteria

1. WHEN scraping phase completes THEN system SHALL scan `outputs/` directory for products with `data.json` files
2. WHEN checking media availability THEN system SHALL filter products based on video profile media requirements
3. WHEN product lacks required media THEN system SHALL skip it in production phase and log reason
4. WHEN building production list THEN system SHALL log transition message with count of ready products
5. WHEN no products ready THEN system SHALL exit gracefully with message "No products with sufficient media for video production"

### Requirement 5: Video Production Phase Execution

**User Story:** As a content creator, I want all successfully scraped products to be processed through video production automatically, so that I get complete videos without additional commands.

#### Acceptance Criteria

1. WHEN production phase starts THEN system SHALL log "VIDEO PRODUCTION PHASE" header with total products
2. WHEN processing each product THEN system SHALL display `[N/total] Producing video: {product_id}` format
3. WHEN video profile specified THEN system SHALL apply that profile to all products
4. WHEN profile randomization enabled THEN system SHALL select random profile per product with deterministic seeding
5. WHEN video production completes THEN system SHALL log success with output path
6. WHEN video production fails THEN system SHALL log error and continue (unless fail-fast enabled)
7. WHEN production phase completes THEN system SHALL log summary: attempted, successful, failed, skipped

### Requirement 6: Error Handling and Resilience

**User Story:** As a content creator, I want the pipeline to handle errors gracefully and continue processing, so that one failure doesn't stop my entire batch operation.

#### Acceptance Criteria

1. WHEN scraping fails for product AND `--fail-fast` not enabled THEN system SHALL log error, mark as failed, continue to next product
2. WHEN scraping phase fails AND `--fail-fast` enabled THEN system SHALL stop pipeline immediately before video production
3. WHEN video production fails for product AND `--fail-fast` not enabled THEN system SHALL log error, mark as failed, continue
4. WHEN video production fails AND `--fail-fast` enabled THEN system SHALL stop pipeline immediately
5. WHEN partial success occurs THEN system SHALL generate summary showing both successful and failed products
6. WHEN scraping succeeds but video production fails THEN system SHALL preserve scraped data (no rollback)

### Requirement 7: Configuration Management

**User Story:** As a content creator, I want flexible configuration through CLI and YAML, so that I can choose between runtime customization and persistent pipeline definitions.

#### Acceptance Criteria

1. WHEN YAML `global_batch.enabled: true` THEN system SHALL use YAML configuration for pipeline
2. WHEN CLI `--product-ids` provided THEN system SHALL override `global_batch.scraper.product_ids`
3. WHEN CLI `--keywords` provided THEN system SHALL override `global_batch.scraper.keywords`
4. WHEN CLI `--profile` provided THEN system SHALL override `global_batch.video.profile`
5. WHEN CLI `--random-profile` provided THEN system SHALL override `global_batch.video.random_profile`
6. WHEN no input provided (no product IDs or keywords) THEN system SHALL error "No product IDs or keywords provided"
7. WHEN configuration validated THEN system SHALL check all profiles exist before pipeline starts

### Requirement 8: Comprehensive Summary Reporting

**User Story:** As a content creator, I want detailed summary statistics for both phases and overall pipeline, so that I can understand complete pipeline performance and identify issues.

#### Acceptance Criteria

1. WHEN scraping phase completes THEN system SHALL report: total attempted, successful (with data.json created), failed (with product IDs)
2. WHEN scraping phase completes THEN system SHALL report media collection statistics
3. WHEN production phase completes THEN system SHALL report: total attempted, successful (with output paths), failed (with product IDs), skipped (insufficient media)
4. WHEN profile randomization enabled THEN system SHALL report profile usage distribution
5. WHEN pipeline completes THEN system SHALL report end-to-end success count (scraped AND produced)
6. WHEN pipeline completes THEN system SHALL report partial success count (scraped but not produced)
7. WHEN pipeline completes THEN system SHALL report total failures and total pipeline duration

### Requirement 9: Performance and Resource Management

**User Story:** As a content creator, I want the pipeline to efficiently manage resources across both phases, so that processing is fast and doesn't consume excessive memory.

#### Acceptance Criteria

1. WHEN pipeline executes THEN system SHALL share HTTP session pool across both scraping and video production
2. WHEN pipeline executes THEN system SHALL maintain background processor context across entire pipeline
3. WHEN scraping phase completes THEN system SHALL clean up scraper-specific resources before video production
4. WHEN video production completes for product THEN system SHALL clean up per-product resources while maintaining global context
5. WHEN pipeline timeout configured THEN system SHALL apply timeout per product (not entire pipeline)

## Non-Functional Requirements

### Code Architecture and Modularity
- **Single Responsibility Principle**: Global batch orchestrator coordinates existing scraper and producer modules without duplicating logic
- **Modular Design**: Pipeline module should invoke existing scraper and producer as black boxes
- **Dependency Management**: Minimize coupling between pipeline orchestrator and individual phase implementations
- **Clear Interfaces**: Define clean contracts between pipeline controller and phase executors

### Performance
- **Sequential Phase Execution**: Complete scraping before starting video production
- **Sequential Product Processing**: Process products one at a time within each phase
- **Resource Pooling**: Share HTTP sessions and background processor across entire pipeline
- **Memory Efficiency**: Clean up phase-specific resources after each phase completion

### Security
- **Configuration Validation**: Validate all inputs before starting any processing
- **Path Validation**: Sanitize product IDs before creating file paths
- **API Key Protection**: Never log or expose API keys in pipeline output

### Reliability
- **Phase Isolation**: Scraping failures don't prevent video production of successful scrapes
- **Product Isolation**: Individual product failures don't stop pipeline for other products
- **Data Integrity**: Preserve scraped data even if video production fails
- **Error Recovery**: Provide clear error messages for troubleshooting

### Usability
- **Clear Progress**: Real-time progress updates with phase headers and `[N/total]` format
- **Comprehensive Logging**: Detailed logs with DEBUG mode for troubleshooting
- **Unified Command**: Single command with intuitive arguments
- **Helpful Errors**: Clear validation errors with actionable guidance
