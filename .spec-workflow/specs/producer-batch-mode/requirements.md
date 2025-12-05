# Requirements Document

## Introduction

The producer batch mode feature enables automated multi-product video generation by processing all products in the `outputs/` directory that have been previously scraped. This enhancement extends the existing single-product video producer to support batch operations with configurable video profiles, including profile randomization for diverse output.

The feature addresses the need for efficient bulk video production workflows, reducing manual intervention and enabling automated content pipelines at scale.

## Alignment with Product Vision

This feature directly supports ContentEngineAI's vision of automated, scalable video production by:
- **Automation**: Eliminating manual per-product video generation through batch discovery
- **Scalability**: Enabling video production for multiple products in single execution
- **Flexibility**: Supporting both fixed profiles and randomized profile selection
- **Efficiency**: Reducing operational overhead through unified batch workflows

## Requirements

### Requirement 1: Automatic Product Discovery

**User Story:** As a content creator, I want the video producer to automatically discover all scraped products in the outputs directory, so that I can generate videos for all products without manually specifying each one.

#### Acceptance Criteria

1. WHEN batch mode enabled THEN system SHALL scan `outputs/` directory for product subdirectories containing `data.json` files
2. WHEN scanning directories THEN system SHALL skip global directories (cache, logs, reports, coverage, error_logs, output, outputs, performance_history, unknown_product)
3. WHEN `data.json` contains single product object THEN system SHALL parse as single product
4. WHEN `data.json` contains array of products THEN system SHALL parse as product list
5. WHEN product identification needed THEN system SHALL extract ID from ASIN, title, or fallback to directory name
6. WHEN `--product-index` specified THEN system SHALL process only that zero-based index from discovered products

### Requirement 2: Batch Execution with Profile Consistency

**User Story:** As a content creator, I want to apply the same video profile to all products in a batch, so that I have consistent video style across my product catalog.

#### Acceptance Criteria

1. WHEN `--batch` flag provided THEN system SHALL enable batch processing mode
2. WHEN `--batch-profile` specified THEN system SHALL apply that profile to all products in batch
3. WHEN batch mode enabled AND no `--batch-profile` provided THEN system SHALL error with message "Batch profile required"
4. WHEN processing products THEN system SHALL process one at a time sequentially
5. WHEN inter-product delay configured THEN system SHALL wait random time between `inter_product_delay_min_sec` and `inter_product_delay_max_sec` after each product

### Requirement 3: Profile Randomization

**User Story:** As a content creator, I want to randomly assign different video profiles to products in my batch, so that I have diverse video styles across my catalog without manual profile selection.

#### Acceptance Criteria

1. WHEN `--random-profile` flag provided THEN system SHALL enable random profile selection per product
2. WHEN `--profile-pool` specified THEN system SHALL randomly select from that list of profiles
3. WHEN `--random-profile` enabled AND no `--profile-pool` THEN system SHALL use all available profiles from VideoConfig
4. WHEN profile selected THEN system SHALL use product ID as random seed for deterministic selection (same product always gets same profile)
5. WHEN `--batch-profile` and `--random-profile` both provided THEN system SHALL error with "Cannot use both --batch-profile and --random-profile"
6. WHEN profile selected for product THEN system SHALL log "Selected profile: {profile_name} for product: {product_id}"
7. WHEN profile incompatible with available media THEN system SHALL skip product and log as "SKIPPED" (not failure)
8. WHEN batch completes THEN system SHALL report profile usage distribution (e.g., "slideshow_images1: 3, product_video_sequential: 2")

### Requirement 4: Progress Tracking and Logging

**User Story:** As a content creator, I want detailed progress updates during batch video production, so that I can monitor the operation and understand what's happening at each step.

#### Acceptance Criteria

1. WHEN batch starts THEN system SHALL log total products discovered and log file path
2. WHEN processing each product THEN system SHALL display `[N/total] Processing product: {product_id}` format
3. WHEN product video completes THEN system SHALL log "Successfully completed video for: {product_id}"
4. WHEN product fails THEN system SHALL log "Failed to produce video for: {product_id}"
5. WHEN product skipped THEN system SHALL log "Skipped product: {product_id} - {reason}"
6. WHEN using profile randomization THEN system SHALL include selected profile in progress logs
7. WHEN batch completes THEN system SHALL generate summary report with counts

### Requirement 5: Error Handling and Resilience

**User Story:** As a content creator, I want batch video production to continue when individual products fail, so that one failure doesn't stop the entire batch operation.

#### Acceptance Criteria

1. WHEN individual product fails AND `--fail-fast` not enabled THEN system SHALL log error and continue with next product
2. WHEN `--fail-fast` flag enabled AND any product fails THEN system SHALL stop entire batch immediately and log reason
3. WHEN product has insufficient media THEN system SHALL return "SKIPPED" special value and not count as failure
4. WHEN timeout occurs for product THEN system SHALL log timeout duration, mark as failed, continue to next product
5. WHEN exception occurs for product THEN system SHALL log with full traceback in debug mode, mark as failed, continue to next
6. WHEN partial results available THEN system SHALL accept partial batch completion with summary

### Requirement 6: Configuration and CLI Interface

**User Story:** As a content creator, I want flexible configuration through CLI arguments and YAML files, so that I can choose between runtime customization and persistent batch configurations.

#### Acceptance Criteria

1. WHEN `--batch` flag provided THEN system SHALL enable batch processing mode
2. WHEN `--batch-profile` specified THEN system SHALL use that profile for all products
3. WHEN `--random-profile` flag provided THEN system SHALL enable profile randomization
4. WHEN `--profile-pool` specified THEN system SHALL use space-separated list of profile names for randomization
5. WHEN `--outputs-dir` specified THEN system SHALL scan that directory instead of default "outputs"
6. WHEN `--fail-fast` flag provided THEN system SHALL stop on first failure
7. WHEN batch mode enabled AND single-product args provided THEN system SHALL error "Cannot use products_file/profile with --batch"
8. WHEN `--product-index` used with `--batch` THEN system SHALL error "Cannot use --product-index with batch mode"
9. WHEN YAML `profile_pool` configured AND CLI override provided THEN system SHALL use CLI arguments

### Requirement 7: Summary Reporting

**User Story:** As a content creator, I want comprehensive summary statistics after batch completion, so that I can quickly understand the results and identify any issues.

#### Acceptance Criteria

1. WHEN batch completes THEN system SHALL report total products discovered
2. WHEN batch completes THEN system SHALL report succeeded count with product IDs
3. WHEN batch completes THEN system SHALL report failed count with product IDs and reasons
4. WHEN batch completes THEN system SHALL report skipped count with product IDs
5. WHEN profile randomization enabled THEN system SHALL report profile usage distribution
6. WHEN failures occur THEN system SHALL list all failed products with their product IDs

### Requirement 8: Performance and Resource Management

**User Story:** As a content creator, I want batch video production to efficiently manage resources across multiple products, so that processing is fast and doesn't consume excessive memory.

#### Acceptance Criteria

1. WHEN processing batch THEN system SHALL reuse global background processor across all products
2. WHEN processing batch THEN system SHALL share HTTP session pool for all products
3. WHEN processing batch THEN system SHALL track per-product metrics (duration, memory, CPU)
4. WHEN product completes THEN system SHALL clean up per-product resources while maintaining global context
5. WHEN timeout configured THEN system SHALL apply `pipeline_timeout_sec` per product (not entire batch)

## Non-Functional Requirements

### Code Architecture and Modularity
- **Single Responsibility Principle**: Separate batch orchestration from single-product video generation
- **Modular Design**: Batch processing module should reuse existing producer components without modification
- **Dependency Management**: Minimize coupling between batch orchestrator and video pipeline
- **Clear Interfaces**: Define clean contracts between batch controller and video producer

### Performance
- **Sequential Processing**: Process products one at a time to avoid resource contention
- **Resource Pooling**: Maintain background processor and HTTP sessions across batch
- **Memory Efficiency**: Clean up per-product resources after each completion
- **Inter-Product Delays**: Configurable delays (1.5-4.0 sec) to avoid rate limiting

### Security
- **Path Validation**: Sanitize product IDs before creating file paths
- **Configuration Validation**: Validate profile names exist before starting batch
- **Input Validation**: Verify outputs directory exists and is readable

### Reliability
- **Graceful Degradation**: Continue batch on individual failures (unless fail-fast)
- **Data Integrity**: Ensure partial batch results are valid and usable
- **Error Recovery**: Provide clear error messages for troubleshooting failed products
- **Deterministic Behavior**: Profile randomization uses product ID as seed for reproducibility

### Usability
- **Clear Progress**: Real-time progress updates with `[N/total]` format
- **Comprehensive Logging**: Detailed logs with DEBUG mode for troubleshooting
- **Helpful Errors**: Clear validation errors with actionable guidance
- **Profile Discovery**: Automatic profile pool from VideoConfig when not specified
