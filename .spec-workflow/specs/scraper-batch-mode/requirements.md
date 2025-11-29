# Requirements Document

## Introduction

The scraper batch mode feature enables automated multi-product data collection from e-commerce platforms using configurable lists of product IDs and keywords. This enhancement extends the existing single-product scraper to support batch operations, allowing users to scrape multiple products in a single execution with comprehensive progress tracking, error handling, and summary reporting.

The feature addresses the need for efficient bulk data collection workflows, reducing manual intervention and enabling automated content pipelines for video production at scale.

## Alignment with Product Vision

This feature directly supports ContentEngineAI's vision of automated, scalable video production by:
- **Automation**: Eliminating manual per-product scraping through batch input lists
- **Scalability**: Enabling collection of data for multiple products in single execution
- **Efficiency**: Reducing operational overhead through unified batch workflows
- **Flexibility**: Supporting both direct product ID lookups and search-based discovery

## Requirements

### Requirement 1: Product ID List Processing

**User Story:** As a content creator, I want to scrape multiple products by providing a list of product IDs (ASINs for Amazon), so that I can collect data for all products in a single execution without running the scraper multiple times.

#### Acceptance Criteria

1. WHEN user provides `--product-ids B0ASIN1 B0ASIN2 B0ASIN3` THEN system SHALL scrape each product ID sequentially and create separate output directories
2. IF `product_ids` list is defined in `config/scraper.yaml` AND no CLI override provided THEN system SHALL use YAML configuration for batch input
3. WHEN CLI argument `--product-ids` is provided THEN system SHALL override any YAML configuration
4. WHEN product ID fails validation THEN system SHALL log warning, skip invalid ID, and continue with remaining IDs
5. WHEN duplicate product IDs exist in list THEN system SHALL automatically deduplicate before processing
6. IF both `product_ids` and `keywords` lists are empty THEN system SHALL error with clear message

### Requirement 2: Keyword List Processing

**User Story:** As a content creator, I want to scrape products using multiple keyword searches in a single execution, so that I can discover products across different search terms efficiently.

#### Acceptance Criteria

1. WHEN user provides `--keywords "keyword1" "keyword2" "keyword3"` THEN system SHALL execute search for each keyword sequentially
2. IF `keywords` list is defined in `config/scraper.yaml` AND no CLI override provided THEN system SHALL use YAML configuration
3. WHEN CLI argument `--keywords` is provided THEN system SHALL override any YAML configuration
4. WHEN same filters applied to all keyword searches THEN system SHALL apply `--min-price`, `--max-price`, `--min-rating`, `--prime-only` consistently
5. WHEN `--max-products` limit is specified THEN system SHALL honor limit across ALL keyword searches combined (not per keyword)
6. WHEN duplicate products discovered across multiple keywords THEN system SHALL deduplicate by product ID

### Requirement 3: Mixed Input Mode

**User Story:** As a content creator, I want to combine explicit product IDs with keyword searches in a single batch run, so that I can scrape both known products and discover new ones simultaneously.

#### Acceptance Criteria

1. WHEN both `--product-ids` and `--keywords` are provided THEN system SHALL process explicit product IDs first, then keyword searches
2. WHEN combining sources THEN system SHALL deduplicate results by product ID across both input types
3. WHEN `--max-products` specified THEN system SHALL count products from both sources toward the limit
4. WHEN media collection target reached THEN system SHALL stop processing remaining keywords

### Requirement 4: Progress Tracking and Logging

**User Story:** As a content creator, I want detailed progress updates during batch scraping, so that I can monitor the operation and understand what's happening at each step.

#### Acceptance Criteria

1. WHEN batch starts THEN system SHALL log total product IDs and keywords to process
2. WHEN processing each item THEN system SHALL display `[N/total] Scraping product: {product_id}` format
3. WHEN product scraping completes THEN system SHALL log status (success/failure) with product ID
4. WHEN batch completes THEN system SHALL generate summary report with success/failure counts per input type
5. WHEN `--debug` flag enabled THEN system SHALL include detailed scraping state and error information

### Requirement 5: Error Handling and Resilience

**User Story:** As a content creator, I want batch scraping to continue when individual products fail, so that one failure doesn't stop the entire batch operation.

#### Acceptance Criteria

1. WHEN individual product scraping fails AND `--fail-fast` not enabled THEN system SHALL log error and continue with next product
2. WHEN `--fail-fast` flag is enabled AND any product fails THEN system SHALL stop entire batch immediately
3. WHEN search for keyword fails THEN system SHALL log error and continue to next keyword
4. WHEN partial results available THEN system SHALL accept partial success and generate summary
5. WHEN invalid product IDs detected THEN system SHALL skip with warning and continue processing valid IDs

### Requirement 6: Configuration and CLI Interface

**User Story:** As a content creator, I want flexible configuration through both CLI arguments and YAML files, so that I can choose between runtime customization and persistent batch job configurations.

#### Acceptance Criteria

1. WHEN `--product-ids` CLI argument provided THEN system SHALL override `scraper.product_ids` YAML configuration
2. WHEN `--keywords` CLI argument provided THEN system SHALL override `scraper.keywords` YAML configuration
3. WHEN no CLI override THEN system SHALL use YAML configuration lists for batch input
4. WHEN validating configuration THEN system SHALL check all product IDs match platform format (Amazon: 10-character alphanumeric)
5. IF neither CLI nor YAML provide input THEN system SHALL error with message "No product IDs or keywords provided"

### Requirement 7: Summary Reporting

**User Story:** As a content creator, I want comprehensive summary statistics after batch completion, so that I can quickly understand the results and identify any issues.

#### Acceptance Criteria

1. WHEN batch completes THEN system SHALL report total products attempted from product IDs
2. WHEN batch completes THEN system SHALL report total products attempted from keyword searches
3. WHEN batch completes THEN system SHALL list successful scrapes with data.json created count
4. WHEN batch completes THEN system SHALL list failed scrapes with product IDs
5. WHEN batch completes THEN system SHALL include media collection statistics (images, videos per product)
6. WHEN failures occur THEN system SHALL report failure counts per input source type

## Non-Functional Requirements

### Code Architecture and Modularity
- **Single Responsibility Principle**: Separate batch orchestration logic from single-product scraping logic
- **Modular Design**: Batch processing module should reuse existing scraper components without modification
- **Dependency Management**: Minimize coupling between batch orchestrator and platform-specific scrapers
- **Clear Interfaces**: Define clean contracts between batch controller and product scraper implementations

### Performance
- **Sequential Processing**: Process products one at a time to respect platform rate limits
- **Rate Limiting**: Apply configurable inter-request delays to avoid detection
- **Memory Efficiency**: Clean up per-product resources while maintaining session pool
- **Session Reuse**: Maintain single HTTP session across all products in batch

### Security
- **API Key Protection**: Never log or expose API keys in batch output
- **Input Validation**: Validate all product IDs and keywords against injection attacks
- **File Path Safety**: Sanitize product IDs before creating output directories

### Reliability
- **Graceful Degradation**: Continue batch on individual failures (unless fail-fast enabled)
- **Data Integrity**: Ensure partial batch results are valid and usable
- **Error Recovery**: Provide clear error messages for troubleshooting failed products
- **Idempotency**: Support re-running batch with same inputs (skip already scraped products)

### Usability
- **Clear Progress**: Real-time progress updates with `[N/total]` format
- **Comprehensive Logging**: Detailed logs with DEBUG mode for troubleshooting
- **Intuitive CLI**: Consistent argument naming with existing producer batch mode
- **Helpful Errors**: Clear validation errors with actionable guidance
