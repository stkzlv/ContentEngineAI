"""Global batch pipeline package for end-to-end automation.

This package orchestrates the complete workflow from scraping products
to generating promotional videos in a single unified command.

Architecture:
    - global_batch: Main orchestrator coordinating scraping and video production
    - config: Configuration models and loading with CLI > YAML > defaults precedence

Three-Phase Pipeline:
    1. Scraping Phase: Process product IDs/keywords through Amazon scraper
    2. Handoff Phase: Scan outputs/ for products with data.json, filter by media
    3. Video Production Phase: Generate videos for ready products

Usage:
    # CLI execution
    python -m src.pipeline.global_batch --product-ids B0ABC123 --profile slideshow_images1

    # Programmatic execution
    from src.pipeline.global_batch import GlobalPipelineOrchestrator
    from src.pipeline.config import GlobalBatchConfig

    config = GlobalBatchConfig(...)
    orchestrator = GlobalPipelineOrchestrator(config)
    summary = await orchestrator.run_pipeline()
"""

__all__ = [
    "GlobalPipelineOrchestrator",
    "GlobalBatchConfig",
]
