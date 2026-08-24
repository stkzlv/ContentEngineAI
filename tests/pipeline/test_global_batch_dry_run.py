"""`--dry-run` reports what a run would do, and does nothing else.

Two defects, both reachable from a documented flag. The plan printer read
`profile.strategy` and `profile.resolution`, neither of which `VideoProfile`
declares, so the fixed-profile branch raised `AttributeError` — the flag
worked only when you did not know which profile would run. And the `--clean`
block sat above the dry-run branch, so `--dry-run --clean` removed the
product directories and then printed a plan for producing them.
"""

import pytest

from src.video.config.visual_models import VideoProfile


@pytest.mark.unit
class TestTheProfileSummaryUsesRealFields:
    """A field the model does not declare raises rather than printing."""

    def test_the_printed_fields_exist_on_the_model(self):
        declared = set(VideoProfile.model_fields)
        for field in (
            "description",
            "use_scraped_images",
            "use_scraped_videos",
            "use_stock_images",
            "use_stock_videos",
            "stock_image_count",
            "stock_video_count",
        ):
            assert field in declared, field

    def test_a_fixed_profile_prints_a_plan(self, capsys, tmp_path):
        from unittest.mock import MagicMock

        from src.pipeline.global_batch import GlobalPipelineOrchestrator

        profile = VideoProfile(
            description="Stock-only slideshow",
            use_scraped_images=False,
            use_scraped_videos=False,
            use_stock_images=True,
            use_stock_videos=False,
            stock_image_count=8,
        )
        video_config = MagicMock()
        video_config.video_profiles = {"slideshow_stock": profile}

        config = MagicMock()
        config.profile = "slideshow_stock"
        config.random_profile = False
        config.profile_pool = None
        config.outputs_dir = tmp_path
        config.product_ids = []
        config.keywords = ["a keyword"]
        config.skip_publish = True

        orchestrator = GlobalPipelineOrchestrator.__new__(GlobalPipelineOrchestrator)
        orchestrator.config = config
        orchestrator.display_execution_plan(video_config)

        printed = capsys.readouterr().out
        assert "slideshow_stock" in printed
        assert "Stock-only slideshow" in printed
        assert "8 stock images" in printed


@pytest.mark.unit
class TestDryRunDeletesNothing:
    """The flag that reports must not be the one that removes the inputs."""

    def test_dry_run_with_clean_keeps_the_product_directories(
        self, tmp_path, monkeypatch, capsys
    ):
        import asyncio
        import sys

        from src.pipeline import global_batch

        outputs = tmp_path / "outputs"
        product = outputs / "B0DRYRUN001"
        product.mkdir(parents=True)
        (product / "data.json").write_text("[]")

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "global_batch",
                "--product-ids",
                "B0DRYRUN001",
                "--profile",
                "slideshow_images1",
                "--outputs-dir",
                str(outputs),
                "--skip-publish",
                "--clean",
                "--dry-run",
            ],
        )

        # `main` installs a root log handler over the captured streams and
        # never removes it; left behind, it breaks any later test that logs.
        import logging

        saved = logging.root.handlers[:]
        saved_level = logging.root.level
        try:
            with pytest.raises(SystemExit) as exit_info:
                asyncio.run(global_batch.main())
        finally:
            for handler in logging.root.handlers[:]:
                if handler not in saved:
                    logging.root.removeHandler(handler)
                    handler.close()
            logging.root.handlers[:] = saved
            logging.root.setLevel(saved_level)

        assert exit_info.value.code == 0
        assert product.exists(), "--dry-run --clean removed the product directory"
        assert (product / "data.json").exists()
        assert "PHASE 3: VIDEO PRODUCTION" in capsys.readouterr().out
