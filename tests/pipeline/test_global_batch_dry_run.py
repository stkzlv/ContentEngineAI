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

        # `main` opens outputs/logs/global_pipeline.log relative to the cwd,
        # in write mode, and reads the developer's real `.env`. Neither is
        # under test here, and the first destroys the log the project's own
        # runbooks tell you to grep after a batch run.
        monkeypatch.setattr(
            "src.utils.logging_setup.setup_debug_logging", lambda **kwargs: None
        )
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)

        with pytest.raises(SystemExit) as exit_info:
            asyncio.run(global_batch.main())

        assert exit_info.value.code == 0
        assert product.exists(), "--dry-run --clean removed the product directory"
        assert (product / "data.json").exists()
        assert "PHASE 3: VIDEO PRODUCTION" in capsys.readouterr().out


@pytest.mark.unit
class TestTheCleanPreviewMatchesTheClean:
    """The preview and the deletion read one list, so they cannot diverge."""

    @staticmethod
    def _outputs(tmp_path):
        outputs = tmp_path / "outputs"
        for name in ("B0PRODUCT1", "B0PRODUCT2", "TESTASIN9", "cache", "logs"):
            (outputs / name).mkdir(parents=True)
        (outputs / "published_products.json").write_text("[]")
        return outputs

    def test_only_product_directories_are_named(self, tmp_path):
        from src.pipeline.global_batch import _clean_targets

        outputs = self._outputs(tmp_path)
        names = {t.name for t in _clean_targets(outputs, [])}
        assert names == {"B0PRODUCT1", "B0PRODUCT2", "TESTASIN9"}

    def test_named_products_narrow_the_set(self, tmp_path):
        from src.pipeline.global_batch import _clean_targets

        outputs = self._outputs(tmp_path)
        names = {t.name for t in _clean_targets(outputs, ["B0PRODUCT1"])}
        assert names == {"B0PRODUCT1"}

    def test_a_missing_outputs_root_removes_nothing(self, tmp_path):
        from src.pipeline.global_batch import _clean_targets

        assert _clean_targets(tmp_path / "absent", ["B0PRODUCT1"]) == []

    def test_the_plan_lists_what_would_go(self, tmp_path, capsys):
        from unittest.mock import MagicMock

        from src.pipeline.global_batch import GlobalPipelineOrchestrator

        outputs = self._outputs(tmp_path)
        config = MagicMock()
        config.profile = None
        config.random_profile = False
        config.profile_pool = None
        config.clean = True
        config.outputs_dir = outputs
        config.product_ids = []
        config.keywords = ["a keyword"]
        config.skip_publish = True

        orchestrator = GlobalPipelineOrchestrator.__new__(GlobalPipelineOrchestrator)
        orchestrator.config = config
        orchestrator.display_execution_plan(MagicMock(video_profiles={}))

        printed = capsys.readouterr().out
        assert "Would remove 3 product directories" in printed
        assert "B0PRODUCT1" in printed
        # The directories that survive a clean must not be advertised as
        # going away.
        assert "cache" not in printed
        assert "published_products.json" not in printed
