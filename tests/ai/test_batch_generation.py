"""Unit tests for batch metadata generation."""

import asyncio
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ai.platform_metadata.batch import (
    BatchGenerationResult,
    BatchMetadataGenerator,
    ProductGenerationResult,
)
from src.ai.platform_metadata.models import (
    BatchGenerationSettings,
    PlatformMetadata,
    PlatformMetadataSettings,
)


@dataclass
class MockProduct:
    """Mock product data for testing."""

    asin: str = "B0TESTASIN"
    title: str = "Test Product Title"
    description: str = "Test product description"
    price: str = "29.99"
    brand: str = "TestBrand"


class TestBatchGenerationSettings:
    """Test BatchGenerationSettings Pydantic model."""

    def test_default_settings(self):
        """Test default batch generation settings."""
        settings = BatchGenerationSettings()

        assert settings.enabled is True
        assert settings.max_concurrent == 3
        assert settings.log_progress is True

    def test_custom_settings(self):
        """Test custom batch generation settings."""
        settings = BatchGenerationSettings(
            enabled=False,
            max_concurrent=10,
            log_progress=False,
        )

        assert settings.enabled is False
        assert settings.max_concurrent == 10
        assert settings.log_progress is False

    def test_max_concurrent_validation_min(self):
        """Test max_concurrent minimum validation."""
        with pytest.raises(ValueError):
            BatchGenerationSettings(max_concurrent=0)

    def test_max_concurrent_validation_max(self):
        """Test max_concurrent maximum validation."""
        with pytest.raises(ValueError):
            BatchGenerationSettings(max_concurrent=21)

    def test_max_concurrent_bounds(self):
        """Test max_concurrent at valid bounds."""
        settings_min = BatchGenerationSettings(max_concurrent=1)
        settings_max = BatchGenerationSettings(max_concurrent=20)

        assert settings_min.max_concurrent == 1
        assert settings_max.max_concurrent == 20


class TestProductGenerationResult:
    """Test ProductGenerationResult dataclass."""

    def test_result_creation(self):
        """Test creating a generation result."""
        metadata = PlatformMetadata.create(
            platform="youtube",
            title="Test Title",
            description="Test description",
            hashtags=["#test", "#ad"],
            keywords=["test"],
            product_id="B0TESTASIN",
        )

        result = ProductGenerationResult(
            product_id="B0TESTASIN",
            success=True,
            metadata={"youtube": metadata},
            errors={},
            duration_seconds=1.5,
            from_cache={"youtube": False},
        )

        assert result.product_id == "B0TESTASIN"
        assert result.success is True
        youtube_meta = result.metadata["youtube"]
        assert youtube_meta is not None
        assert youtube_meta.platform == "youtube"
        assert result.duration_seconds == 1.5
        assert result.from_cache["youtube"] is False

    def test_result_with_errors(self):
        """Test result with errors."""
        result = ProductGenerationResult(
            product_id="B0TESTASIN",
            success=False,
            metadata={"youtube": None},
            errors={"youtube": "Generation failed"},
            duration_seconds=0.5,
        )

        assert result.success is False
        assert result.errors["youtube"] == "Generation failed"

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        metadata = PlatformMetadata.create(
            platform="youtube",
            title="Test Title",
            description="Test description",
            hashtags=["#test", "#ad"],
            keywords=["test"],
            product_id="B0TESTASIN",
        )

        result = ProductGenerationResult(
            product_id="B0TESTASIN",
            success=True,
            metadata={"youtube": metadata},
            errors={},
            duration_seconds=1.5,
            from_cache={"youtube": True},
        )

        data = result.to_dict()

        assert data["product_id"] == "B0TESTASIN"
        assert data["success"] is True
        assert data["metadata"]["youtube"]["platform"] == "youtube"
        assert data["duration_seconds"] == 1.5
        assert data["from_cache"]["youtube"] is True

    def test_result_to_dict_with_none_metadata(self):
        """Test converting result with None metadata."""
        result = ProductGenerationResult(
            product_id="B0TESTASIN",
            success=False,
            metadata={"youtube": None},
            errors={"youtube": "Failed"},
        )

        data = result.to_dict()

        assert data["metadata"]["youtube"] is None


class TestBatchGenerationResult:
    """Test BatchGenerationResult dataclass."""

    def test_batch_result_creation(self):
        """Test creating batch generation result."""
        product_result = ProductGenerationResult(
            product_id="B0TEST001",
            success=True,
            metadata={},
        )

        batch_result = BatchGenerationResult(
            total_products=10,
            successful_products=8,
            failed_products=2,
            results=[product_result],
            total_duration_seconds=15.5,
            started_at="2025-01-15T12:00:00+00:00",
            completed_at="2025-01-15T12:00:15+00:00",
        )

        assert batch_result.total_products == 10
        assert batch_result.successful_products == 8
        assert batch_result.failed_products == 2
        assert len(batch_result.results) == 1

    def test_batch_result_success_rate(self):
        """Test success rate calculation."""
        batch_result = BatchGenerationResult(
            total_products=10,
            successful_products=8,
            failed_products=2,
            results=[],
            total_duration_seconds=10.0,
            started_at="",
            completed_at="",
        )

        assert batch_result.success_rate == 80.0

    def test_batch_result_success_rate_zero_products(self):
        """Test success rate with zero products."""
        batch_result = BatchGenerationResult(
            total_products=0,
            successful_products=0,
            failed_products=0,
            results=[],
            total_duration_seconds=0.0,
            started_at="",
            completed_at="",
        )

        assert batch_result.success_rate == 0.0

    def test_batch_result_to_dict(self):
        """Test converting batch result to dictionary."""
        product_result = ProductGenerationResult(
            product_id="B0TEST001",
            success=True,
            metadata={},
        )

        batch_result = BatchGenerationResult(
            total_products=5,
            successful_products=4,
            failed_products=1,
            results=[product_result],
            total_duration_seconds=10.5,
            started_at="2025-01-15T12:00:00+00:00",
            completed_at="2025-01-15T12:00:10+00:00",
        )

        data = batch_result.to_dict()

        assert data["total_products"] == 5
        assert data["successful_products"] == 4
        assert data["failed_products"] == 1
        assert len(data["results"]) == 1
        assert data["total_duration_seconds"] == 10.5


class TestBatchMetadataGenerator:
    """Test BatchMetadataGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create batch generator instance."""
        return BatchMetadataGenerator(max_concurrent=3)

    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session."""
        return MagicMock()

    @pytest.fixture
    def mock_settings(self):
        """Create mock LLM settings."""
        return MagicMock()

    @pytest.fixture
    def platform_settings(self):
        """Create platform settings dictionary."""
        return {
            "youtube": {"title_length_max": 60},
            "tiktok": {"caption_length_optimal": 150},
        }

    @pytest.fixture
    def intermediate_paths(self):
        """Create intermediate paths dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            return {"output": Path(tmpdir)}

    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = BatchMetadataGenerator(max_concurrent=5)

        assert generator.max_concurrent == 5
        assert generator.progress_callback is None

    def test_generator_max_concurrent_bounds(self):
        """Test max_concurrent is bounded to 1-20."""
        generator_low = BatchMetadataGenerator(max_concurrent=0)
        generator_high = BatchMetadataGenerator(max_concurrent=100)

        assert generator_low.max_concurrent == 1
        assert generator_high.max_concurrent == 20

    def test_generator_with_callback(self):
        """Test generator with progress callback."""
        progress_calls = []

        def callback(current, total, product_id, status):
            progress_calls.append((current, total, product_id, status))

        generator = BatchMetadataGenerator(
            max_concurrent=3,
            progress_callback=callback,
        )

        assert generator.progress_callback is not None

    @pytest.mark.asyncio
    async def test_generate_batch_empty_products(
        self, generator, mock_settings, mock_session, platform_settings
    ):
        """Test batch generation with empty products list."""
        result = await generator.generate_batch(
            products=[],
            settings=mock_settings,
            secrets={},
            session=mock_session,
            platform_settings=platform_settings,
            intermediate_paths={},
        )

        assert result.total_products == 0
        assert result.successful_products == 0
        assert result.failed_products == 0
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_generate_batch_single_product(
        self,
        generator,
        mock_settings,
        mock_session,
        platform_settings,
        intermediate_paths,
    ):
        """Test batch generation with single product."""
        product = MockProduct()

        # Mock the factory's generate_multi_platform
        mock_metadata = PlatformMetadata.create(
            platform="youtube",
            title="Test Title",
            description="Test description",
            hashtags=["#ad"],
            keywords=[],
            product_id="B0TESTASIN",
        )

        with patch(
            "src.ai.platform_metadata.PlatformMetadataFactory.generate_multi_platform",
            new_callable=AsyncMock,
        ) as mock_generate:
            mock_generate.return_value = {
                "youtube": mock_metadata,
                "tiktok": mock_metadata,
            }

            result = await generator.generate_batch(
                products=[product],
                settings=mock_settings,
                secrets={},
                session=mock_session,
                platform_settings=platform_settings,
                intermediate_paths=intermediate_paths,
            )

        assert result.total_products == 1
        assert result.successful_products == 1
        assert result.failed_products == 0
        assert len(result.results) == 1
        assert result.results[0].product_id == "B0TESTASIN"
        assert result.results[0].success is True

    @pytest.mark.asyncio
    async def test_generate_batch_multiple_products(
        self,
        generator,
        mock_settings,
        mock_session,
        platform_settings,
        intermediate_paths,
    ):
        """Test batch generation with multiple products."""
        products = [
            MockProduct(asin="B0TEST001"),
            MockProduct(asin="B0TEST002"),
            MockProduct(asin="B0TEST003"),
        ]

        mock_metadata = PlatformMetadata.create(
            platform="youtube",
            title="Test",
            description="Test",
            hashtags=["#ad"],
            keywords=[],
            product_id="test",
        )

        with patch(
            "src.ai.platform_metadata.PlatformMetadataFactory.generate_multi_platform",
            new_callable=AsyncMock,
        ) as mock_generate:
            mock_generate.return_value = {
                "youtube": mock_metadata,
                "tiktok": mock_metadata,
            }

            result = await generator.generate_batch(
                products=products,
                settings=mock_settings,
                secrets={},
                session=mock_session,
                platform_settings=platform_settings,
                intermediate_paths=intermediate_paths,
            )

        assert result.total_products == 3
        assert result.successful_products == 3
        assert result.failed_products == 0

    @pytest.mark.asyncio
    async def test_generate_batch_with_failures(
        self,
        generator,
        mock_settings,
        mock_session,
        platform_settings,
        intermediate_paths,
    ):
        """Test batch generation with some failures."""
        products = [
            MockProduct(asin="B0TEST001"),
            MockProduct(asin="B0TEST002"),
        ]

        mock_metadata = PlatformMetadata.create(
            platform="youtube",
            title="Test",
            description="Test",
            hashtags=["#ad"],
            keywords=[],
            product_id="test",
        )

        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"youtube": mock_metadata, "tiktok": mock_metadata}
            else:
                return {"youtube": None, "tiktok": None}

        with patch(
            "src.ai.platform_metadata.PlatformMetadataFactory.generate_multi_platform",
            side_effect=mock_generate,
        ):
            result = await generator.generate_batch(
                products=products,
                settings=mock_settings,
                secrets={},
                session=mock_session,
                platform_settings=platform_settings,
                intermediate_paths=intermediate_paths,
            )

        assert result.total_products == 2
        assert result.successful_products == 1
        assert result.failed_products == 1

    @pytest.mark.asyncio
    async def test_generate_batch_progress_callback(
        self,
        mock_settings,
        mock_session,
        platform_settings,
        intermediate_paths,
    ):
        """Test that progress callback is called."""
        progress_calls = []

        def callback(current, total, product_id, status):
            progress_calls.append((current, total, product_id, status))

        generator = BatchMetadataGenerator(
            max_concurrent=3,
            progress_callback=callback,
        )

        products = [MockProduct(asin="B0TEST001")]

        mock_metadata = PlatformMetadata.create(
            platform="youtube",
            title="Test",
            description="Test",
            hashtags=["#ad"],
            keywords=[],
            product_id="test",
        )

        with patch(
            "src.ai.platform_metadata.PlatformMetadataFactory.generate_multi_platform",
            new_callable=AsyncMock,
        ) as mock_generate:
            mock_generate.return_value = {"youtube": mock_metadata}

            await generator.generate_batch(
                products=products,
                settings=mock_settings,
                secrets={},
                session=mock_session,
                platform_settings=platform_settings,
                intermediate_paths=intermediate_paths,
            )

        # Should have at least 2 calls: starting and completion status
        assert len(progress_calls) >= 2
        assert progress_calls[0][2] == "B0TEST001"
        assert progress_calls[0][3] == "starting"

    @pytest.mark.asyncio
    async def test_generate_batch_with_exception(
        self,
        generator,
        mock_settings,
        mock_session,
        platform_settings,
        intermediate_paths,
    ):
        """Test batch generation handles exceptions gracefully."""
        products = [MockProduct(asin="B0TEST001")]

        with patch(
            "src.ai.platform_metadata.PlatformMetadataFactory.generate_multi_platform",
            new_callable=AsyncMock,
        ) as mock_generate:
            mock_generate.side_effect = Exception("LLM API Error")

            result = await generator.generate_batch(
                products=products,
                settings=mock_settings,
                secrets={},
                session=mock_session,
                platform_settings=platform_settings,
                intermediate_paths=intermediate_paths,
            )

        assert result.total_products == 1
        assert result.successful_products == 0
        assert result.failed_products == 1
        assert "_generation" in result.results[0].errors

    @pytest.mark.asyncio
    async def test_generate_batch_concurrency_limit(
        self,
        mock_settings,
        mock_session,
        platform_settings,
        intermediate_paths,
    ):
        """Test that concurrency is limited by max_concurrent."""
        generator = BatchMetadataGenerator(max_concurrent=2)

        products = [MockProduct(asin=f"B0TEST{i:03d}") for i in range(5)]

        concurrent_count = 0
        max_concurrent_observed = 0

        async def mock_generate(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent_observed
            concurrent_count += 1
            max_concurrent_observed = max(max_concurrent_observed, concurrent_count)
            await asyncio.sleep(0.1)  # Simulate work
            concurrent_count -= 1
            mock_meta = PlatformMetadata.create(
                platform="youtube",
                title="Test",
                description="Test",
                hashtags=["#ad"],
                keywords=[],
                product_id="test",
            )
            return {"youtube": mock_meta, "tiktok": mock_meta}

        with patch(
            "src.ai.platform_metadata.PlatformMetadataFactory.generate_multi_platform",
            side_effect=mock_generate,
        ):
            result = await generator.generate_batch(
                products=products,
                settings=mock_settings,
                secrets={},
                session=mock_session,
                platform_settings=platform_settings,
                intermediate_paths=intermediate_paths,
            )

        assert result.total_products == 5
        assert result.successful_products == 5
        # Max concurrent should not exceed the limit (2)
        assert max_concurrent_observed <= 2

    @pytest.mark.asyncio
    async def test_generate_batch_with_cache(
        self,
        generator,
        mock_settings,
        mock_session,
        platform_settings,
        intermediate_paths,
    ):
        """Test batch generation with cache."""
        products = [MockProduct(asin="B0TEST001")]

        mock_cache = MagicMock()
        mock_metadata = PlatformMetadata.create(
            platform="youtube",
            title="Cached Title",
            description="Cached description",
            hashtags=["#ad"],
            keywords=[],
            product_id="B0TEST001",
        )

        # Return cached metadata for all platforms
        mock_cache.get.return_value = mock_metadata

        result = await generator.generate_batch(
            products=products,
            settings=mock_settings,
            secrets={},
            session=mock_session,
            platform_settings=platform_settings,
            intermediate_paths=intermediate_paths,
            cache=mock_cache,
        )

        assert result.total_products == 1
        assert result.successful_products == 1
        # All platforms should be from cache
        assert result.results[0].from_cache["youtube"] is True
        assert result.results[0].from_cache["tiktok"] is True


class TestPlatformMetadataSettingsBatch:
    """Test that batch settings are integrated into PlatformMetadataSettings."""

    def test_batch_settings_in_platform_metadata_settings(self):
        """Test that batch settings are part of PlatformMetadataSettings."""
        settings = PlatformMetadataSettings()

        assert hasattr(settings, "batch")
        assert isinstance(settings.batch, BatchGenerationSettings)
        assert settings.batch.enabled is True
        assert settings.batch.max_concurrent == 3
        assert settings.batch.log_progress is True

    def test_custom_batch_settings_in_platform_metadata_settings(self):
        """Test custom batch settings in PlatformMetadataSettings."""
        batch_settings = BatchGenerationSettings(
            enabled=False,
            max_concurrent=10,
            log_progress=False,
        )

        settings = PlatformMetadataSettings(batch=batch_settings)

        assert settings.batch.enabled is False
        assert settings.batch.max_concurrent == 10
        assert settings.batch.log_progress is False
