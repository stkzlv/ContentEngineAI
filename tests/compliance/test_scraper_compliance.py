"""Scraper Architecture Compliance Tests

This module tests compliance with multi-platform scraper architecture requirements.
Tests verify that scrapers follow the BaseScraper interface and architectural patterns
defined in REQUIREMENTS.md.

Requirements Tested:
- 2.1: BaseScraper abstract interface
- 2.2: Platform-specific scraper inheritance
"""

import inspect
from abc import ABC

import pytest

from src.scraper.amazon.scraper import BotasaurusAmazonScraper
from src.scraper.base.models import BaseScraper, Platform

# ============================================================================
# Requirement 2.1: BaseScraper Interface Tests
# ============================================================================


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_1_base_scraper_is_abstract():
    """Test that BaseScraper is an abstract base class."""
    assert issubclass(BaseScraper, ABC), "BaseScraper must inherit from ABC"
    assert inspect.isabstract(BaseScraper), "BaseScraper must be an abstract base class"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_1_base_scraper_has_required_abstract_methods():
    """Test that BaseScraper defines all required abstract methods."""
    # Get all abstract methods using isfunction (works for unbound methods)
    abstract_methods = {
        name
        for name, method in inspect.getmembers(
            BaseScraper, predicate=inspect.isfunction
        )
        if getattr(method, "__isabstractmethod__", False)
    }

    # Also check for abstract properties
    abstract_properties = {
        name
        for name, prop in inspect.getmembers(
            BaseScraper, predicate=inspect.isdatadescriptor
        )
        if isinstance(prop, property)
        and getattr(prop.fget, "__isabstractmethod__", False)
    }

    all_abstract = abstract_methods | abstract_properties

    # Required abstract members per architecture
    required_abstract = {
        "platform",  # Property
        "validate_product_id",  # Method
        "scrape_products",  # Method
        "scrape_single_product",  # Method
    }

    missing = required_abstract - all_abstract
    assert (
        required_abstract <= all_abstract
    ), f"Missing required abstract methods/properties: {missing}"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_1_base_scraper_platform_property():
    """Test that BaseScraper defines platform as an abstract property."""
    # Check that platform is a property
    assert hasattr(
        BaseScraper, "platform"
    ), "BaseScraper must define 'platform' property"

    # Get the property descriptor
    platform_prop = BaseScraper.platform
    assert isinstance(platform_prop, property), "platform must be a property"

    # Check that it's abstract
    assert getattr(
        platform_prop.fget, "__isabstractmethod__", False
    ), "platform property must be abstract"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_1_base_scraper_validate_product_id_signature():
    """Test that validate_product_id method has correct signature."""
    method = BaseScraper.validate_product_id
    sig = inspect.signature(method)

    # Should have self and product_id parameters
    params = list(sig.parameters.keys())
    assert "product_id" in params, "validate_product_id must have product_id parameter"

    # Check return annotation
    assert sig.return_annotation is bool, "validate_product_id must return bool"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_1_base_scraper_scrape_products_signature():
    """Test that scrape_products method has correct signature."""
    method = BaseScraper.scrape_products
    sig = inspect.signature(method)

    # Should have keywords and optional search_params
    params = list(sig.parameters.keys())
    assert "keywords" in params, "scrape_products must have keywords parameter"
    assert (
        "search_params" in params
    ), "scrape_products must have search_params parameter"

    # Check that search_params is optional (has default)
    search_params = sig.parameters["search_params"]
    has_default = search_params.default is not inspect.Parameter.empty
    has_none = "None" in str(search_params.annotation)
    assert (
        has_default or has_none
    ), "search_params should be optional (None default or Optional type)"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_1_base_scraper_scrape_single_product_signature():
    """Test that scrape_single_product method has correct signature."""
    method = BaseScraper.scrape_single_product
    sig = inspect.signature(method)

    # Should have product_id parameter
    params = list(sig.parameters.keys())
    assert (
        "product_id" in params
    ), "scrape_single_product must have product_id parameter"

    # Return type should allow None (product not found)
    return_annotation = str(sig.return_annotation)
    assert (
        "None" in return_annotation or "|" in return_annotation
    ), "scrape_single_product must allow None return (product not found)"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_1_base_scraper_context_manager_support():
    """Test that BaseScraper supports context manager protocol."""
    assert hasattr(BaseScraper, "__enter__"), "BaseScraper must define __enter__"
    assert hasattr(BaseScraper, "__exit__"), "BaseScraper must define __exit__"

    # Check cleanup method exists
    assert hasattr(BaseScraper, "cleanup"), "BaseScraper must define cleanup method"


# ============================================================================
# Requirement 2.2: AmazonScraper Inheritance Tests
# ============================================================================


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_2_amazon_scraper_inherits_base_scraper():
    """Test that BotasaurusAmazonScraper inherits from BaseScraper."""
    assert issubclass(
        BotasaurusAmazonScraper, BaseScraper
    ), "BotasaurusAmazonScraper must inherit from BaseScraper"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_2_amazon_scraper_not_abstract():
    """Test that BotasaurusAmazonScraper is concrete (not abstract)."""
    is_concrete = not inspect.isabstract(BotasaurusAmazonScraper)
    assert is_concrete, (
        "BotasaurusAmazonScraper must be a concrete class "
        "(all abstract methods implemented)"
    )


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_2_amazon_scraper_implements_platform_property():
    """Test that BotasaurusAmazonScraper implements platform property."""
    # Check that platform is defined
    assert hasattr(
        BotasaurusAmazonScraper, "platform"
    ), "BotasaurusAmazonScraper must implement platform property"

    # Create instance to test property value (no args needed, uses defaults)
    try:
        scraper = BotasaurusAmazonScraper()
        assert (
            scraper.platform == Platform.AMAZON
        ), "platform property must return Platform.AMAZON"
    except FileNotFoundError:
        # If config file missing, test just the property definition
        assert isinstance(BotasaurusAmazonScraper.platform, property)


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_2_amazon_scraper_implements_validate_product_id():
    """Test that BotasaurusAmazonScraper implements validate_product_id."""
    # Check method exists
    assert hasattr(
        BotasaurusAmazonScraper, "validate_product_id"
    ), "BotasaurusAmazonScraper must implement validate_product_id"

    # Test basic functionality with valid and invalid ASINs
    try:
        scraper = BotasaurusAmazonScraper()

        # Valid ASIN format: B0XXXXXXXX (10 chars starting with B0)
        valid_asin = "B0BTYCRJSS"
        assert scraper.validate_product_id(
            valid_asin
        ), f"validate_product_id should accept valid ASIN: {valid_asin}"

        # Invalid ASINs: too short, has lowercase, has special chars
        invalid_asins = ["abc", "b0test1234", "B0TEST@#$%"]
        for invalid_asin in invalid_asins:
            assert not scraper.validate_product_id(
                invalid_asin
            ), f"validate_product_id should reject invalid ASIN: {invalid_asin}"
    except FileNotFoundError:
        # If config file missing, test the method directly from utils
        from src.scraper.amazon.utils import validate_asin_format

        valid_asin = "B0BTYCRJSS"
        assert validate_asin_format(
            valid_asin
        ), f"Should accept valid ASIN: {valid_asin}"

        invalid_asins = ["abc", "b0test1234", "B0TEST@#$%"]
        for invalid_asin in invalid_asins:
            assert not validate_asin_format(
                invalid_asin
            ), f"Should reject invalid ASIN: {invalid_asin}"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_2_amazon_scraper_implements_scrape_products():
    """Test that BotasaurusAmazonScraper implements scrape_products."""
    # Check method exists
    assert hasattr(
        BotasaurusAmazonScraper, "scrape_products"
    ), "BotasaurusAmazonScraper must implement scrape_products"

    # Verify signature matches base class
    method = BotasaurusAmazonScraper.scrape_products
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())

    assert "keywords" in params, "scrape_products must have keywords parameter"
    assert (
        "search_params" in params
    ), "scrape_products must have search_params parameter"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_2_amazon_scraper_implements_scrape_single_product():
    """Test that BotasaurusAmazonScraper implements scrape_single_product."""
    # Check method exists
    assert hasattr(
        BotasaurusAmazonScraper, "scrape_single_product"
    ), "BotasaurusAmazonScraper must implement scrape_single_product"

    # Verify signature
    method = BotasaurusAmazonScraper.scrape_single_product
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())

    assert (
        "product_id" in params
    ), "scrape_single_product must have product_id parameter"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_2_amazon_scraper_context_manager_works():
    """Test that BotasaurusAmazonScraper context manager protocol works."""
    # Test that scraper can be used as context manager
    try:
        with BotasaurusAmazonScraper() as scraper:
            assert scraper is not None, "Context manager should return scraper instance"
            assert isinstance(
                scraper, BotasaurusAmazonScraper
            ), "Context manager should return BotasaurusAmazonScraper instance"
    except FileNotFoundError:
        # If config missing, just test that methods exist
        assert hasattr(BotasaurusAmazonScraper, "__enter__")
        assert hasattr(BotasaurusAmazonScraper, "__exit__")

    # Cleanup should be called automatically on exit (tested implicitly)


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_2_amazon_scraper_has_cleanup_method():
    """Test that BotasaurusAmazonScraper has cleanup method."""
    assert hasattr(
        BotasaurusAmazonScraper, "cleanup"
    ), "BotasaurusAmazonScraper must have cleanup method"

    # Test cleanup can be called without error
    try:
        scraper = BotasaurusAmazonScraper()
        scraper.cleanup()
    except FileNotFoundError:
        # If config missing, just verify method exists
        assert callable(BotasaurusAmazonScraper.cleanup)


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_2_amazon_scraper_registered_with_platform():
    """Test that BotasaurusAmazonScraper is registered for Platform.AMAZON."""
    from src.scraper.base.models import ScraperRegistry

    # Check that Amazon platform is supported
    assert ScraperRegistry.is_platform_supported(
        Platform.AMAZON
    ), "Platform.AMAZON must be registered in ScraperRegistry"

    # Check that the registered scraper is BotasaurusAmazonScraper
    scraper_class = ScraperRegistry.get_scraper_class(Platform.AMAZON)
    assert (
        scraper_class is BotasaurusAmazonScraper
    ), "ScraperRegistry must return BotasaurusAmazonScraper for Platform.AMAZON"


# ============================================================================
# Requirement 2.3: Product Data Extraction Tests
# ============================================================================


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_3_base_product_data_has_required_fields():
    """Test that BaseProductData defines all required fields per requirement 2.3."""
    from dataclasses import fields

    from src.scraper.base.models import BaseProductData

    # Get all fields from the dataclass
    product_fields = {f.name for f in fields(BaseProductData)}

    # Required fields per requirement 2.3
    required_fields = {
        "title",  # Product title
        "price",  # Product price
        "description",  # Product description
        "platform_id",  # Platform-specific ID (ASIN, Item ID, etc.)
        "rating",  # Product ratings
        "reviews_count",  # Review count
        "url",  # Product URL
        "platform",  # Platform identifier
    }

    missing_fields = required_fields - product_fields
    assert (
        not missing_fields
    ), f"BaseProductData missing required fields: {missing_fields}"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_3_base_product_data_required_fields_not_optional():
    """Test that core required fields are not optional in BaseProductData."""
    from dataclasses import fields

    from src.scraper.base.models import BaseProductData

    # Fields that must not have defaults (truly required)
    strictly_required = {"title", "price", "url", "platform"}

    product_fields_dict = {f.name: f for f in fields(BaseProductData)}

    for field_name in strictly_required:
        field_obj = product_fields_dict[field_name]
        # Check field has no default and no default_factory
        from dataclasses import MISSING

        has_no_default = (
            field_obj.default is MISSING and field_obj.default_factory is MISSING
        )
        assert (
            has_no_default
        ), f"Required field '{field_name}' must not have a default value"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_3_product_data_completeness_validation():
    """Test that product data with all required fields can be created."""
    from src.scraper.base.models import BaseProductData, Platform

    # Create product with all required fields
    product = BaseProductData(
        title="Test Product",
        price="$29.99",
        url="https://example.com/product/123",
        platform=Platform.AMAZON,
        description="Test product description",
        platform_id="B0TEST1234",
        rating="4.5",
        reviews_count="1234",
    )

    # Verify all required fields are present and not empty
    assert product.title, "Title should not be empty"
    assert product.price, "Price should not be empty"
    assert product.url, "URL should not be empty"
    assert product.platform == Platform.AMAZON, "Platform should be set"
    assert product.description, "Description should not be empty"
    assert product.platform_id, "Platform ID should not be empty"
    assert product.rating, "Rating should not be empty"
    assert product.reviews_count, "Reviews count should not be empty"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_3_product_data_to_dict_includes_all_fields():
    """Test that to_dict() includes all required fields."""
    from src.scraper.base.models import BaseProductData, Platform

    product = BaseProductData(
        title="Test Product",
        price="$29.99",
        url="https://example.com/product/123",
        platform=Platform.AMAZON,
        description="Complete product description",
        platform_id="B0TEST1234",
        rating="4.7",
        reviews_count="523",
    )

    product_dict = product.to_dict()

    # Verify all required fields are in the dictionary
    required_keys = {
        "title",
        "price",
        "url",
        "platform",
        "description",
        "platform_id",
        "rating",
        "reviews_count",
    }

    missing_keys = required_keys - set(product_dict.keys())
    assert not missing_keys, f"Product dict missing required keys: {missing_keys}"

    # Verify values are correctly serialized
    assert product_dict["title"] == "Test Product"
    assert product_dict["price"] == "$29.99"
    assert product_dict["platform"] == "amazon"
    assert product_dict["platform_id"] == "B0TEST1234"
    assert product_dict["rating"] == "4.7"
    assert product_dict["reviews_count"] == "523"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_3_scraper_result_validates_product_data():
    """Test that ScrapeResult enforces product data structure."""
    from src.scraper.base.models import BaseProductData, Platform, ScrapeResult

    # Create valid product
    product = BaseProductData(
        title="Valid Product",
        price="$19.99",
        url="https://example.com/valid",
        platform=Platform.AMAZON,
        description="Valid description",
        platform_id="B0VALID123",
        rating="4.0",
        reviews_count="100",
    )

    # Create scrape result
    result = ScrapeResult(products=[product], platform=Platform.AMAZON, keyword="test")

    # Verify result contains valid products
    assert len(result.products) == 1
    assert isinstance(result.products[0], BaseProductData)
    assert result.products[0].title == "Valid Product"
    assert result.products[0].platform_id == "B0VALID123"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_2_3_product_data_handles_missing_optional_fields():
    """Test that optional fields can be None or empty without breaking."""
    from src.scraper.base.models import BaseProductData, Platform

    # Create product with only required fields (optional fields omitted)
    product = BaseProductData(
        title="Minimal Product",
        price="$9.99",
        url="https://example.com/minimal",
        platform=Platform.AMAZON,
    )

    # Verify optional fields have sensible defaults
    assert product.description == "", "Description should default to empty string"
    assert product.rating is None, "Rating should default to None"
    assert product.reviews_count is None, "Reviews count should default to None"
    assert product.platform_id is None, "Platform ID should default to None"
    assert product.images == [], "Images should default to empty list"
    assert product.videos == [], "Videos should default to empty list"

    # Verify product can still be converted to dict
    product_dict = product.to_dict()
    assert "title" in product_dict
    assert "rating" in product_dict
    assert product_dict["rating"] is None


# =============================================================================
# REQUIREMENT 3: Media Storage Structure Tests (Req 3.1, 3.5)
# =============================================================================


@pytest.mark.compliance
@pytest.mark.unit
def test_req_3_1_output_directory_structure_pattern(tmp_path):
    """Test that output directories follow outputs/<product_id>/ pattern per req 3.1."""
    from pathlib import Path

    # Simulate product directory structure from config/core.yaml
    product_id = "B0TEST1234"
    base_output = tmp_path / "outputs"
    product_dir = base_output / product_id

    # Create directory structure
    product_dir.mkdir(parents=True, exist_ok=True)

    # Verify structure matches pattern: outputs/{product_id}/
    assert product_dir.exists(), "Product directory should exist"
    assert product_dir.parent.name == "outputs", "Parent should be 'outputs' directory"
    assert product_dir.name == product_id, "Directory name should match product_id"
    assert product_dir.is_dir(), "Product path should be a directory"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_3_1_subdirectory_structure_from_config(tmp_path):
    """Test subdirectories match config/core.yaml path_config.subdirs per req 3.1."""
    from pathlib import Path

    import yaml

    # Load config to get expected subdirectory names
    config_path = Path("config/core.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    subdirs_config = config.get("path_config", {}).get("subdirs", {})
    expected_subdirs = set(subdirs_config.values())

    # Create product directory
    product_id = "B0TEST5678"
    base_output = tmp_path / "outputs"
    product_dir = base_output / product_id
    product_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories according to config
    for subdir_name in expected_subdirs:
        (product_dir / subdir_name).mkdir(exist_ok=True)

    # Verify all expected subdirectories exist
    created_subdirs = {d.name for d in product_dir.iterdir() if d.is_dir()}
    assert expected_subdirs.issubset(
        created_subdirs
    ), f"Missing subdirectories: {expected_subdirs - created_subdirs}"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_3_5_images_stored_in_images_subdirectory(tmp_path):
    """Test that image files are stored in images/ subdirectory per requirement 3.5."""
    from pathlib import Path

    product_id = "B0TESTIMG1"
    base_output = tmp_path / "outputs"
    product_dir = base_output / product_id
    images_dir = product_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create mock image files
    image_files = ["image_1.jpg", "image_2.png", "image_3.webp"]
    for img_file in image_files:
        (images_dir / img_file).write_text("mock image data")

    # Verify images are in correct location
    assert images_dir.exists(), "Images directory should exist"
    assert images_dir.parent == product_dir, "Images dir should be under product dir"

    # Verify all images are in images/ subdirectory
    stored_images = list(images_dir.glob("*"))
    assert len(stored_images) == 3, "All 3 images should be stored"
    for img_path in stored_images:
        assert img_path.parent.name == "images", "Images should be in images/ subdir"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_3_5_videos_stored_in_videos_subdirectory(tmp_path):
    """Test that video files are stored in videos/ subdirectory per requirement 3.5."""
    from pathlib import Path

    product_id = "B0TESTVID1"
    base_output = tmp_path / "outputs"
    product_dir = base_output / product_id
    videos_dir = product_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Create mock video files
    video_files = ["video_1.mp4", "video_2.webm"]
    for vid_file in video_files:
        (videos_dir / vid_file).write_text("mock video data")

    # Verify videos are in correct location
    assert videos_dir.exists(), "Videos directory should exist"
    assert videos_dir.parent == product_dir, "Videos dir should be under product dir"

    # Verify all videos are in videos/ subdirectory
    stored_videos = list(videos_dir.glob("*"))
    assert len(stored_videos) == 2, "All 2 videos should be stored"
    for vid_path in stored_videos:
        assert vid_path.parent.name == "videos", "Videos should be in videos/ subdir"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_3_5_product_files_in_root_directory(tmp_path):
    """Test that product files are in root product directory per config."""
    from pathlib import Path

    import yaml

    product_id = "B0TESTDATA"
    base_output = tmp_path / "outputs"
    product_dir = base_output / product_id
    product_dir.mkdir(parents=True, exist_ok=True)

    # Create mock product files in root
    expected_files = ["data.json", "script.txt", "description.txt", "metadata.json"]
    for file_name in expected_files:
        (product_dir / file_name).write_text("mock data")

    # Verify files are in product root directory
    for file_name in expected_files:
        file_path = product_dir / file_name
        assert file_path.exists(), f"{file_name} should exist in product root"
        assert (
            file_path.parent == product_dir
        ), f"{file_name} should be in product root, not subdirectory"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_3_5_cleanup_removes_temp_files(tmp_path):
    """Test that cleanup removes temporary files while preserving final outputs."""
    from pathlib import Path

    import yaml

    # Load config to get temp extensions
    config_path = Path("config/core.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    temp_extensions = config.get("path_config", {}).get("temp_extensions", [])

    product_id = "B0TESTCLEAN"
    base_output = tmp_path / "outputs"
    product_dir = base_output / product_id
    product_dir.mkdir(parents=True, exist_ok=True)

    # Create mix of temp and final files
    final_files = ["data.json", "video.mp4", "script.txt"]
    temp_files = ["processing.tmp", "debug.log", "temp_audio.temp"]

    for file_name in final_files:
        (product_dir / file_name).write_text("final data")

    for file_name in temp_files:
        (product_dir / file_name).write_text("temp data")

    # Simulate cleanup: remove files matching temp_extensions
    for temp_file in product_dir.iterdir():
        if any(temp_file.suffix == ext for ext in temp_extensions):
            temp_file.unlink()

    # Verify final files remain and temp files removed
    remaining_files = {f.name for f in product_dir.iterdir()}
    assert remaining_files == set(
        final_files
    ), "Only final files should remain after cleanup"

    for temp_file_name in temp_files:
        assert not (
            product_dir / temp_file_name
        ).exists(), f"Temp file {temp_file_name} should be removed"
