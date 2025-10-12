"""Unit tests for URL shortening utilities."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import aiohttp

from src.utils.url_shortener import (
    BaseURLShortener,
    ShortenedURL,
    URLShortenerError,
    URLShortenerProvider,
    PicseeURLShortener,
    URLShortenerRegistry,
    create_url_shortener,
    register_shortener,
)


class TestURLShortenerProvider:
    """Test URL shortener provider enum."""

    def test_provider_values(self):
        """Test that provider enum has expected values."""
        assert URLShortenerProvider.PICSEE.value == "picsee"
        assert URLShortenerProvider.BITLY.value == "bitly"
        assert URLShortenerProvider.TINYURL.value == "tinyurl"

    def test_provider_from_string(self):
        """Test creating provider from string value."""
        provider = URLShortenerProvider("picsee")
        assert provider == URLShortenerProvider.PICSEE

    def test_provider_invalid_string(self):
        """Test creating provider from invalid string raises ValueError."""
        with pytest.raises(ValueError):
            URLShortenerProvider("invalid_provider")


class TestShortenedURL:
    """Test ShortenedURL dataclass."""

    def test_shortened_url_creation(self):
        """Test creating a ShortenedURL object."""
        url = ShortenedURL(
            original_url="https://example.com/long-url",
            short_url="https://psee.io/abc123",
            provider=URLShortenerProvider.PICSEE,
            metadata={"picsee_id": "123", "created_at": "2024-01-01"},
        )

        assert url.original_url == "https://example.com/long-url"
        assert url.short_url == "https://psee.io/abc123"
        assert url.provider == URLShortenerProvider.PICSEE
        assert url.metadata["picsee_id"] == "123"

    def test_shortened_url_without_metadata(self):
        """Test creating ShortenedURL without metadata."""
        url = ShortenedURL(
            original_url="https://example.com/long-url",
            short_url="https://psee.io/abc123",
            provider=URLShortenerProvider.PICSEE,
        )

        assert url.metadata is None


class TestURLShortenerRegistry:
    """Test URL shortener registry functionality."""

    def test_register_shortener(self):
        """Test registering a shortener class."""

        class TestShortener(BaseURLShortener):
            @property
            def provider(self):
                return URLShortenerProvider.PICSEE

            async def shorten(self, url, custom_alias=None):
                pass

            async def shorten_bulk(self, urls):
                pass

            async def validate_api_key(self):
                pass

        # Clear registry first
        URLShortenerRegistry._providers.clear()

        URLShortenerRegistry.register(URLShortenerProvider.PICSEE, TestShortener)
        assert URLShortenerRegistry.is_provider_supported(URLShortenerProvider.PICSEE)

    def test_get_shortener_class(self):
        """Test retrieving a registered shortener class."""

        class TestShortener(BaseURLShortener):
            @property
            def provider(self):
                return URLShortenerProvider.PICSEE

            async def shorten(self, url, custom_alias=None):
                pass

            async def shorten_bulk(self, urls):
                pass

            async def validate_api_key(self):
                pass

        URLShortenerRegistry._providers.clear()
        URLShortenerRegistry.register(URLShortenerProvider.PICSEE, TestShortener)

        shortener_class = URLShortenerRegistry.get_shortener_class(
            URLShortenerProvider.PICSEE
        )
        assert shortener_class == TestShortener

    def test_get_available_providers(self):
        """Test getting list of available providers."""
        URLShortenerRegistry._providers.clear()

        class TestShortener1(BaseURLShortener):
            @property
            def provider(self):
                return URLShortenerProvider.PICSEE

            async def shorten(self, url, custom_alias=None):
                pass

            async def shorten_bulk(self, urls):
                pass

            async def validate_api_key(self):
                pass

        class TestShortener2(BaseURLShortener):
            @property
            def provider(self):
                return URLShortenerProvider.BITLY

            async def shorten(self, url, custom_alias=None):
                pass

            async def shorten_bulk(self, urls):
                pass

            async def validate_api_key(self):
                pass

        URLShortenerRegistry.register(URLShortenerProvider.PICSEE, TestShortener1)
        URLShortenerRegistry.register(URLShortenerProvider.BITLY, TestShortener2)

        providers = URLShortenerRegistry.get_available_providers()
        assert len(providers) == 2
        assert URLShortenerProvider.PICSEE in providers
        assert URLShortenerProvider.BITLY in providers

    def test_is_provider_supported(self):
        """Test checking if provider is supported."""
        URLShortenerRegistry._providers.clear()

        assert not URLShortenerRegistry.is_provider_supported(URLShortenerProvider.PICSEE)


class TestRegisterShortenerDecorator:
    """Test register_shortener decorator."""

    def test_decorator_registers_class(self):
        """Test that decorator registers the shortener class."""
        URLShortenerRegistry._providers.clear()

        @register_shortener(URLShortenerProvider.PICSEE)
        class TestShortener(BaseURLShortener):
            @property
            def provider(self):
                return URLShortenerProvider.PICSEE

            async def shorten(self, url, custom_alias=None):
                pass

            async def shorten_bulk(self, urls):
                pass

            async def validate_api_key(self):
                pass

        assert URLShortenerRegistry.is_provider_supported(URLShortenerProvider.PICSEE)
        shortener_class = URLShortenerRegistry.get_shortener_class(
            URLShortenerProvider.PICSEE
        )
        assert shortener_class == TestShortener


class TestCreateURLShortener:
    """Test URL shortener factory function."""

    def test_create_url_shortener_with_enum(self):
        """Test creating shortener with provider enum."""
        URLShortenerRegistry._providers.clear()

        @register_shortener(URLShortenerProvider.PICSEE)
        class TestShortener(BaseURLShortener):
            def __init__(self, api_key, session=None, **kwargs):
                self.api_key = api_key
                self.session = session

            @property
            def provider(self):
                return URLShortenerProvider.PICSEE

            async def shorten(self, url, custom_alias=None):
                pass

            async def shorten_bulk(self, urls):
                pass

            async def validate_api_key(self):
                pass

        shortener = create_url_shortener(
            provider=URLShortenerProvider.PICSEE, api_key="test-api-key"
        )

        assert isinstance(shortener, TestShortener)
        assert shortener.api_key == "test-api-key"

    def test_create_url_shortener_with_string(self):
        """Test creating shortener with provider string."""
        URLShortenerRegistry._providers.clear()

        @register_shortener(URLShortenerProvider.PICSEE)
        class TestShortener(BaseURLShortener):
            def __init__(self, api_key, session=None, **kwargs):
                self.api_key = api_key

            @property
            def provider(self):
                return URLShortenerProvider.PICSEE

            async def shorten(self, url, custom_alias=None):
                pass

            async def shorten_bulk(self, urls):
                pass

            async def validate_api_key(self):
                pass

        shortener = create_url_shortener(provider="picsee", api_key="test-api-key")

        assert isinstance(shortener, TestShortener)
        assert shortener.api_key == "test-api-key"

    def test_create_url_shortener_invalid_string(self):
        """Test creating shortener with invalid provider string."""
        with pytest.raises(ValueError, match="Invalid provider"):
            create_url_shortener(provider="invalid_provider", api_key="test-api-key")

    def test_create_url_shortener_unregistered_provider(self):
        """Test creating shortener with unregistered provider."""
        URLShortenerRegistry._providers.clear()

        with pytest.raises(ValueError, match="not registered"):
            create_url_shortener(
                provider=URLShortenerProvider.PICSEE, api_key="test-api-key"
            )


class TestPicseeURLShortener:
    """Test Picsee.io URL shortener implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock aiohttp session."""
        session = MagicMock(spec=aiohttp.ClientSession)
        session.closed = False
        return session

    @pytest.fixture
    def picsee_shortener(self, mock_session):
        """Create a PicseeURLShortener instance for testing."""
        return PicseeURLShortener(
            api_key="test-api-key", session=mock_session, timeout=30
        )

    def test_picsee_shortener_initialization(self, picsee_shortener):
        """Test Picsee shortener initialization."""
        assert picsee_shortener.api_key == "test-api-key"
        assert picsee_shortener.timeout == 30
        assert picsee_shortener.custom_domain is None

    def test_picsee_shortener_with_custom_domain(self):
        """Test Picsee shortener with custom domain."""
        shortener = PicseeURLShortener(
            api_key="test-api-key", custom_domain="example.com"
        )
        assert shortener.custom_domain == "example.com"

    def test_picsee_provider_property(self, picsee_shortener):
        """Test that provider property returns correct value."""
        assert picsee_shortener.provider == URLShortenerProvider.PICSEE

    def test_picsee_get_headers(self, picsee_shortener):
        """Test header generation for API requests."""
        headers = picsee_shortener._get_headers()
        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_picsee_shorten_success(self, picsee_shortener, mock_session):
        """Test successful URL shortening."""
        # Mock the API response
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "shortLink": "https://psee.io/abc123",
                "id": "test-id",
                "createdAt": "2024-01-01T00:00:00Z",
            },
        }
        mock_response.raise_for_status = MagicMock()

        # Mock the session.post context manager
        mock_session.post.return_value.__aenter__.return_value = mock_response

        result = await picsee_shortener.shorten("https://example.com/long-url")

        assert isinstance(result, ShortenedURL)
        assert result.original_url == "https://example.com/long-url"
        assert result.short_url == "https://psee.io/abc123"
        assert result.provider == URLShortenerProvider.PICSEE
        assert result.metadata["picsee_id"] == "test-id"

    @pytest.mark.asyncio
    async def test_picsee_shorten_with_custom_alias(
        self, picsee_shortener, mock_session
    ):
        """Test URL shortening with custom alias."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "shortLink": "https://psee.io/custom-alias",
                "id": "test-id",
                "createdAt": "2024-01-01T00:00:00Z",
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        result = await picsee_shortener.shorten(
            "https://example.com/long-url", custom_alias="custom-alias"
        )

        assert result.short_url == "https://psee.io/custom-alias"

    @pytest.mark.asyncio
    async def test_picsee_shorten_api_error(self, picsee_shortener, mock_session):
        """Test handling of API error responses."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "success": False,
            "message": "Invalid API key",
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(URLShortenerError, match="Picsee API error"):
            await picsee_shortener.shorten("https://example.com/long-url")

    @pytest.mark.asyncio
    async def test_picsee_shorten_network_error(self, picsee_shortener, mock_session):
        """Test handling of network errors."""
        mock_session.post.side_effect = aiohttp.ClientError("Network error")

        with pytest.raises(URLShortenerError, match="Failed to shorten URL"):
            await picsee_shortener.shorten("https://example.com/long-url")

    @pytest.mark.asyncio
    async def test_picsee_shorten_bulk_success(self, picsee_shortener, mock_session):
        """Test successful bulk URL shortening."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "success": True,
            "data": [
                {
                    "shortLink": "https://psee.io/abc123",
                    "originalUrl": "https://example.com/url1",
                    "id": "id1",
                    "createdAt": "2024-01-01T00:00:00Z",
                },
                {
                    "shortLink": "https://psee.io/def456",
                    "originalUrl": "https://example.com/url2",
                    "id": "id2",
                    "createdAt": "2024-01-01T00:00:00Z",
                },
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        urls = ["https://example.com/url1", "https://example.com/url2"]
        results = await picsee_shortener.shorten_bulk(urls)

        assert len(results) == 2
        assert results[0].original_url == "https://example.com/url1"
        assert results[0].short_url == "https://psee.io/abc123"
        assert results[1].original_url == "https://example.com/url2"
        assert results[1].short_url == "https://psee.io/def456"

    @pytest.mark.asyncio
    async def test_picsee_shorten_bulk_exceeds_limit(self, picsee_shortener):
        """Test bulk shortening with too many URLs."""
        urls = [f"https://example.com/url{i}" for i in range(101)]  # 101 URLs

        with pytest.raises(URLShortenerError, match="exceeds maximum"):
            await picsee_shortener.shorten_bulk(urls)

    @pytest.mark.asyncio
    async def test_picsee_validate_api_key_success(
        self, picsee_shortener, mock_session
    ):
        """Test API key validation with valid key."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "shortLink": "https://psee.io/test",
                "id": "test-id",
                "createdAt": "2024-01-01T00:00:00Z",
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        result = await picsee_shortener.validate_api_key()
        assert result is True

    @pytest.mark.asyncio
    async def test_picsee_validate_api_key_failure(
        self, picsee_shortener, mock_session
    ):
        """Test API key validation with invalid key."""
        mock_session.post.side_effect = URLShortenerError("Invalid API key")

        result = await picsee_shortener.validate_api_key()
        assert result is False

    @pytest.mark.asyncio
    async def test_picsee_cleanup(self, mock_session):
        """Test cleanup of owned session."""
        shortener = PicseeURLShortener(api_key="test-api-key")
        shortener._session = mock_session
        shortener._owns_session = True

        await shortener.cleanup()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_picsee_context_manager(self):
        """Test async context manager usage."""
        async with PicseeURLShortener(api_key="test-api-key") as shortener:
            assert shortener.api_key == "test-api-key"
        # cleanup should be called automatically on exit


class TestURLShortenerIntegration:
    """Integration tests for URL shortener functionality."""

    def test_end_to_end_flow(self):
        """Test complete URL shortener workflow."""
        # Clear registry
        URLShortenerRegistry._providers.clear()

        # Register Picsee provider
        register_shortener(URLShortenerProvider.PICSEE)(PicseeURLShortener)

        # Verify registration
        assert URLShortenerRegistry.is_provider_supported(URLShortenerProvider.PICSEE)

        # Create shortener instance
        shortener = create_url_shortener(
            provider="picsee", api_key="test-api-key", timeout=30
        )

        assert isinstance(shortener, PicseeURLShortener)
        assert shortener.api_key == "test-api-key"
        assert shortener.timeout == 30
