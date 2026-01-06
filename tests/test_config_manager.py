"""Unit tests for configuration manager validation."""

import os
from unittest.mock import patch

import pytest

from src.config_manager import (
    SECRETS_REGISTRY,
    SecretDefinition,
    SecretsValidationResult,
    UnifiedConfigManager,
)


class TestSecretDefinition:
    """Test SecretDefinition dataclass."""

    @pytest.mark.unit
    def test_get_value_primary_name(self, monkeypatch):
        """Test getting secret value from primary name."""
        monkeypatch.setenv("TEST_API_KEY", "test_value_123")
        secret = SecretDefinition(
            name="TEST_API_KEY",
            description="Test API key",
            required=True,
            setup_url="https://example.com",
        )
        assert secret.get_value() == "test_value_123"

    @pytest.mark.unit
    def test_get_value_alternative_name(self, monkeypatch):
        """Test getting secret value from alternative name."""
        monkeypatch.setenv("ALT_API_KEY", "alt_value_456")
        secret = SecretDefinition(
            name="PRIMARY_API_KEY",
            description="Test API key",
            required=True,
            setup_url="https://example.com",
            alternative_names=("ALT_API_KEY",),
        )
        assert secret.get_value() == "alt_value_456"

    @pytest.mark.unit
    def test_get_value_missing(self):
        """Test getting missing secret value returns None."""
        secret = SecretDefinition(
            name="NONEXISTENT_KEY",
            description="Missing key",
            required=True,
            setup_url="https://example.com",
        )
        with patch.dict(os.environ, {}, clear=True):
            assert secret.get_value() is None

    @pytest.mark.unit
    def test_get_value_primary_takes_precedence(self, monkeypatch):
        """Test primary name takes precedence over alternative."""
        monkeypatch.setenv("PRIMARY_KEY", "primary_value")
        monkeypatch.setenv("ALT_KEY", "alt_value")
        secret = SecretDefinition(
            name="PRIMARY_KEY",
            description="Test",
            required=True,
            setup_url="https://example.com",
            alternative_names=("ALT_KEY",),
        )
        assert secret.get_value() == "primary_value"


class TestSecretsValidationResult:
    """Test SecretsValidationResult dataclass."""

    @pytest.mark.unit
    def test_valid_when_no_missing_required(self):
        """Test result is valid when no required secrets missing."""
        result = SecretsValidationResult(
            valid=True,
            missing_required=[],
            missing_optional=[],
            present=[],
        )
        assert result.valid is True

    @pytest.mark.unit
    def test_invalid_when_missing_required(self):
        """Test result is invalid when required secrets missing."""
        missing = SecretDefinition(
            name="REQUIRED_KEY",
            description="Required",
            required=True,
            setup_url="https://example.com",
        )
        result = SecretsValidationResult(
            valid=False,
            missing_required=[missing],
            missing_optional=[],
            present=[],
        )
        assert result.valid is False
        assert len(result.missing_required) == 1


class TestValidateRequiredSecrets:
    """Test validate_required_secrets() method."""

    @pytest.fixture
    def config_manager(self):
        """Create a config manager instance."""
        return UnifiedConfigManager()

    @pytest.mark.unit
    def test_all_secrets_present(self, config_manager, monkeypatch):
        """Test validation passes when all secrets are present."""
        # Set all required secrets
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test1234567890")
        monkeypatch.setenv("PEXELS_API_KEY", "pexels-test-key-1234")
        monkeypatch.setenv("FREESOUND_API_KEY", "freesound-test-key")
        # Set optional secrets too
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/creds.json")
        monkeypatch.setenv("LATE_API_KEY", "late-api-key-test")
        monkeypatch.setenv("PICSEE_API_KEY", "picsee-key-test")

        result = config_manager.validate_required_secrets(exit_on_missing=False)

        assert result.valid is True
        assert len(result.missing_required) == 0
        assert len(result.missing_optional) == 0
        assert len(result.present) == len(SECRETS_REGISTRY)

    @pytest.mark.unit
    def test_required_secret_missing(self, config_manager, monkeypatch):
        """Test validation fails when required secret is missing."""
        # Clear all env vars and set only some required ones
        for secret in SECRETS_REGISTRY:
            monkeypatch.delenv(secret.name, raising=False)
            for alt in secret.alternative_names:
                monkeypatch.delenv(alt, raising=False)

        # Set only PEXELS and FREESOUND, missing OPENROUTER
        monkeypatch.setenv("PEXELS_API_KEY", "pexels-test-key")
        monkeypatch.setenv("FREESOUND_API_KEY", "freesound-test-key")

        result = config_manager.validate_required_secrets(exit_on_missing=False)

        assert result.valid is False
        assert len(result.missing_required) == 1
        assert result.missing_required[0].name == "OPENROUTER_API_KEY"

    @pytest.mark.unit
    def test_optional_secret_missing_still_valid(self, config_manager, monkeypatch):
        """Test validation passes when only optional secrets are missing."""
        # Clear all env vars
        for secret in SECRETS_REGISTRY:
            monkeypatch.delenv(secret.name, raising=False)
            for alt in secret.alternative_names:
                monkeypatch.delenv(alt, raising=False)

        # Set all required secrets
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test1234567890")
        monkeypatch.setenv("PEXELS_API_KEY", "pexels-test-key-1234")
        monkeypatch.setenv("FREESOUND_API_KEY", "freesound-test-key")
        # Don't set optional secrets

        result = config_manager.validate_required_secrets(exit_on_missing=False)

        assert result.valid is True
        assert len(result.missing_required) == 0
        assert len(result.missing_optional) == 3  # GOOGLE, LATE, PICSEE
        assert len(result.present) == 3

    @pytest.mark.unit
    def test_alternative_name_accepted(self, config_manager, monkeypatch):
        """Test that alternative env var names are accepted."""
        # Clear all env vars
        for secret in SECRETS_REGISTRY:
            monkeypatch.delenv(secret.name, raising=False)
            for alt in secret.alternative_names:
                monkeypatch.delenv(alt, raising=False)

        # Set required secrets
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test1234567890")
        monkeypatch.setenv("PEXELS_API_KEY", "pexels-test-key-1234")
        monkeypatch.setenv("FREESOUND_API_KEY", "freesound-test-key")
        # Use alternative name for LATE_API_KEY
        monkeypatch.setenv("PUBLISHER_API_KEY", "publisher-key-via-alt")

        result = config_manager.validate_required_secrets(exit_on_missing=False)

        assert result.valid is True
        # LATE_API_KEY should be in present list (found via alternative name)
        present_names = [s.name for s in result.present]
        assert "LATE_API_KEY" in present_names

    @pytest.mark.unit
    def test_exit_on_missing_required(self, config_manager, monkeypatch):
        """Test that exit_on_missing=True causes SystemExit."""
        # Clear all env vars
        for secret in SECRETS_REGISTRY:
            monkeypatch.delenv(secret.name, raising=False)
            for alt in secret.alternative_names:
                monkeypatch.delenv(alt, raising=False)

        with pytest.raises(SystemExit) as exc_info:
            config_manager.validate_required_secrets(exit_on_missing=True)

        assert exc_info.value.code == 1

    @pytest.mark.unit
    def test_no_exit_when_valid(self, config_manager, monkeypatch):
        """Test that exit_on_missing=True does not exit when valid."""
        # Set all required secrets
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test1234567890")
        monkeypatch.setenv("PEXELS_API_KEY", "pexels-test-key-1234")
        monkeypatch.setenv("FREESOUND_API_KEY", "freesound-test-key")

        # Should not raise
        result = config_manager.validate_required_secrets(exit_on_missing=True)
        assert result.valid is True

    @pytest.mark.unit
    def test_multiple_required_missing(self, config_manager, monkeypatch):
        """Test validation reports all missing required secrets."""
        # Clear all env vars
        for secret in SECRETS_REGISTRY:
            monkeypatch.delenv(secret.name, raising=False)
            for alt in secret.alternative_names:
                monkeypatch.delenv(alt, raising=False)

        # Set only one required secret
        monkeypatch.setenv("FREESOUND_API_KEY", "freesound-test-key")

        result = config_manager.validate_required_secrets(exit_on_missing=False)

        assert result.valid is False
        assert len(result.missing_required) == 2
        missing_names = [s.name for s in result.missing_required]
        assert "OPENROUTER_API_KEY" in missing_names
        assert "PEXELS_API_KEY" in missing_names


class TestSecretsRegistry:
    """Test SECRETS_REGISTRY configuration."""

    @pytest.mark.unit
    def test_registry_has_required_secrets(self):
        """Test registry contains expected required secrets."""
        required_names = [s.name for s in SECRETS_REGISTRY if s.required]
        assert "OPENROUTER_API_KEY" in required_names
        assert "PEXELS_API_KEY" in required_names
        assert "FREESOUND_API_KEY" in required_names

    @pytest.mark.unit
    def test_registry_has_optional_secrets(self):
        """Test registry contains expected optional secrets."""
        optional_names = [s.name for s in SECRETS_REGISTRY if not s.required]
        assert "GOOGLE_APPLICATION_CREDENTIALS" in optional_names
        assert "LATE_API_KEY" in optional_names
        assert "PICSEE_API_KEY" in optional_names

    @pytest.mark.unit
    def test_all_secrets_have_setup_urls(self):
        """Test all secrets have setup URLs."""
        for secret in SECRETS_REGISTRY:
            assert secret.setup_url, f"{secret.name} missing setup_url"
            assert secret.setup_url.startswith(
                "https://"
            ), f"{secret.name} has invalid URL"

    @pytest.mark.unit
    def test_all_secrets_have_descriptions(self):
        """Test all secrets have descriptions."""
        for secret in SECRETS_REGISTRY:
            assert secret.description, f"{secret.name} missing description"
            assert (
                len(secret.description) > 10
            ), f"{secret.name} has too short description"
