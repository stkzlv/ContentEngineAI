"""Tests for multi-account support in publisher configuration."""

import pytest

from src.publisher.config import (
    _apply_cli_overrides,
    _parse_accounts,
    load_publisher_config,
)
from src.publisher.models import AccountConfig, Platform, PublisherConfig


class TestAccountConfig:
    """Tests for AccountConfig dataclass."""

    def test_valid_account_config(self):
        """Test creating a valid account configuration."""
        account = AccountConfig(
            name="main",
            api_key="sk_live_test123456",
            vercel_token="vercel_token_123",  # noqa: S106
            description="Main production account",
        )

        assert account.name == "main"
        assert account.api_key == "sk_live_test123456"
        assert account.vercel_token == "vercel_token_123"  # noqa: S105
        assert account.description == "Main production account"
        assert account.default_platforms == []

    def test_account_with_platforms(self):
        """Test account with default platforms configured."""
        account = AccountConfig(
            name="youtube_only",
            api_key="sk_live_test123456",
            default_platforms=[Platform.YOUTUBE, Platform.TIKTOK],
        )

        assert len(account.default_platforms) == 2
        assert Platform.YOUTUBE in account.default_platforms
        assert Platform.TIKTOK in account.default_platforms

    def test_account_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Account name cannot be empty"):
            AccountConfig(name="", api_key="sk_live_test123456")

    def test_account_empty_api_key_raises(self):
        """Test that empty API key raises ValueError."""
        with pytest.raises(ValueError, match="api_key cannot be empty"):
            AccountConfig(name="test", api_key="")

    def test_account_short_api_key_raises(self):
        """Test that short API key raises ValueError."""
        with pytest.raises(ValueError, match="Invalid API key format"):
            AccountConfig(name="test", api_key="short")

    def test_account_to_dict_masks_credentials(self):
        """Test that to_dict masks sensitive credentials."""
        account = AccountConfig(
            name="main",
            api_key="sk_live_very_long_api_key",
            vercel_token="vercel_very_long_token",  # noqa: S106
        )

        result = account.to_dict()

        assert result["api_key"] == "sk_l..."
        assert result["vercel_token"] == "verc..."  # noqa: S105
        assert result["name"] == "main"

    def test_account_to_dict_none_credentials(self):
        """Test to_dict with no vercel token."""
        account = AccountConfig(
            name="main",
            api_key="sk_live_test123456",
        )

        result = account.to_dict()

        assert result["vercel_token"] is None


class TestParseAccounts:
    """Tests for _parse_accounts function."""

    def test_parse_multi_account_config(self):
        """Test parsing multi-account configuration from YAML."""
        config = {
            "provider": "late",
            "accounts": {
                "main": {
                    "api_key": "sk_live_main_key_123",
                    "vercel_token": "vercel_main_token",
                    "description": "Main account",
                },
                "secondary": {
                    "api_key": "sk_live_secondary_key",
                    "description": "Secondary account",
                },
            },
            "default_account": "main",
        }

        result = _parse_accounts(config)

        assert "accounts" in result
        assert len(result["accounts"]) == 2
        assert "main" in result["accounts"]
        assert "secondary" in result["accounts"]
        assert result["active_account"] == "main"
        assert result["api_key"] == "sk_live_main_key_123"
        assert result["vercel_token"] == "vercel_main_token"  # noqa: S105

    def test_parse_multi_account_uses_first_when_no_default(self):
        """Test that first account is used when no default specified."""
        config = {
            "provider": "late",
            "accounts": {
                "first": {
                    "api_key": "sk_live_first_key_123",
                },
                "second": {
                    "api_key": "sk_live_second_key_123",
                },
            },
        }

        result = _parse_accounts(config)

        assert result["active_account"] == "first"
        assert result["api_key"] == "sk_live_first_key_123"

    def test_parse_single_account_legacy_mode(self):
        """Test parsing legacy single-account configuration."""
        config = {
            "provider": "late",
            "api_key": "sk_live_legacy_key_123",
            "vercel_token": "vercel_legacy_token",
        }

        result = _parse_accounts(config)

        assert "accounts" in result
        assert "default" in result["accounts"]
        assert result["active_account"] == "default"
        assert result["accounts"]["default"].api_key == "sk_live_legacy_key_123"

    def test_parse_account_with_platforms(self):
        """Test parsing account with default platforms."""
        config = {
            "provider": "late",
            "accounts": {
                "youtube_focus": {
                    "api_key": "sk_live_youtube_key_123",
                    "default_platforms": ["youtube", "tiktok"],
                },
            },
        }

        result = _parse_accounts(config)

        account = result["accounts"]["youtube_focus"]
        assert len(account.default_platforms) == 2
        assert Platform.YOUTUBE in account.default_platforms

    def test_parse_skips_invalid_accounts(self):
        """Test that invalid accounts are skipped with warning."""
        config = {
            "provider": "late",
            "accounts": {
                "valid": {
                    "api_key": "sk_live_valid_key_123",
                },
                "invalid": "not a dict",
                "missing_key": {
                    "description": "No API key",
                },
            },
        }

        result = _parse_accounts(config)

        assert len(result["accounts"]) == 1
        assert "valid" in result["accounts"]

    def test_parse_removes_raw_sections(self):
        """Test that raw YAML sections are removed after parsing."""
        config = {
            "provider": "late",
            "accounts": {
                "main": {"api_key": "sk_live_main_key_123"},
            },
            "default_account": "main",
        }

        result = _parse_accounts(config)

        assert "default_account" not in result


class TestCliAccountOverride:
    """Tests for account selection via CLI override."""

    def test_cli_account_override_switches_account(self):
        """Test that --account CLI flag switches active account."""
        config = {
            "provider": "late",
            "api_key": "sk_live_first_key_123",
            "accounts": {
                "first": AccountConfig(name="first", api_key="sk_live_first_key_123"),
                "second": AccountConfig(
                    name="second",
                    api_key="sk_live_second_key_456",
                    vercel_token="vercel_second_token",  # noqa: S106
                ),
            },
            "active_account": "first",
        }

        result = _apply_cli_overrides(config, {"account": "second"})

        assert result["active_account"] == "second"
        assert result["api_key"] == "sk_live_second_key_456"
        assert result["vercel_token"] == "vercel_second_token"  # noqa: S105

    def test_cli_account_override_invalid_account_raises(self):
        """Test that invalid account name raises ValueError."""
        config = {
            "provider": "late",
            "accounts": {
                "main": AccountConfig(name="main", api_key="sk_live_main_key_123"),
            },
            "active_account": "main",
        }

        with pytest.raises(ValueError, match="Account 'nonexistent' not found"):
            _apply_cli_overrides(config, {"account": "nonexistent"})

    def test_cli_account_override_shows_available(self):
        """Test that error message shows available accounts."""
        config = {
            "provider": "late",
            "accounts": {
                "main": AccountConfig(name="main", api_key="sk_live_main_key_123"),
                "backup": AccountConfig(
                    name="backup", api_key="sk_live_backup_key_123"
                ),
            },
            "active_account": "main",
        }

        with pytest.raises(ValueError) as exc_info:
            _apply_cli_overrides(config, {"account": "invalid"})

        assert "main" in str(exc_info.value)
        assert "backup" in str(exc_info.value)


class TestPublisherConfigAccounts:
    """Tests for PublisherConfig account methods."""

    def test_get_account_by_name(self):
        """Test getting account by name."""
        main_account = AccountConfig(name="main", api_key="sk_live_main_key_123")
        config = PublisherConfig(
            provider="late",
            api_key="sk_live_main_key_123",
            accounts={"main": main_account},
            active_account="main",
        )

        result = config.get_account("main")

        assert result == main_account

    def test_get_account_uses_active_when_none(self):
        """Test that get_account uses active_account when name is None."""
        main_account = AccountConfig(name="main", api_key="sk_live_main_key_123")
        config = PublisherConfig(
            provider="late",
            api_key="sk_live_main_key_123",
            accounts={"main": main_account},
            active_account="main",
        )

        result = config.get_account()

        assert result == main_account

    def test_get_account_returns_none_for_unknown(self):
        """Test that get_account returns None for unknown account."""
        config = PublisherConfig(
            provider="late",
            api_key="sk_live_test_key_123",
            accounts={},
        )

        result = config.get_account("nonexistent")

        assert result is None

    def test_list_accounts(self):
        """Test listing all account names."""
        config = PublisherConfig(
            provider="late",
            api_key="sk_live_main_key_123",
            accounts={
                "main": AccountConfig(name="main", api_key="sk_live_main_key_123"),
                "backup": AccountConfig(
                    name="backup", api_key="sk_live_backup_key_123"
                ),
            },
        )

        result = config.list_accounts()

        assert len(result) == 2
        assert "main" in result
        assert "backup" in result

    def test_to_dict_includes_accounts(self):
        """Test that to_dict includes accounts section."""
        config = PublisherConfig(
            provider="late",
            api_key="sk_live_main_key_123",
            accounts={
                "main": AccountConfig(name="main", api_key="sk_live_main_key_123"),
            },
            active_account="main",
        )

        result = config.to_dict()

        assert "accounts" in result
        assert "main" in result["accounts"]
        assert result["active_account"] == "main"


class TestBackwardCompatibility:
    """Tests for backward compatibility with single-account mode."""

    def test_single_api_key_creates_default_account(self, tmp_path):
        """Test that single api_key creates 'default' account."""
        config_file = tmp_path / "publisher.yaml"
        config_file.write_text(
            """
provider: late
api_key: sk_live_single_key_123
vercel_token: vercel_single_token
"""
        )

        config = load_publisher_config(config_path=config_file)

        assert config.api_key == "sk_live_single_key_123"
        assert config.active_account == "default"
        assert "default" in config.accounts
        assert config.accounts["default"].api_key == "sk_live_single_key_123"

    def test_multi_account_mode_takes_precedence(self, tmp_path):
        """Test that accounts section takes precedence over api_key."""
        config_file = tmp_path / "publisher.yaml"
        config_file.write_text(
            """
provider: late
api_key: sk_live_ignored_key
accounts:
  main:
    api_key: sk_live_main_key_123
default_account: main
"""
        )

        config = load_publisher_config(config_path=config_file)

        # api_key should be from accounts, not root level
        assert config.api_key == "sk_live_main_key_123"
        assert config.active_account == "main"


class TestLoadPublisherConfigWithAccounts:
    """Integration tests for load_publisher_config with accounts."""

    def test_load_multi_account_yaml(self, tmp_path):
        """Test loading full multi-account configuration."""
        config_file = tmp_path / "publisher.yaml"
        config_file.write_text(
            """
provider: late
accounts:
  production:
    api_key: sk_live_prod_key_12345
    vercel_token: vercel_prod_token
    description: Production account
  staging:
    api_key: sk_live_staging_key_12
    description: Staging account
    default_platforms:
      - youtube
default_account: production
immediate_publish: false
max_retries: 5
"""
        )

        config = load_publisher_config(config_path=config_file)

        assert config.provider == "late"
        assert config.active_account == "production"
        assert config.api_key == "sk_live_prod_key_12345"
        assert config.vercel_token == "vercel_prod_token"  # noqa: S105
        assert len(config.accounts) == 2
        assert config.immediate_publish is False
        assert config.max_retries == 5

        # Check staging account
        staging = config.get_account("staging")
        assert staging is not None
        assert staging.api_key == "sk_live_staging_key_12"
        assert Platform.YOUTUBE in staging.default_platforms

    def test_load_with_cli_account_override(self, tmp_path):
        """Test loading config with CLI account override."""
        config_file = tmp_path / "publisher.yaml"
        config_file.write_text(
            """
provider: late
accounts:
  main:
    api_key: sk_live_main_key_12345
  overflow:
    api_key: sk_live_overflow_key_1
    vercel_token: vercel_overflow
default_account: main
"""
        )

        config = load_publisher_config(
            config_path=config_file,
            cli_overrides={"account": "overflow"},
        )

        assert config.active_account == "overflow"
        assert config.api_key == "sk_live_overflow_key_1"
        assert config.vercel_token == "vercel_overflow"  # noqa: S105
