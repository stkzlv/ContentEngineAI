"""Configuration management for video publisher.

This module provides three-tier configuration precedence:
1. YAML file (config/publisher.yaml) - lowest precedence
2. Environment variables - medium precedence
3. CLI arguments - highest precedence
"""

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from src.publisher.models import (
    DEFAULT_PLATFORMS,
    AccountConfig,
    BlobRetentionConfig,
    CleanupConfig,
    FirstCommentConfig,
    LinkInBioConfig,
    Platform,
    PublisherConfig,
    RecurringSlot,
    ScheduleConfig,
    TikTokContentSettings,
)
from src.video.config.constants import LATE_API_KEY_MIN_LENGTH

logger = logging.getLogger(__name__)


def load_publisher_config(
    config_path: Path | str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> PublisherConfig:
    """Load publisher configuration with three-tier precedence.

    Precedence order (highest to lowest):
    1. CLI arguments (cli_overrides parameter)
    2. Environment variables (LATE_API_KEY, LATE_VERCEL_TOKEN, etc.)
    3. YAML config file (config/publisher.yaml)

    Args:
    ----
        config_path: Path to YAML config file (default: config/publisher.yaml)
        cli_overrides: CLI argument overrides (highest precedence)

    Returns:
    -------
        PublisherConfig instance with all precedence rules applied

    Raises:
    ------
        FileNotFoundError: If config file doesn't exist and no env/CLI
            overrides provided
        ValueError: If required fields missing (provider, api_key)
        ValidationError: If config validation fails

    Example:
    -------
        >>> config = load_publisher_config(
        ...     cli_overrides={
        ...         "provider": "late",
        ...         "immediate_publish": True
        ...     }
        ... )
        >>> print(config.provider)
        late

    """
    # Determine config file path
    if config_path is None:
        config_path = Path("config/publisher.yaml")
    elif isinstance(config_path, str):
        config_path = Path(config_path)

    logger.info("Loading publisher configuration from %s", config_path)

    # Load YAML config (lowest precedence)
    yaml_config = _load_yaml_config(config_path)

    # Parse schedule and cleanup configurations from YAML
    yaml_config = _parse_schedule_and_cleanup_config(yaml_config)

    # Parse accounts (multi-account support)
    yaml_config = _parse_accounts(yaml_config)

    # Apply environment variable overrides (medium precedence)
    config_dict = _apply_env_overrides(yaml_config)

    # Apply CLI overrides (highest precedence)
    if cli_overrides:
        config_dict = _apply_cli_overrides(config_dict, cli_overrides)

    # Set defaults for missing optional fields
    config_dict = _apply_defaults(config_dict)

    # Validate required fields
    _validate_required_fields(config_dict)

    # Strip keys not accepted by PublisherConfig (e.g. deprecated backoff_multiplier)
    _known = {f.name for f in dataclasses.fields(PublisherConfig)}
    config_dict = {k: v for k, v in config_dict.items() if k in _known}

    # Convert to PublisherConfig dataclass
    try:
        config = PublisherConfig(**config_dict)
        logger.info(
            "Configuration loaded: provider=%s, immediate_publish=%s, max_retries=%d",
            config.provider,
            config.immediate_publish,
            config.max_retries,
        )
        return config
    except (ValueError, TypeError) as e:
        logger.error("Configuration validation failed: %s", e)
        raise ValueError(f"Invalid publisher configuration: {e}") from e


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
    ----
        config_path: Path to YAML config file

    Returns:
    -------
        Configuration dictionary from YAML, or empty dict if file doesn't exist

    """
    if not config_path.exists():
        logger.warning(
            "Config file not found: %s, using defaults and env vars", config_path
        )
        return {}

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                logger.warning(
                    "Invalid YAML structure in %s, using empty config", config_path
                )
                return {}
            logger.debug("Loaded YAML config from %s", config_path)
            return config
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML file %s: %s", config_path, e)
        return {}
    except OSError as e:
        logger.error("Error loading config file %s: %s", config_path, e)
        return {}


def _parse_schedule_and_cleanup_config(config: dict[str, Any]) -> dict[str, Any]:
    """Parse schedule and cleanup configuration sections from YAML.

    Parses three sections:
    1. recurring_schedule: enabled, timezone, slots (list of RecurringSlot)
    2. schedule_validation: min_post_spacing_hours, prevent_duplicates, etc.
    3. cleanup: enabled, verify_before_delete, require_all_platforms, etc.

    Args:
    ----
        config: Raw YAML configuration dictionary

    Returns:
    -------
        Configuration with schedule_config and cleanup_config objects

    """
    result = config.copy()

    # Parse recurring_schedule section
    recurring_schedule = config.get("recurring_schedule", {})
    schedule_validation = config.get("schedule_validation", {})

    # Merge recurring_schedule and schedule_validation into ScheduleConfig
    schedule_config_dict = {}

    # From recurring_schedule section (enabled, timezone, slots)
    if "enabled" in recurring_schedule:
        schedule_config_dict["enabled"] = recurring_schedule["enabled"]
    if "timezone" in recurring_schedule:
        schedule_config_dict["timezone"] = recurring_schedule["timezone"]

    # Parse slots if present
    slots_data = recurring_schedule.get("slots", [])
    if slots_data and isinstance(slots_data, list):
        try:
            slots = []
            for slot_dict in slots_data:
                default_tz = recurring_schedule.get("timezone", "UTC")
                slot = RecurringSlot(
                    day_of_week=slot_dict["day_of_week"],
                    time=slot_dict["time"],
                    timezone=slot_dict.get("timezone", default_tz),
                )
                slots.append(slot)
            schedule_config_dict["slots"] = slots
            logger.debug("Parsed %d recurring slots from config", len(slots))
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse recurring slots: %s, using empty slots", e)
            schedule_config_dict["slots"] = []

    # From schedule_validation section
    if "min_post_spacing_hours" in schedule_validation:
        schedule_config_dict["min_post_spacing_hours"] = schedule_validation[
            "min_post_spacing_hours"
        ]
    if "prevent_duplicates" in schedule_validation:
        schedule_config_dict["prevent_duplicates"] = schedule_validation[
            "prevent_duplicates"
        ]
    if "allow_past_schedules" in schedule_validation:
        schedule_config_dict["allow_past_schedules"] = schedule_validation[
            "allow_past_schedules"
        ]
    if "max_posts_per_day" in schedule_validation:
        schedule_config_dict["max_posts_per_day"] = schedule_validation[
            "max_posts_per_day"
        ]

    # From top-level config (content strategy)
    if "use_platform_specific_content" in config:
        schedule_config_dict["use_platform_specific_content"] = config[
            "use_platform_specific_content"
        ]

    # Create ScheduleConfig if any config provided
    if schedule_config_dict:
        try:
            result["schedule_config"] = ScheduleConfig(**schedule_config_dict)
            logger.debug("Parsed schedule config: %s", schedule_config_dict)
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse schedule config: %s, using defaults", e)
            result["schedule_config"] = ScheduleConfig()
    else:
        result["schedule_config"] = ScheduleConfig()

    # Parse cleanup section
    cleanup_section = config.get("cleanup", {})
    cleanup_config_dict = {}

    for key in [
        "enabled",
        "verify_before_delete",
        "require_all_platforms",
        "archive_before_delete",
        "keep_published_days",
        "preserve_metadata",
        "preserve_logs",
    ]:
        if key in cleanup_section:
            cleanup_config_dict[key] = cleanup_section[key]

    # Handle archive_dir specially (convert to Path)
    if "archive_dir" in cleanup_section:
        cleanup_config_dict["archive_dir"] = Path(cleanup_section["archive_dir"])

    # Create CleanupConfig if any config provided
    if cleanup_config_dict:
        try:
            result["cleanup_config"] = CleanupConfig(**cleanup_config_dict)
            logger.debug("Parsed cleanup config: %s", cleanup_config_dict)
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse cleanup config: %s, using defaults", e)
            result["cleanup_config"] = CleanupConfig()
    else:
        result["cleanup_config"] = CleanupConfig()

    # Parse link_in_bio config
    link_in_bio_section = result.get("link_in_bio", {})
    if link_in_bio_section:
        try:
            result["link_in_bio_config"] = LinkInBioConfig(**link_in_bio_section)
            logger.debug("Parsed link_in_bio config: %s", link_in_bio_section)
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse link_in_bio config: %s, using defaults", e)
            result["link_in_bio_config"] = LinkInBioConfig()
    else:
        result["link_in_bio_config"] = LinkInBioConfig()

    # Parse tiktok_settings config
    tiktok_section = result.get("tiktok_settings", {})
    if tiktok_section:
        try:
            result["tiktok_settings"] = TikTokContentSettings(**tiktok_section)
            logger.debug("Parsed tiktok_settings config: %s", tiktok_section)
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse tiktok_settings: %s, using defaults", e)
            result["tiktok_settings"] = TikTokContentSettings()
    else:
        result["tiktok_settings"] = TikTokContentSettings()

    # Parse first_comment config
    first_comment_section = result.get("first_comment", {})
    if first_comment_section:
        try:
            result["first_comment_config"] = FirstCommentConfig(**first_comment_section)
            logger.debug("Parsed first_comment config: %s", first_comment_section)
        except (ValueError, TypeError) as e:
            logger.warning(
                "Failed to parse first_comment config: %s, using defaults", e
            )
            result["first_comment_config"] = FirstCommentConfig()
    else:
        result["first_comment_config"] = FirstCommentConfig()

    # Parse blob_retention config
    blob_retention_section = result.get("blob_retention", {})
    if blob_retention_section:
        try:
            result["blob_retention_config"] = BlobRetentionConfig(
                **blob_retention_section
            )
            logger.debug("Parsed blob_retention config: %s", blob_retention_section)
        except (ValueError, TypeError) as e:
            logger.warning(
                "Failed to parse blob_retention config: %s, using defaults", e
            )
            result["blob_retention_config"] = BlobRetentionConfig()
    else:
        result["blob_retention_config"] = BlobRetentionConfig()

    # Remove raw YAML sections (already parsed into objects)
    result.pop("recurring_schedule", None)
    result.pop("schedule_validation", None)
    result.pop("cleanup", None)
    result.pop("link_in_bio", None)
    result.pop("first_comment", None)
    # Keep use_platform_specific_content for PublisherConfig (don't pop)

    return result


def _parse_accounts(config: dict[str, Any]) -> dict[str, Any]:
    """Parse accounts section from YAML configuration.

    Supports both multi-account mode (accounts section) and single-account mode
    (api_key at root level) for backward compatibility.

    Multi-account YAML format:
        accounts:
          main:
            api_key: sk_live_...
            vercel_token: vercel_...
            description: "Main production account"
          secondary:
            api_key: sk_live_...
            description: "Overflow account"
        default_account: main

    Single-account (legacy) YAML format:
        api_key: sk_live_...
        vercel_token: vercel_...

    Args:
    ----
        config: Raw YAML configuration dictionary

    Returns:
    -------
        Configuration with parsed accounts dict

    """
    result = config.copy()
    accounts_dict: dict[str, AccountConfig] = {}

    # Check for multi-account configuration
    accounts_section = config.get("accounts", {})

    if accounts_section and isinstance(accounts_section, dict):
        # Multi-account mode
        for name, account_data in accounts_section.items():
            if not isinstance(account_data, dict):
                logger.warning("Invalid account config for '%s', skipping", name)
                continue

            # Get API key (supports env var reference)
            api_key = account_data.get("api_key")
            if not api_key:
                logger.warning("Account '%s' missing api_key, skipping", name)
                continue

            # Get vercel token
            vercel_token = account_data.get("vercel_token")

            # Get description
            description = account_data.get("description", "")

            # Parse default platforms for this account
            platforms_data = account_data.get("default_platforms", [])
            default_platforms = []
            if platforms_data:
                try:
                    default_platforms = [
                        Platform(p.lower()) if isinstance(p, str) else p
                        for p in platforms_data
                    ]
                except ValueError as e:
                    logger.warning(
                        "Invalid platform in account '%s': %s, " "using empty list",
                        name,
                        e,
                    )

            try:
                accounts_dict[name] = AccountConfig(
                    name=name,
                    api_key=api_key,
                    vercel_token=vercel_token,
                    description=description,
                    default_platforms=default_platforms,
                )
                logger.debug("Parsed account: %s", name)
            except ValueError as e:
                logger.warning("Failed to create account '%s': %s", name, e)

        if accounts_dict:
            result["accounts"] = accounts_dict
            logger.info("Loaded %d account(s) from config", len(accounts_dict))

            # Set default account if specified
            default_account = config.get("default_account")
            if default_account and default_account in accounts_dict:
                result["active_account"] = default_account
                # Set api_key and vercel_token from default account
                default_acc = accounts_dict[default_account]
                result["api_key"] = default_acc.api_key
                result["vercel_token"] = default_acc.vercel_token
                logger.info("Using default account: %s", default_account)
            elif accounts_dict:
                # Use first account if no default specified
                first_account = next(iter(accounts_dict.values()))
                result["active_account"] = first_account.name
                result["api_key"] = first_account.api_key
                result["vercel_token"] = first_account.vercel_token
                logger.info(
                    "No default_account specified, using first: %s", first_account.name
                )
    else:
        # Single-account mode (legacy) - create "default" account if api_key exists
        api_key = config.get("api_key")
        if api_key:
            vercel_token = config.get("vercel_token")
            try:
                accounts_dict["default"] = AccountConfig(
                    name="default",
                    api_key=api_key,
                    vercel_token=vercel_token,
                    description="Default account (legacy single-account mode)",
                )
                result["accounts"] = accounts_dict
                result["active_account"] = "default"
                logger.debug("Created default account from legacy config")
            except ValueError as e:
                logger.debug("Could not create default account: %s", e)

    # Remove raw accounts section (already parsed)
    result.pop("default_account", None)
    if "accounts" in result and not isinstance(result["accounts"], dict):
        result.pop("accounts", None)

    return result


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to configuration.

    Environment variables (medium precedence):
    - LATE_API_KEY / PUBLISHER_API_KEY
    - BLOB_READ_WRITE_TOKEN / LATE_VERCEL_TOKEN / PUBLISHER_VERCEL_TOKEN
    - PUBLISHER_PROVIDER
    - PUBLISHER_IMMEDIATE
    - PUBLISHER_MAX_RETRIES
    - PUBLISHER_TIMEOUT
    - PUBLISHER_DEFAULT_PLATFORMS (comma-separated: youtube,tiktok,instagram)

    Args:
    ----
        config: Base configuration from YAML

    Returns:
    -------
        Configuration with environment variable overrides applied

    """
    result = config.copy()

    # API Key (check multiple env vars for flexibility)
    api_key = (
        os.environ.get("LATE_API_KEY")
        or os.environ.get("PUBLISHER_API_KEY")
        or config.get("api_key")
    )
    if api_key:
        result["api_key"] = api_key

    # Vercel Token (supports official Vercel SDK naming: BLOB_READ_WRITE_TOKEN)
    vercel_token = (
        os.environ.get("BLOB_READ_WRITE_TOKEN")
        or os.environ.get("LATE_VERCEL_TOKEN")
        or os.environ.get("PUBLISHER_VERCEL_TOKEN")
        or config.get("vercel_token")
    )
    if vercel_token:
        result["vercel_token"] = vercel_token

    # Provider
    provider = os.environ.get("PUBLISHER_PROVIDER") or config.get("provider")
    if provider:
        result["provider"] = provider

    # Immediate publish
    immediate = os.environ.get("PUBLISHER_IMMEDIATE")
    if immediate is not None:
        result["immediate_publish"] = immediate.lower() in ("true", "1", "yes")

    # Max retries
    max_retries = os.environ.get("PUBLISHER_MAX_RETRIES")
    if max_retries is not None:
        try:
            result["max_retries"] = int(max_retries)
        except ValueError:
            logger.warning("Invalid PUBLISHER_MAX_RETRIES: %s", max_retries)

    # Timeout
    timeout = os.environ.get("PUBLISHER_TIMEOUT")
    if timeout is not None:
        try:
            result["timeout"] = float(timeout)
        except ValueError:
            logger.warning("Invalid PUBLISHER_TIMEOUT: %s", timeout)

    # Default platforms (comma-separated)
    platforms_str = os.environ.get("PUBLISHER_DEFAULT_PLATFORMS")
    if platforms_str:
        try:
            platform_list = [p.strip() for p in platforms_str.split(",")]
            result["default_platforms"] = [Platform(p.lower()) for p in platform_list]
        except ValueError as e:
            logger.warning("Invalid PUBLISHER_DEFAULT_PLATFORMS: %s", e)

    # Privacy settings (e.g., PUBLISHER_PRIVACY_YOUTUBE=public)
    for platform in ["youtube", "tiktok", "instagram"]:
        env_var = f"PUBLISHER_PRIVACY_{platform.upper()}"
        privacy_value = os.environ.get(env_var)
        if privacy_value:
            if "privacy_settings" not in result:
                result["privacy_settings"] = {}
            result["privacy_settings"][Platform(platform)] = privacy_value

    logger.debug("Applied environment variable overrides")
    return result


def _apply_cli_overrides(
    config: dict[str, Any], cli_overrides: dict[str, Any]
) -> dict[str, Any]:
    """Apply CLI argument overrides to configuration.

    CLI arguments have highest precedence and override both YAML and env vars.

    Common CLI override keys:
    - provider: Publishing service provider
    - api_key: API key for the provider
    - vercel_token: Vercel token for large uploads
    - platforms: List of platform names or Platform enums
    - immediate: Boolean for immediate publishing
    - max_retries: Integer for retry attempts
    - timeout: Float for request timeout
    - account: Name of account to use (multi-account support)

    Args:
    ----
        config: Configuration with YAML and env overrides
        cli_overrides: CLI argument overrides

    Returns:
    -------
        Configuration with CLI overrides applied

    """
    result = config.copy()

    for key, value in cli_overrides.items():
        if value is None:
            continue

        # Handle special cases
        if key == "platforms":
            # Convert platform strings to Platform enums
            if isinstance(value, list):
                try:
                    result["default_platforms"] = [
                        Platform(p) if isinstance(p, str) else p for p in value
                    ]
                except ValueError as e:
                    logger.warning("Invalid platform in CLI: %s", e)
            continue

        if key == "immediate":
            result["immediate_publish"] = value
            continue

        if key == "account":
            # Switch to specified account
            accounts = result.get("accounts", {})
            if value in accounts:
                account = accounts[value]
                result["active_account"] = value
                result["api_key"] = account.api_key
                result["vercel_token"] = account.vercel_token
                logger.info("Switched to account: %s", value)
            else:
                available = list(accounts.keys()) if accounts else []
                raise ValueError(
                    f"Account '{value}' not found. "
                    f"Available accounts: {available or 'none configured'}"
                )
            continue

        # Direct mapping for other keys
        result[key] = value

    logger.debug("Applied CLI overrides: %s", list(cli_overrides.keys()))
    return result


def _apply_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Apply sensible defaults for missing optional fields.

    Defaults:
    - provider: "late"
    - immediate_publish: True
    - max_retries: 3
    - timeout: 120.0 (matches YAML default, allows for slow video processing)
    - stagger_delay_min: 30
    - stagger_delay_max: 60
    - default_platforms: [youtube, tiktok, instagram]
    - privacy_settings: {}
    - accounts: {} (empty dict if no accounts configured)
    - active_account: None
    - schedule_config: ScheduleConfig() with defaults
    - cleanup_config: CleanupConfig() with defaults

    Args:
    ----
        config: Configuration with YAML/env/CLI overrides

    Returns:
    -------
        Configuration with defaults applied for missing fields

    """
    defaults = {
        "provider": "late",
        "immediate_publish": True,
        "max_retries": 3,
        "timeout": 120.0,  # TikTok video processing can take 60-120 seconds
        "stagger_delay_min": 30,
        "stagger_delay_max": 60,
        "default_platforms": list(DEFAULT_PLATFORMS),
        "privacy_settings": {},
        "accounts": {},
        "active_account": None,
        "schedule_config": ScheduleConfig(),
        "cleanup_config": CleanupConfig(),
        "link_in_bio_config": LinkInBioConfig(),
        "tiktok_settings": TikTokContentSettings(),
        "first_comment_config": FirstCommentConfig(),
    }

    for key, default_value in defaults.items():
        if key not in config or config[key] is None:
            config[key] = default_value

    logger.debug("Applied default configuration values")
    return config


def _validate_required_fields(config: dict[str, Any]) -> None:
    """Validate that required configuration fields are present.

    Required fields:
    - provider: Publishing service provider
    - api_key: API key for authentication

    Args:
    ----
        config: Configuration dictionary to validate

    Raises:
    ------
        ValueError: If required fields are missing

    """
    # Validate provider
    if "provider" not in config or not config["provider"]:
        raise ValueError(
            "Missing required field: 'provider'. "
            "Set via YAML (provider: late), env var (PUBLISHER_PROVIDER=late), "
            "or CLI argument (--provider late)"
        )

    # Validate API key
    if "api_key" not in config or not config["api_key"]:
        raise ValueError(
            "Missing required field: 'api_key'. "
            "Set via YAML (api_key: sk_live_...), env var (LATE_API_KEY=sk_live_...), "
            "or CLI argument (--api-key sk_live_...)"
        )

    # Validate API key format (basic check)
    api_key = config["api_key"]
    if not isinstance(api_key, str) or len(api_key) < LATE_API_KEY_MIN_LENGTH:
        raise ValueError(
            f"Invalid API key format: must be string with at least "
            f"{LATE_API_KEY_MIN_LENGTH} characters "
            f"(got {len(api_key) if isinstance(api_key, str) else 'non-string'})"
        )

    logger.debug("Required fields validated successfully")


def create_default_config_file(
    output_path: Path | str = Path("config/publisher.yaml"),
) -> None:
    """Create a default publisher.yaml configuration file.

    Args:
    ----
        output_path: Path where to create the config file

    Example:
    -------
        >>> create_default_config_file()
        Created default config: config/publisher.yaml

    """
    if isinstance(output_path, str):
        output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sep = "# " + "=" * 77
    template = f"""\
# Publisher Configuration

{sep}
# SINGLE ACCOUNT MODE (Legacy - use env vars for credentials)
{sep}
provider: late
api_key_env_var: "LATE_API_KEY"
vercel_token_env_var: "LATE_VERCEL_TOKEN"

{sep}
# MULTI-ACCOUNT MODE (uncomment to enable)
{sep}
# accounts:
#   main:
#     api_key: ${{LATE_API_KEY}}  # Use env var reference
#     vercel_token: ${{LATE_VERCEL_TOKEN}}
#     description: "Main production account"
#   secondary:
#     api_key: ${{LATE_API_KEY_2}}
#     vercel_token: ${{LATE_VERCEL_TOKEN_2}}
#     description: "Overflow account for high volume"
#     default_platforms:
#       - youtube
#       - tiktok
# default_account: main  # Which account to use by default

{sep}
# PUBLISHING BEHAVIOR
{sep}
immediate_publish: true
default_platforms:
  - youtube
  - tiktok
  - instagram

# Retry and timeout settings
max_retries: 3
timeout: 120.0

# Batch publishing delays (seconds)
stagger_delay_min: 30
stagger_delay_max: 60

# Privacy settings per platform
privacy_settings:
  youtube: public
  tiktok: public
  instagram: everyone
"""
    output_path.write_text(template, encoding="utf-8")

    logger.info("Created default config file: %s", output_path)
    print(f"Created default publisher config: {output_path}")
