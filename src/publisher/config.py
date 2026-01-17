"""Configuration management for video publisher.

This module provides three-tier configuration precedence:
1. YAML file (config/publisher.yaml) - lowest precedence
2. Environment variables - medium precedence
3. CLI arguments - highest precedence
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from src.publisher.models import (
    AccountConfig,
    CleanupConfig,
    Platform,
    PublisherConfig,
    RecurringSlot,
    ScheduleConfig,
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

    logger.info(f"Loading publisher configuration from {config_path}")

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

    # Convert to PublisherConfig Pydantic model
    try:
        config = PublisherConfig(**config_dict)
        logger.info(
            f"Configuration loaded: provider={config.provider}, "
            f"immediate_publish={config.immediate_publish}, "
            f"max_retries={config.max_retries}"
        )
        return config
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
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
            f"Config file not found: {config_path}, using defaults and env vars"
        )
        return {}

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                logger.warning(
                    f"Invalid YAML structure in {config_path}, using empty config"
                )
                return {}
            logger.debug(f"Loaded YAML config from {config_path}")
            return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
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
            logger.debug(f"Parsed {len(slots)} recurring slots from config")
        except Exception as e:
            logger.warning(f"Failed to parse recurring slots: {e}, using empty slots")
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
            logger.debug(f"Parsed schedule config: {schedule_config_dict}")
        except Exception as e:
            logger.warning(f"Failed to parse schedule config: {e}, using defaults")
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
            logger.debug(f"Parsed cleanup config: {cleanup_config_dict}")
        except Exception as e:
            logger.warning(f"Failed to parse cleanup config: {e}, using defaults")
            result["cleanup_config"] = CleanupConfig()
    else:
        result["cleanup_config"] = CleanupConfig()

    # Remove raw YAML sections (already parsed into objects)
    result.pop("recurring_schedule", None)
    result.pop("schedule_validation", None)
    result.pop("cleanup", None)
    result.pop("use_platform_specific_content", None)

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
                logger.warning(f"Invalid account config for '{name}', skipping")
                continue

            # Get API key (supports env var reference)
            api_key = account_data.get("api_key")
            if not api_key:
                logger.warning(f"Account '{name}' missing api_key, skipping")
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
                        f"Invalid platform in account '{name}': {e}, using empty list"
                    )

            try:
                accounts_dict[name] = AccountConfig(
                    name=name,
                    api_key=api_key,
                    vercel_token=vercel_token,
                    description=description,
                    default_platforms=default_platforms,
                )
                logger.debug(f"Parsed account: {name}")
            except ValueError as e:
                logger.warning(f"Failed to create account '{name}': {e}")

        if accounts_dict:
            result["accounts"] = accounts_dict
            logger.info(f"Loaded {len(accounts_dict)} account(s) from config")

            # Set default account if specified
            default_account = config.get("default_account")
            if default_account and default_account in accounts_dict:
                result["active_account"] = default_account
                # Set api_key and vercel_token from default account
                default_acc = accounts_dict[default_account]
                result["api_key"] = default_acc.api_key
                result["vercel_token"] = default_acc.vercel_token
                logger.info(f"Using default account: {default_account}")
            elif accounts_dict:
                # Use first account if no default specified
                first_account = next(iter(accounts_dict.values()))
                result["active_account"] = first_account.name
                result["api_key"] = first_account.api_key
                result["vercel_token"] = first_account.vercel_token
                logger.info(
                    f"No default_account specified, using first: {first_account.name}"
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
                logger.debug(f"Could not create default account: {e}")

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
            logger.warning(f"Invalid PUBLISHER_MAX_RETRIES: {max_retries}")

    # Timeout
    timeout = os.environ.get("PUBLISHER_TIMEOUT")
    if timeout is not None:
        try:
            result["timeout"] = float(timeout)
        except ValueError:
            logger.warning(f"Invalid PUBLISHER_TIMEOUT: {timeout}")

    # Default platforms (comma-separated)
    platforms_str = os.environ.get("PUBLISHER_DEFAULT_PLATFORMS")
    if platforms_str:
        try:
            platform_list = [p.strip() for p in platforms_str.split(",")]
            result["default_platforms"] = [Platform(p.lower()) for p in platform_list]
        except ValueError as e:
            logger.warning(f"Invalid PUBLISHER_DEFAULT_PLATFORMS: {e}")

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
                    logger.warning(f"Invalid platform in CLI: {e}")
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
                logger.info(f"Switched to account: {value}")
            else:
                available = list(accounts.keys()) if accounts else []
                raise ValueError(
                    f"Account '{value}' not found. "
                    f"Available accounts: {available or 'none configured'}"
                )
            continue

        # Direct mapping for other keys
        result[key] = value

    logger.debug(f"Applied CLI overrides: {list(cli_overrides.keys())}")
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
        "default_platforms": [Platform.YOUTUBE, Platform.TIKTOK, Platform.INSTAGRAM],
        "privacy_settings": {},
        "accounts": {},
        "active_account": None,
        "schedule_config": ScheduleConfig(),
        "cleanup_config": CleanupConfig(),
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

    with open(output_path, "w", encoding="utf-8") as f:
        # Write with comments manually for better formatting
        sep = "# " + "=" * 77 + "\n"
        f.write("# Publisher Configuration\n\n")

        f.write(sep)
        f.write("# SINGLE ACCOUNT MODE (Legacy - use env vars for credentials)\n")
        f.write(sep)
        f.write("provider: late\n")
        f.write('api_key_env_var: "LATE_API_KEY"\n')
        f.write('vercel_token_env_var: "LATE_VERCEL_TOKEN"\n\n')

        f.write(sep)
        f.write("# MULTI-ACCOUNT MODE (uncomment to enable)\n")
        f.write(sep)
        f.write("# accounts:\n")
        f.write("#   main:\n")
        f.write("#     api_key: ${LATE_API_KEY}  # Use env var reference\n")
        f.write("#     vercel_token: ${LATE_VERCEL_TOKEN}\n")
        f.write('#     description: "Main production account"\n')
        f.write("#   secondary:\n")
        f.write("#     api_key: ${LATE_API_KEY_2}\n")
        f.write("#     vercel_token: ${LATE_VERCEL_TOKEN_2}\n")
        f.write('#     description: "Overflow account for high volume"\n')
        f.write("#     default_platforms:\n")
        f.write("#       - youtube\n")
        f.write("#       - tiktok\n")
        f.write("# default_account: main  # Which account to use by default\n\n")

        f.write(sep)
        f.write("# PUBLISHING BEHAVIOR\n")
        f.write(sep)
        f.write("immediate_publish: true\n")
        f.write("default_platforms:\n")
        f.write("  - youtube\n")
        f.write("  - tiktok\n")
        f.write("  - instagram\n\n")

        f.write("# Retry and timeout settings\n")
        f.write("max_retries: 3\n")
        f.write("timeout: 120.0\n\n")

        f.write("# Batch publishing delays (seconds)\n")
        f.write("stagger_delay_min: 30\n")
        f.write("stagger_delay_max: 60\n\n")

        f.write("# Privacy settings per platform\n")
        f.write("privacy_settings:\n")
        f.write("  youtube: public\n")
        f.write("  tiktok: public\n")
        f.write("  instagram: everyone\n")

    logger.info(f"Created default config file: {output_path}")
    print(f"Created default publisher config: {output_path}")
