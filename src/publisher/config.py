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

from src.publisher.models import Platform, PublisherConfig

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
        FileNotFoundError: If config file doesn't exist and no env/CLI overrides provided
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


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to configuration.

    Environment variables (medium precedence):
    - LATE_API_KEY / PUBLISHER_API_KEY
    - LATE_VERCEL_TOKEN / PUBLISHER_VERCEL_TOKEN
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

    # Vercel Token
    vercel_token = (
        os.environ.get("LATE_VERCEL_TOKEN")
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
    - timeout: 30.0
    - backoff_multiplier: 2.0
    - stagger_delay_min: 30
    - stagger_delay_max: 60
    - default_platforms: [youtube, tiktok, instagram]
    - privacy_settings: {}

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
        "timeout": 30.0,
        "backoff_multiplier": 2.0,
        "stagger_delay_min": 30,
        "stagger_delay_max": 60,
        "default_platforms": [Platform.YOUTUBE, Platform.TIKTOK, Platform.INSTAGRAM],
        "privacy_settings": {},
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
    if not isinstance(api_key, str) or len(api_key) < 10:
        raise ValueError(
            f"Invalid API key format: must be string with at least 10 characters "
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

    default_config = {
        "# Publisher Configuration": None,
        "# API credentials (use environment variables for security)": None,
        "provider": "late",
        "api_key": "${LATE_API_KEY}",
        "vercel_token": "${LATE_VERCEL_TOKEN}",
        "# Publishing behavior": None,
        "immediate_publish": True,
        "default_platforms": ["youtube", "tiktok", "instagram"],
        "# Retry and timeout settings": None,
        "max_retries": 3,
        "timeout": 30.0,
        "backoff_multiplier": 2.0,
        "# Batch publishing delays (seconds)": None,
        "stagger_delay_min": 30,
        "stagger_delay_max": 60,
        "# Privacy settings per platform": None,
        "privacy_settings": {
            "youtube": "public",
            "tiktok": "public",
            "instagram": "everyone",
        },
    }

    # Filter out comment keys (starting with #)
    yaml_config = {k: v for k, v in default_config.items() if not k.startswith("#")}

    with open(output_path, "w", encoding="utf-8") as f:
        # Write with comments manually for better formatting
        f.write("# Publisher Configuration\n")
        f.write("# API credentials (use environment variables for security)\n")
        f.write("provider: late\n")
        f.write("api_key: ${LATE_API_KEY}\n")
        f.write("vercel_token: ${LATE_VERCEL_TOKEN}\n\n")
        f.write("# Publishing behavior\n")
        f.write("immediate_publish: true\n")
        f.write("default_platforms:\n")
        f.write("  - youtube\n")
        f.write("  - tiktok\n")
        f.write("  - instagram\n\n")
        f.write("# Retry and timeout settings\n")
        f.write("max_retries: 3\n")
        f.write("timeout: 30.0\n")
        f.write("backoff_multiplier: 2.0\n\n")
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
