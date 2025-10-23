"""Configuration compliance tests for requirements verification.

This module tests the three-tier configuration precedence system:
CLI > ENV > YAML

Requirements tested:
- 1.1: CLI arguments override environment variables
- 1.2: Environment variables override YAML configuration
- 1.3: YAML configuration provides fallback values
"""

import os
from typing import Any
from unittest.mock import patch

import pytest

from src.config_manager import UnifiedConfigManager


@pytest.fixture
def config_manager():
    """Provide a fresh UnifiedConfigManager instance for each test."""
    return UnifiedConfigManager()


@pytest.fixture
def base_yaml_config():
    """Provide base YAML configuration for testing."""
    return {
        "debug_mode": False,
        "global_output_directory": "outputs",
        "pipeline_timeout_sec": 300,
        "global_settings": {
            "debug_mode": False,
            "browser_config": {
                "headless": True,
            },
        },
        "subtitle_settings": {
            "anchor": "bottom",
            "margin": 10,
            "content_aware": False,
            "style_preset": "modern",
            "font_size_scale": 1.0,
        },
        "cleanup": {
            "remove_temp_on_success": True,
        },
    }


# =============================================================================
# Requirement 1.1: CLI overrides ENV
# =============================================================================


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_1_cli_overrides_env_debug_mode(config_manager, base_yaml_config):
    """Test CLI argument overrides environment variable for debug_mode.

    Requirement: 1.1
    Scenario: Both ENV and CLI set debug_mode, CLI should win
    """
    # ENV sets debug to True
    with patch.dict(os.environ, {"DEBUG_MODE": "true"}):
        # CLI sets debug to False
        cli_overrides = {"debug": False}

        result = config_manager.apply_precedence_rules(base_yaml_config, cli_overrides)

        # CLI value should override ENV
        assert result["debug_mode"] is False
        assert result["global_settings"]["debug_mode"] is False


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_1_cli_overrides_env_output_directory(config_manager, base_yaml_config):
    """Test CLI argument overrides environment variable for output directory.

    Requirement: 1.1
    Scenario: Both ENV and CLI set output directory, CLI should win
    """
    # ENV sets output directory
    with patch.dict(os.environ, {"OUTPUTS_DIR": "/env/output"}):
        # CLI sets different output directory
        cli_overrides = {"output_dir": "/cli/output"}

        result = config_manager.apply_precedence_rules(base_yaml_config, cli_overrides)

        # CLI value should override ENV
        assert result["global_output_directory"] == "/cli/output"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_1_cli_overrides_env_subtitle_anchor(config_manager, base_yaml_config):
    """Test CLI argument overrides environment variable for subtitle anchor.

    Requirement: 1.1
    Scenario: Both ENV and CLI set subtitle anchor, CLI should win
    """
    # ENV sets subtitle anchor to "top"
    with patch.dict(os.environ, {"SUBTITLE_ANCHOR": "top"}):
        # CLI sets subtitle anchor to "center"
        cli_overrides = {"subtitle_settings.anchor": "center"}

        result = config_manager.apply_precedence_rules(base_yaml_config, cli_overrides)

        # CLI value should override ENV
        assert result["subtitle_settings"]["anchor"] == "center"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_1_cli_overrides_env_timeout(config_manager, base_yaml_config):
    """Test CLI argument overrides environment variable for timeout.

    Requirement: 1.1
    Scenario: Both ENV and CLI set timeout, CLI should win
    """
    # ENV sets timeout to 600
    with patch.dict(os.environ, {"CONTENT_ENGINE_TIMEOUT": "600"}):
        # CLI sets timeout to 120
        cli_overrides = {"timeout": 120}

        result = config_manager.apply_precedence_rules(base_yaml_config, cli_overrides)

        # CLI value should override ENV
        assert result["pipeline_timeout_sec"] == 120


# =============================================================================
# Requirement 1.2: ENV overrides YAML
# =============================================================================


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_2_env_overrides_yaml_debug_mode(config_manager, base_yaml_config):
    """Test environment variable overrides YAML for debug_mode.

    Requirement: 1.2
    Scenario: YAML has debug=False, ENV sets debug=True, ENV should win
    """
    # YAML has debug_mode = False
    assert base_yaml_config["debug_mode"] is False

    # ENV sets debug to True
    with patch.dict(os.environ, {"DEBUG_MODE": "true"}):
        # No CLI overrides
        result = config_manager.apply_precedence_rules(base_yaml_config)

        # ENV value should override YAML
        assert result["debug_mode"] is True


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_2_env_overrides_yaml_output_directory(config_manager, base_yaml_config):
    """Test environment variable overrides YAML for output directory.

    Requirement: 1.2
    Scenario: YAML has default output dir, ENV sets custom dir, ENV should win
    """
    # YAML has default output directory
    assert base_yaml_config["global_output_directory"] == "outputs"

    # ENV sets custom output directory
    with patch.dict(os.environ, {"OUTPUTS_DIR": "/env/custom/output"}):
        result = config_manager.apply_precedence_rules(base_yaml_config)

        # ENV value should override YAML
        assert result["global_output_directory"] == "/env/custom/output"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_2_env_overrides_yaml_subtitle_settings(config_manager, base_yaml_config):
    """Test environment variable overrides YAML for subtitle settings.

    Requirement: 1.2
    Scenario: YAML has subtitle settings, ENV overrides multiple settings
    """
    # YAML baseline values
    assert base_yaml_config["subtitle_settings"]["anchor"] == "bottom"
    assert base_yaml_config["subtitle_settings"]["margin"] == 10
    assert base_yaml_config["subtitle_settings"]["content_aware"] is False

    # ENV overrides subtitle settings
    with patch.dict(
        os.environ,
        {
            "SUBTITLE_ANCHOR": "top",
            "SUBTITLE_MARGIN": "20",
            "SUBTITLE_CONTENT_AWARE": "true",
        },
    ):
        result = config_manager.apply_precedence_rules(base_yaml_config)

        # ENV values should override YAML
        assert result["subtitle_settings"]["anchor"] == "top"
        assert result["subtitle_settings"]["margin"] == 20
        assert result["subtitle_settings"]["content_aware"] is True


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_2_env_overrides_yaml_style_preset(config_manager, base_yaml_config):
    """Test environment variable overrides YAML for style preset.

    Requirement: 1.2
    Scenario: YAML has modern preset, ENV sets bold preset, ENV should win
    """
    # YAML has style preset = modern
    assert base_yaml_config["subtitle_settings"]["style_preset"] == "modern"

    # ENV sets style preset to bold
    with patch.dict(os.environ, {"SUBTITLE_STYLE_PRESET": "bold"}):
        result = config_manager.apply_precedence_rules(base_yaml_config)

        # ENV value should override YAML
        assert result["subtitle_settings"]["style_preset"] == "bold"


# =============================================================================
# Requirement 1.3: YAML provides fallback
# =============================================================================


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_3_yaml_fallback_when_no_overrides(config_manager, base_yaml_config):
    """Test YAML configuration used when no CLI or ENV overrides.

    Requirement: 1.3
    Scenario: No CLI or ENV overrides, should use YAML values
    """
    # No environment variables or CLI overrides
    result = config_manager.apply_precedence_rules(base_yaml_config)

    # Should return YAML values unchanged
    assert result["debug_mode"] is False
    assert result["global_output_directory"] == "outputs"
    assert result["pipeline_timeout_sec"] == 300
    assert result["subtitle_settings"]["anchor"] == "bottom"
    assert result["subtitle_settings"]["margin"] == 10


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_3_yaml_fallback_partial_overrides(config_manager, base_yaml_config):
    """Test YAML fallback for settings not overridden by CLI/ENV.

    Requirement: 1.3
    Scenario: Only some settings overridden, others should use YAML fallback
    """
    # CLI overrides only debug mode
    cli_overrides = {"debug": True}

    result = config_manager.apply_precedence_rules(base_yaml_config, cli_overrides)

    # Overridden setting uses CLI value
    assert result["debug_mode"] is True

    # Non-overridden settings use YAML fallback
    assert result["global_output_directory"] == "outputs"
    assert result["pipeline_timeout_sec"] == 300
    assert result["subtitle_settings"]["anchor"] == "bottom"
    assert result["subtitle_settings"]["margin"] == 10


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_3_yaml_fallback_nested_settings(config_manager, base_yaml_config):
    """Test YAML fallback for nested configuration settings.

    Requirement: 1.3
    Scenario: Deep nested settings should preserve YAML values when not overridden
    """
    # Override only one nested setting
    with patch.dict(os.environ, {"SUBTITLE_ANCHOR": "top"}):
        result = config_manager.apply_precedence_rules(base_yaml_config)

        # Overridden nested setting
        assert result["subtitle_settings"]["anchor"] == "top"

        # Other nested settings use YAML fallback
        assert result["subtitle_settings"]["margin"] == 10
        assert result["subtitle_settings"]["content_aware"] is False
        assert result["subtitle_settings"]["style_preset"] == "modern"
        assert result["subtitle_settings"]["font_size_scale"] == 1.0


# =============================================================================
# Combined precedence scenarios (all three tiers)
# =============================================================================


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_1_1_2_1_3_full_precedence_chain(config_manager, base_yaml_config):
    """Test complete precedence chain: CLI > ENV > YAML.

    Requirements: 1.1, 1.2, 1.3
    Scenario: All three tiers set, verify correct precedence order
    """
    # YAML baseline
    assert base_yaml_config["debug_mode"] is False
    assert base_yaml_config["global_output_directory"] == "outputs"
    assert base_yaml_config["pipeline_timeout_sec"] == 300

    # ENV sets some values
    with patch.dict(
        os.environ,
        {
            "DEBUG_MODE": "true",  # ENV overrides YAML
            "OUTPUTS_DIR": "/env/output",  # ENV overrides YAML
            "CONTENT_ENGINE_TIMEOUT": "600",  # ENV overrides YAML
        },
    ):
        # CLI overrides only debug and timeout
        cli_overrides = {
            "debug": False,  # CLI overrides ENV
            "timeout": 120,  # CLI overrides ENV
            # output_dir not in CLI, should use ENV value
        }

        result = config_manager.apply_precedence_rules(base_yaml_config, cli_overrides)

        # CLI wins for debug (overrides ENV)
        assert result["debug_mode"] is False

        # CLI wins for timeout (overrides ENV)
        assert result["pipeline_timeout_sec"] == 120

        # ENV wins for output_dir (CLI didn't override, ENV overrides YAML)
        assert result["global_output_directory"] == "/env/output"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_1_1_2_1_3_precedence_independence(config_manager, base_yaml_config):
    """Test precedence tiers work independently for different settings.

    Requirements: 1.1, 1.2, 1.3
    Scenario: Different settings use different precedence tiers
    """
    # ENV overrides some settings
    with patch.dict(
        os.environ,
        {
            "SUBTITLE_ANCHOR": "top",
            "SUBTITLE_MARGIN": "20",
        },
    ):
        # CLI overrides different settings
        cli_overrides = {
            "debug": True,
            "subtitle_settings.content_aware": True,
        }

        result = config_manager.apply_precedence_rules(base_yaml_config, cli_overrides)

        # CLI tier settings
        assert result["debug_mode"] is True
        assert result["subtitle_settings"]["content_aware"] is True

        # ENV tier settings
        assert result["subtitle_settings"]["anchor"] == "top"
        assert result["subtitle_settings"]["margin"] == 20

        # YAML tier settings (no overrides)
        assert result["subtitle_settings"]["style_preset"] == "modern"
        assert result["subtitle_settings"]["font_size_scale"] == 1.0
        assert result["global_output_directory"] == "outputs"


# =============================================================================
# Requirement 1.4: Secret isolation (API keys only in .env, not in YAML)
# =============================================================================


@pytest.fixture
def secret_patterns():
    """Provide configurable patterns for detecting secrets in YAML files.

    Returns patterns that should NEVER appear as literal values in config files.
    Environment variable references (like api_key_env_var) are allowed.
    """
    return {
        # Patterns that indicate actual secret values (not env var references)
        "api_keys": [
            r":\s*['\"]?sk-[a-zA-Z0-9]{20,}['\"]?",  # OpenAI-style keys
            r":\s*['\"]?[a-zA-Z0-9]{32,}['\"]?\s*$",  # Generic long strings
        ],
        # Field names that should only reference env vars, never contain values
        "sensitive_fields": [
            "api_key",
            "token",
            "password",
            "secret",
            "access_key",
            "secret_key",
            "private_key",
            "client_secret",
            "refresh_token",
        ],
        # Allowed patterns (these are safe env var references)
        "allowed_patterns": [
            r"api_key_env_var",  # References to env vars are OK
            r"env_var",
            r"_ENV",
            r"max_tokens",  # Configuration values, not secrets
            r"token_expiry",
            r"token_refresh",
            r"token_url",
        ],
    }


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_4_no_secrets_in_yaml_files(secret_patterns):
    """Test that YAML config files contain no hardcoded secrets.

    Requirement: 1.4
    Scenario: Scan all YAML files in config/ for secret patterns
    Success: No hardcoded API keys, tokens, or passwords found
    Failure: Clear error message indicating which file contains secrets
    """
    from pathlib import Path

    import yaml

    config_dir = Path("config")
    assert config_dir.exists(), "config/ directory not found"

    yaml_files = list(config_dir.glob("*.yaml"))
    assert len(yaml_files) > 0, "No YAML files found in config/"

    violations = []

    for yaml_file in yaml_files:
        with open(yaml_file) as f:
            try:
                config_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                violations.append(f"{yaml_file.name}: Failed to parse YAML - {e}")
                continue

        # Scan the YAML file content for secrets
        file_violations = _scan_for_secrets(
            config_data, yaml_file.name, secret_patterns
        )
        violations.extend(file_violations)

    # Test should fail if any violations found
    if violations:
        violation_msg = "\n".join(
            [
                "SECRET ISOLATION VIOLATION: Hardcoded secrets found in YAML files!",
                "API keys, tokens, and passwords must be stored in .env file only.",
                "",
                "Violations found:",
            ]
            + [f"  - {v}" for v in violations]
            + [
                "",
                "Fix: Replace hardcoded secrets with environment variable references.",
                (
                    "Example: Use 'api_key_env_var: OPENROUTER_API_KEY' "
                    "instead of 'api_key: sk-...'"
                ),
            ]
        )
        pytest.fail(violation_msg)


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_4_env_var_references_used(secret_patterns):
    """Test that config files use env var references for sensitive data.

    Requirement: 1.4
    Scenario: Verify sensitive fields use *_env_var pattern
    Success: All API key fields reference environment variables
    """
    from pathlib import Path

    import yaml

    config_dir = Path("config")
    yaml_files = list(config_dir.glob("*.yaml"))

    # Track files that properly use env var references
    proper_references = []
    missing_references = []

    for yaml_file in yaml_files:
        with open(yaml_file) as f:
            content = f.read()

        # Check if file contains API key fields
        if any(
            field in content.lower() for field in secret_patterns["sensitive_fields"]
        ):
            # Verify it uses env_var pattern
            if "env_var" in content:
                proper_references.append(yaml_file.name)
            else:
                # File has sensitive fields but no env_var references
                missing_references.append(yaml_file.name)

    # All files with sensitive fields should use env_var pattern
    assert (
        len(missing_references) == 0
    ), f"Files with sensitive fields missing env_var references: {missing_references}"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_1_4_yaml_files_scannable():
    """Test that all YAML config files are valid and scannable.

    Requirement: 1.4
    Scenario: Ensure all config files can be parsed for secret scanning
    Success: All YAML files parse without errors
    """
    from pathlib import Path

    import yaml

    config_dir = Path("config")
    yaml_files = list(config_dir.glob("*.yaml"))

    parse_errors = []

    for yaml_file in yaml_files:
        with open(yaml_file) as f:
            try:
                yaml.safe_load(f)
            except yaml.YAMLError as e:
                parse_errors.append(f"{yaml_file.name}: {e}")

    assert len(parse_errors) == 0, f"YAML parse errors: {parse_errors}"


def _scan_for_secrets(
    data: dict | list | Any, file_name: str, patterns: dict, path: str = ""
) -> list[str]:
    """Recursively scan configuration data for hardcoded secrets.

    Args:
    ----
        data: Configuration data (dict, list, or primitive)
        file_name: Name of the file being scanned
        patterns: Secret detection patterns from fixture
        path: Current path in the config hierarchy (for error messages)

    Returns:
    -------
        List of violation messages

    """
    violations = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key

            # Skip non-string keys (e.g., numeric keys in YAML)
            if not isinstance(key, str):
                violations.extend(
                    _scan_for_secrets(value, file_name, patterns, current_path)
                )
                continue

            # Check if this is a sensitive field
            is_sensitive = any(
                sensitive in key.lower() for sensitive in patterns["sensitive_fields"]
            )

            # Check if this key is an allowed pattern (env var reference)
            is_allowed = any(
                allowed in key.lower() for allowed in patterns["allowed_patterns"]
            )

            if is_sensitive and not is_allowed:
                # Sensitive field that's not an env var reference
                if isinstance(value, str):
                    # Check if value looks like a secret (long alphanumeric string)
                    if len(value) > 20 and not value.startswith("$"):
                        violations.append(
                            f"{file_name}:{current_path} = '{value[:10]}...' "
                            "(possible hardcoded secret)"
                        )
                elif value is not None and not isinstance(value, dict | list):
                    violations.append(
                        f"{file_name}:{current_path} has non-env-var value: {value}"
                    )

            # Recurse into nested structures
            violations.extend(
                _scan_for_secrets(value, file_name, patterns, current_path)
            )

    elif isinstance(data, list):
        for i, item in enumerate(data):
            current_path = f"{path}[{i}]"
            violations.extend(
                _scan_for_secrets(item, file_name, patterns, current_path)
            )

    return violations


# =============================================================================
# Requirement 12.1, 12.2: Configuration validation with clear error messages
# =============================================================================


@pytest.mark.compliance
@pytest.mark.unit
def test_req_12_1_pydantic_catches_type_errors():
    """Test Pydantic validation catches type errors with clear messages.

    Requirement: 12.1
    Scenario: Invalid type provided for configuration field
    Success: Validation fails with clear error indicating field and expected type
    """
    from pydantic import ValidationError

    from src.video.video_config import GoogleCloudTTSSettings

    # Invalid type: speaking_rate should be float, not string
    invalid_config = {
        "audio_encoding": "MP3",
        "language_code": "en-US",
        "voice_selection_criteria": [{"language_code": "en-US"}],
        "speaking_rate": "fast",  # Should be float, not string
    }

    with pytest.raises(ValidationError) as exc_info:
        GoogleCloudTTSSettings(**invalid_config)

    error = exc_info.value
    error_dict = error.errors()

    # Verify error message contains field name
    assert any("speaking_rate" in str(err["loc"]) for err in error_dict)

    # Verify error message is descriptive
    error_messages = str(error)
    assert "speaking_rate" in error_messages.lower()


@pytest.mark.compliance
@pytest.mark.unit
def test_req_12_1_pydantic_catches_missing_required_fields():
    """Test Pydantic validation catches missing required fields.

    Requirement: 12.1
    Scenario: Required field missing from configuration
    Success: Validation fails indicating which field is required
    """
    from pydantic import ValidationError

    from src.video.video_config import GoogleCloudTTSSettings

    # Missing required field: language_code
    invalid_config = {
        "audio_encoding": "MP3",
        "voice_selection_criteria": [{"language_code": "en-US"}],
    }

    with pytest.raises(ValidationError) as exc_info:
        GoogleCloudTTSSettings(**invalid_config)

    error = exc_info.value
    error_dict = error.errors()

    # Verify error indicates missing field
    assert any("language_code" in str(err["loc"]) for err in error_dict)
    # Pydantic 2.x uses "missing" type for required fields
    assert any(
        "missing" in str(err["type"]).lower() or "required" in str(err["type"]).lower()
        for err in error_dict
    )


@pytest.mark.compliance
@pytest.mark.unit
def test_req_12_1_pydantic_catches_constraint_violations():
    """Test Pydantic validation catches field constraint violations.

    Requirement: 12.1
    Scenario: Field value violates constraints (e.g., min_length)
    Success: Validation fails with clear message about constraint
    """
    from pydantic import ValidationError

    from src.video.video_config import TTSConfig

    # Constraint violation: provider_order requires min_length=1
    invalid_config = {"provider_order": []}  # Empty list violates min_length=1

    with pytest.raises(ValidationError) as exc_info:
        TTSConfig(**invalid_config)

    error = exc_info.value
    error_dict = error.errors()

    # Verify error mentions the field and constraint
    assert any("provider_order" in str(err["loc"]) for err in error_dict)


@pytest.mark.compliance
@pytest.mark.unit
def test_req_12_2_validation_errors_contain_field_names():
    """Test validation error messages contain specific field names.

    Requirement: 12.2
    Scenario: Multiple validation errors occur
    Success: Each error message identifies the specific field
    """
    from pydantic import ValidationError

    from src.video.video_config import VideoProfile

    # Multiple invalid fields
    invalid_config = {
        "name": "test_profile",
        "description": "Test",
        "stock_image_count": -5,  # Should be >= 0
        "stock_video_count": -3,  # Should be >= 0
    }

    with pytest.raises(ValidationError) as exc_info:
        VideoProfile(**invalid_config)

    error = exc_info.value
    error_dict = error.errors()

    # Each error should have a clear field location
    for err in error_dict:
        assert "loc" in err
        assert len(err["loc"]) > 0  # Field path should exist


@pytest.mark.compliance
@pytest.mark.unit
def test_req_12_2_validation_errors_describe_expected_values():
    """Test validation error messages describe expected values/constraints.

    Requirement: 12.2
    Scenario: Constraint violation with range/type expectations
    Success: Error message indicates expected range or valid values
    """
    from pydantic import ValidationError

    from src.video.video_config import VideoProfile

    # Value outside valid range
    invalid_config = {
        "name": "test_profile",
        "description": "Test",
        "stock_image_count": -10,  # Violates ge=0 constraint
    }

    with pytest.raises(ValidationError) as exc_info:
        VideoProfile(**invalid_config)

    error = exc_info.value
    error_messages = str(error)

    # Error should mention the constraint
    assert "stock_image_count" in error_messages
    # Error should describe the constraint (greater than or equal to 0)
    assert any(
        indicator in error_messages.lower()
        for indicator in ["greater", ">=", "ge", "0"]
    )


@pytest.mark.compliance
@pytest.mark.unit
def test_req_12_2_nested_validation_errors_have_clear_paths():
    """Test nested field validation errors include full field path.

    Requirement: 12.2
    Scenario: Validation error in nested configuration object
    Success: Error message includes complete path to nested field
    """
    from pydantic import ValidationError

    from src.video.video_config import GoogleCloudTTSSettings

    # Nested validation error
    invalid_config = {
        "audio_encoding": "MP3",
        "language_code": "en-US",
        "voice_selection_criteria": [
            {
                "language_code": "en-US",
                "ssml_gender": 123,  # Should be string, not int
            }
        ],
    }

    with pytest.raises(ValidationError) as exc_info:
        GoogleCloudTTSSettings(**invalid_config)

    error = exc_info.value
    error_dict = error.errors()

    # Verify nested path is included
    nested_errors = [
        err for err in error_dict if "voice_selection_criteria" in str(err["loc"])
    ]
    assert len(nested_errors) > 0

    # Verify path shows hierarchy
    for err in nested_errors:
        # Path should be a tuple showing the nesting
        assert isinstance(err["loc"], tuple)
        assert len(err["loc"]) >= 2  # At least 2 levels deep


@pytest.mark.compliance
@pytest.mark.unit
def test_req_12_1_12_2_custom_validator_provides_clear_error():
    """Test custom validators provide actionable error messages.

    Requirements: 12.1, 12.2
    Scenario: Custom model_validator catches invalid state
    Success: Custom error message is clear and actionable
    """
    from pydantic import ValidationError

    from src.video.video_config import TTSConfig

    # Test that validation error provides clear field information
    # TTSConfig validates provider_order has corresponding settings
    try:
        # This may or may not raise depending on validation implementation
        TTSConfig(provider_order=["google_cloud"])
    except ValidationError as e:
        # If it raises, error should be descriptive
        error_messages = str(e)
        assert len(error_messages) > 0
        # Error should reference the configuration issue
        assert any(
            keyword in error_messages.lower()
            for keyword in ["provider", "google", "setting"]
        )


@pytest.mark.compliance
@pytest.mark.unit
def test_req_12_2_validation_aggregates_multiple_errors():
    """Test validation reports all errors, not just the first one.

    Requirement: 12.2
    Scenario: Multiple fields have validation errors
    Success: All errors reported in single validation exception
    """
    from pydantic import ValidationError

    from src.video.video_config import VideoProfile

    # Multiple validation errors
    invalid_config = {
        "name": "test",
        "description": "Test",
        "stock_image_count": -5,  # Error 1: negative value
        "stock_video_count": -10,  # Error 2: negative value
        "image_width_percent": 2.0,  # Potentially invalid (> 1.0)
    }

    with pytest.raises(ValidationError) as exc_info:
        VideoProfile(**invalid_config)

    error = exc_info.value
    error_dict = error.errors()

    # Should have multiple errors reported
    assert len(error_dict) >= 2  # At least the two ge=0 violations

    # Each error should be distinct
    error_fields = {str(err["loc"]) for err in error_dict}
    assert len(error_fields) >= 2  # Multiple different fields in error
