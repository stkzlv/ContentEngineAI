# src/video/config/llm_settings.py
"""LLM configuration settings - extracted to avoid circular imports."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScriptTemplateConfig(BaseModel):
    """Config for multi-template script generation."""

    enabled: bool = False
    templates_dir: str = "src/ai/prompts/scripts"
    template_pool: list[str] = Field(default_factory=list)
    fixed_template: str | None = None
    # Pillar -> list of template names. A template may appear under more
    # than one pillar. When a pillar is selected at runtime, deterministic
    # MD5 selection picks from the matching list instead of the full pool.
    # Empty dict disables pillar filtering (legacy behavior).
    pillars: dict[str, list[str]] = Field(default_factory=dict)
    # Pillar -> preamble string. When a pillar is set at runtime and the map
    # has an entry for it, the preamble is prepended to the LLM prompt so the
    # model leans into that pillar's framing angle. Templates themselves stay
    # pillar-agnostic. Empty dict disables preamble injection.
    pillar_preambles: dict[str, str] = Field(default_factory=dict)
    # Channel-wide voice direction. When non-empty, prepended to every script
    # prompt above any pillar preamble. Carries the rules every template
    # would otherwise duplicate (banned words, word target, narrator persona,
    # anti-AI-tells). Empty string disables narrator profile injection.
    narrator_profile: str = ""
    # Pillar -> target-audience override. When a pillar is set at runtime and
    # the map has an entry for it, the {AUDIENCE} placeholder uses this value
    # instead of LLMSettings.target_audience. Lets each pillar speak to the
    # buyer it's actually written for. Empty dict falls back to the global
    # target_audience.
    pillar_audiences: dict[str, str] = Field(default_factory=dict)


class ScriptValidationConfig(BaseModel):
    """Thresholds for script completeness validation."""

    min_chars: int = Field(200)
    min_words: int = Field(50)


class LLMSettings(BaseModel):
    model_config = {"protected_namespaces": ()}

    provider: Literal["openrouter", "gemini"]
    api_key_env_var: str
    models: list[str] = Field(..., min_length=1)
    prompt_template_path: str
    target_audience: str = Field("General audience")
    base_url: str | None = Field(None)
    auto_select_free_model: bool = Field(True)
    random_model_selection: bool = Field(False)  # False = try models in order
    fallback_discover_any_free: bool = Field(True)  # Try any free model as fallback
    max_tokens: int = Field(600)  # Matches ai_services.yaml default
    temperature: float = Field(0.7)  # Sensible default, configurable via YAML
    timeout_seconds: int = Field(60)  # Sensible default, configurable via YAML
    # Retry settings (used by tenacity in generators)
    retry_attempts: int = Field(3)
    retry_min_wait_sec: int = Field(1)
    retry_max_wait_sec: int = Field(30)
    # OpenRouter model discovery filters
    model_blocklist: list[str] = Field(
        default_factory=lambda: [
            "liquid/lfm-2.5-1.2b-instruct:free",
            "liquid/lfm-2.5-1.2b-instruct",
        ]
    )
    min_context_length: int = Field(8000)  # Filter out tiny models
    # Script validation thresholds
    script_validation: ScriptValidationConfig = Field(
        default_factory=ScriptValidationConfig  # type: ignore[arg-type]
    )
    script_templates: ScriptTemplateConfig = Field(default_factory=ScriptTemplateConfig)
    fallback_provider: LLMSettings | None = Field(None)


LLMSettings.model_rebuild()
