# src/video/config/llm_settings.py
"""LLM configuration settings - extracted to avoid circular imports."""

from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    provider: str
    api_key_env_var: str
    models: list[str] = Field(..., min_length=1)
    prompt_template_path: str
    target_audience: str = Field("General audience")
    base_url: str | None = Field(None)
    auto_select_free_model: bool = Field(True)
    max_tokens: int = Field(4096)  # Sensible default, configurable via YAML
    temperature: float = Field(0.7)  # Sensible default, configurable via YAML
    timeout_seconds: int = Field(60)  # Sensible default, configurable via YAML
