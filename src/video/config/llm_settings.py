# src/video/config/llm_settings.py
"""LLM configuration settings - extracted to avoid circular imports."""

from pydantic import BaseModel, Field

from src.video.config.constants import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)


class LLMSettings(BaseModel):
    provider: str
    api_key_env_var: str
    models: list[str] = Field(..., min_length=1)
    prompt_template_path: str
    target_audience: str = Field("General audience")
    base_url: str | None = Field(None)
    auto_select_free_model: bool = Field(True)
    max_tokens: int = Field(LLM_MAX_TOKENS)
    temperature: float = Field(LLM_TEMPERATURE)
    timeout_seconds: int = Field(LLM_TIMEOUT_SECONDS)
