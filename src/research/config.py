"""Settings for the research stages, from `config/research.yaml`."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PATH = Path("config/research.yaml")


class ProductResearch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor: str
    drop_below: float = Field(default=0.25, ge=0.0, le=1.0)
    seeds: list[str] = Field(default_factory=list)


class TopicResearch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor: str
    suggest_stems: list[str] = Field(default_factory=list)


class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    countries: list[str] = Field(default_factory=lambda: ["US"], min_length=1)
    request_pause_sec: float = Field(default=10.0, ge=0.0)
    max_retries: int = Field(default=2, ge=0)
    products: ProductResearch
    topics: TopicResearch


def load_research_config(path: Path = DEFAULT_PATH) -> ResearchConfig:
    return ResearchConfig.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )
