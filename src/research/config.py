"""Settings for the research stages, from `config/research.yaml`."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_PATH = Path("config/research.yaml")


class WikipediaArticle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str
    article: str


class ProductResearch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor: str
    drop_below: float = Field(default=0.25, ge=0.0, le=1.0)
    # Rising searches next to this many of the strongest keywords become
    # add candidates, once measured against the anchor.
    related_from: int = Field(default=5, ge=0)
    # The English Wikipedia article whose monthly views are a keyword's
    # second, absolute demand signal. Redirects are followed. A keyword with
    # no specific article is left out: a broad one measures something else.
    wikipedia: list[WikipediaArticle] = Field(default_factory=list)


class QuestionSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    site: str  # a Stack Exchange site's API name, such as "apple"
    tagged: str = ""  # one tag; empty means every question on the site


class StackExchangeResearch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[QuestionSource] = Field(default_factory=list)
    # Questions asked in this many days, ranked by views per day.
    days: int = Field(default=90, ge=1)
    # Pages of 100 per source; without a key the API allows 300 a day.
    pages: int = Field(default=3, ge=1, le=10)
    top: int = Field(default=10, ge=1)


class TopicResearch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor: str
    suggest_stems: list[str] = Field(default_factory=list)
    stack_exchange: StackExchangeResearch = Field(default_factory=StackExchangeResearch)


VARIANTS = ("shipped", "step_lists", "task_answer_first")


class SampleResearch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topics: int = Field(default=16, ge=0)
    products: int = Field(default=8, ge=0)
    # Any profile: the script step only needs its name for the run paths.
    profile: str = "slideshow_stock"
    variants: list[str] = Field(default_factory=lambda: list(VARIANTS), min_length=1)
    # The narrator profiles ask for 30-40 seconds, roughly 75-100 words.
    band: tuple[int, int] = (75, 100)

    @field_validator("variants")
    @classmethod
    def _known(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(VARIANTS))
        if unknown:
            raise ValueError(f"unknown variant(s): {unknown}; known: {list(VARIANTS)}")
        if "shipped" not in value:
            raise ValueError("variants must include shipped, the baseline")
        return value


class VerifyResearch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A model with Google Search grounding, as the step lists use.
    model: str = "gemini-3.7-flash"
    timeout_seconds: float = Field(default=90, gt=0)


class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    countries: list[str] = Field(default_factory=lambda: ["US"], min_length=1)
    request_pause_sec: float = Field(default=10.0, ge=0.0)
    max_retries: int = Field(default=2, ge=0)
    products: ProductResearch
    topics: TopicResearch
    sample: SampleResearch = Field(default_factory=SampleResearch)
    verify: VerifyResearch = Field(default_factory=VerifyResearch)


def load_research_config(path: Path = DEFAULT_PATH) -> ResearchConfig:
    return ResearchConfig.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )
