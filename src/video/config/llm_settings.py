# src/video/config/llm_settings.py
"""LLM configuration settings - extracted to avoid circular imports."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.video.config.constants import MIN_PHRASE_WORDS


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
    # Templates used when the record came from a topic rather than a scraped
    # product. Selected instead of the pillar pool, because a pillar narrows
    # which product angle to take while this decides whether the script is
    # about a product at all. Empty list falls back to the normal pool, which
    # produces a product pitch about a subject.
    topic_templates: list[str] = Field(default_factory=list)
    # Narrator direction for topic scripts. The default profile is written for
    # someone describing a thing they bought, down to the CTA options, so a
    # topic script inherits an affiliate call to action it has no basis for.
    # Empty string falls back to narrator_profile.
    narrator_profile_topic: str = ""
    # The closing calls to action, one of which every script must end on.
    # Structured rather than prose inside the narrator profile, because prose
    # forty lines from the task did not bind: five of five scheduled renders
    # ended on the template's closing beat and never reached a CTA. These are
    # rendered into `{CTA_RULE}` inside each template's Rules, adjacent to the
    # closing-beat rule, and the validator refuses a script that does not end
    # on one of them. The topic list exists because the product list implies
    # something to buy.
    cta_options: list[str] = Field(default_factory=list)
    cta_options_topic: list[str] = Field(default_factory=list)

    @field_validator("cta_options", "cta_options_topic")
    @classmethod
    def _each_option_carries_words(cls, options: list[str]) -> list[str]:
        """Refuse an option the ending check could never match sensibly.

        An empty or punctuation-only entry would match every script tail or
        none of them; either way the tenth paid render is the wrong place to
        find out. Fail at load.
        """
        for option in options:
            if not re.sub(r"[^a-z0-9]", "", option.lower()):
                raise ValueError(f"CTA option carries no words: {option!r}")
        return options

    def cta_options_for(self, is_topic: bool) -> list[str]:
        """The CTA lines a render may close on, by the same rule as the voice."""
        if is_topic and self.cta_options_topic:
            return self.cta_options_topic
        return self.cta_options

    def narrator_for(self, is_topic: bool) -> str:
        """The narrator profile a render should use.

        Every consumer of the profile has to make this choice, not just the
        script generator: the hook overlay and the per-platform caption prompts
        take it too, and the hook overlay is on by default. Resolving it at each
        call site meant a topic render's burned-in headline kept the purchase
        voice while only the spoken script changed.

        Falls back to the product profile when no topic one is configured, which
        is the pre-existing behaviour for anyone who has not set one.
        """
        if is_topic and self.narrator_profile_topic:
            return self.narrator_profile_topic
        return self.narrator_profile

    # Pillar -> target-audience override. When a pillar is set at runtime and
    # the map has an entry for it, the {AUDIENCE} placeholder uses this value
    # instead of LLMSettings.target_audience. Lets each pillar speak to the
    # buyer it's actually written for. Empty dict falls back to the global
    # target_audience.
    pillar_audiences: dict[str, str] = Field(default_factory=dict)
    # Topic counterparts to the two maps above. The product versions are
    # written about a thing being shown -- "the product fixes a specific
    # annoyance", "practical buyers" -- so pairing one with a topic template
    # gives the model a prompt that argues with itself: the template says
    # never invent a product, the preamble assumes one exists. Same keys, so
    # --pillar takes the same values on both families and a later taxonomy
    # change moves one key list rather than two vocabularies. Empty falls back
    # to the product map, which is the pre-existing behaviour.
    pillar_preambles_topic: dict[str, str] = Field(default_factory=dict)
    pillar_audiences_topic: dict[str, str] = Field(default_factory=dict)

    def preambles_for(self, is_topic: bool) -> dict[str, str]:
        """The pillar preamble map a render should use."""
        if is_topic and self.pillar_preambles_topic:
            return self.pillar_preambles_topic
        return self.pillar_preambles

    def audiences_for(self, is_topic: bool) -> dict[str, str]:
        """The pillar audience map a render should use."""
        if is_topic and self.pillar_audiences_topic:
            return self.pillar_audiences_topic
        return self.pillar_audiences


class ScriptValidationConfig(BaseModel):
    """Thresholds for script completeness validation."""

    min_chars: int = Field(200)
    min_words: int = Field(50)


class VisualSearchTermsConfig(BaseModel):
    """Deriving stock search phrases from the script that will be narrated.

    Only consulted when the profile draws every visual from stock, because
    that is the case where the search terms are the whole visual layer. A
    profile showing product photography ignores this.
    """

    enabled: bool = True
    # One Pexels search per phrase, so this is how many different shots the
    # render can draw on. The provider joins a multi-term list into a single
    # query string, which the library answers with results skewed toward
    # whichever phrase dominates, leaving the others unrepresented, so phrases
    # are searched separately rather than concatenated.
    max_phrases: int = Field(3, ge=1, le=6)
    # Words per phrase. A long phrase is matched only in part, so the extra
    # words narrow nothing and the result drifts from what was asked for.
    #
    # Floored at the minimum rather than at a number of its own: a maximum
    # below it renders a prompt asking for "3 to 2 words" and drops every
    # phrase that obeys it, falling the render back to a title-only search
    # with nothing logged. Equal to it is coherent -- exactly-three-word
    # phrases -- so the bound is `ge`, not `gt`.
    max_words_per_phrase: int = Field(5, ge=MIN_PHRASE_WORDS, le=8)


class StockRelevanceConfig(BaseModel):
    """Score every stock candidate the library returns against the script.

    The judgement is of the thumbnail, by the multimodal model, because every
    text signal was measured and refuted on #307 and the whole pool is scored
    because provider rank was measured to be noise on #341. `min_score` is a
    floor a candidate is used below only when there are not enough above it:
    a stock shortfall skips the render, which is the worse loss.
    """

    enabled: bool = False
    model: str = Field("gemini-2.5-flash-lite")
    min_score: int = Field(2, ge=0, le=3)
    max_candidates: int = Field(80, ge=1, le=80)
    concurrency: int = Field(6, ge=1, le=16)
    timeout_seconds: int = Field(20, ge=1)


class ScriptFactCheckConfig(BaseModel):
    """Check a tutorial script's falsifiable claims against grounded search.

    A shipped render sent viewers to a Windows menu node that does not exist.
    Measured on 17 topic scripts (#380): 83 falsifiable claims, 16 flagged, 14
    of the 16 correct on hand review, four distinct invented methods. The
    prompt fix in 0.103.1 removed the demand that caused most of them; this
    catches what a prompt rule cannot, a value that is simply wrong.

    `model` is a flash tier rather than the lite one the rest of the pipeline
    uses, and `LLMSettings.thinking_budget` is deliberately not applied to it.
    The repo turns thinking off everywhere on the grounds that nothing it asks
    for is a reasoning problem; this call is the one exception.

    `timeout_seconds` is generous against a measured 10.6s because a grounded
    call makes a search round trip inside the request, so it is not comparable
    to the thumbnail judge's budget.

    There is no round count to configure. The module makes exactly one
    grounded call per script because it contains no loop, so the revised
    sentence is never re-checked -- a real gap, stated rather than hidden. A
    `max_rounds` field existed here briefly and was removed: it could not
    raise the query count above what the code does, so it promised a second
    round that never happened while duplicating `enabled` at zero.
    """

    enabled: bool = False
    model: str = Field("gemini-2.5-flash")
    topics_only: bool = True
    max_flags_to_revise: int = Field(3, ge=1, le=10)
    max_length_drift: float = Field(0.25, ge=0.0, le=1.0)
    timeout_seconds: int = Field(45, ge=1)


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
    # Gemini only. `0` disables the model's internal reasoning pass; `None`
    # leaves the model's own default in place.
    #
    # The flash tier spends around a thousand thinking tokens on a task whose
    # visible output is sixteen, which is where the fortyfold gap between the
    # lite and flash tiers comes from -- not the headline rate. Every
    # generation this project asks for is short and structured: a script of
    # about a hundred and twenty words, a caption, a handful of search
    # phrases. None of them are reasoning problems.
    #
    # Applied to the script path as well, deliberately. It is the longest
    # output here and still not a reasoning task, and leaving it out would
    # mean a future switch to a flash model quietly costing forty times more
    # on the one call that runs on every render. That protection holds on the
    # 2.5 flash tier only; the 3.x flash models ignore the budget and their
    # lowest thinking level still bills.
    thinking_budget: int | None = Field(None)
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
    script_fact_check: ScriptFactCheckConfig = Field(
        default_factory=ScriptFactCheckConfig  # type: ignore[arg-type]
    )
    stock_relevance: StockRelevanceConfig = Field(
        default_factory=StockRelevanceConfig  # type: ignore[arg-type]
    )
    visual_search_terms: VisualSearchTermsConfig = Field(
        default_factory=VisualSearchTermsConfig  # type: ignore[arg-type]
    )
    fallback_provider: LLMSettings | None = Field(None)


LLMSettings.model_rebuild()
