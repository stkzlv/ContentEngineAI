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
    # The floor is 4 rather than 2 because the sanitizer refuses anything
    # under `MIN_PHRASE_WORDS` (3): a maximum below that renders a prompt
    # asking for "3 to 2 words" and drops every phrase that obeys it, which
    # falls the render back to a title-only search with nothing logged.
    max_words_per_phrase: int = Field(5, ge=4, le=8)


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
    visual_search_terms: VisualSearchTermsConfig = Field(
        default_factory=VisualSearchTermsConfig  # type: ignore[arg-type]
    )
    fallback_provider: LLMSettings | None = Field(None)


LLMSettings.model_rebuild()
