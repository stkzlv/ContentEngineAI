"""Typed configuration for the URL-shortening providers in this package.

Lives here rather than in `src/video/config/` because the scraper reads it at
construction. Importing any submodule of that package runs its `__init__`,
which eagerly loads the whole video configuration from five cwd-relative YAML
files -- so a scraper built from another directory, or on a machine with an
unrelated video-config error, would fail to construct at all.

The model mirrors `config/url_shortener.yaml` field for field. It used to be a
flattened paraphrase (`api_timeout_sec` for `api.timeout_sec`) that nothing
populated and nothing read, while the scraper walked the YAML dict itself with
its own defaults -- so the two default sets drifted, and a typo in the file was
swallowed rather than reported.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

# The file sits beside the other config, at the repo root. Anchored on this
# module rather than on the working directory: the scraper is invoked from
# anywhere, and a cwd-relative miss would silently load the `bare` no-op
# instead of the operator's provider.
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "url_shortener.yaml"
)


class URLShortenerApiSettings(BaseModel):
    """Network behaviour shared by every shortening provider."""

    model_config = {"extra": "forbid"}

    timeout_sec: int = 30
    max_retries: int = 3
    retry_delay_sec: float = 2.0
    retry_backoff_multiplier: float = 2.0


class URLShortenerProviderSettings(BaseModel):
    """One provider's block. `bare` needs none of it, which is why all optional."""

    model_config = {"extra": "forbid"}

    # The old consumer defaulted this name rather than the value, so a
    # `picsee` block omitting it still found the key. Leaving it None made
    # such a config skip shortening with only a debug-gated warning.
    api_key_env_var: str = "PICSEE_API_KEY"
    api_base_url: str | None = None
    custom_domain: str | None = None
    max_bulk_size: int = 100
    bulk_timeout_multiplier: float = 2.0


class URLShortenerIntegrationSettings(BaseModel):
    """Where shortening happens, as opposed to how."""

    model_config = {"extra": "forbid"}

    # Whether the scraper shortens as it goes, as opposed to a later pass.
    # False for the same reason `enabled` is: the old consumer read an absent
    # key as off.
    shorten_on_scrape: bool = False
    # Keep the long URL when shortening fails, rather than losing the link.
    fallback_to_original: bool = True


class URLShortenerSettings(BaseModel):
    """Configuration for URL shortening services.

    Mirrors `config/url_shortener.yaml` field for field. The previous model was
    a flattened paraphrase (`api_timeout_sec` for `api.timeout_sec`) that
    nothing populated and nothing read, while the scraper walked the YAML dict
    itself with its own defaults -- so the two default sets drifted, and a
    typo in the file was swallowed rather than reported.

    Providers are declared rather than collected from unknown keys: an
    undeclared key has to be a mistake, and treating it as a provider block
    would mean a misspelled section silently configured nothing.
    """

    model_config = {"extra": "forbid"}

    # False, matching what the old consumer defaulted to when the key was
    # absent. A partial override file that omits it must not start making
    # third-party API calls on every scrape.
    enabled: bool = False
    # `bare` is a no-op that returns the canonical URL unchanged. `picsee` is
    # opt-in and needs an API key.
    provider: str = "bare"

    api: URLShortenerApiSettings = Field(default_factory=URLShortenerApiSettings)
    integration: URLShortenerIntegrationSettings = Field(
        default_factory=URLShortenerIntegrationSettings
    )

    bare: URLShortenerProviderSettings = Field(
        default_factory=URLShortenerProviderSettings
    )
    picsee: URLShortenerProviderSettings = Field(
        default_factory=URLShortenerProviderSettings
    )

    @model_validator(mode="after")
    def _provider_is_declared(self) -> URLShortenerSettings:
        """A provider name with no block is a misconfiguration, not a default.

        Left unchecked it resolves to an empty block, so a typo'd `provider`
        would silently shorten nothing while reporting shortening as enabled.
        """
        if self.provider not in self.provider_names():
            raise ValueError(
                f"Unknown url_shortener.provider {self.provider!r}. "
                f"Declared providers: {', '.join(self.provider_names())}"
            )
        return self

    @classmethod
    def provider_names(cls) -> list[str]:
        return [
            name
            for name, field in cls.model_fields.items()
            if field.annotation is URLShortenerProviderSettings
        ]

    def active_provider(self) -> URLShortenerProviderSettings:
        """The block for `provider`, validated to exist at load."""
        block = getattr(self, self.provider)
        assert isinstance(block, URLShortenerProviderSettings)
        return block


def load_url_shortener_settings(
    config_path: Path | str | None = None,
) -> URLShortenerSettings:
    """Read `config/url_shortener.yaml` into the typed model.

    A missing file is not an error: shortening then runs on the model defaults,
    which is the `bare` no-op. A malformed one is, because a swallowed typo is
    what this replaces.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return URLShortenerSettings()

    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    return URLShortenerSettings(**(loaded.get("url_shortener") or {}))
