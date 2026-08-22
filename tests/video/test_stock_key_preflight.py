"""Tests for the stock-provider key pre-flight check.

Without the key, the fetcher warns, returns nothing, and the run dies three
steps later with "No visual inputs were found or gathered for this profile" —
a message that names neither the provider nor the variable to set. For a
profile that draws every visual from stock, that is a whole render lost to one
missing environment variable, reported as a media problem.
"""

from types import SimpleNamespace

import pytest

from src.video.config_validator import check_stock_media_key
from src.video.producer.utils import profile_needs_stock_media

STOCK = SimpleNamespace(
    use_stock_images=True,
    stock_image_count=8,
    use_stock_videos=False,
    stock_video_count=0,
)
SCRAPED = SimpleNamespace(
    use_stock_images=False,
    stock_image_count=0,
    use_stock_videos=False,
    stock_video_count=0,
)
STOCK_VIDEO = SimpleNamespace(
    use_stock_images=False,
    stock_image_count=0,
    use_stock_videos=True,
    stock_video_count=2,
)
# Enabled but asking for nothing: the step builds no fetcher, so no key needed.
STOCK_ZERO = SimpleNamespace(
    use_stock_images=True,
    stock_image_count=0,
    use_stock_videos=False,
    stock_video_count=0,
)


def _config(**profiles):
    return SimpleNamespace(
        stock_media_settings=SimpleNamespace(pexels_api_key_env_var="PEXELS_API_KEY"),
        video_profiles=dict(profiles),
    )


@pytest.mark.unit
class TestNeedsStockMedia:
    """Shared with `step_gather_visuals`, so the check and the step it
    protects cannot disagree about which profiles need a key.
    """

    def test_a_stock_image_profile_needs_it(self):
        assert profile_needs_stock_media(STOCK)

    def test_a_stock_video_profile_needs_it(self):
        assert profile_needs_stock_media(STOCK_VIDEO)

    def test_a_scraped_profile_does_not(self):
        assert not profile_needs_stock_media(SCRAPED)

    def test_enabled_but_asking_for_zero_does_not(self):
        """The step builds no fetcher in this case, so neither does the check."""
        assert not profile_needs_stock_media(STOCK_ZERO)


@pytest.mark.unit
class TestKeyCheck:
    def test_a_missing_key_names_the_variable_and_the_profiles(self, monkeypatch):
        monkeypatch.delenv("PEXELS_API_KEY", raising=False)
        error = check_stock_media_key(
            _config(slideshow_stock=STOCK), ["slideshow_stock"]
        )
        assert error is not None
        assert "PEXELS_API_KEY" in error
        assert "slideshow_stock" in error

    def test_a_present_key_passes(self, monkeypatch):
        monkeypatch.setenv("PEXELS_API_KEY", "k")
        assert check_stock_media_key(_config(s=STOCK), ["s"]) is None

    def test_a_key_in_the_secrets_dict_counts(self, monkeypatch):
        """The caller may have built its secrets before the environment is
        read, and a key there is just as usable.
        """
        monkeypatch.delenv("PEXELS_API_KEY", raising=False)
        error = check_stock_media_key(
            _config(s=STOCK), ["s"], secrets={"PEXELS_API_KEY": "k"}
        )
        assert error is None

    def test_a_scraped_profile_needs_no_key(self, monkeypatch):
        monkeypatch.delenv("PEXELS_API_KEY", raising=False)
        assert check_stock_media_key(_config(s=SCRAPED), ["s"]) is None

    def test_any_profile_in_a_pool_triggers_it(self, monkeypatch):
        """A random draw can pick the stock profile, so failing only when it
        does would make the error intermittent rather than deterministic.
        """
        monkeypatch.delenv("PEXELS_API_KEY", raising=False)
        error = check_stock_media_key(
            _config(a=SCRAPED, b=STOCK, c=SCRAPED), ["a", "b", "c"]
        )
        assert error is not None
        assert "b" in error

    def test_an_unknown_profile_name_is_ignored(self, monkeypatch):
        """Validation of the name itself belongs to the config validator; this
        check must not raise a second, less useful error about it.
        """
        monkeypatch.delenv("PEXELS_API_KEY", raising=False)
        assert check_stock_media_key(_config(s=SCRAPED), ["nonexistent"]) is None

    def test_no_profiles_is_not_an_error(self, monkeypatch):
        monkeypatch.delenv("PEXELS_API_KEY", raising=False)
        assert check_stock_media_key(_config(), []) is None

    def test_an_empty_string_key_does_not_count(self, monkeypatch):
        """`PEXELS_API_KEY=` in a .env reads as set but is unusable."""
        monkeypatch.setenv("PEXELS_API_KEY", "")
        assert check_stock_media_key(_config(s=STOCK), ["s"]) is not None

    def test_the_configured_variable_name_is_honoured(self, monkeypatch):
        """The env var is a config field, so the message must not hardcode it."""
        monkeypatch.delenv("OTHER_KEY", raising=False)
        config = _config(s=STOCK)
        config.stock_media_settings.pexels_api_key_env_var = "OTHER_KEY"
        error = check_stock_media_key(config, ["s"])
        assert error is not None
        assert "OTHER_KEY" in error


@pytest.mark.unit
class TestOnlyStockOnlyProfilesBlock:
    """A profile that also draws scraped media renders fine without the key.

    The fetcher warns and returns nothing; the scraped images carry the video.
    `docs/configuration.md` documents exactly such a profile, so refusing it
    would block a configuration that works — worse than the silent gap this
    check closes.
    """

    def _mixed(self):
        return SimpleNamespace(
            use_scraped_images=True,
            use_scraped_videos=False,
            use_stock_images=True,
            stock_image_count=5,
            use_stock_videos=False,
            stock_video_count=2,
        )

    def test_supplementary_stock_is_allowed(self, monkeypatch):
        monkeypatch.delenv("PEXELS_API_KEY", raising=False)
        assert check_stock_media_key(_config(mixed=self._mixed()), ["mixed"]) is None

    def test_stock_only_still_blocks(self, monkeypatch):
        monkeypatch.delenv("PEXELS_API_KEY", raising=False)
        stock_only = SimpleNamespace(
            use_scraped_images=False,
            use_scraped_videos=False,
            use_stock_images=True,
            stock_image_count=5,
            use_stock_videos=False,
            stock_video_count=0,
        )
        assert check_stock_media_key(_config(s=stock_only), ["s"]) is not None

    def test_a_pool_mixing_both_blocks_on_the_stock_only_one(self, monkeypatch):
        monkeypatch.delenv("PEXELS_API_KEY", raising=False)
        stock_only = SimpleNamespace(
            use_scraped_images=False,
            use_scraped_videos=False,
            use_stock_images=True,
            stock_image_count=5,
            use_stock_videos=False,
            stock_video_count=0,
        )
        error = check_stock_media_key(
            _config(mixed=self._mixed(), pure=stock_only), ["mixed", "pure"]
        )
        assert error is not None
        assert "pure" in error
        assert "mixed" not in error


@pytest.mark.unit
class TestCandidateProfiles:
    """Which profiles the producer decides a run might select.

    This is where the candidate set is derived, and where a raise would escape
    into a traceback rather than the CLI's own error handling.
    """

    def _args(self, **kw):
        base = {
            "random_profile": False,
            "profile_pool": None,
            "batch_profile": None,
            "profile": None,
        }
        base.update(kw)
        return SimpleNamespace(**base)

    def _config_with(self, *names):
        return SimpleNamespace(
            video_profiles=dict.fromkeys(names, SCRAPED), profile_pool=None
        )

    def test_a_named_profile(self):
        from src.video.producer.cli import _profiles_this_run_may_use

        args = self._args(profile="slideshow_images1")
        assert _profiles_this_run_may_use(args, self._config_with("a")) == [
            "slideshow_images1"
        ]

    def test_a_batch_profile_wins_over_the_positional(self):
        from src.video.producer.cli import _profiles_this_run_may_use

        args = self._args(batch_profile="b", profile="a")
        assert _profiles_this_run_may_use(args, self._config_with("a", "b")) == ["b"]

    def test_no_profile_named_is_no_candidates(self):
        from src.video.producer.cli import _profiles_this_run_may_use

        assert _profiles_this_run_may_use(self._args(), self._config_with("a")) == []

    def test_a_random_run_returns_the_pool(self):
        from src.video.producer.cli import _profiles_this_run_may_use

        args = self._args(random_profile=True, profile_pool=["a", "b"])
        assert sorted(
            _profiles_this_run_may_use(args, self._config_with("a", "b"))
        ) == [
            "a",
            "b",
        ]

    def test_an_unusable_pool_does_not_raise(self):
        """`load_profile_pool` raises on an unknown name.

        Letting that escape replaces the CLI's own "Invalid profile pool
        configuration" message with a traceback, and the log-file line is
        never written.
        """
        from src.video.producer.cli import _profiles_this_run_may_use

        args = self._args(random_profile=True, profile_pool=["nope"])
        assert _profiles_this_run_may_use(args, self._config_with("a")) == []
