"""The static upper line, on both engines (#88).

The two-part subtitle system already rendered a line above the visual, but
only under FFmpeg: `step_generate_subtitles` disables two-part mode when the
engine is pycaps, so a profile that switched engines lost the line with a
single warning. pycaps has one caption track and no static element, so it
could never carry it.

The line is static for the whole video and needs no subtitle engine. Drawn as
an overlay in the assembler, it survives both engines, the pycaps burn that
composes over the assembler's output, and the FFmpeg caption fallback -- the
case where a subtitle-side implementation would lose it exactly when the
fallback is doing its job.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.video.assembler.overlay_builder import (
    apply_disclosure_overlay,
    apply_upper_line_overlay,
    resolve_upper_line_text,
)
from src.video.assembler.visual_band import upper_line_bottom, visual_band
from src.video.config.visual_models import (
    DisclosureSettings,
    PartialUpperLine,
    UpperLineSettings,
)

FRAME_H = 1920


def product(**fields):
    base = {"shortened_affiliate_link": "", "affiliate_link": "", "url": ""}
    return SimpleNamespace(**{**base, **fields})


@pytest.mark.unit
class TestResolvingTheText:
    def test_the_affiliate_link_is_preferred_shortened(self) -> None:
        text, reason = resolve_upper_line_text(
            UpperLineSettings(source="affiliate_link"),
            product(shortened_affiliate_link="https://a.co/x", url="https://long"),
        )
        assert text == "https://a.co/x" and "affiliate_link" in reason

    def test_it_falls_back_through_the_link_fields(self) -> None:
        text, _ = resolve_upper_line_text(
            UpperLineSettings(source="affiliate_link"),
            product(url="https://amazon.com/dp/B0X"),
        )
        assert text == "https://amazon.com/dp/B0X"

    def test_a_topic_render_has_no_link_and_says_so(self) -> None:
        """Ordinary, not an error: a topic has nothing to link to. The reason
        is what keeps an empty box off the frame.
        """
        text, reason = resolve_upper_line_text(
            UpperLineSettings(source="affiliate_link"), product()
        )
        assert text is None and "no affiliate link" in reason

    def test_the_bio_url_comes_from_the_environment(self) -> None:
        """Never from the bundled YAML: the public config carries no
        account-specific value, and the pipeline has no other route to that
        address -- the link-in-bio module holds OAuth credentials, not a URL.
        """
        text, reason = resolve_upper_line_text(
            UpperLineSettings(source="link_in_bio"),
            product(),
            env={"LINK_IN_BIO_URL": "https://lnk.bio/example"},
        )
        assert text == "https://lnk.bio/example" and reason == "link_in_bio"

    def test_an_unset_bio_url_names_both_variables(self) -> None:
        text, reason = resolve_upper_line_text(
            UpperLineSettings(source="link_in_bio"), product(), env={}
        )
        assert text is None
        assert "LINK_IN_BIO_URL" in reason and "SUBTITLE_BUSINESS_URL" in reason

    def test_the_existing_business_url_still_works(self) -> None:
        """The two-part upper line has read SUBTITLE_BUSINESS_URL since it
        shipped, and six bundled profiles point at it. A second variable for
        the same value would have silently lost the line for an installation
        that already had one -- and this feature turns two-part's line off.
        """
        text, reason = resolve_upper_line_text(
            UpperLineSettings(source="link_in_bio"),
            product(),
            env={"SUBTITLE_BUSINESS_URL": "https://lnk.bio/legacy"},
        )
        assert text == "https://lnk.bio/legacy"
        assert "SUBTITLE_BUSINESS_URL" in reason

    def test_the_new_variable_wins_when_both_are_set(self) -> None:
        text, _ = resolve_upper_line_text(
            UpperLineSettings(source="link_in_bio"),
            product(),
            env={
                "LINK_IN_BIO_URL": "https://lnk.bio/new",
                "SUBTITLE_BUSINESS_URL": "https://lnk.bio/legacy",
            },
        )
        assert text == "https://lnk.bio/new"

    def test_custom_text_is_rendered_verbatim(self) -> None:
        text, _ = resolve_upper_line_text(
            UpperLineSettings(source="custom", custom_text="Full guide in bio"),
            product(shortened_affiliate_link="https://a.co/x"),
        )
        assert text == "Full guide in bio"

    def test_empty_custom_text_draws_nothing(self) -> None:
        text, reason = resolve_upper_line_text(
            UpperLineSettings(source="custom"), product()
        )
        assert text is None and "empty" in reason


@pytest.mark.unit
class TestTheDrawnFilter:
    def _apply(self, tmp_path, **over):
        settings = UpperLineSettings(enabled=True, **over)
        return apply_upper_line_overlay(
            ["[v_0]copy[v_out]"],
            settings,
            over.pop("text", None) or "https://a.co/example",
            60,
            FRAME_H,
            tmp_path,
        )

    def test_the_text_goes_through_a_textfile(self, tmp_path) -> None:
        """Inline `text=` corrupts inside the assembler's multi-filter chain
        when the text carries an apostrophe: FFmpeg swallows the filter's own
        trailing args. A URL or operator prose can carry one.
        """
        out = self._apply(tmp_path)
        assert "drawtext=textfile=" in out[-1]
        # The inline form specifically. `x=(w-text_w)/2` contains "text=".
        assert "drawtext=text=" not in out[-1]
        assert (tmp_path / "upper_line_text.txt").exists()

    def test_percent_and_backslash_are_escaped_in_the_file(self, tmp_path) -> None:
        """`textfile=` removes the quoting layer but not drawtext's text
        expansion. A raw `%` makes FFmpeg draw nothing for the line and exit
        0; a raw backslash is swallowed. Both are silent.
        """
        apply_upper_line_overlay(
            ["[v_0]copy[v_out]"],
            UpperLineSettings(enabled=True),
            r"50% off C:\deals",
            60,
            FRAME_H,
            tmp_path,
        )
        written = (tmp_path / "upper_line_text.txt").read_text(encoding="utf-8")
        assert r"\%" in written and r"\\" in written

    def test_it_produces_the_terminal_output_stream(self, tmp_path) -> None:
        out = self._apply(tmp_path)
        assert out[-1].endswith("[v_out]")
        assert out[-1].startswith("[v_0]")

    def test_a_disabled_line_changes_nothing(self, tmp_path) -> None:
        chain = ["[v_0]copy[v_out]"]
        assert (
            apply_upper_line_overlay(
                chain, UpperLineSettings(enabled=False), "x", 60, FRAME_H, tmp_path
            )
            == chain
        )

    def test_no_text_changes_nothing(self, tmp_path) -> None:
        """The resolver returning None must not leave a bare background box
        over the visual.
        """
        chain = ["[v_0]copy[v_out]"]
        assert (
            apply_upper_line_overlay(
                chain, UpperLineSettings(enabled=True), None, 60, FRAME_H, tmp_path
            )
            == chain
        )

    def test_long_prose_is_trimmed_on_a_word_boundary(self, tmp_path) -> None:
        """Drawtext does not wrap; the overflow is drawn past the frame edge
        with no sign the line was cut.
        """
        apply_upper_line_overlay(
            ["[v_0]copy[v_out]"],
            UpperLineSettings(enabled=True, max_chars=20),
            "the full guide and every link lives in my bio",
            60,
            FRAME_H,
            tmp_path,
        )
        written = (tmp_path / "upper_line_text.txt").read_text(encoding="utf-8")
        assert len(written) <= 24 and written.endswith("...")

    def test_an_unbreakable_overlong_url_draws_nothing(self, tmp_path) -> None:
        """A URL has no spaces, so trimming cuts mid-address and holds a link
        nobody can use on screen for the whole video -- the one thing this
        line exists to do. The third resolver rung reaches a 500-character
        Amazon SERP address, so this is not hypothetical.
        """
        chain = ["[v_0]copy[v_out]"]
        out = apply_upper_line_overlay(
            chain,
            UpperLineSettings(enabled=True, max_chars=20),
            "https://www.amazon.com/dp/B0EXAMPLE?tag=something-20",
            60,
            FRAME_H,
            tmp_path,
        )
        assert out == chain
        assert not (tmp_path / "upper_line_text.txt").exists()

    def test_it_composes_with_the_disclosure(self, tmp_path) -> None:
        """Both rewrite the chain's terminal, so whichever runs second has to
        normalise what the first left. One overlay silently replacing the
        other is the failure this pins.
        """
        chain = apply_upper_line_overlay(
            ["[v_0]copy[v_out]"],
            UpperLineSettings(enabled=True),
            "https://a.co/x",
            60,
            FRAME_H,
            tmp_path,
        )
        chain = apply_disclosure_overlay(
            chain, DisclosureSettings(enabled=True), 60, tmp_path
        )
        joined = "\n".join(chain)

        assert "upper_line_text.txt" in joined
        assert "disclosure_text.txt" in joined
        assert chain[-1].endswith("[v_out]")
        assert sum(1 for f in chain if f.endswith("[v_out]")) == 1

    def test_it_applies_on_the_content_aware_ass_terminal(self, tmp_path) -> None:
        """That path ends with `ass='...'[v_out]` and no rewritable no-op,
        which is how both overlays used to be dropped silently.
        """
        out = apply_upper_line_overlay(
            ["[v_0]ass='/tmp/x.ass'[v_out]"],
            UpperLineSettings(enabled=True),
            "https://a.co/x",
            60,
            FRAME_H,
            tmp_path,
        )
        assert "upper_line_text.txt" in "\n".join(out)
        assert out[-1].endswith("[v_out]")


@pytest.mark.unit
class TestTheImageMakesRoom:
    def test_the_band_starts_below_the_line(self) -> None:
        """Without this the image is fitted under the header and the line is
        drawn on top of it -- the same defect the band exists to prevent at
        the caption end.
        """
        settings = UpperLineSettings(enabled=True)
        reserved = upper_line_bottom(FRAME_H, settings, 60)
        band = visual_band(
            FRAME_H,
            caption_top=1200,
            top_offset=0,
            centred=True,
            upper_line_bottom_px=reserved,
        )
        assert band.top > reserved

    def test_a_disabled_line_reserves_nothing(self) -> None:
        assert upper_line_bottom(FRAME_H, UpperLineSettings(enabled=False), 60) == 0
        with_line = visual_band(FRAME_H, caption_top=1200, top_offset=0, centred=True)
        assert (
            with_line.top
            == visual_band(
                FRAME_H,
                caption_top=1200,
                top_offset=0,
                centred=True,
                upper_line_bottom_px=0,
            ).top
        )

    def test_the_caption_block_is_untouched(self) -> None:
        """The line takes rows from the top. The captions keep theirs."""
        plain = visual_band(FRAME_H, caption_top=1200, top_offset=0, centred=True)
        with_line = visual_band(
            FRAME_H,
            caption_top=1200,
            top_offset=0,
            centred=True,
            upper_line_bottom_px=upper_line_bottom(
                FRAME_H, UpperLineSettings(enabled=True), 60
            ),
        )
        assert plain.bottom == with_line.bottom


@pytest.mark.unit
class TestItSupersedesTheTwoPartUpperLine:
    """Both draw a static line above the visual.

    Leaving both on renders the text twice under FFmpeg, and the overlay is
    the half that survives pycaps. The voiceover-synced lower line is a
    different thing and stays on.
    """

    def _settings(self, two_part=True, upper=True):
        from src.video.config.subtitle_models import SubtitleSettings

        s = SubtitleSettings()
        s.two_part_subtitles.enabled = two_part
        s.two_part_subtitles.upper_line.enabled = upper
        return s

    def test_the_two_part_upper_line_is_turned_off(self) -> None:
        from src.video.producer.steps import supersede_two_part_upper_line

        settings = self._settings()
        assert supersede_two_part_upper_line(
            UpperLineSettings(enabled=True), settings, "https://a.co/x"
        )
        assert settings.two_part_subtitles.upper_line.enabled is False

    def test_the_lower_line_is_untouched(self) -> None:
        from src.video.producer.steps import supersede_two_part_upper_line

        settings = self._settings()
        supersede_two_part_upper_line(
            UpperLineSettings(enabled=True), settings, "https://a.co/x"
        )
        assert settings.two_part_subtitles.lower_line.enabled is True
        assert settings.two_part_subtitles.enabled is True

    def test_an_off_overlay_changes_nothing(self) -> None:
        from src.video.producer.steps import supersede_two_part_upper_line

        settings = self._settings()
        assert not supersede_two_part_upper_line(
            UpperLineSettings(enabled=False), settings, "https://a.co/x"
        )
        assert settings.two_part_subtitles.upper_line.enabled is True

    def test_two_part_already_off_is_not_reported_as_superseded(self) -> None:
        """The caller logs on a True return, and a log line saying it turned
        off something already off is noise that reads as a real decision.
        """
        from src.video.producer.steps import supersede_two_part_upper_line

        assert not supersede_two_part_upper_line(
            UpperLineSettings(enabled=True),
            self._settings(two_part=False),
            "https://a.co/x",
        )


@pytest.mark.unit
class TestTheProfileDecides:
    def test_a_partial_override_keeps_the_other_fields(self) -> None:
        """A profile turning the line on must not reset the styling to the
        model defaults, which is what the whole-field override map does.
        """
        base = UpperLineSettings(enabled=False, size_factor=0.9, font_color="yellow")
        merged = PartialUpperLine(enabled=True).merge_into(base)

        assert merged.enabled is True
        assert merged.size_factor == 0.9
        assert merged.font_color == "yellow"

    def test_an_unknown_key_is_refused_at_load(self) -> None:
        with pytest.raises(ValueError):
            PartialUpperLine(enable=True)

    def test_the_code_default_is_off(self) -> None:
        assert UpperLineSettings().enabled is False

    def test_a_profile_override_reaches_the_consumer(self) -> None:
        """The merge existing is not the same as anything reading it.

        Every consumer took the global `config.video_settings.upper_line`
        while the merged value sat unread, so a profile enabling the line was
        a complete no-op and a profile disabling it did nothing -- all three
        declaration conditions satisfied and the fourth, that something reads
        the target, missed. Driving a consumer is what tells the two apart.
        """
        from src.video.config import load_video_config_modular

        config = load_video_config_modular()
        name = next(iter(config.video_profiles))
        profile = config.video_profiles[name]
        assert config.video_settings.upper_line.enabled is False

        profile.upper_line = PartialUpperLine(enabled=True, size_factor=0.8)
        try:
            merged = config.get_profile_merged_settings(name)
            reserved = upper_line_bottom(
                FRAME_H,
                merged.video_settings.upper_line,
                max(
                    8,
                    int(
                        round(FRAME_H * config.video_settings.base_font_height_percent)
                    ),
                ),
            )
            assert reserved > 0, "the profile's line reserves no rows"

            band = visual_band(
                FRAME_H,
                caption_top=1200,
                top_offset=0,
                centred=True,
                upper_line_bottom_px=reserved,
            )
            plain = visual_band(FRAME_H, caption_top=1200, top_offset=0, centred=True)
            assert band.top > plain.top, "the image did not move down for it"
        finally:
            profile.upper_line = None
