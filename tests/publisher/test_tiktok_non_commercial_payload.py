"""A non-commercial TikTok render still affirms consent and preview.

`for_render` clears the commercial-content declaration for a render with no
material connection. The open question was whether it should also clear
`content_preview_confirmed` and `express_consent_given` -- they look like part
of the commercial-content flow, and sending them alongside "not commercial"
might be rejected the way a missing disclosure option is.

It is not. TikTok's Content Sharing Guidelines make both unconditional
requirements of the Direct Post API, for every post:

    "API Clients must only start sending content materials to TikTok after the
    user has expressly consent to the upload."
    "API Clients should display a preview of the to-be-posted content."

The rejection they were confused with -- "Commercial content disclosure is
enabled but no option selected" -- fires only when the disclosure toggle is ON
and neither option is chosen. `commercial_content_type="none"` with
`is_brand_organic_post=False` is the opposite of that state.

So clearing them would assert that consent was not obtained and no preview was
shown, on a post the API requires both for.
"""

from __future__ import annotations

import pytest

from src.publisher.models import TikTokContentSettings


def settings() -> TikTokContentSettings:
    return TikTokContentSettings()


class TestTheCommercialDeclarationIsCleared:
    def test_a_non_commercial_render_declares_none(self):
        render = settings().for_render(carries_affiliate_content=False)

        assert render.commercial_content_type == "none"
        assert render.is_brand_organic_post is False

    def test_an_affiliate_render_keeps_the_configured_declaration(self):
        render = settings().for_render(carries_affiliate_content=True)

        assert render.commercial_content_type == "brand_organic"
        assert render.is_brand_organic_post is True


class TestConsentAndPreviewSurvive:
    """The answer to the open question, pinned so it is not "tidied" later."""

    @pytest.mark.parametrize("carries", [True, False])
    def test_both_affirmations_are_sent_either_way(self, carries):
        render = settings().for_render(carries_affiliate_content=carries)

        assert render.content_preview_confirmed is True, (
            "clearing this asserts no preview was shown, on a post whose API "
            "requires one for every upload, commercial or not"
        )
        assert render.express_consent_given is True, (
            "clearing this asserts consent was not obtained, which the Direct "
            "Post API requires before any content is sent"
        )

    def test_the_payload_carries_them_to_the_sdk(self):
        """A field kept on the object but dropped from the dict is the same
        thing as clearing it, from TikTok's point of view.
        """
        payload = settings().for_render(False).to_sdk_dict()

        assert payload["content_preview_confirmed"] is True
        assert payload["express_consent_given"] is True
        assert payload["commercial_content_type"] == "none"
        assert payload["is_brand_organic_post"] is False

    def test_only_the_two_commercial_fields_differ_between_renders(self):
        """Scoping the change is what makes it safe.

        Anything else diverging between an affiliate and a topic render would
        be a behaviour difference nobody asked for.
        """
        from dataclasses import asdict

        commercial = asdict(settings().for_render(True))
        plain = asdict(settings().for_render(False))

        differing = {k for k in commercial if commercial[k] != plain[k]}
        assert differing == {"commercial_content_type", "is_brand_organic_post"}
