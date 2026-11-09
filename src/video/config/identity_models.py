"""A recurring narrator identity per content kind (REQ-CNT-168 to REQ-CNT-170).

A recurring narrator is recognised by the same voice, sign-off, sound and
caption look in every video. Each field pins one of those for every render of
its kind; an unset field leaves that choice to the usual draw. The bundled
config leaves every kind empty, and an installation sets its own values in its
overlay (decision 0017).
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.video.config.audio_models import SignatureSting


class NarratorIdentity(BaseModel):
    """What one content kind pins. Every field is optional."""

    model_config = ConfigDict(extra="forbid")

    # A TTS voice profile name; replaces the pinned default and empties the pool.
    voice_profile: str | None = None
    # Spoken on every render, just before the closing call to action.
    signoff: str | None = None
    # Replaces `audio_settings.signature_sting` for the kind.
    sting: SignatureSting | None = None
    # Local beds drawn from instead of the provider chain.
    music_files: list[Path] = Field(default_factory=list)
    # Search queries the music providers use instead of their own.
    music_queries: list[str] = Field(default_factory=list)
    # A pycaps template name, used instead of drawing from the pool.
    caption_template: str | None = None
    # The narrator is synthetic: its scripts never claim a physical act
    # (REQ-CNT-173). Not a pinned dimension; it fixes no drawn choice.
    virtual: bool = False

    @field_validator("signoff")
    @classmethod
    def _one_sentence(cls, value: str | None) -> str | None:
        """One sentence with words: the script step places it as one."""
        if value is None:
            return None
        from src.utils.script_signoff import split_sentences

        if not re.sub(r"[^a-z0-9]", "", value.lower()):
            raise ValueError(f"signoff carries no words: {value!r}")
        if len(split_sentences(value.strip())) != 1:
            raise ValueError(f"signoff must be one sentence: {value!r}")
        return value.strip()

    def pinned_dimensions(self) -> list[str]:
        """The variety-report dimensions this identity fixes to one value."""
        pinned = []
        if self.voice_profile:
            pinned.append("voice_profile")
        if self.caption_template:
            pinned.append("caption_template")
        if len(self.music_files) == 1:
            pinned.append("music")
        return pinned


class NarratorIdentities(BaseModel):
    """One identity per content kind."""

    model_config = ConfigDict(extra="forbid")

    product: NarratorIdentity = Field(default_factory=NarratorIdentity)
    topic: NarratorIdentity = Field(default_factory=NarratorIdentity)
    # A topic record marked `kind: commentary` (REQ-CNT-166).
    commentary: NarratorIdentity = Field(default_factory=NarratorIdentity)

    def for_kind(self, kind: str) -> NarratorIdentity:
        identity: NarratorIdentity = getattr(self, kind)
        return identity


def content_kind(product: object) -> str:
    """`product`, `topic` or `commentary`, from the record.

    A correction (REQ-CNT-175) and a reply (REQ-CNT-176) render in the
    commentary identity.
    """
    if getattr(product, "topic_kind", None) in ("commentary", "correction", "reply"):
        return "commentary"
    return "topic" if getattr(product, "topic", None) else "product"
