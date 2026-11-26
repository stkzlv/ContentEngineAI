"""Named recurring series (REQ-CNT-193 to REQ-CNT-195).

The shipped config holds none; an installation names its series in its
overlay, and a topic marks which series it belongs to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VerdictBeat(BaseModel):
    """The stamp card on the verdict sentence (REQ-VID-167)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # `text` and `background` FFmpeg colours; None, or a missing key, takes
    # the hook overlay's.
    colours: dict[str, str] | None = None
    duration_sec: float = Field(default=1.5, gt=0.0, le=10.0)
    # Font size as a multiple of the narration caption font.
    size_factor: float = Field(default=2.5, gt=0.0, le=6.0)
    # The card's top edge, as a fraction of the frame height.
    y_percent: float = Field(default=0.35, ge=0.0, le=0.9)

    @field_validator("colours")
    @classmethod
    def _known_colour_keys(cls, value: dict[str, str] | None) -> object:
        unknown = set(value or {}) - {"text", "background"}
        if unknown:
            raise ValueError(
                f"verdict_beat.colours takes text and background, not {sorted(unknown)}"
            )
        return value


class CreditLine(BaseModel):
    """The lower-third line naming the commenter (REQ-VID-168)."""

    model_config = ConfigDict(extra="forbid")

    # `{credit}` is the topic's credit, such as a commenter's handle.
    format: str = "Suggested by {credit}"
    # `text` and `background` FFmpeg colours; a missing key takes the hook
    # overlay's.
    colours: dict[str, str] | None = None
    # Font size as a multiple of the narration caption font.
    size_factor: float = Field(default=0.9, gt=0.0, le=4.0)
    # The line's top edge, as a fraction of the frame height; kept above the
    # captions, which sit low.
    y_percent: float = Field(default=0.62, ge=0.0, le=0.95)

    @field_validator("format")
    @classmethod
    def _names_the_credit(cls, value: str) -> str:
        if "{credit}" not in value:
            raise ValueError("credit_line.format must contain {credit}")
        return value

    @field_validator("colours")
    @classmethod
    def _known_colour_keys(cls, value: dict[str, str] | None) -> object:
        unknown = set(value or {}) - {"text", "background"}
        if unknown:
            raise ValueError(
                f"credit_line.colours takes text and background, not {sorted(unknown)}"
            )
        return value


class SeriesGroups(BaseModel):
    """Where a live episode is grouped on each platform (REQ-PUB-169)."""

    model_config = ConfigDict(extra="forbid")

    youtube_playlist_id: str | None = None
    tiktok_playlist: str | None = None
    instagram_link: bool = False


class SeriesConfig(BaseModel):
    """One named series."""

    model_config = ConfigDict(extra="forbid")

    # Starts the title and the platform captions, e.g. "Myth check: ".
    title_prefix: str = ""
    # The script's first sentence, word for word, with `{claim}` filled in.
    opening_line: str = ""
    # The verdict labels a topic may carry, e.g. [YES, NO, PARTLY].
    verdicts: list[str] = Field(default_factory=list)
    verdict_beat: VerdictBeat = Field(default_factory=VerdictBeat)
    numbering: Literal["none", "episode"] = "none"
    serial: bool = False
    queue_file: Path | None = None
    credit_commenter: bool = False
    credit_line: CreditLine = Field(default_factory=CreditLine)
    groups: SeriesGroups = Field(default_factory=SeriesGroups)

    @field_validator("verdicts", mode="before")
    @classmethod
    def _labels_are_strings(cls, value: object) -> object:
        """YAML reads an unquoted YES, NO, TRUE or FALSE as a boolean."""
        if isinstance(value, list) and any(isinstance(v, bool) for v in value):
            raise ValueError(
                "a verdict label was read as true/false: quote labels such as "
                '"YES", "NO", "TRUE" and "FALSE" in YAML'
            )
        return value

    @model_validator(mode="after")
    def _episode_numbers_need_a_queue(self) -> SeriesConfig:
        if self.numbering == "episode" and self.queue_file is None:
            raise ValueError("numbering: episode requires a queue_file")
        return self
