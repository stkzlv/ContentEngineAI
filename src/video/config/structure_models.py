"""Script-shape targets per content kind (REQ-CNT-189 to REQ-CNT-191).

Every kind ships disabled; an installation sets its targets in its overlay.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StructureTargets(BaseModel):
    """Times by which a script's beats should start, in seconds."""

    model_config = ConfigDict(extra="forbid")

    turn_by_sec: float | None = Field(None, gt=0)
    payoff_by_sec: float | None = Field(None, gt=0)
    signoff_max_sec: float | None = Field(None, gt=0)


class ScriptStructure(StructureTargets):
    """One kind's targets, prompt rules and variants."""

    enabled: bool = False
    # Words a minute for the estimate before voicing; a generic rate, and an
    # installation sets its voice's measured one.
    wpm: float = Field(150.0, gt=0)
    # `misconception` asks for the setup framed as what most people believe.
    setup: Literal["any", "misconception"] = "any"
    # Ask for no summary line before the closing question.
    no_recap: bool = False
    # Named overrides of the targets; one is drawn per product.
    variants: dict[str, StructureTargets] = Field(default_factory=dict)


class ScriptStructures(BaseModel):
    """One set per content kind."""

    model_config = ConfigDict(extra="forbid")

    product: ScriptStructure = Field(
        default_factory=ScriptStructure  # type: ignore[arg-type]
    )
    topic: ScriptStructure = Field(
        default_factory=ScriptStructure  # type: ignore[arg-type]
    )
    commentary: ScriptStructure = Field(
        default_factory=ScriptStructure  # type: ignore[arg-type]
    )

    def for_kind(self, kind: str) -> ScriptStructure:
        structure: ScriptStructure = getattr(self, kind)
        return structure
