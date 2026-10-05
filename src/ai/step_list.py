"""A sourced step list for a topic script (design 0017).

One grounded call asks for the task's steps, each with its action, exact UI
path, expected result and the URL of the page that states it. The script is
then written from the list, and its length follows the step count.

A step with no source is refused, and the topic with it: a tutorial with a
step missing cannot be followed. A topic is also dropped when nothing could
be sourced, and one whose steps fork by device, or need more steps than a
short video holds, is set aside for a series. Nothing here raises: a failed
call returns None, which the caller treats as unsourced.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "topic_step_list.md"
SCRIPT_PROMPT_PATH = Path(__file__).parent / "prompts" / "topic_from_steps.md"
# Words for a short (one or two steps) and a longer (three or more) tutorial,
# at roughly 150 words a minute: 20-30 s and 40-75 s.
SHORT_WORDS = (50, 75)
LONG_WORDS = (100, 185)


@dataclass(frozen=True)
class Step:
    action: str
    ui_path: str
    expected: str
    source: str


@dataclass
class StepList:
    start_screen: str
    platform: str
    forks: bool
    steps: list[Step]
    mistake_step: int | None = None
    mistake: str | None = None
    refused: list[Step] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _sourced(source: str) -> bool:
    return source.startswith(("https://", "http://"))


def parse_step_list(answer: str | None) -> StepList | None:
    """The model's step list, unsourced steps refused; None when unreadable."""
    if not answer:
        return None
    body = answer.strip().strip("`").removeprefix("json").strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(body[start : end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return None
    kept: list[Step] = []
    refused: list[Step] = []
    # The model numbers steps in its own list; map to positions in `kept`.
    position: dict[int, int] = {}
    for number, raw in enumerate(data["steps"], start=1):
        if not isinstance(raw, dict):
            continue
        step = Step(
            _text(raw.get("action")),
            _text(raw.get("ui_path")),
            _text(raw.get("expected")),
            _text(raw.get("source")),
        )
        if not step.action:
            continue
        if _sourced(step.source):
            kept.append(step)
            position[number] = len(kept)
        else:
            refused.append(step)
    mistake = data.get("common_mistake")
    raw_step = mistake.get("step") if isinstance(mistake, dict) else None
    # A mistake whose own step was refused, or that names no step, is dropped.
    mistake_step = position.get(raw_step) if isinstance(raw_step, int) else None
    mistake_text = _text(mistake.get("mistake")) if isinstance(mistake, dict) else ""
    return StepList(
        start_screen=_text(data.get("start_screen")),
        platform=_text(data.get("platform")),
        forks=data.get("forks") is True,
        steps=kept,
        mistake_step=mistake_step,
        mistake=(mistake_text or None) if mistake_step else None,
        refused=refused,
    )


def drop_reason(step_list: StepList | None, max_steps: int) -> str | None:
    """Why the topic is not rendered from this list, or None to go ahead."""
    if step_list is None:
        return "no step list came back"
    if not step_list.steps:
        return "no step could be sourced"
    if step_list.refused:
        # A refused step leaves a gap the viewer cannot get past.
        return "a step could not be sourced"
    if step_list.forks or len(step_list.steps) > max_steps:
        return "its steps fork by device or exceed the limit; set aside for a series"
    return None


def word_range(step_count: int) -> tuple[int, int]:
    return SHORT_WORDS if step_count <= 2 else LONG_WORDS


# A draft this far under the floor is retried; the model tends to run short.
LENGTH_TOLERANCE = 0.8


def too_short(script: str, step_count: int) -> bool:
    """Whether a script falls well short of the length its steps call for."""
    return len(script.split()) < word_range(step_count)[0] * LENGTH_TOLERANCE


def render_steps(step_list: StepList) -> dict[str, str]:
    """The `<<...>>` fills for the script prompt."""
    lines = []
    for n, step in enumerate(step_list.steps, start=1):
        path = f" ({step.ui_path})" if step.ui_path else ""
        expected = f" (Viewer then sees: {step.expected}.)" if step.expected else ""
        lines.append(f"{n}. {step.action}{path}.{expected}")
    count = len(step_list.steps)
    if (
        step_list.mistake
        and step_list.mistake_step
        and 1 <= (step_list.mistake_step) <= count
    ):
        mistake = (
            f"At step {step_list.mistake_step}, name this common mistake in one "
            f"sentence: {step_list.mistake}"
        )
    else:
        mistake = "Name no mistake the steps do not state."
    low, high = word_range(count)
    return {
        "STEP_LIST": "\n".join(lines),
        "START_SCREEN": step_list.start_screen or "the screen the first step opens",
        "MISTAKE_RULE": mistake,
        "WORD_RANGE": f"{low}-{high}",
    }


def fill(prompt: str, fills: dict[str, str]) -> str:
    """Replace `<<KEY>>` markers after `str.format`, so braces in a step stay."""
    for key, value in fills.items():
        prompt = prompt.replace(f"<<{key}>>", value)
    return prompt


async def build_step_list(
    topic: str, detail: str, *, api_key: str, settings: Any
) -> StepList | None:
    """One grounded call for the topic's steps; None when it fails."""
    try:
        from google import genai
        from google.genai import errors as genai_errors

        client = genai.Client(api_key=api_key)
        config = genai.types.GenerateContentConfig(
            tools=[genai.types.Tool(google_search=genai.types.GoogleSearch())],
            temperature=0.0,
        )
    except (ImportError, ValueError, OSError, RuntimeError) as e:
        logger.warning("Step list unavailable: %s", e)
        return None
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(
        TOPIC_TITLE=topic, TOPIC_DETAIL=detail, MAX_STEPS=settings.max_steps
    )
    try:
        async with asyncio.timeout(settings.timeout_seconds):
            answer = await client.aio.models.generate_content(
                model=settings.model, contents=prompt, config=config
            )
    except (
        aiohttp.ClientError,
        TimeoutError,
        OSError,
        ValueError,
        RuntimeError,
        genai_errors.APIError,
    ) as e:
        logger.warning("Step list call failed for '%s': %s", topic, e)
        return None
    finally:
        aclose = getattr(client.aio, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except (OSError, RuntimeError) as e:  # closing is best effort
                logger.debug("Closing the step list client failed: %s", e)
    return parse_step_list(answer.text)
