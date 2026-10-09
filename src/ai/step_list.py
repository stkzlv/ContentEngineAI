"""A sourced step list for a topic script (design 0017).

One grounded call asks for the task's steps, each with its action, exact UI
path, expected result and the URL of the page that states it. The script is
then written from the list, and its length follows the step count.

A step with no source is refused, and the topic with it: a tutorial with a
step missing cannot be followed. A topic is also dropped when nothing could
be sourced, and one whose steps fork by device, or need more steps than a
short video holds, is set aside for a series. The same call judges the
topic itself (specific, searchable, demonstrable, non-default, and no health,
financial or legal advice), and a topic that fails is dropped too. Nothing
here raises: a failed call returns None, which the caller treats as
unsourced.
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
# A concept explainer ("Why ...") is a cause, then one or two checks, in about
# 100-160 words (docs/explanation/tutorials.md, "Length"; REQ-VID-161).
EXPLAINER_PROMPT_PATH = (
    Path(__file__).parent / "prompts" / "topic_explainer_from_steps.md"
)
EXPLAINER_WORDS = (100, 160)
# Words for a single setting or shortcut (one or two steps, 15-30 s) and a
# multi-step fix (three to six, 40-75 s), from the tutorial research's length
# table (docs/explanation/tutorials.md, "Length").
SHORT_WORDS = (40, 80)
LONG_WORDS = (110, 200)


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
    # The topic filter's failed criteria; None when no check came back.
    topic_failures: list[str] | None = None
    # A "Why ..." topic, written as a cause and its checks (REQ-VID-161).
    explainer: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _sourced(source: str) -> bool:
    return source.startswith(("https://", "http://"))


TOPIC_CRITERIA = ("specific", "searchable", "demonstrable", "non_default")


def _topic_failures(check: Any) -> list[str] | None:
    """The criteria a topic fails; None when the check is missing or unreadable."""
    if not isinstance(check, dict):
        return None
    failures = [
        f"not {name.replace('_', '-')}"
        for name in TOPIC_CRITERIA
        if check.get(name) is not True
    ]
    advice = check.get("advice")
    # Strict like the criteria: only the string "none" passes.
    if not isinstance(advice, str) or not advice.strip():
        failures.append("advice unanswered")
    elif advice.strip().lower() != "none":
        failures.append(f"asks for {advice.strip().lower()} advice")
    return failures


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
            # A step with nothing to say is a missing step, like an unsourced one.
            refused.append(Step("", "", "", ""))
            continue
        step = Step(
            _text(raw.get("action")),
            _text(raw.get("ui_path")),
            _text(raw.get("expected")),
            _text(raw.get("source")),
        )
        if step.action and _sourced(step.source):
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
        topic_failures=_topic_failures(data.get("topic_check")),
    )


def drop_reason(step_list: StepList | None, max_steps: int) -> str | None:
    """Why the topic is not rendered from this list, or None to go ahead."""
    if step_list is None:
        return "no step list came back"
    if step_list.topic_failures is None:
        return "no topic check came back"
    if step_list.topic_failures:
        return "fails the topic filter: " + ", ".join(step_list.topic_failures)
    if not step_list.steps:
        return "no step could be sourced"
    if step_list.refused:
        # A refused step leaves a gap the viewer cannot get past.
        return "a step could not be sourced"
    if step_list.forks or len(step_list.steps) > max_steps:
        return "its steps fork by device or exceed the limit; set aside for a series"
    return None


def word_range(step_count: int, explainer: bool = False) -> tuple[int, int]:
    if explainer:
        return EXPLAINER_WORDS
    return SHORT_WORDS if step_count <= 2 else LONG_WORDS


def is_explainer_title(title: str | None) -> bool:
    """Whether a topic asks why something happens, not how to do a task."""
    return bool(title) and str(title).strip().lower().startswith("why ")


def script_prompt_path(step_list: StepList) -> Path:
    """The prompt a script is written from for this step list."""
    return EXPLAINER_PROMPT_PATH if step_list.explainer else SCRIPT_PROMPT_PATH


# A draft this far under the floor is retried; the model tends to run short.
LENGTH_TOLERANCE = 0.8


def too_short(script: str, step_count: int, explainer: bool = False) -> bool:
    """Whether a script falls well short of the length its steps call for."""
    floor = word_range(step_count, explainer)[0]
    return len(script.split()) < floor * LENGTH_TOLERANCE


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
    low, high = word_range(count, step_list.explainer)
    return {
        "STEP_LIST": "\n".join(lines),
        "PLATFORM": step_list.platform or "the device the steps were checked on",
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
    # A timeout, a dropped connection or a server error is retried: about
    # one grounded call in five ran past the timeout, and a topic day lost to
    # it posts nothing. A 4xx (bad request, no credit) would fail again.
    attempts = max(1, getattr(settings, "attempts", 1))
    answer = None
    try:
        for attempt in range(1, attempts + 1):
            try:
                async with asyncio.timeout(settings.timeout_seconds):
                    answer = await client.aio.models.generate_content(
                        model=settings.model, contents=prompt, config=config
                    )
                break
            except (
                aiohttp.ClientError,
                TimeoutError,
                OSError,
                ValueError,
                RuntimeError,
                genai_errors.APIError,
            ) as e:
                retry = attempt < attempts and not isinstance(
                    e, ValueError | genai_errors.ClientError
                )
                # A timeout's message is empty, so name the error's type.
                logger.warning(
                    "Step list call failed for '%s' (attempt %d of %d): %s %s",
                    topic,
                    attempt,
                    attempts,
                    type(e).__name__,
                    e,
                )
                if not retry:
                    return None
    finally:
        aclose = getattr(client.aio, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except (OSError, RuntimeError) as e:  # closing is best effort
                logger.debug("Closing the step list client failed: %s", e)
    if answer is None:
        return None
    return parse_step_list(answer.text)
