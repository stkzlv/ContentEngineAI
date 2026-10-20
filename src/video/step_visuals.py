"""Step visuals for a tutorial render: a screen per step (REQ-VID-123).

With step visuals on, a topic render with a step list shows each step as it
is spoken: a full-frame mockup of the screen the step happens on, with the
step's exact menu labels and the one to tap marked (design 0018). Stock
footage keeps only the opening, at most `opening_max_sec`; the time from
there to the first step shows the start screen, and the video ends on a
recap of every step's path and the last step's result.

The mockup is generic in its chrome and exact in its labels: the only text
it carries comes from the step list, which cites a source for every step.
Rows it can't name are drawn as blank bars rather than invented labels.

All or nothing: if any step's narration can't be found, the render keeps its
stock visuals, since a screen shown at the wrong time teaches the wrong step.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.video.step_cards import (
    find_step_words,
    first_segment_word,
    path_segments,
    sentence_starts,
    step_keys,
)

logger = logging.getLogger(__name__)

# Shortest time the last step's screen keeps before the recap takes over.
_LAST_STEP_MIN_SEC = 1.0

_BACKGROUND = "#1b1f27"
_PANEL = "#f5f6f8"
_INK = "#1d2330"
_MUTED = "#9aa1ad"
_BAR = "#e3e6eb"
_ACCENT = "#2f6fec"
_TARGET_FILL = "#e3edff"
_DONE = "#1f9d55"


@dataclass(frozen=True)
class Screen:
    """One full-frame visual and when it is on screen, in seconds.

    `kind` is "start", "step" or "recap". A step screen marks `target` among
    its rows; a recap lists `rows` and ends on `result`.
    """

    kind: str
    start: float
    end: float
    title: str = ""
    header: str = ""
    target: str = ""
    rows: list[str] = field(default_factory=list)
    result: str = ""


def _step_start_words(
    steps: list[dict[str, Any]],
    spoken: list[dict[str, Any]],
    found: dict[int, int],
) -> list[int]:
    """Where each step's screen starts: the sentence that first names it.

    The matched word is usually the path's last segment, often spoken at the end of
    the step ("Now, find Auto-Lock. Open the list. Select 30."), so the
    screen moves back to the earliest segment named since the previous step,
    then to the start of that word's sentence. It never starts at or before
    the previous step's matched word, so each screen holds its own key word,
    and never in the first sentence, which names the topic and often a
    step's segment ("... in Accessibility settings.").
    """
    sentences = sentence_starts(spoken)
    starts: list[int] = []
    # The first sentence's last word, where `find_step_words` starts too.
    previous = sentences[1] - 1 if len(sentences) > 1 else -1
    for number, step in enumerate(steps, start=1):
        key = found[number]
        earliest = first_segment_word(step, spoken, previous, key)
        index = key if earliest is None else min(earliest, key)
        sentence = max((s for s in sentences if s <= index), default=index)
        starts.append(sentence if sentence > previous else index)
        previous = key
    return starts


def _recap_word(spoken: list[dict[str, Any]], last_key: int) -> int | None:
    """Where the spoken recap starts: two sentences after the last step's.

    The last step's sentence, then its result ("Your settings are saved"),
    then the closing recap the step-list script ends on.
    """
    later = [s for s in sentence_starts(spoken) if s > last_key]
    return later[1] if len(later) > 1 else None


def plan_step_visuals(
    steps: list[dict[str, Any]],
    spoken: list[dict[str, Any]],
    *,
    start_screen: str,
    video_end: float,
    opening_max_sec: float,
    recap_sec: float,
) -> list[Screen]:
    """The screens for a render, in order; none unless every step is found.

    A step's screen runs from the start of the sentence that first names it
    to the next step's. The start screen fills any time between
    `opening_max_sec` and the first step. The recap runs from where the
    spoken recap starts to the end, at least `recap_sec`, leaving the last
    step at least a second.
    """
    found = find_step_words(steps, spoken)
    missing = [n for n in range(1, len(steps) + 1) if n not in found]
    if not steps or missing:
        logger.info(
            "Step visuals skipped: narration of step(s) %s not found",
            "; ".join(
                f"{n} ({', '.join(step_keys(steps[n - 1], (steps + [{}])[n]))})"
                for n in missing
            )
            or "-",
        )
        return []
    starts = [
        float(spoken[i]["start_time"]) for i in _step_start_words(steps, spoken, found)
    ]
    recap_word = _recap_word(spoken, found[len(steps)])
    spoken_recap = (
        float(spoken[recap_word]["start_time"]) if recap_word is not None else video_end
    )
    recap_start = max(
        min(spoken_recap, video_end - recap_sec), starts[-1] + _LAST_STEP_MIN_SEC
    )
    total = len(steps)

    screens: list[Screen] = []
    if starts[0] > opening_max_sec:
        screens.append(Screen("start", opening_max_sec, starts[0], header=start_screen))
    for number, step in enumerate(steps, start=1):
        segments = path_segments(str(step.get("ui_path") or ""))
        target = segments[-1] if segments else str(step.get("action") or "")
        # The screen the step happens on, where the step list names it: the
        # path's parent, or the start screen for the first step. Otherwise
        # the header stays blank rather than guessed.
        if len(segments) > 1:
            header = segments[-2]
        elif number == 1:
            header = start_screen
        else:
            header = ""
        end = starts[number] if number < total else recap_start
        screens.append(
            Screen(
                "step",
                starts[number - 1],
                min(end, video_end),
                title=f"Step {number} of {total}",
                header=header,
                target=target,
            )
        )
    if recap_start < video_end:
        screens.append(
            Screen(
                "recap",
                recap_start,
                video_end,
                header="Recap",
                rows=[
                    " > ".join(path_segments(str(s.get("ui_path") or "")))
                    or str(s.get("action") or "")
                    for s in steps
                ],
                result=str(steps[-1].get("expected") or ""),
            )
        )
    return screens


@cache
def _font_file(bold: bool) -> str | None:
    """A TrueType face from fontconfig, or None when it can't say."""
    try:
        proc = subprocess.run(
            ["fc-match", "-f", "%{file}", "Sans:bold" if bold else "Sans"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    path = proc.stdout.strip()
    return path if proc.returncode == 0 and path.endswith((".ttf", ".otf")) else None


def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _font_file(bold)
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _fit(
    draw: ImageDraw.ImageDraw, text: str, width: int, size: int, bold: bool
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, str]:
    """The largest font down to 60% of `size` that fits, then an ellipsis."""
    for current in range(size, int(size * 0.6) - 1, -2):
        font = _font(current, bold)
        if draw.textlength(text, font=font) <= width:
            return font, text
    font = _font(int(size * 0.6), bold)
    while text and draw.textlength(text + "...", font=font) > width:
        text = text[:-1]
    return font, text.rstrip() + "..."


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    width: int,
    size: int,
    *,
    bold: bool = False,
    fill: str = _INK,
    anchor: str = "lm",
) -> None:
    font, shown = _fit(draw, text, width, size, bold)
    draw.text(xy, shown, font=font, fill=fill, anchor=anchor)


def draw_screen(screen: Screen, size: tuple[int, int]) -> Image.Image:
    """One screen as an opaque frame-sized image."""
    w, h = size
    image = Image.new("RGB", size, _BACKGROUND)
    draw = ImageDraw.Draw(image)
    unit = h / 1920
    if screen.title:
        _text(
            draw,
            (w // 2, int(0.15 * h)),
            screen.title,
            int(0.8 * w),
            int(64 * unit),
            bold=True,
            fill="white",
            anchor="mm",
        )

    left, right = int(0.1 * w), int(0.9 * w)
    top, bottom = int(0.21 * h), int(0.56 * h)
    pad = int(40 * unit)
    draw.rounded_rectangle(
        (left, top, right, bottom), radius=int(40 * unit), fill=_PANEL
    )
    header_h = int(120 * unit)
    # A back chevron on the screens a step happens on; the recap is no screen.
    back = 0 if screen.kind == "recap" else int(60 * unit)
    if back:
        _text(
            draw,
            (left + pad, top + header_h // 2),
            "<",
            back,
            int(52 * unit),
            bold=True,
            fill=_MUTED,
        )
    if screen.header:
        _text(
            draw,
            (left + pad + back, top + header_h // 2),
            screen.header,
            right - left - 2 * pad - back,
            int(48 * unit),
            bold=True,
        )
    else:
        # A screen the step list doesn't name: a blank title, like the rows.
        bar_h = int(30 * unit)
        x0 = left + pad + back
        draw.rounded_rectangle(
            (
                x0,
                top + (header_h - bar_h) // 2,
                x0 + int(0.35 * (right - left)),
                top + (header_h + bar_h) // 2,
            ),
            radius=bar_h // 2,
            fill=_BAR,
        )
    draw.line(
        (left, top + header_h, right, top + header_h),
        fill=_BAR,
        width=max(1, int(3 * unit)),
    )

    rows_top = top + header_h + int(24 * unit)
    row_w = right - left - 2 * pad
    if screen.kind == "recap":
        _draw_recap(draw, screen, left + pad, rows_top, row_w, bottom - pad, unit)
        return image

    row_h, gap = int(110 * unit), int(18 * unit)
    bar_widths = (0.6, 0.45, 0.52, 0.38)
    for i, share in enumerate(bar_widths):
        y0 = rows_top + i * (row_h + gap)
        box = (left + pad, y0, left + pad + row_w, y0 + row_h)
        if screen.kind == "step" and i == 1:
            draw.rounded_rectangle(
                box,
                radius=int(22 * unit),
                fill=_TARGET_FILL,
                outline=_ACCENT,
                width=max(2, int(5 * unit)),
            )
            ring = int(34 * unit)
            cx, cy = box[2] - pad - ring, y0 + row_h // 2
            _text(
                draw,
                (box[0] + int(28 * unit), cy),
                screen.target,
                row_w - int(28 * unit) - 2 * ring - 2 * pad,
                int(46 * unit),
                bold=True,
            )
            draw.ellipse(
                (cx - ring, cy - ring, cx + ring, cy + ring),
                outline=_ACCENT,
                width=max(2, int(6 * unit)),
            )
            dot = ring // 3
            draw.ellipse((cx - dot, cy - dot, cx + dot, cy + dot), fill=_ACCENT)
            continue
        bar_h = int(26 * unit)
        draw.rounded_rectangle(
            (
                box[0] + int(28 * unit),
                y0 + (row_h - bar_h) // 2,
                box[0] + int(28 * unit) + int(share * row_w),
                y0 + (row_h + bar_h) // 2,
            ),
            radius=bar_h // 2,
            fill=_BAR,
        )
    return image


def _draw_recap(
    draw: ImageDraw.ImageDraw,
    screen: Screen,
    x: int,
    top: int,
    width: int,
    bottom: int,
    unit: float,
) -> None:
    """The steps' paths, numbered, then the result with a check mark."""
    lines = len(screen.rows) + (1 if screen.result else 0)
    row_h = min(int(100 * unit), (bottom - top) // max(lines, 1))
    size = int(min(44 * unit, row_h * 0.45))
    for i, row in enumerate(screen.rows):
        cy = top + i * row_h + row_h // 2
        _text(draw, (x, cy), f"{i + 1}", int(50 * unit), size, bold=True, fill=_ACCENT)
        _text(draw, (x + int(56 * unit), cy), row, width - int(56 * unit), size)
    if screen.result:
        cy = top + len(screen.rows) * row_h + row_h // 2
        r = int(size * 0.55)
        draw.ellipse((x, cy - r, x + 2 * r, cy + r), fill=_DONE)
        stroke = max(2, int(5 * unit))
        draw.line(
            (
                x + int(r * 0.5),
                cy,
                x + int(r * 0.9),
                cy + int(r * 0.4),
                x + int(r * 1.5),
                cy - int(r * 0.4),
            ),
            fill="white",
            width=stroke,
        )
        _text(
            draw,
            (x + int(56 * unit), cy),
            screen.result,
            width - int(56 * unit),
            size,
            fill=_DONE,
        )


def load_step_visuals(
    step_list_file: Path,
    spoken: list[dict[str, Any]] | None,
    out_dir: Path,
    *,
    size: tuple[int, int],
    video_end: float,
    opening_max_sec: float,
    recap_sec: float,
) -> list[tuple[Path, float, float]]:
    """Each screen's image and its times, drawn into `out_dir`; none on any gap.

    No step list, no word timings, or a step not found leaves the render's
    visuals as they are.
    """
    if not step_list_file.exists():
        return []
    if not spoken:
        logger.info("Step visuals skipped: no word timings")
        return []
    try:
        data = json.loads(step_list_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Step visuals skipped: unreadable %s: %s", step_list_file, exc)
        return []
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list) or not steps:
        return []
    steps = [s for s in steps if isinstance(s, dict)]
    screens = plan_step_visuals(
        steps,
        spoken,
        start_screen=str(data.get("start_screen") or ""),
        video_end=video_end,
        opening_max_sec=opening_max_sec,
        recap_sec=recap_sec,
    )
    drawn: list[tuple[Path, float, float]] = []
    for i, screen in enumerate(screens):
        path = out_dir / f"step_visual_{i}.png"
        try:
            draw_screen(screen, size).save(path)
        except OSError as exc:
            logger.warning("Step visuals skipped: could not write %s: %s", path, exc)
            return []
        drawn.append((path, screen.start, screen.end))
    if drawn:
        logger.info(
            "Step visuals: %d screen(s) from %.2fs to %.2fs",
            len(drawn),
            drawn[0][1],
            drawn[-1][2],
        )
    return drawn
