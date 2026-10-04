"""Prefer clean product images over text-heavy seller infographics (design 0012).

Listing images are mostly marketing composites with dense overlaid text, and
the render draws captions and a hook headline on top. With curation on, each
scraped image is judged once by the multimodal model the stock relevance
judge uses, for the share of the frame covered by overlaid text, and the
text-heavy ones are dropped while enough others remain.

Nothing here raises. A failed judgement is unknown, and an unknown image is
never dropped: losing a usable image to a network error is worse than
keeping one with text on it.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from PIL import Image

logger = logging.getLogger(__name__)

# Measured on a ten-image listing: gemini-2.5-flash scored the four plain
# product shots 0.0 and the six marketing images 0.2-0.3 with this prompt;
# flash-lite counted the watch's own screen as text and could not separate
# them, so the model is a setting of its own.
_PROMPT = (
    "This is a product listing image. Estimate the share of the image area "
    "covered by text or graphics ADDED ON TOP of the photo by the seller: "
    "headlines, marketing copy, spec callouts, icons, badges, arrows, "
    "comparison panels. Text that belongs to the product itself does not "
    "count: a watch or phone screen, a printed label, packaging. A plain "
    "photo of the product scores 0. Also say whether the image is a "
    "composite of several photos or panels.\n"
    'Answer with JSON only: {"text_share": 0.0-1.0, "composite": true|false}'
)
_SHARE_RE = re.compile(r'"text_share"\s*:\s*([0-9.]+)')
# The long edge an image is sent at; enough to see text, cheap to send.
_JUDGE_EDGE = 768
_CACHE_SUFFIX = ".text_score.json"


@dataclass
class ImageScore:
    text_share: float | None
    composite: bool | None = None


def parse_judgement(text: str | None) -> ImageScore:
    """The model's answer, or an unknown score when it answered otherwise."""
    if not text:
        return ImageScore(None)
    body = text.strip().strip("`").removeprefix("json").strip()
    try:
        data = json.loads(body)
        share = data.get("text_share")
        composite = data.get("composite")
        if isinstance(share, int | float) and 0 <= share <= 1:
            return ImageScore(
                float(share), composite if isinstance(composite, bool) else None
            )
    except (ValueError, AttributeError, TypeError):
        pass
    match = _SHARE_RE.search(body)
    if match:
        try:
            share = float(match.group(1))
        except ValueError:
            return ImageScore(None)
        if 0 <= share <= 1:
            return ImageScore(share)
    return ImageScore(None)


def _cache_path(image: Path) -> Path:
    return image.with_name(image.name + _CACHE_SUFFIX)


def _stamp(image: Path) -> dict[str, int]:
    st = image.stat()
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def read_cached(image: Path) -> ImageScore | None:
    """The cached score, when it was taken of this exact file."""
    try:
        data = json.loads(_cache_path(image).read_text(encoding="utf-8"))
        if data.get("stamp") != _stamp(image):
            return None
        share = data.get("text_share")
        if share is None or not isinstance(share, int | float):
            return None
        composite = data.get("composite")
        return ImageScore(
            float(share), composite if isinstance(composite, bool) else None
        )
    except (OSError, ValueError, AttributeError):
        return None


def write_cache(image: Path, score: ImageScore) -> None:
    """Keep a known score beside the image; an unknown one is retried."""
    if score.text_share is None:
        return
    try:
        _cache_path(image).write_text(
            json.dumps(
                {
                    "text_share": score.text_share,
                    "composite": score.composite,
                    "stamp": _stamp(image),
                }
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.debug("Could not cache the text score of %s: %s", image.name, exc)


def _jpeg_bytes(image: Path) -> bytes:
    with Image.open(image) as source:
        rgb = source.convert("RGB")
    rgb.thumbnail((_JUDGE_EDGE, _JUDGE_EDGE))
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


async def score_images(
    images: list[Path],
    *,
    api_key: str,
    model: str,
    concurrency: int,
    timeout_seconds: int,
) -> list[ImageScore]:
    """One score per image, in order, from the cache or the model."""
    scores = [read_cached(image) for image in images]
    missing = [i for i, score in enumerate(scores) if score is None]
    if not missing:
        return [s or ImageScore(None) for s in scores]
    try:
        from google import genai
        from google.genai import errors as genai_errors

        client = genai.Client(api_key=api_key)
        config = genai.types.GenerateContentConfig(
            max_output_tokens=40,
            temperature=0.0,
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        )
    except (ImportError, ValueError, OSError, RuntimeError) as e:
        logger.warning("Image curation judge unavailable: %s", e)
        return [s or ImageScore(None) for s in scores]
    semaphore = asyncio.Semaphore(concurrency)

    async def judge(image: Path) -> ImageScore:
        try:
            data = await asyncio.to_thread(_jpeg_bytes, image)
            content = genai.types.Content(
                role="user",
                parts=[
                    genai.types.Part.from_bytes(data=data, mime_type="image/jpeg"),
                    genai.types.Part.from_text(text=_PROMPT),
                ],
            )
            async with semaphore, asyncio.timeout(timeout_seconds):
                answer = await client.aio.models.generate_content(
                    model=model, config=config, contents=content
                )
        except (
            aiohttp.ClientError,
            TimeoutError,
            OSError,
            ValueError,
            RuntimeError,
            Image.DecompressionBombError,
            genai_errors.APIError,
        ) as e:
            logger.debug("Image curation judgement failed for %s: %s", image, e)
            return ImageScore(None)
        score = parse_judgement(answer.text)
        write_cache(image, score)
        return score

    try:
        judged = await asyncio.gather(*(judge(images[i]) for i in missing))
    finally:
        aclose = getattr(client.aio, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except (OSError, RuntimeError) as e:  # closing is best effort
                logger.debug("Closing the curation judge's client failed: %s", e)
    for i, score in zip(missing, judged, strict=True):
        scores[i] = score
    return [s or ImageScore(None) for s in scores]


def curate(
    images: list[Path],
    scores: list[ImageScore],
    max_text_share: float,
    keep_at_least: int,
) -> tuple[list[Path], list[Path]]:
    """(kept, dropped): clean images first, text-heavy ones only to fill.

    An image is clean at or below `max_text_share`. Text-heavy images are
    dropped while at least `keep_at_least` others remain; otherwise the least
    text-heavy of them are kept to reach it. An unknown score is never
    dropped, and sorts after the clean ones.
    """
    paired = list(zip(images, scores, strict=True))
    clean = sorted(
        (
            p
            for p in paired
            if p[1].text_share is not None and p[1].text_share <= max_text_share
        ),
        key=lambda p: p[1].text_share or 0.0,
    )
    unknown = [p for p in paired if p[1].text_share is None]
    heavy = sorted(
        (
            p
            for p in paired
            if p[1].text_share is not None and p[1].text_share > max_text_share
        ),
        key=lambda p: p[1].text_share or 0.0,
    )
    fill = max(0, keep_at_least - len(clean) - len(unknown))
    kept = [p[0] for p in clean + unknown + heavy[:fill]]
    dropped = [p[0] for p in heavy[fill:]]
    return kept, dropped
