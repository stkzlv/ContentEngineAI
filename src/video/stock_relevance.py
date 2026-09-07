"""Judge stock candidates by what the photo shows, not what its caption says.

Every text signal for picking stock footage was measured and refuted on #307:
word overlap, rarity-weighted overlap and the provider's own rank all put an
oriental tea box with a decorative fan first for `laptop fan always on`,
because its caption honestly contains both words. Scoring the *thumbnail*
with the multimodal model separates them (#341).

Measured on the full 80-candidate pools the library returns (#341): for a
concrete subject, fits are common and provider rank is noise -- 31 of 80 for
`laptop fan clogged dust`, at a median rank near 45, so a rank-based
shortlist would have held two of them. For an abstract subject, fits are
scarce -- three or four per pool for the wifi phrases -- and a random sample
of eight expects to land none, while the judge takes all of them. So the
whole pool is scored, never a shortlist, and the judgement is against the
script with the query naming the moment, which is more lenient than the
query alone in the right direction: a heat-pipe photo scores 0 against
`dried out thermal paste` and 2 against the script it serves.

Nothing here raises. A relevance check that can break a render is worse than
a mismatched shot, so a failed judgement is an unknown score and a pool with
no known scores falls back to the random sample the caller already had.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import TYPE_CHECKING, Any

import aiohttp

if TYPE_CHECKING:
    from src.video.config.llm_settings import StockRelevanceConfig

logger = logging.getLogger(__name__)

_PROMPT = (
    "You choose stock photos for a short how-to video. Score this photo's "
    "fitness 0-3 for the moment of the SCRIPT that the SEARCH QUERY was "
    "written for: 0 wrong subject, 1 loosely related, 2 fits, 3 exactly what "
    "to show.\nSEARCH QUERY: {query}\nSCRIPT: {script}\n"
    'Answer with JSON only: {{"score": n}}'
)
_SCORE_RE = re.compile(r"[0-3]")
UNKNOWN = -1  # sorts after every known score, so a failed judgement is last resort


def parse_score(text: str | None) -> int | None:
    """The 0-3 the model answered, or None when it answered something else."""
    if not text:
        return None
    body = text.strip().strip("`").removeprefix("json").strip()
    try:
        value = json.loads(body).get("score")
        if isinstance(value, int | float) and 0 <= value <= 3:
            return int(value)
    except (ValueError, AttributeError, TypeError):
        pass
    match = _SCORE_RE.search(body)
    return int(match.group()) if match else None


async def score_candidates(
    candidates: list[dict[str, Any]],
    query: str,
    script: str,
    *,
    api_key: str,
    settings: StockRelevanceConfig,
    session: Any,
) -> list[int | None]:
    """One score per candidate, in order; None where the judgement failed.

    Candidates past `max_candidates`, and any without a thumbnail, are
    unknown rather than fetched at full size to be judged.
    """
    if not candidates:
        return []
    unknown: list[int | None] = [None] * len(candidates)
    try:
        from google import genai
        from google.genai import errors as genai_errors

        client = genai.Client(api_key=api_key)
        config = genai.types.GenerateContentConfig(
            max_output_tokens=20,
            temperature=0.0,
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        )
    except (ImportError, ValueError, OSError, RuntimeError) as e:
        # A SOCKS proxy without socksio raises ImportError from the
        # constructor alone; the judge is then unavailable, not broken.
        logger.warning("Stock relevance judge unavailable: %s", e)
        return unknown
    prompt = _PROMPT.format(query=query, script=script)
    semaphore = asyncio.Semaphore(settings.concurrency)

    async def judge(candidate: dict[str, Any]) -> int | None:
        url = candidate.get("thumbnail")
        if not url:
            return None
        try:
            async with semaphore, asyncio.timeout(settings.timeout_seconds):
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    data = await response.read()
                    mime = response.content_type or "image/jpeg"
                content = genai.types.Content(
                    role="user",
                    parts=[
                        genai.types.Part.from_bytes(data=data, mime_type=mime),
                        genai.types.Part.from_text(text=prompt),
                    ],
                )
                answer = await client.aio.models.generate_content(
                    model=settings.model, config=config, contents=content
                )
        except (
            aiohttp.ClientError,
            TimeoutError,
            OSError,
            ValueError,
            RuntimeError,
            genai_errors.APIError,
        ) as e:
            logger.debug("Stock relevance judgement failed for %s: %s", url, e)
            return None
        return parse_score(answer.text)

    try:
        judged = await asyncio.gather(
            *(judge(c) for c in candidates[: settings.max_candidates])
        )
    finally:
        # One session per judge call; left open it is reported at collection.
        aclose = getattr(client.aio, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except (OSError, RuntimeError) as e:  # closing is best effort
                logger.debug("Closing the judge's client failed: %s", e)
    return list(judged) + [None] * (len(candidates) - len(judged))


def select_by_relevance(
    candidates: list[dict[str, Any]],
    scores: list[int | None],
    count: int,
    min_score: int,
    rng: random.Random | None = None,
) -> list[dict[str, Any]] | None:
    """The `count` best-scoring candidates, each stamped with its score.

    Random within a score, so repeated renders still vary. Candidates below
    `min_score` are used only when there are not enough above it, because a
    stock shortfall skips the render and a loosely related shot is the better
    loss; a warning names how many. Returns None when no candidate has a
    known score, which is the signal to keep the caller's own sampling.
    """
    if not any(s is not None for s in scores):
        return None
    draw = rng.random if rng is not None else random.random
    keyed = [
        (s if s is not None else UNKNOWN, draw(), c)
        for c, s in zip(candidates, scores, strict=True)
    ]
    keyed.sort(key=lambda k: (-k[0], k[1]))
    above = [k for k in keyed if k[0] >= min_score]
    chosen = above[:count]
    if len(chosen) < count:
        fill = [k for k in keyed if k[0] < min_score][: count - len(chosen)]
        if fill:
            logger.warning(
                "Stock relevance: %d of %d judged candidates scored at least %d; "
                "filling %d from below the floor",
                len(above),
                sum(1 for s in scores if s is not None),
                min_score,
                len(fill),
            )
        chosen += fill
    result = []
    for score, _, candidate in chosen:
        item = dict(candidate)
        item["score"] = None if score == UNKNOWN else score
        result.append(item)
    return result
