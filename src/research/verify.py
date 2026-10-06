"""Grounded verification of the sampled scripts (REQ-OPS-111).

One Gemini call with Google Search per topic script asks for a verdict per
step and claim, with the page that settles it. A verdict with no source
counts as unverified: a model judging its own kind of output without a
document is the failure the fact check already showed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).with_name("verify_prompt.md")
VERDICTS = ("correct", "wrong", "outdated", "unverified")


def parse_verdicts(answer: str | None) -> list[dict[str, str]] | None:
    """The model's verdicts, each with a source, or None when unreadable."""
    if not answer:
        return None
    # A grounded answer is free text: take the first complete JSON array,
    # skipping bracketed prose before it ("Here is [the] list") and after it.
    raw = None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", answer):
        try:
            found, _ = decoder.raw_decode(answer, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(found, list) and (not found or isinstance(found[0], dict)):
            raw = found
            break
    if raw is None:
        return None
    out = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("claim"), str):
            continue
        verdict = str(item.get("verdict", "")).strip().lower()
        source = str(item.get("source") or "").strip()
        if verdict not in VERDICTS or not source.startswith("http"):
            # Without a document behind it, a verdict is the model's opinion.
            verdict = "unverified"
        out.append(
            {
                "claim": item["claim"],
                "verdict": verdict,
                "source": source,
                "quote": str(item.get("quote") or ""),
                "correction": str(item.get("correction") or ""),
            }
        )
    return out


async def verify_script(
    title: str, script: str, *, api_key: str, model: str, timeout: float
) -> dict[str, Any]:
    """Verdicts for one script, or the error that stopped the call."""
    try:
        from google import genai
        from google.genai import errors as genai_errors

        client = genai.Client(api_key=api_key)
        config = genai.types.GenerateContentConfig(
            tools=[genai.types.Tool(google_search=genai.types.GoogleSearch())],
            temperature=0.0,
        )
    except (ImportError, ValueError, OSError, RuntimeError) as e:
        return {"error": f"verification unavailable: {e}"}
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(TITLE=title, SCRIPT=script)
    try:
        async with asyncio.timeout(timeout):
            answer = await client.aio.models.generate_content(
                model=model, contents=prompt, config=config
            )
    except (
        aiohttp.ClientError,
        TimeoutError,
        OSError,
        ValueError,
        RuntimeError,
        genai_errors.APIError,
    ) as e:
        return {"error": f"verification call failed: {e}"}
    finally:
        aclose = getattr(client.aio, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except (OSError, RuntimeError) as e:  # closing is best effort
                logger.debug("Closing the verification client failed: %s", e)
    verdicts = parse_verdicts(answer.text)
    if verdicts is None:
        return {"error": "verification answer was not readable JSON"}
    return {"verdicts": verdicts}


def tally(verdicts: list[dict[str, str]]) -> dict[str, int]:
    return {v: sum(d["verdict"] == v for d in verdicts) for v in VERDICTS}


async def run_verify(
    records: list[dict[str, Any]], *, api_key: str, model: str, timeout: float
) -> list[dict[str, Any]]:
    """Verify every topic sample that has a script; others are skipped.

    Products are checked against their listing by the pipeline's own fact
    check, which the sample already records.
    """
    out = []
    for r in records:
        if r.get("kind") != "topic" or not r.get("script"):
            continue
        result = await verify_script(
            r["title"], r["script"], api_key=api_key, model=model, timeout=timeout
        )
        if "verdicts" in result:
            result["tally"] = tally(result["verdicts"])
        out.append(
            {"variant": r["variant"], "id": r["id"], "title": r["title"], **result}
        )
    return out
