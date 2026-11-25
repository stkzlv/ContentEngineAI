"""Voice delivery segments through the Gemini API speech model (REQ-CNT-181).

One request per segment to the Interactions endpoint: the text with its
events as inline tags and a `speech_metadata` annotation holding its style.
The WAV replies are joined with each segment's planned gap.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiohttp

from src.video.delivery_plan import DeliverySegment, text_with_tags

logger = logging.getLogger(__name__)


class GeminiApiTTSError(RuntimeError):
    """A segment could not be voiced; the whole voice fails."""


def request_body(
    text: str, style: str, voice: str | None, model: str
) -> dict[str, Any]:
    content: dict[str, Any] = {"type": "text", "text": text}
    if style:
        content["annotations"] = [{"type": "speech_metadata", "style": style}]
    body: dict[str, Any] = {
        "model": model,
        "input": [{"type": "user_input", "content": [content]}],
        "response_format": {"type": "audio"},
    }
    if voice:
        body["generation_config"] = {"speech_config": [{"voice": voice}]}
    return body


def audio_of(reply: dict[str, Any]) -> bytes:
    """The WAV bytes of an Interactions reply (`steps[].content[].data`)."""
    for step in reply.get("steps") or []:
        for part in (step or {}).get("content") or []:
            data = (part or {}).get("data")
            if isinstance(data, str) and data:
                try:
                    return base64.b64decode(data, validate=True)
                except (binascii.Error, ValueError) as e:
                    raise GeminiApiTTSError(f"undecodable audio: {e}") from e
    raise GeminiApiTTSError(f"no audio in reply (status {reply.get('status')!r})")


def join_wavs(parts: list[bytes], gaps_ms: list[int], output_path: Path) -> None:
    """Concatenate WAV parts with silence after each, all one format."""
    params: tuple[int, int, int] | None = None
    frames: list[bytes] = []
    for i, part in enumerate(parts):
        try:
            with wave.open(io.BytesIO(part)) as w:
                p = (w.getnchannels(), w.getsampwidth(), w.getframerate())
                data = w.readframes(w.getnframes())
        except (wave.Error, EOFError) as e:
            raise GeminiApiTTSError(f"segment {i}: not a WAV reply: {e}") from e
        if params is None:
            params = p
        elif p != params:
            raise GeminiApiTTSError(f"segment {i}: format {p} differs from {params}")
        frames.append(data)
        if i < len(parts) - 1 and gaps_ms[i] > 0:
            channels, width, rate = params
            frames.append(b"\x00" * (rate * gaps_ms[i] // 1000) * channels * width)
    if params is None:
        raise GeminiApiTTSError("no segments")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(params[0])
        out.setsampwidth(params[1])
        out.setframerate(params[2])
        out.writeframes(b"".join(frames))


async def _voice_one(
    session: aiohttp.ClientSession,
    endpoint: str,
    api_key: str,
    body: dict[str, Any],
    retries: int,
    retry_delay_sec: float,
) -> bytes:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with session.post(
                endpoint,
                json=body,
                headers={"x-goog-api-key": api_key},
            ) as resp:
                if resp.status != 200:
                    detail = (await resp.text())[:300]
                    raise GeminiApiTTSError(f"HTTP {resp.status}: {detail}")
                return audio_of(await resp.json())
        except (aiohttp.ClientError, TimeoutError, ValueError, GeminiApiTTSError) as e:
            last = e
            logger.warning("Gemini API speech attempt %d failed: %s", attempt + 1, e)
            if attempt < retries:
                await asyncio.sleep(retry_delay_sec)
    raise GeminiApiTTSError(str(last))


async def synthesize_segments(
    segments: list[DeliverySegment],
    output_path: Path,
    *,
    api_key: str,
    model: str,
    voice: str | None,
    endpoint: str,
    timeout_sec: float = 120,
    retries: int = 1,
    retry_delay_sec: float = 2,
    prepare: Callable[[str], str] | None = None,
) -> Path:
    """Voice each segment and join them; raises GeminiApiTTSError on failure.

    `prepare` rewrites each segment's tagged text for the voice (the
    normalisation the other providers get) before it is sent.
    """
    if not segments:
        raise GeminiApiTTSError("no segments")
    parts: list[bytes] = []
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, segment in enumerate(segments):
            text = text_with_tags(segment)
            if prepare is not None:
                text = prepare(text)
            body = request_body(text, segment.style, voice, model)
            parts.append(
                await _voice_one(
                    session,
                    endpoint,
                    api_key,
                    body,
                    retries,
                    retry_delay_sec,
                )
            )
            logger.debug("Voiced segment %d of %d", i + 1, len(segments))
    join_wavs(parts, [s.gap_after_ms for s in segments], output_path)
    return output_path
