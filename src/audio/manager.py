"""Audio manager that orchestrates provider chain with local file fallback."""

import asyncio
import logging
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

import aiohttp

from src.utils import ensure_dirs_exist
from src.utils.circuit_breaker import CircuitBreakerError

from .base import AudioTrack, BaseAudioProvider

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")


def query_terms(query: str) -> list[str]:
    """The words of a search query, lower-cased, in order, without repeats."""
    seen: list[str] = []
    for word in _WORD.findall(query.lower()):
        if word not in seen:
            seen.append(word)
    return seen


def matched_terms(track: AudioTrack, terms: list[str]) -> list[str]:
    """The query terms the track's title or tags carry.

    A term matches a word it begins ("calm" matches "calming", "chill"
    matches "chillout"), so a provider's compound tags count.
    """
    words = set(_WORD.findall(track.name.lower()))
    for tag in track.tags:
        words.update(_WORD.findall(tag.lower()))
    return [term for term in terms if any(word.startswith(term) for word in words)]


def rank_by_mood(
    tracks: list[AudioTrack], terms: list[str]
) -> list[tuple[AudioTrack, list[str]]]:
    """Tracks that match at least one term, most matches first, ties shuffled.

    A provider that ranks by rating or popularity alone returns whatever
    high-rated sound the text search loosely matched, which is how a
    "calm ambient instrumental" query produced a drill instrumental. With
    no terms every track passes, in random order.
    """
    scored = [(track, matched_terms(track, terms)) for track in tracks]
    if terms:
        scored = [(track, matched) for track, matched in scored if matched]
    random.shuffle(scored)
    scored.sort(key=lambda pair: len(pair[1]), reverse=True)
    return scored


class BudgetSpentError(Exception):
    """The chain's time budget ran out; the caller falls back to local files."""


class AudioManager:
    """Try each configured provider in order, fall back to local files.

    `budget_sec` bounds the whole chain: every search and download runs
    inside the time left, and when it is spent the local fallback is used.
    The step once spent 208 s of a 304 s render on failed refreshes and
    stalled downloads; the budget sits under the per-step warning threshold.
    """

    def __init__(
        self,
        providers: list[BaseAudioProvider],
        local_paths: list[Path] | None = None,
        budget_sec: float | None = None,
    ) -> None:
        self._providers = providers
        self._local_paths = local_paths or []
        self._budget_sec = budget_sec
        self._deadline: float | None = None

    def _remaining(self) -> float | None:
        if self._deadline is None:
            return None
        return self._deadline - time.monotonic()

    async def _within_budget(self, coroutine, what: str):
        """Run a search or download inside the time left, or stop the chain."""
        remaining = self._remaining()
        if remaining is None:
            return await coroutine
        if remaining <= 0:
            coroutine.close()
            raise BudgetSpentError(what)
        try:
            return await asyncio.wait_for(coroutine, timeout=remaining)
        except TimeoutError as exc:
            # A provider's own timeout with budget left is its problem and
            # the caller moves to the next candidate; only the deadline
            # passing ends the chain.
            left = self._remaining()
            if left is not None and left <= 0:
                raise BudgetSpentError(what) from exc
            raise

    async def find_music(
        self,
        query: str,
        min_duration: float,
        max_duration: float,
        max_results: int,
        output_dir: Path,
        session: aiohttp.ClientSession,
    ) -> dict[str, Any] | None:
        """Search providers in order, download first suitable track.

        Returns attribution dict or None if nothing found.
        """
        t0 = time.monotonic()
        self._deadline = t0 + self._budget_sec if self._budget_sec else None
        tried_providers: list[str] = []

        for provider in self._providers:
            tried_providers.append(provider.provider_name)
            try:
                result = await self._try_provider(
                    provider,
                    query,
                    min_duration,
                    max_duration,
                    max_results,
                    output_dir,
                    session,
                )
                if result:
                    self._log_summary(
                        result, query, tried_providers, time.monotonic() - t0
                    )
                    return result
            except BudgetSpentError as exc:
                logger.warning(
                    "Music budget of %.0fs spent during %s, falling back to local "
                    "files",
                    self._budget_sec or 0,
                    exc,
                )
                break
            except CircuitBreakerError:
                logger.warning(
                    "Circuit breaker open for %s, skipping",
                    provider.provider_name,
                )
            except (
                RuntimeError,
                OSError,
                TimeoutError,
                aiohttp.ClientError,
            ) as exc:
                logger.warning(
                    "Provider %s failed: %s",
                    provider.provider_name,
                    exc,
                )

        fallback = self._try_local_fallback(output_dir)
        if fallback:
            tried_providers.append("local")
        self._log_summary(fallback, query, tried_providers, time.monotonic() - t0)
        return fallback

    @staticmethod
    def _log_summary(
        result: dict[str, Any] | None,
        query: str,
        tried: list[str],
        elapsed: float,
    ) -> None:
        logger.info("--- AUDIO SUMMARY ---")
        if result:
            logger.info(
                "Provider: %s (query: %s)",
                result.get("source", "unknown"),
                query,
            )
            logger.info(
                "Track: %s by %s",
                result.get("name", "unknown"),
                result.get("author", "unknown"),
            )
            if "matched_terms" in result:
                logger.info("Matched: %s", ", ".join(result["matched_terms"]) or "none")
        else:
            logger.info("Result: no track found (tried: %s)", ", ".join(tried))
        logger.info("Duration: %.1fs", elapsed)
        logger.info("---")

    async def _try_provider(
        self,
        provider: BaseAudioProvider,
        query: str,
        min_duration: float,
        max_duration: float,
        max_results: int,
        output_dir: Path,
        session: aiohttp.ClientSession,
    ) -> dict[str, Any] | None:
        tracks = await self._within_budget(
            provider.search(
                query,
                min_duration,
                max_duration,
                max_results,
                session,
            ),
            f"{provider.provider_name} search",
        )
        if not tracks:
            logger.info(
                "No tracks from %s, trying next provider",
                provider.provider_name,
            )
            return None

        eligible = [t for t in tracks if t.duration >= min_duration]
        if not eligible:
            logger.info(
                "No tracks from %s meet min duration %.0fs",
                provider.provider_name,
                min_duration,
            )
            return None
        # Judge the provider on the terms it searched: Jamendo draws one of
        # its own queries, and "soft background" shares no word with the
        # query this chain was asked for.
        terms = query_terms(provider.last_query or query)
        ranked = rank_by_mood(eligible, terms)
        if not ranked:
            logger.info(
                "No track from %s matches the query terms (%s), trying next provider",
                provider.provider_name,
                ", ".join(terms),
            )
            return None

        for track, matched in ranked:
            logger.info(
                "Trying track '%s' (%.0fs) from %s, matches: %s",
                track.name,
                track.duration,
                provider.provider_name,
                ", ".join(matched) or "none",
            )
            try:
                result = await self._within_budget(
                    provider.download(track, output_dir, session),
                    f"{provider.provider_name} download of '{track.name}'",
                )
                if result:
                    _, found = result
                    attribution: dict[str, Any] = dict(found)
                    attribution["matched_terms"] = matched
                    return attribution
            except (RuntimeError, OSError, TimeoutError) as exc:
                logger.warning(
                    "Download failed for '%s' from %s: %s",
                    track.name,
                    provider.provider_name,
                    exc,
                )

        logger.info(
            "No suitable track downloaded from %s",
            provider.provider_name,
        )
        return None

    def _try_local_fallback(self, output_dir: Path) -> dict[str, Any] | None:
        existing = [p for p in self._local_paths if p.exists()]
        if not existing:
            logger.warning("No background music from any source.")
            return None

        local_path = random.choice(existing)  # noqa: S311
        ensure_dirs_exist(output_dir)
        dest_path = output_dir / local_path.name
        shutil.copy(local_path, dest_path)

        logger.info("Using local fallback: %s", local_path.name)
        return {
            "source": "Local",
            "type": "Music",
            "path": str(dest_path),
            "name": local_path.stem,
            "author": "Unknown",
            "license": "Local File",
            "url": "",
            "id": "",
        }
