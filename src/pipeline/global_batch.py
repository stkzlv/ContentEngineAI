"""Global batch pipeline orchestrator.

This module orchestrates the complete end-to-end workflow from scraping
products to generating and publishing promotional videos in four sequential phases:

1. Scraping Phase: Process product IDs/keywords through Amazon scraper
2. Handoff Phase: Discover products ready for video production
3. Video Production Phase: Generate videos for ready products
4. Publishing Phase: Publish videos to social media platforms (optional)

The orchestrator treats scraper, producer, and publisher as black boxes,
coordinating their execution without modifying their internals.

Usage:
    from src.pipeline.global_batch import GlobalPipelineOrchestrator
    from src.pipeline.config import GlobalBatchConfig

    config = GlobalBatchConfig(...)
    orchestrator = GlobalPipelineOrchestrator(config)
    summary = await orchestrator.run_pipeline()
"""

import asyncio
import logging
import re
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.pipeline.config import (
    GlobalBatchConfig,
    PipelinePhase,
    PipelineState,
    PipelineSummary,
    ProductionPhaseSummary,
    PublishingPhaseSummary,
    ScrapingPhaseSummary,
    clear_pipeline_state,
    load_pipeline_state,
    save_pipeline_state,
)
from src.pipeline.phases.production import run_production_phase
from src.pipeline.phases.publishing import run_publishing_phase
from src.pipeline.phases.scraping import run_scraping_phase
from src.pipeline.webhooks import WebhookConfig, WebhookNotifier
from src.scraper.amazon.models import ProductData
from src.video.config_adapter import load_video_config_modular
from src.video.producer.shared_cli import (
    subtitle_render_overrides,
)
from src.video.producer.topic_input import (
    TOPIC_ID_PREFIX,
    materialise_topics,
    topic_product_id,
)
from src.video.producer.utils import (
    eligible_random_profiles,
)

# Every function-local import below now genuinely defers something: the
# publisher (the Zernio SDK), the scraper (Botasaurus and its Chromium
# driver) and the producer orchestration (Whisper, torch) are each absent
# from this module's import closure until a path that needs them runs. That
# holds only because the package `__init__` re-exports resolve lazily --
# they used to load all three regardless -- so hoisting one of these to
# module scope puts its whole stack back into `--help` and every dry run.
# `tests/utils/test_import_closure.py` fails if that happens.

if TYPE_CHECKING:
    from src.publisher.models import PublisherConfig

logger = logging.getLogger(__name__)


# Stands in for the provider credential when only the settings are being
# read. Never sent anywhere: the publishing phase takes its key from the
# environment and refuses to run without one.
_NO_CREDENTIAL = "settings-read-no-credential"


@lru_cache(maxsize=1)
def _publisher_settings() -> "PublisherConfig":
    """The publisher config, typed, loaded once, resolved absolutely.

    Four places in this module used to open `config/publisher.yaml` with a
    relative path and read raw keys out of it. Two consequences, both silent:
    a run from anywhere but the repository root got the hardcoded fallbacks,
    and each site carried its own default, so `immediate_publish` fell back to
    True here against a shipped false. Reading one typed object also removes
    the hand-parsing that let `tiktok_settings` go missing from the batch for
    several releases while `single` had it.

    A config that fails to load is not caught here, deliberately. Swallowing
    it and returning the dataclass defaults reads like caution and is the
    opposite: one typo'd platform name would discard the whole file, and the
    run would then publish on the spot -- to whichever platforms the defaults
    name, with the first comment, the disclosures, the stagger and the
    retention policy all silently reverted at the same time. A missing file
    needs no fallback either: the loader returns an empty mapping for that
    without raising.

    The one exception is an absent credential, which says nothing about
    whether the rest of the file is usable. The batch reads settings here and
    takes its key from `LATE_API_KEY` at publish time, having checked it
    before the run started. So it retries with a placeholder rather than
    falling back to defaults, and keeps every configured setting; a typo'd
    platform still raises out of the second call.
    """
    from src.publisher.config import MissingApiKeyError, load_publisher_config

    try:
        return load_publisher_config()
    except MissingApiKeyError as exc:
        logger.debug("No publisher credential while reading settings: %s", exc)
        return load_publisher_config(cli_overrides={"api_key": _NO_CREDENTIAL})


# A run directory is a scraped product (ASIN-shaped, or a TEST fixture) or a
# topic. Topics were absent, so `--clean` walked past them and the dry-run plan
# under-reported what the run would remove.
_RUN_DIR_PATTERN = re.compile(r"^([A-Z0-9]{10}|TEST[A-Z0-9]+|topic-[a-z0-9-]+)$")


def resumed_record_kinds(config: "GlobalBatchConfig") -> tuple[bool, bool]:
    """What kinds of record a `--resume` is picking up: (topics, products).

    Topics are not persisted: the identifier carries a one-way digest of the
    title, so the specs cannot be read back out of the state. The ids can, and
    the prefix tells the two kinds apart.

    Both halves are needed, not just the first. A run can carry topics and
    scraped products together, and the config's own `keywords` cannot settle it
    -- a resume inherits the configured keywords whatever it is resuming, and
    the completed scraping phase already ignored them. Reading the state is the
    only way to know whether this resume has products in it.

    Separated from `main` so it can be tested. Left inline it was the one
    piece of this feature no test touched, and deleting it left the suite
    green while every resumed topic rendered under a product profile.
    """
    if not config.resume or config.topics:
        return (False, False)

    saved = load_pipeline_state(config.outputs_dir)
    if saved is None:
        return (False, False)

    ids = saved.scraping_completed_products or []
    topics = any(pid.startswith(TOPIC_ID_PREFIX) for pid in ids)
    products = any(not pid.startswith(TOPIC_ID_PREFIX) for pid in ids)
    return (topics, products)


def apply_resume_record_kinds(config: "GlobalBatchConfig") -> None:
    """Stamp what a `--resume` is picking up onto the config.

    Both flags, not just the first. `topics_resume` narrows the profile pool
    to stock-sourced profiles, and `resume_has_products` is what stops that
    narrowing on a resume that also carries scraped products -- which would
    otherwise render them from generic footage, ignoring the photography
    scraped for them.

    A function rather than two lines in `main` so the values can be tested.
    Inline, the only reachable guard was an AST check that an assignment
    existed, which passes just as well when the assignment is a constant.
    """
    resuming_topics, resuming_products = resumed_record_kinds(config)
    if not resuming_topics:
        return

    logger.info("Resuming a topics run (recognised from saved state)")
    config.topics_resume = True
    config.resume_has_products = resuming_products


def _named_run_ids(config: "GlobalBatchConfig") -> list[str]:
    """The run directories this invocation named, if it named any.

    Topics are a named input exactly like product ids, and were falling into
    the unnamed branch below, which removes every run directory in `outputs/`.
    A `--topic X --clean` run would delete every scraped product the machine
    held, along with any rendered-but-unpublished video in them.

    Keywords name nothing: which products they produce is not known until the
    search runs, so a run carrying them is a sweep and the answer is the empty
    list. That has to be checked first, and both other kinds have to be
    unioned. Returning the first non-empty kind meant a run with a topic and
    keywords -- which the bundled config now produces with no flags at all --
    named only the topic, and `--clean` silently spared every product
    directory the operator asked it to remove.
    """
    if config.keywords:
        return []

    named = list(config.product_ids)
    named += [topic_product_id(spec.title) for spec in config.topics]
    return named


def _clean_targets(outputs_dir: Path, product_ids: list[str] | None) -> list[Path]:
    """The product directories `--clean` would remove.

    Shared with the dry-run plan so the preview cannot describe something
    other than what the run does.
    """
    if not outputs_dir.exists():
        return []
    if product_ids:
        # Deduplicated: the caller may name a product twice, and hoisting
        # the selection ahead of the deletion means the second occurrence
        # would try to remove a directory the first one already took.
        return [
            outputs_dir / pid
            for pid in dict.fromkeys(product_ids)
            if (outputs_dir / pid).is_dir()
        ]
    return sorted(
        item
        for item in outputs_dir.iterdir()
        if item.is_dir() and _RUN_DIR_PATTERN.match(item.name)
    )


def _merge_scraping_summaries(
    *summaries: ScrapingPhaseSummary | None,
) -> ScrapingPhaseSummary:
    """Fold the topic and scrape phases into the one summary the state holds.

    Both write into the same directory shape and both feed the same handoff, so
    downstream reads one list of prepared ids. Reported as a single phase
    because that is what resume, the saved state and the phase summaries
    already understand.

    Duration is summed rather than maxed: the phases run one after the other.
    """
    present = [s for s in summaries if s is not None]
    if len(present) == 1:
        return present[0]

    media: dict[str, int] = {}
    for summary in present:
        for key, value in summary.media_stats.items():
            media[key] = media.get(key, 0) + value

    return ScrapingPhaseSummary(
        total_attempted=sum(s.total_attempted for s in present),
        successful=sum(s.successful for s in present),
        failed=sum(s.failed for s in present),
        successful_products=[p for s in present for p in s.successful_products],
        failed_products=[p for s in present for p in s.failed_products],
        dead_queries=[q for s in present for q in s.dead_queries],
        throttled_inputs=[i for s in present for i in s.throttled_inputs],
        media_stats=media,
        duration_sec=sum(s.duration_sec for s in present),
    )


class GlobalPipelineOrchestrator:
    """Orchestrates scraping, video production, and publishing phases sequentially.

    Coordinates the complete pipeline from scraping to publishing,
    treating scraper, producer, and publisher as black boxes and managing
    the handoff between phases.

    Supports resume capability through state persistence, allowing interrupted
    pipelines to continue from the last successful checkpoint.

    Attributes
    ----------
        config: Unified pipeline configuration
        state: Pipeline state for tracking progress and enabling resume
        webhook_notifier: Optional webhook notifier for event notifications

    """

    def __init__(
        self,
        config: GlobalBatchConfig,
        state: PipelineState | None = None,
        webhook_notifier: WebhookNotifier | None = None,
        video_config: Any = None,
    ):
        """Initialize orchestrator with unified configuration.

        Args:
        ----
            config: Global batch configuration with scraper and producer settings
            state: Optional pipeline state for resume capability
            webhook_notifier: Optional webhook notifier for pipeline events
            video_config: Video configuration for profile-aware scraper validation

        """
        self.config = config
        self.state = state or PipelineState.create_new(config)
        self.webhook_notifier = webhook_notifier
        self.video_config = video_config

    def _save_state(self) -> None:
        """Save current pipeline state to disk."""
        save_pipeline_state(self.state, self.config.outputs_dir)

    def _build_cli_overrides(self) -> dict[str, Any] | None:
        """Build CLI overrides dict from pipeline config.

        Keys here must match the dotted override keys consumed by
        ``VideoConfig.get_profile_merged_settings`` — keep this in sync with
        ``src.video.producer.cli._build_cli_overrides`` per the
        Module/Batch Alignment Rule in CLAUDE.md.
        """
        overrides: dict[str, Any] = {}
        if self.config.voice_profile:
            overrides["voice_profile"] = self.config.voice_profile
        if self.config.script_template:
            overrides["script_template"] = self.config.script_template
        if self.config.cta:
            overrides["cta"] = self.config.cta
        if self.config.pillar:
            overrides["pillar"] = self.config.pillar
        # Shared subtitle-engine overrides: the batch config carries the same
        # attribute names the producer's argparse namespace does, so one
        # helper serves both (declare-and-apply lives in shared_cli).
        overrides.update(subtitle_render_overrides(self.config))
        return overrides or None

    def _resolve_profile_uses_videos(self) -> bool | None:
        """Check if the target profile(s) use scraped videos.

        Returns False if any profile in the selection doesn't use videos
        (strictest requirement wins). Returns None when no profile info
        is available.
        """
        if not self.video_config:
            return None

        if self.config.profile:
            profile = self.video_config.video_profiles.get(self.config.profile)
            if profile:
                return bool(profile.use_scraped_videos)
            return None

        if self.config.random_profile:
            pool = self.config.profile_pool or eligible_random_profiles(
                self.video_config
            )
            for name in pool:
                profile = self.video_config.video_profiles.get(name)
                if profile and not profile.use_scraped_videos:
                    return False
            return True

        return None

    async def _notify_webhook(
        self,
        event: str,
        data: dict[str, Any],
    ) -> None:
        """Send webhook notification (non-blocking).

        Args:
        ----
            event: Event type (e.g., "phase.complete")
            data: Event data payload

        """
        if self.webhook_notifier and self.webhook_notifier.is_ready():
            try:
                await self.webhook_notifier.notify(event, data)
            except Exception as e:
                # Never let webhook failures affect the pipeline
                logger.warning("Webhook notification failed: %s", e)

    def _resumed_topic_ids(self) -> list[str]:
        """Topic ids from the saved state, for a `--resume` plan.

        `config.topics` is empty on a resume, so the plan has nothing to name
        without reading them back.
        """
        if not self.config.topics_resume:
            return []

        saved = load_pipeline_state(self.config.outputs_dir)
        if saved is None:
            return []
        return [
            pid
            for pid in (saved.scraping_completed_products or [])
            if pid.startswith(TOPIC_ID_PREFIX)
        ]

    def display_execution_plan(self, video_config: Any) -> None:
        """Print the dry-run plan; the body lives in `src.pipeline.plan`."""
        from src.pipeline.plan import display_execution_plan

        display_execution_plan(self.config, video_config, self._resumed_topic_ids)

    async def run_pipeline(self) -> PipelineSummary:
        """Execute complete pipeline: scrape → handoff → produce → publish.

        Orchestrates four sequential phases:
        1. Scraping Phase: Scrape products using configured inputs
        2. Handoff Phase: Discover products ready for video production
        3. Video Production Phase: Generate videos for ready products
        4. Publishing Phase: Publish produced videos to social media platforms

        Supports resume capability: if --resume flag is set and state file exists,
        skips already-completed phases and products.

        Returns
        -------
            PipelineSummary with aggregated statistics from all phases

        """
        pipeline_start = time.time()

        # Log resume status
        if self.config.resume:
            logger.info("Resuming pipeline run: %s", self.state.run_id)
            logger.info("  Started: %s", self.state.started_at)
            completed = ", ".join(self.state.completed_phases) or "none"
            logger.info("  Completed phases: %s", completed)

        # Save initial state
        self._save_state()

        # Phase 1: Scraping
        if self.state.is_phase_completed(PipelinePhase.SCRAPING):
            logger.info("=" * 80)
            logger.info("SCRAPING PHASE (SKIPPED - Already completed)")
            logger.info("=" * 80)
            # Reconstruct summary from state
            scraping_summary = ScrapingPhaseSummary(
                **self.state.scraping_summary  # type: ignore[arg-type]
            )
            logger.info(
                "Using cached results: %s successful, %s failed",
                scraping_summary.successful,
                scraping_summary.failed,
            )
        else:
            logger.info("=" * 80)
            logger.info("SCRAPING PHASE")
            logger.info("=" * 80)
            self.state.advance_phase(PipelinePhase.SCRAPING)
            self._save_state()

            # A topic has no listing to scrape. Its records are built here
            # instead, into the same directory shape the scraper writes, so the
            # handoff and everything after it are unchanged. A run can carry
            # both kinds of input -- that is what a configured `topics:`
            # section alongside `keywords:` produces -- so this is two phases
            # that both may run, not a choice between them.
            topic_summary = (
                self._materialise_topics_phase() if self.config.topics else None
            )
            scrape_summary = (
                await self._execute_scraping_phase()
                if (self.config.keywords or self.config.product_ids)
                else None
            )
            scraping_summary = _merge_scraping_summaries(topic_summary, scrape_summary)

            # Update state with scraping results
            self.state.scraping_completed_products = (
                scraping_summary.successful_products
            )
            self.state.scraping_failed_products = scraping_summary.failed_products
            self.state.scraping_summary = asdict(scraping_summary)
            self.state.mark_phase_complete(PipelinePhase.SCRAPING)
            self._save_state()

            # Notify webhook of scraping phase completion
            await self._notify_webhook(
                "phase.complete",
                {"phase": "scraping", "summary": asdict(scraping_summary)},
            )

        # Phase 2: Handoff
        self.state.advance_phase(PipelinePhase.HANDOFF)
        self._save_state()

        ready_products = self._execute_handoff_phase(
            scraping_summary.successful_products
        )

        self.state.mark_phase_complete(PipelinePhase.HANDOFF)
        self._save_state()

        # Check if any products are ready
        if not ready_products:
            logger.warning("No products with sufficient media for video production")
            # Return early with empty production summary.
            #
            # The drop count has to be carried here as well as into the
            # rendering path, and this is the branch that matters: when the
            # guard drops *every* product there is nothing to render, so the
            # `nothing new` outcome fired only on runs that were never the
            # problem. Getting this wrong left the headline symptom -- a
            # correct run reporting PIPELINE FAILED and exiting 1 -- in place
            # while three documents said it was fixed.
            dropped = getattr(self, "_skipped_as_published", [])
            production_summary = ProductionPhaseSummary(
                total_attempted=0,
                successful=0,
                failed=0,
                skipped=0,
                failed_products=[],
                skipped_products=[],
                profile_distribution=None,
                duration_sec=0.0,
                already_published=len(dropped),
                already_published_products=list(dropped),
            )
            produced_videos: list[tuple[Path, str]] = []
        else:
            # Phase 3: Video Production
            if self.state.is_phase_completed(PipelinePhase.PRODUCTION):
                logger.info("=" * 80)
                logger.info("VIDEO PRODUCTION PHASE (SKIPPED - Already completed)")
                logger.info("=" * 80)
                # Reconstruct summary from state
                production_summary = ProductionPhaseSummary(
                    **self.state.production_summary  # type: ignore[arg-type]
                )
                msg = (
                    f"→ Using cached results: "
                    f"{production_summary.successful} successful, "
                    f"{production_summary.failed} failed"
                )
                logger.info(msg)
                # Reconstruct produced_videos from state, minus anything the
                # handoff guard just dropped.
                #
                # The path is resolved rather than composed. This built
                # `outputs/<id>/video.mp4`, a name nothing writes -- renders
                # are `video_<id>_<profile>.mp4` -- so every product on a
                # resumed publish failed at `upload_media` before reaching
                # the scheduler. `sole_render_for_product` is the resolver
                # `single`, `schedule` and the immediate batch already share,
                # so a resume now picks the same cut they would.
                #
                # The guard runs on a resume (handoff is never skipped), but
                # this branch rebuilds the publish list from the *saved* run,
                # which still names the products the guard removed. Filtering
                # here is what keeps a dropped product out of the one path
                # that reaches publishing without passing through production.
                from src.publisher.video_selector import sole_render_for_product

                # Routed the way the publisher routes it, not by taking the
                # alphabetically first file. A product rendered under two
                # profiles has two files in its directory, and the bare
                # resolver would hand the resumed publish a different cut
                # from the one the publish phase would have chosen.
                #
                # Parity with the other discoverers is the goal here, not
                # replaying the interrupted run: the state records product
                # ids and not paths, so which cut *that* run produced is not
                # recoverable. `schedule` and the immediate batch accept any
                # one render per product by the same rule, so a resume now
                # answers the question the same way they do.
                dropped = getattr(self, "_skipped_as_published", [])
                dropped_ids = set(dropped)
                produced_videos = []
                # The publishing phase below is gated on the same flag, so
                # this list is discarded on a `--skip-publish` run. Building
                # it anyway reads the publisher config, which now raises on a
                # file it cannot parse -- aborting a resume that had no
                # intention of publishing, over a file it would not use.
                if not self.config.skip_publish:
                    platforms_for_resume = [
                        platform.lower()
                        for platform in (
                            self.config.platforms or self._default_platforms()
                        )
                    ]
                    for pid in self.state.production_completed_products:
                        if pid in dropped_ids:
                            continue
                        render = sole_render_for_product(
                            self.config.outputs_dir / pid,
                            self._publisher_profiles(),
                            platforms_for_resume[0] if platforms_for_resume else "",
                        )
                        if render is None:
                            logger.warning(
                                "Resume: no render found for %s, skipping publish", pid
                            )
                            continue
                        produced_videos.append((render, pid))
                if dropped:
                    # The cached summary predates the drop, so it reports
                    # zero and contradicts the log line above it.
                    #
                    # Count and list both come from `dropped`, not from the
                    # set: the other two routes list in discovery order, and
                    # taking the count from the set while listing from
                    # elsewhere lets the two disagree if a product ever
                    # appears twice.
                    production_summary.already_published = len(dropped)
                    production_summary.already_published_products = list(dropped)
            else:
                logger.info("=" * 80)
                logger.info("VIDEO PRODUCTION PHASE")
                logger.info("=" * 80)
                self.state.advance_phase(PipelinePhase.PRODUCTION)
                self._save_state()

                (
                    production_summary,
                    produced_videos,
                ) = await self._execute_production_phase(ready_products)

                # Update state with production results
                self.state.production_completed_products = [
                    pid for _, pid in produced_videos
                ]
                self.state.production_failed_products = (
                    production_summary.failed_products
                )
                self.state.production_skipped_products = (
                    production_summary.skipped_products
                )
                self.state.production_summary = asdict(production_summary)
                self.state.mark_phase_complete(PipelinePhase.PRODUCTION)
                self._save_state()

                # Notify webhook of production phase completion
                await self._notify_webhook(
                    "phase.complete",
                    {"phase": "production", "summary": asdict(production_summary)},
                )

        # Phase 4: Publishing (conditional)
        publishing_summary = None
        if not self.config.skip_publish and produced_videos:
            if self.state.is_phase_completed(PipelinePhase.PUBLISHING):
                logger.info("=" * 80)
                logger.info("PUBLISHING PHASE (SKIPPED - Already completed)")
                logger.info("=" * 80)
                # Reconstruct summary from state
                publishing_summary = PublishingPhaseSummary(
                    **self.state.publishing_summary  # type: ignore[arg-type]
                )
                msg = (
                    f"→ Using cached results: "
                    f"{publishing_summary.successful} successful, "
                    f"{publishing_summary.failed} failed"
                )
                logger.info(msg)
            else:
                logger.info("=" * 80)
                logger.info("PUBLISHING PHASE")
                logger.info("=" * 80)
                self.state.advance_phase(PipelinePhase.PUBLISHING)
                self._save_state()

                publishing_summary = await self._execute_publishing_phase(
                    produced_videos
                )

                # Update state with publishing results
                self.state.publishing_completed_products = [
                    pid
                    for pid in self.state.production_completed_products
                    if pid not in publishing_summary.failed_videos
                ]
                self.state.publishing_failed_products = publishing_summary.failed_videos
                self.state.publishing_summary = asdict(publishing_summary)
                self.state.mark_phase_complete(PipelinePhase.PUBLISHING)
                self._save_state()

                # Notify webhook of publishing phase completion
                await self._notify_webhook(
                    "phase.complete",
                    {"phase": "publishing", "summary": asdict(publishing_summary)},
                )
        elif self.config.skip_publish:
            logger.info("Skipping publishing phase (--skip-publish)")
        else:
            logger.info("No videos to publish")

        # Mark pipeline as completed
        self.state.advance_phase(PipelinePhase.COMPLETED)
        self._save_state()

        # Generate final summary
        pipeline_duration = time.time() - pipeline_start
        final_summary = self._generate_final_summary(
            scraping_summary, production_summary, publishing_summary, pipeline_duration
        )

        # Notify webhook of pipeline completion
        await self._notify_webhook(
            "pipeline.complete",
            {"summary": final_summary.to_dict()},
        )

        # Clear state file on successful completion
        clear_pipeline_state(self.config.outputs_dir)
        logger.info("Pipeline completed successfully - state file cleared")

        return final_summary

    async def _execute_scraping_phase(self) -> ScrapingPhaseSummary:
        """Scrape the inputs; the body lives in `src.pipeline.phases.scraping`."""
        return await run_scraping_phase(self.config, self._resolve_profile_uses_videos)

    def _materialise_topics_phase(self) -> ScrapingPhaseSummary:
        """Stand in for the scraping phase when the inputs are topics.

        Reported as the scraping phase rather than as a new one so resume,
        state and the phase summaries keep working unchanged; what differs is
        where the records come from, not what the pipeline does with them.

        Returns a summary whose `successful_products` are the topic
        identifiers, which is what the handoff phase filters on.

        `--fail-fast` is deliberately not honoured here, unlike the scraping
        phase it stands in for. That flag exists to stop a run before it pays
        for more scrapes after one has failed; writing a record costs nothing,
        and the only way this fails is an unwritable outputs directory, which
        is not a reason to discard the topics that did write.
        """
        phase_start = time.time()
        logger.info("Preparing %s topic(s) (no scraping)", len(self.config.topics))

        config = load_video_config_modular()
        # `materialise_topics` needs a profile only to resolve the run paths,
        # and the data.json it writes is profile-independent. A random-profile
        # run has no single profile yet, so any valid one resolves the same
        # directory.
        profile = self.config.profile or next(iter(config.video_profiles))

        prepared: list[str] = []
        failed: list[str] = []
        for spec in self.config.topics:
            try:
                ((topic_dir, product),) = materialise_topics(
                    [spec], config, profile, outputs_dir=self.config.outputs_dir
                )
            except OSError as e:
                # One unwritable directory must not lose the rest of the batch.
                logger.error("Could not prepare topic %r: %s", spec.title, e)
                failed.append(spec.title)
                continue
            logger.info("Prepared topic %r in %s", spec.title, topic_dir)
            prepared.append(str(product.asin))

        return ScrapingPhaseSummary(
            total_attempted=len(self.config.topics),
            successful=len(prepared),
            failed=len(failed),
            successful_products=prepared,
            failed_products=failed,
            media_stats={"total_images": 0, "total_videos": 0},
            duration_sec=time.time() - phase_start,
        )

    def _execute_handoff_phase(
        self, scraped_product_ids: list[str]
    ) -> list[tuple[Path, ProductData]]:
        """Discover products ready for video production.

        Scans outputs directory for products with data.json and filters
        by scraped product IDs unless process_all_products is enabled.

        Args:
        ----
            scraped_product_ids: List of ASINs scraped in current run

        Returns:
        -------
            List of (product_dir, ProductData) tuples for ready products

        """
        from src.video.producer.cli import discover_products_for_batch

        logger.info("Discovering products ready for video production...")

        # Use existing discover_products_for_batch function
        # A topic run named its inputs, so its directories are what it asked
        # for. Discovery skips them by default, which would drop every topic
        # between the phase that wrote them and the phase that renders them.
        # Derived from the ids being handed off, not from the config: a
        # `--resume` carries no input flags, so `config.topics` is empty while
        # the saved state's ids are topics. Reading the config there returned
        # nothing and the resumed run reported PIPELINE FAILED.

        # `topics_resume` is set in `main` before validation, which is what
        # narrows the profile pool. The id check stays as the fallback for a
        # caller that builds the orchestrator directly.
        include_topics = (
            bool(self.config.topics)
            or self.config.topics_resume
            or any(pid.startswith(TOPIC_ID_PREFIX) for pid in scraped_product_ids)
        )
        all_products = discover_products_for_batch(
            self.config.outputs_dir, include_topics=include_topics
        )

        logger.info(
            "Found %s product(s) with data.json in %s",
            len(all_products),
            self.config.outputs_dir,
        )

        # Filter by scraped products unless process_all_products is enabled
        if self.config.process_all_products:
            ready_products = all_products
            logger.info("Processing all products in outputs directory")
        else:
            # Only process products scraped in current run
            scraped_set = set(scraped_product_ids)
            ready_products = [
                (path, data)
                for path, data in all_products
                if hasattr(data, "asin") and data.asin in scraped_set
            ]
            logger.info(
                "Processing %s product(s) from current scraping run",
                len(ready_products),
            )

        ready_products = self._drop_already_published(ready_products)

        # Log transition
        if ready_products:
            logger.info("%s product(s) ready for video production", len(ready_products))
        else:
            logger.warning("No products ready for video production")

        return ready_products

    def _drop_already_published(
        self, products: list[tuple[Path, Any]]
    ) -> list[tuple[Path, Any]]:
        """Drop products already recorded as published on every platform.

        The batch had no duplicate guard at all. `single` and `schedule` skip
        an already-published product; the batch's publish phase did not, so a
        re-scraped product was rendered and then published a second time --
        a duplicate Zernio post, with the tracking row overwritten by the new
        `post_id` while the older post stayed live. So this stops the
        duplicate as well as the render, which is why it is on by default and
        why `--force` exists to get the old behaviour back.

        `publish_history.json` is the file that backs the guard, keyed by
        `<asin>:<platform>`; `published_products.json` records what was
        produced rather than what went live and cannot answer this. A product
        published to some platforms but not all is kept, since the run still
        has somewhere to send it.

        Skipped entirely when `--force` is set, which is the flag that already
        means "publish it again" on the single and schedule paths, and when
        `--skip-publish` is set, where there is no duplicate to prevent and
        re-rendering a published product is the point of the run.

        Topics are never dropped. A topic's id is a pure function of its
        title, so once published it would be skipped on every later run and
        the tutorial arm would stop producing silently -- the bundled config
        ships two topics at one per run, so that lands on day three.

        This assumes `default_platforms` names platforms the install actually
        has accounts for. The publish phase only records a platform it could
        publish to, so a platform listed here with no connected account never
        accumulates a tracking key, `all(...)` is never satisfied, and the
        guard quietly never fires. That install has a louder problem -- every
        such publish is already counted a failure -- but it is worth knowing
        that this goes silent rather than half-firing.
        """
        from src.publisher.tracking import is_already_published

        if (
            getattr(self.config, "force", False)
            or getattr(self.config, "skip_publish", False)
            or not products
        ):
            return products

        # The list the publish phase will actually target, read the same way
        # it reads it. A hardcoded triple demanded tiktok of an install whose
        # `default_platforms` omits it, so a product complete for that install
        # was re-rendered and re-published (#126 tracks folding the three
        # inline reads in this module into the loaded config).
        # Lowercased: `record_publish` always writes `Platform(...).value`,
        # and `is_already_published` does an exact key match. Config
        # validation accepts mixed case and keeps the original spelling, so
        # `platforms: [YouTube]` would look up a key nothing ever writes and
        # silently keep every product.
        platforms = [
            platform.lower()
            for platform in (self.config.platforms or self._default_platforms())
        ]
        kept, skipped = [], []
        for path, data in products:
            asin = getattr(data, "asin", None)
            if (
                asin
                and not asin.startswith(TOPIC_ID_PREFIX)
                and all(
                    is_already_published(asin, platform, self.config.outputs_dir)
                    for platform in platforms
                )
            ):
                skipped.append(asin)
            else:
                kept.append((path, data))

        if skipped:
            logger.info(
                "Skipping %s already-published product(s) before render: %s",
                len(skipped),
                ", ".join(skipped),
            )
        self._skipped_as_published = skipped
        return kept

    def _publisher_profiles(self) -> dict[str, str] | None:
        """Per-platform render profiles, as the publish phase would read them.

        Only used to resolve which cut of a product a resumed publish sends,
        so it has to answer the same question `BatchPublisher` asks.
        """
        profiles = _publisher_settings().profiles
        return profiles if isinstance(profiles, dict) else None

    def _default_platforms(self) -> list[str]:
        """The platforms a publish would target, absent a CLI override.

        Reads the loaded config, as the plan printout and the publishing
        phase do. The duplicate guard has to ask the same question they do, or
        it decides completeness against platforms this install never publishes
        to. The literal fallback lives in `PublisherConfig.__post_init__` now,
        so there is only one of it.
        """
        return [p.value for p in _publisher_settings().default_platforms]

    async def _execute_production_phase(
        self, products: list[tuple[Path, ProductData]]
    ) -> tuple[ProductionPhaseSummary, list[tuple[Path, str]]]:
        """Render each product; the body lives in `src.pipeline.phases.production`."""
        return await run_production_phase(
            self.config,
            products,
            self._build_cli_overrides,
            getattr(self, "_skipped_as_published", []),
        )

    async def _execute_publishing_phase(
        self, produced_videos: list[tuple[Path, str]]
    ) -> PublishingPhaseSummary:
        """Publish each video; the body lives in `src.pipeline.phases.publishing`."""
        return await run_publishing_phase(self.config, produced_videos)

    def _generate_final_summary(
        self,
        scraping: ScrapingPhaseSummary,
        production: ProductionPhaseSummary,
        publishing: PublishingPhaseSummary | None,
        total_duration: float,
    ) -> PipelineSummary:
        """Generate end-to-end pipeline summary.

        Calculates derived statistics from phase summaries:
        - End-to-end success: Products scraped, produced, and published (if enabled)
        - Partial success: Products scraped but not fully produced/published
        - Total failures: Products that failed in any phase

        Logs formatted summary with all phase statistics and
        end-to-end metrics.

        Args:
        ----
            scraping: Scraping phase summary
            production: Video production phase summary
            publishing: Publishing phase summary (None if --skip-publish)
            total_duration: Total pipeline duration in seconds

        Returns:
        -------
            PipelineSummary with aggregated end-to-end statistics

        """
        # Calculate end-to-end metrics
        if publishing:
            # Full pipeline: scraped, produced, AND published successfully
            end_to_end_success = publishing.successful
            partial_success = scraping.successful - publishing.total_attempted
            total_failures = scraping.failed + production.failed + publishing.failed
        else:
            # Publishing skipped: scraped AND produced successfully
            end_to_end_success = production.successful
            partial_success = scraping.successful - production.total_attempted
            total_failures = scraping.failed + production.failed

        summary = PipelineSummary(
            scraping=scraping,
            production=production,
            publishing=publishing,
            end_to_end_success=end_to_end_success,
            partial_success=partial_success,
            total_failures=total_failures,
            total_duration_sec=total_duration,
        )

        # Log formatted summary
        logger.info(summary.format())

        return summary


if __name__ == "__main__":
    from src.pipeline.cli import main

    asyncio.run(main())
