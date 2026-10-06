"""`python -m src.research <stage>`: run a research stage (design 0023)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.research.checks import run_checks
from src.research.config import DEFAULT_PATH, load_research_config
from src.research.demand import run_demand
from src.research.report import render_report
from src.research.sample import recent_products, sample
from src.research.sources import PytrendsSource, TrendsUnavailableError
from src.scraper.base.keyword_pillars import read_keyword_pillars
from src.utils.outputs_paths import get_project_root, resolve_outputs_dir

SCRAPER_CONFIG = Path("config/scraper.yaml")


def scraper_keywords(path: Path = SCRAPER_CONFIG) -> list[str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    keywords, _ = read_keyword_pillars(((raw or {}).get("batch") or {}).get("keywords"))
    return keywords


def pool_specs() -> list[Any]:
    """The topic pool the batch reads, as the topic filter's tool reads it."""
    from src.pipeline.config import DEFAULT_PIPELINE_CONFIG_PATH, _configured_topics

    raw = yaml.safe_load(DEFAULT_PIPELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    batch = (raw or {}).get("global_batch") or {}
    return list(_configured_topics(batch, DEFAULT_PIPELINE_CONFIG_PATH))


def pool_topics() -> list[tuple[str, str]]:
    return [(t.title, t.search) for t in pool_specs()]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_dir(out: str | None) -> Path:
    if out:
        return Path(out)
    return Path(resolve_outputs_dir(None)) / "reports" / f"research-{date.today()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.research")
    parser.add_argument("stage", choices=["demand", "sample", "check", "report"])
    parser.add_argument("--config", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--out", help="Run directory (default: outputs/reports/research-<date>)"
    )
    args = parser.parse_args(argv)
    # The topic pool's path (`PIPELINE_TOPICS_FILE`) and the outputs root
    # come from the environment, as for the batch.
    load_dotenv(get_project_root() / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = run_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    demand_file = out / "demand.json"

    if args.stage == "demand":
        config = load_research_config(args.config)
        try:
            source = PytrendsSource(config.request_pause_sec, config.max_retries)
        except TrendsUnavailableError:
            print(
                "pytrends is not installed: poetry install --with research",
                file=sys.stderr,
            )
            return 2
        measured = run_demand(config, source, scraper_keywords(), pool_topics())
        demand_file.write_text(json.dumps(measured, indent=1), encoding="utf-8")
        print(f"Wrote {demand_file}")

    samples_file = out / "samples.json"
    checks_file = out / "checks.json"
    if args.stage == "sample":
        settings = load_research_config(args.config).sample
        records = sample(
            settings.variants,
            pool_specs()[: settings.topics],
            recent_products(Path(resolve_outputs_dir(None)), settings.products),
            settings.profile,
            out,
        )
        samples_file.write_text(json.dumps(records, indent=1), encoding="utf-8")
        print(f"Wrote {samples_file}")
    if args.stage in ("sample", "check") and samples_file.exists():
        band = load_research_config(args.config).sample.band
        records = json.loads(samples_file.read_text(encoding="utf-8"))
        checks_file.write_text(
            json.dumps(run_checks(records, band), indent=1), encoding="utf-8"
        )
        print(f"Wrote {checks_file}")

    report = out / "report.md"
    report.write_text(
        render_report(_load(demand_file), _load(checks_file)), encoding="utf-8"
    )
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
