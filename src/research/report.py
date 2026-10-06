"""The research report, rendered from the stage records (design 0023)."""

from __future__ import annotations

import calendar
from typing import Any

from src.research.demand import RISE


def _cell(reading: dict[str, Any] | None) -> str:
    if reading is None:
        return "no data"
    trend = reading.get("recent_ratio")
    if trend is None:
        arrow = ""
    elif trend > RISE:
        arrow = " rising"
    elif trend < 1 / RISE:
        arrow = " falling"
    else:
        arrow = ""
    return f"{reading['share']:.2f}{arrow}"


def _peak(per_geo: dict[str, Any], countries: list[str]) -> str:
    months = {
        calendar.month_abbr[r["peak_month"]]
        for g in countries
        if (r := per_geo.get(g)) and r.get("peak_month")
    }
    return ", ".join(sorted(months)) or "-"


def _table(
    rows: dict[str, dict[str, Any]], countries: list[str], label: str
) -> list[str]:
    head = f"| {label} | " + " | ".join(countries) + " | Peak month |"
    out = [head, "|" + "---|" * (len(countries) + 2)]

    def order(item: tuple[str, dict[str, Any]]) -> float:
        shares = [r["share"] for g in countries if (r := item[1].get(g))]
        return -max(shares) if shares else 0.0

    for name, per_geo in sorted(rows.items(), key=order):
        cells = " | ".join(_cell(per_geo.get(g)) for g in countries)
        out.append(f"| {name} | {cells} | {_peak(per_geo, countries)} |")
    return out


def render_demand(demand: dict[str, Any]) -> str:
    countries = demand["countries"]
    p, t = demand["products"], demand["topics"]
    lines = [
        "## Demand",
        "",
        f"Google Trends relative interest over 12 months, as a share of the anchor "
        f"term (\"{p['anchor']}\" for products, \"{t['anchor']}\" for topics); "
        "\"rising\" and \"falling\" compare the last quarter with the year. "
        "Trends gives no volumes.",
        "",
        "### Scraper keywords",
        "",
        *_table(p["terms"], countries, "Keyword"),
        "",
        "**Drop candidates** (under the configured share of the median in every "
        "country, not rising): " + (", ".join(p["drop_candidates"]) or "none") + ".",
        "",
        "**Add candidates** (rising related searches for the product seeds, not "
        "already keywords):",
        "",
    ]
    lines += [
        f"- {a['query']} (from \"{a['seed']}\", {a['geo']}, +{a['growth']:g}%)"
        for a in p["add_candidates"][:25]
    ] or ["- none"]
    topic_rows = {f"{name} (\"{r['term']}\")": r for name, r in t["terms"].items()}
    lines += [
        "",
        "### Topic pool",
        "",
        *_table(topic_rows, countries, "Topic (search measured)"),
        "",
        "**Searches no pool topic covers** (Google autocomplete for the configured "
        "stems; phrasing, not volume):",
        "",
    ]
    lines += [f"- {u['suggestion']} ({u['geo']})" for u in t["uncovered"][:40]] or [
        "- none"
    ]
    if demand["missing"]:
        lines += ["", "### Missing data", ""]
        lines += [f"- {m}" for m in demand["missing"]]
    return "\n".join(lines) + "\n"


SUMMARY_ROWS = (
    ("samples", "Samples"),
    ("errors", "Failed or dropped"),
    ("median_words", "Median words"),
    ("in_band", "In the word band"),
    ("phrase_in_first_sentence", "Search phrase in the first sentence"),
    ("cta_last", "CTA as the last sentence"),
    ("with_lint_tell", "Failing the script lint (a tell or a long sentence)"),
    ("template_misfit", "Task topic on a symptom or mistake template"),
    ("flagged_claims", "Claims the fact check flagged"),
    ("rewritten", "Scripts the fact check rewrote"),
)
# Shares print as percentages; counts and medians as numbers.
SHARE_ROWS = frozenset(
    {
        "in_band",
        "phrase_in_first_sentence",
        "cta_last",
        "with_lint_tell",
        "template_misfit",
    }
)


def _value(key: str, value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.0%}" if key in SHARE_ROWS else f"{value:g}"


def _escape(cell: str) -> str:
    """A table cell: Amazon titles use " | " as a separator."""
    return cell.replace("|", "\\|").replace("\n", " ")


def _notes(c: dict[str, Any]) -> str:
    found = [
        (not c["in_band"], f"out of band {c['band'][0]}-{c['band'][1]}"),
        (not c["phrase_in_first_sentence"], "no search phrase in first sentence"),
        (not c["cta_last"], "CTA not last"),
        (bool(c["lint"]), f"lint: {c['lint']}"),
        (bool(c["flagged"]), f"{c['flagged']} flagged"),
        (c["rewritten"], "rewritten"),
        (c["template_misfit"], "template misfit"),
    ]
    return ", ".join(text for on, text in found if on) or "ok"


def render_samples(checks: dict[str, Any]) -> str:
    summary = checks["summary"]
    groups = sorted(summary)
    lines = [
        "## Script samples",
        "",
        "Text-only scripts from the producer's own script step, per variant. "
        "Topic variants share one sample, so their columns compare directly; "
        "products run under `shipped` only.",
        "",
        "| Check | " + " | ".join(f"`{g}`" for g in groups) + " |",
        "|---|" + "---|" * len(groups),
    ]
    for key, label in SUMMARY_ROWS:
        cells = " | ".join(_value(key, summary[g].get(key)) for g in groups)
        lines.append(f"| {label} | {cells} |")
    repeated = {g: summary[g]["repeated_openings"] for g in groups}
    if any(repeated.values()):
        lines += ["", "**Openings used three or more times:**", ""]
        lines += [
            f"- `{g}`: " + ", ".join(f'"{o}" ({k})' for o, k in found.items())
            for g, found in repeated.items()
            if found
        ]
    lines += [
        "",
        "### Per sample",
        "",
        "| Variant | Kind | Title | Template | Words | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for row in checks["samples"]:
        c = row["checks"]
        failed = "error" in c
        title = _escape(row["title"][:60])
        lines.append(
            f"| {row['variant']} | {row['kind']} | {title} | "
            f"{row.get('template') or '-'} | {'-' if failed else c['words']} | "
            f"{_escape('failed: ' + c['error'][:80] if failed else _notes(c))} |"
        )
    return "\n".join(lines) + "\n"


def render_report(
    demand: dict[str, Any] | None, checks: dict[str, Any] | None = None
) -> str:
    parts = ["# Content research", ""]
    if demand:
        parts += [
            f"Run {demand['date']}, countries {', '.join(demand['countries'])}.",
            "",
        ]
        parts.append(render_demand(demand))
    if checks:
        parts += ["", render_samples(checks)]
    return "\n".join(parts)
