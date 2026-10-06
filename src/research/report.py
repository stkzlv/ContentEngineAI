"""The research report, rendered from the stage records (design 0023)."""

from __future__ import annotations

import calendar
from typing import Any


def _cell(reading: dict[str, Any] | None) -> str:
    if reading is None:
        return "no data"
    trend = reading.get("recent_ratio")
    arrow = (
        ""
        if trend is None
        else (" rising" if trend > 1.1 else (" falling" if trend < 0.9 else ""))
    )
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


def render_report(demand: dict[str, Any] | None) -> str:
    parts = ["# Content research", ""]
    if demand:
        parts += [
            f"Run {demand['date']}, countries {', '.join(demand['countries'])}.",
            "",
        ]
        parts.append(render_demand(demand))
    return "\n".join(parts)
