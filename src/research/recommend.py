"""Config recommendations from the measured research (REQ-OPS-112).

A recommendation names the file, the key and the value, and the evidence.
Nothing here edits configuration.
"""

from __future__ import annotations

from typing import Any

AI_SERVICES = "config/ai_services.yaml"
SCRAPER = "config/scraper.yaml"
# How many add candidates a recommendation lists.
TOP = 10
NO_KEY = (
    "no key does this yet: routing templates by topic shape needs a pipeline change"
)


def wrong_or_outdated(verification: list[dict[str, Any]], variant: str) -> float | None:
    """The share of a variant's verified scripts with a wrong or outdated step.

    A share, not a count: a variant that drops topics has fewer scripts, and
    a total would reward it for that. None when nothing was verified.
    """
    rows = [v for v in verification if v["variant"] == variant and sourced(v)]
    if not rows:
        return None
    bad = sum(bool(v["tally"]["wrong"] + v["tally"]["outdated"]) for v in rows)
    return round(bad / len(rows), 2)


def sourced(row: dict[str, Any]) -> bool:
    """A script the verifier could judge: at least one verdict with a source.

    One whose verdicts are all unverified says nothing about its steps, so
    it must not count as a clean script.
    """
    tally: dict[str, int] = row.get("tally") or {}
    judged = sum(tally.get(k, 0) for k in ("correct", "wrong", "outdated"))
    return judged > 0


def unjudged(verification: list[dict[str, Any]], variant: str) -> int:
    """Scripts with a readable answer but no sourced verdict."""
    return sum(
        1
        for v in verification
        if v["variant"] == variant and "tally" in v and not sourced(v)
    )


def _variant_key(variant: str, all_tasks: bool) -> tuple[str, str, Any] | None:
    if variant == "step_lists":
        return AI_SERVICES, "llm_settings.topic_scripts.step_list.enabled", True
    if variant == "task_answer_first" and all_tasks:
        # Every pool topic is a task, so one template for all topics is the
        # same thing as routing by shape.
        return (
            AI_SERVICES,
            "llm_settings.script_templates.topic_templates",
            ["topic_answer_first"],
        )
    return None


def variant_recommendations(
    summary: dict[str, Any],
    verification: list[dict[str, Any]],
    all_tasks: bool,
) -> list[dict[str, Any]]:
    """Each variant judged against `shipped` on the same topic sample.

    Recommended when it has fewer wrong or outdated steps without a lower
    share in the word band or a higher share of template misfits.
    """
    base = summary.get("shipped/topic")
    base_wrong = wrong_or_outdated(verification, "shipped")
    out = []
    for group, stats in sorted(summary.items()):
        variant, kind = group.split("/")
        if kind != "topic" or variant == "shipped" or base is None:
            continue
        wrong = wrong_or_outdated(verification, variant)
        evidence = {
            "wrong_or_outdated": [base_wrong, wrong],
            "in_band": [base["in_band"], stats["in_band"]],
            "template_misfit": [base["template_misfit"], stats["template_misfit"]],
            "dropped": [base["errors"], stats["errors"]],
            "unjudged": [
                unjudged(verification, "shipped"),
                unjudged(verification, variant),
            ],
        }
        if wrong is None or base_wrong is None:
            decision, reason = "undecided", "no verification for this comparison"
        elif wrong >= base_wrong:
            decision = "keep"
            reason = "no smaller share of scripts with a wrong or outdated step"
        elif (stats["in_band"] or 0) < (base["in_band"] or 0):
            decision, reason = (
                "keep",
                "more accurate, but fewer scripts in the word band",
            )
        elif (stats["template_misfit"] or 0) > (base["template_misfit"] or 0):
            decision, reason = "keep", "more accurate, but more template misfits"
        else:
            decision, reason = (
                "recommend",
                "fewer scripts with a wrong or outdated step, "
                "no worse on length or fit",
            )
        key = _variant_key(variant, all_tasks)
        out.append(
            {
                "variant": variant,
                "decision": decision,
                "reason": reason,
                "file": key[0] if key else None,
                "key": key[1] if key else None,
                "value": key[2] if key else None,
                "note": None if key else NO_KEY,
                "evidence": evidence,
            }
        )
    return out


def demand_recommendations(demand: dict[str, Any]) -> list[dict[str, Any]]:
    """Keyword and topic changes from the demand stage."""
    out = []
    products, topics = demand["products"], demand["topics"]
    if products["drop_candidates"]:
        out.append(
            {
                "decision": "consider",
                "file": SCRAPER,
                "key": "batch.keywords",
                "change": "drop",
                "items": products["drop_candidates"],
                "reason": "far below the keywords' median in every country, not rising",
            }
        )
    adds = sorted(products["add_candidates"], key=lambda a: -a["growth"])[:TOP]
    if adds:
        out.append(
            {
                "decision": "consider",
                "file": SCRAPER,
                "key": "batch.keywords",
                "change": "add",
                "items": [a["query"] for a in adds],
                "reason": "rising searches next to the strongest keywords that "
                "measure at or above the keywords' median",
            }
        )
    uncovered = [u["suggestion"] for u in topics["uncovered"]][:TOP]
    if uncovered:
        out.append(
            {
                "decision": "consider",
                "file": "the topic pool (PIPELINE_TOPICS_FILE)",
                "key": "topics",
                "change": "add",
                "items": uncovered,
                "reason": "autocomplete searches no pool topic covers; "
                "check each with the topic filter",
            }
        )
    return out
