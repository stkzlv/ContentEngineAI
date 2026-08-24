"""One reader for the pillar-keyed keyword config.

`config/scraper.yaml` groups `batch.keywords` by pillar:

    batch:
      keywords:
        value: ["USB C hub", "smart plug"]
        utility: ["portable ssd"]

Three places used to fold that into a keyword list and a keyword-to-pillar
map, each with its own loop, and they disagreed in ways nothing caught: the
CLI never flattened it at all, so a run with no `--keywords` searched for the
literal strings `value` and `utility` rather than any configured keyword.

Both shapes are accepted. A flat list is the pre-pillar config and attaches no
pillar.
"""

from typing import Any


def normalize_keyword(keyword: Any) -> str:
    """The form a keyword is matched by.

    Case and surrounding whitespace are presentation, not identity: a keyword
    written `USB C hub` in one place and `usb c hub` in another is the same
    search, and a byte-exact lookup silently dropped the pillar for the
    variant. Inner runs of whitespace collapse for the same reason.
    """
    return " ".join(str(keyword).split()).casefold()


def read_keyword_pillars(raw: Any) -> tuple[list[str], dict[str, str]]:
    """Return the configured keywords and their normalized pillar map.

    The keyword list keeps its original spelling, because it is what gets
    searched. The map is keyed by the normalized form, because it is what gets
    looked up.
    """
    keywords: list[str] = []
    pillars: dict[str, str] = {}

    if isinstance(raw, dict):
        for pillar, group in raw.items():
            if not isinstance(group, list):
                continue
            for keyword in group:
                text = str(keyword)
                keywords.append(text)
                pillars[normalize_keyword(text)] = str(pillar)
    elif isinstance(raw, list):
        keywords = [str(keyword) for keyword in raw]

    return keywords, pillars


def pillar_for(keyword: Any, pillars: dict[str, str]) -> str | None:
    """Look a keyword up in a map built by `read_keyword_pillars`."""
    return pillars.get(normalize_keyword(keyword))
