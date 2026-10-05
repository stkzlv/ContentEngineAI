"""A lint for machine-writing tells and over-long scripts (design 0007).

Run after a script passes the completeness check, when
`script_validation.lint.enabled` is set. A failing script goes back into the
retry loop like a missing call to action; one that fails only the lint still
ships, with a warning, rather than losing the render.
"""

from __future__ import annotations

import re
from typing import Any

from src.utils.script_sanitizer import split_sentences


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'’-]+", text)


def lint_script(
    script: str, settings: Any, *, word_cap: bool = True, exempt: str = ""
) -> str | None:
    """Why the script fails the lint, or None when it passes.

    `word_cap` is off for a script whose length another rule sets, such as
    a tutorial sized by its step count. A match that also occurs in `exempt`
    (the product's own title and keyword) is no tell: a product named
    "Seamless" has to be called by its name.
    """
    # The model types curly apostrophes as often as straight ones.
    text = script.replace("\u2019", "'")
    exempt_text = exempt.replace("\u2019", "'").lower()
    for pattern in settings.banned_phrases:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if match.group(0).lower() not in exempt_text:
                return f"uses the phrase {match.group(0)!r}"
    for sentence in split_sentences(text):
        count = len(_words(sentence))
        if count > settings.max_sentence_words:
            return (
                f"a sentence runs {count} words, over " f"{settings.max_sentence_words}"
            )
    if word_cap:
        cap = int(settings.max_words_per_sec * settings.target_duration_sec)
        count = len(_words(text))
        if count > cap:
            return f"{count} words, over {cap} for {settings.target_duration_sec:g} s"
    return None
