"""Which licence a music track carries, as a short code the config can list.

Both providers report the licence as a Creative Commons URL
(`https://creativecommons.org/licenses/by-nc/4.0/`,
`http://creativecommons.org/publicdomain/zero/1.0/`); older records and test
doubles use a short name ("CC0", "CC-BY", "Attribution NonCommercial").
"""

import re

_CC_URL = re.compile(r"creativecommons\.org/(licenses|publicdomain)/([a-z-]+)", re.I)
_SHORT = {"cc0", "by", "by-sa", "by-nc", "by-nd", "by-nc-sa", "by-nc-nd"}
# Older licence names, longest first so "attribution noncommercial" is not
# read as plain "attribution".
_NAMES = (
    ("attribution noncommercial", "by-nc"),
    ("attribution", "by"),
    ("creative commons 0", "cc0"),
    ("free art", "fal"),
)


def licence_code(licence: str | None) -> str | None:
    """`cc0`, `by`, `by-sa`, `by-nc`, `by-nd`, `by-nc-sa`, `by-nc-nd`, `fal`,
    or another publicdomain path (`mark`); None when unrecognised.
    """
    if not licence:
        return None
    text = licence.strip().lower()
    match = _CC_URL.search(text)
    if match:
        kind, code = match.groups()
        return "cc0" if kind == "publicdomain" and code == "zero" else code
    if "artlibre.org" in text:
        return "fal"
    short = text.removeprefix("cc-")
    if short in _SHORT:
        return short
    for name, code in _NAMES:
        if text.startswith(name):
            return code
    return None


def jamendo_licence_params(allowed: list[str] | None) -> dict[str, str]:
    """Jamendo's own licence filters for an allow-list with no NC or ND code.

    Filtering at the API keeps a full page of usable results; filtering only
    afterwards left one or two of twenty, since most of Jamendo's catalogue
    is NonCommercial. The manager still checks every track.
    """
    if allowed is None:
        return {}
    codes = [a.lower() for a in allowed]
    params: dict[str, str] = {}
    if not any("nc" in code.split("-") for code in codes):
        params["ccnc"] = "false"
    if not any("nd" in code.split("-") for code in codes):
        params["ccnd"] = "false"
    return params


def is_allowed(licence: str | None, allowed: list[str] | None) -> bool:
    """True when `allowed` is None (no filter) or lists the licence's code.

    An unrecognised licence is refused: terms that can't be read can't be
    shown to permit commercial use.
    """
    if allowed is None:
        return True
    code = licence_code(licence)
    return code is not None and code in {a.lower() for a in allowed}
