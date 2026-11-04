"""Which licence a music track carries, as a short code the config can list.

Both providers report the licence as a Creative Commons URL
(`https://creativecommons.org/licenses/by-nc/4.0/`,
`http://creativecommons.org/publicdomain/zero/1.0/`); older records and test
doubles use a short name ("CC0", "CC-BY", "Attribution NonCommercial").
"""

import re

_CC_URL = re.compile(r"creativecommons\.org/(licenses|publicdomain)/([a-z-]+)", re.I)
_SHORT = {"cc0", "by", "by-sa", "by-nc", "by-nd", "by-nc-sa", "by-nc-nd"}
# The codes `allowed_music_licenses` may list.
KNOWN_CODES = frozenset(_SHORT | {"fal", "mark", "sampling", "nc-sampling"})
# Whole licence names, after separators collapse to spaces and a trailing
# version is dropped. Matched exactly: a prefix match read
# "Attribution-NonCommercial" as plain "by".
_NAMES = {
    "attribution": "by",
    "attribution sharealike": "by-sa",
    "attribution share alike": "by-sa",
    "attribution noncommercial": "by-nc",
    "attribution noderivs": "by-nd",
    "attribution noderivatives": "by-nd",
    "attribution noncommercial sharealike": "by-nc-sa",
    "attribution noncommercial share alike": "by-nc-sa",
    "attribution noncommercial noderivs": "by-nc-nd",
    "attribution noncommercial noderivatives": "by-nc-nd",
    "creative commons 0": "cc0",
    "free art": "fal",
    "free art license": "fal",
}
_VERSION = re.compile(r"\s+\d+(\.\d+)*$")


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
    # The whole name first: "creative commons 0" ends in what looks like a
    # version.
    spaced = re.sub(r"[\s_-]+", " ", text).strip()
    return _NAMES.get(spaced) or _NAMES.get(_VERSION.sub("", spaced))


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
