"""Checks a commentary script against the facts its topic detail carries.

A commentary topic's detail holds the operator's sourced figures, and the
template forbids anything the detail doesn't state (design 0030). A probe
render spoke a rounded figure in place of the detail's exact one, and the web
fact check passed it as plausible (#810). These checks compare the script with
the detail itself.
"""

import re

_UNITS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS = {
    w: 10 * i
    for i, w in enumerate(
        "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
    )
    if w != "_"
}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 10**6, "billion": 10**9}
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | set(_SCALES) | {"and", "point"}

# A lone spoken count up to ten is a way of talking ("two things stood out",
# "the one next to it"), not a figure the detail must carry.
_MAX_LONE_COUNT = 10

# Digits, with a scale word after them ("62 billion") or an abbreviated scale
# attached ("$62B", "10K", "1.5M").
_DIGITS = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)"
    r"(?:(k|m|bn|b)\b|\s+(hundred|thousand|million|billion)\b)?",
    re.IGNORECASE,
)
_SUFFIXES = {"k": 1_000, "m": 10**6, "b": 10**9, "bn": 10**9}
_HALVES = (
    (re.compile(r"\band a half\b"), "point five"),
    (re.compile(r"\band a quarter\b"), "point two five"),
)
# Rewritten before the digits are read: "1 and a half million", "half a
# million", and "401(k)" spelled the way a script writes it, "401k".
_SPOKEN_FORMS = (
    (re.compile(r"(\d+) and a half\b"), r"\1.5"),
    (re.compile(r"(\d+) and a quarter\b"), r"\1.25"),
    (re.compile(r"\bhalf an? (thousand|million|billion)\b"), r"point five \1"),
    (re.compile(r"(\d)\(k\)"), r"\1k"),
)
# A word, or one punctuation mark: punctuation ends a run of number words, so
# "in twenty twenty-four, nineteen eighty-four" reads as two years.
_TOKEN = re.compile(r"[a-z]+|[^\sa-z]")


def _words_value(words: list[str]) -> float | None:
    """The value of a run of number words, or None if it doesn't parse."""
    if words and words[-1] == "and":
        words = words[:-1]
    if "point" in words:
        scale = 1
        if words[-1] in _SCALES and words[-1] != "hundred":
            scale, words = _SCALES[words[-1]], words[:-1]
        i = words.index("point")
        whole = _words_value(words[:i]) if i else 0
        digits = [_UNITS.get(w) for w in words[i + 1 :]]
        if whole is None or not digits or any(d is None or d > 9 for d in digits):
            return None
        return (whole + float("0." + "".join(str(d) for d in digits))) * scale
    # A spoken year: two pairs of tens and units ("twenty twenty-four").
    pairs = _year_pairs(words)
    if pairs is not None:
        return pairs
    total, current = 0, 0
    for w in words:
        if w == "and":
            continue
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = max(current, 1) * 100
        elif w in _SCALES:
            total += max(current, 1) * _SCALES[w]
            current = 0
        else:
            return None
    return float(total + current)


def _year_pairs(words: list[str]) -> float | None:
    """1900-2099 spoken as two pairs ("nineteen eighty-four"), else None."""
    if "and" in words or any(w in _SCALES for w in words):
        return None
    groups: list[int] = []
    for w in words:
        if w in _TENS:
            groups.append(_TENS[w])
        elif w in _UNITS and groups and groups[-1] >= 20 and groups[-1] % 10 == 0:
            groups[-1] += _UNITS[w]
        elif w in _UNITS:
            groups.append(_UNITS[w])
        else:
            return None
    if len(groups) == 2 and groups[0] in (19, 20) and 0 <= groups[1] <= 99:
        return float(groups[0] * 100 + groups[1])
    return None


def figures(text: str) -> set[float]:
    """Every figure in the text, written in digits or in words."""
    found: set[float] = set()
    text = text.lower()
    for pattern, written_as in _SPOKEN_FORMS:
        text = pattern.sub(written_as, text)
    for m in _DIGITS.finditer(text):
        try:
            value = float(m.group(1).replace(",", "").rstrip("."))
        except ValueError:
            continue
        if m.group(2):
            value *= _SUFFIXES[m.group(2).lower()]
        elif m.group(3):
            value *= _SCALES[m.group(3).lower()]
        found.add(value)
    # Words already read with their digits ("62 billion") aren't read again.
    words_only = _DIGITS.sub(" ", text.lower())
    for pattern, spoken_as in _HALVES:
        words_only = pattern.sub(spoken_as, words_only)
    run: list[str] = []
    for word in _TOKEN.findall(words_only.replace("-", " ")) + [""]:
        # "and" joins a number only after "hundred" or a scale word ("two
        # hundred and seven"); elsewhere it ends one ("between two and three").
        joins = word != "and" or (run and (run[-1] in _SCALES))
        if word in _NUMBER_WORDS and joins:
            run.append(word)
            continue
        if run:
            spoken = _words_value(run)
            lone = len(run) == 1 and spoken is not None and spoken <= _MAX_LONE_COUNT
            if spoken is not None and not lone:
                found.add(spoken)
            run = []
    return found


def unbacked_figure(sentences: list[str], source: str) -> str | None:
    """The first sentence that speaks a figure the source doesn't carry."""
    backed = figures(source)
    for sentence in sentences:
        if figures(sentence) - backed:
            return sentence
    return None


def has_closing_question(sentences: list[str]) -> bool:
    """Whether one of the last two sentences is a question."""
    return any(s.rstrip().rstrip("\"'").endswith("?") for s in sentences[-2:])
