You check a short tech-help video script against current official documentation.

TOPIC: {TITLE}

SCRIPT:
{SCRIPT}

List every instruction, menu path, setting name, button name, model or version claim, and factual claim in the script. For each, search for the current official documentation (Apple, Google, Samsung, Microsoft, or the app's own help pages first) and judge it:

- "correct": the documentation states it as the script does, for the current version.
- "wrong": the documentation says something different.
- "outdated": it was right for an earlier version and the current version names or places it differently.
- "unverified": you found no documentation that settles it.

Answer with JSON only, no prose, in this shape:

[{{"claim": "<the script's words>", "verdict": "correct|wrong|outdated|unverified", "source": "<URL of the page that settles it, or empty>", "quote": "<under 25 words from that page, or empty>", "correction": "<what the documentation says, when wrong or outdated, or empty>"}}]
