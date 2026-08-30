"""Which prompt a record gets: the product one, or its topic variant.

A topic has no product, and the product prompts say so on every line -- "using
the provided product information", a `{FULL_PRODUCT_NAME}` slot, and worked
examples that all end in a product URL and an `#ad`. Measured on a real topic
render, the model copied the example's `https://example.com/product` into the
YouTube description verbatim and tagged `#ad` on all three platforms, on a
record whose `carries_affiliate_content` is False and whose on-frame
disclosure was correctly skipped.

Rewording the product prompts cannot fix that. This project has recorded the
reason twice: examples teach by demonstration, and when an example contradicts
a rule the example wins. So the topic path gets its own files, with its own
examples.

Selection matches how the hook headline already chooses (#230): on the record
carrying a topic, which is the same thing `carries_affiliate_content` keys off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROMPT_DIR = Path("src/ai/prompts")

# Product prompt -> its topic variant. A prompt absent from this map has no
# variant and is used for both kinds of record.
_TOPIC_VARIANTS = {
    "youtube_metadata.md": "youtube_metadata_topic.md",
    "tiktok_caption.md": "tiktok_caption_topic.md",
    "instagram_caption.md": "instagram_caption_topic.md",
    "video_description.md": "video_description_topic.md",
}


def is_topic_record(product: Any) -> bool:
    """Whether this record describes a topic rather than a scraped product."""
    return bool(getattr(product, "topic", None))


def prompt_path_for(product: Any, product_prompt: str | Path) -> Path:
    """The prompt this record should be rendered with.

    `product_prompt` is the product-shaped path a caller would otherwise use,
    so a call site reads as "this prompt, or its topic variant".
    """
    path = Path(product_prompt)
    if not is_topic_record(product):
        return path

    variant = _TOPIC_VARIANTS.get(path.name)
    if variant is None:
        return path

    candidate = path.with_name(variant)
    # A missing variant falls back rather than raising: a caller that has not
    # been given one yet still renders, product framing and all, which is what
    # it did before this existed.
    return candidate if candidate.exists() else path
