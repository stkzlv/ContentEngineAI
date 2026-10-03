"""`--script-template` applies only to a render of its own kind (REQ-CNT-013).

The override was applied before the topic-or-product check, so a forced
product template ran on a topic, pitching a product the topic didn't have,
and a topic template ran on a scraped product.
"""

from __future__ import annotations

import pytest

from src.ai.script_generator import select_script_template
from src.video.config import config


@pytest.fixture
def templates():
    st = config.llm_settings.script_templates
    assert st.topic_templates, "no topic templates configured"
    saved = st.fixed_template
    yield st
    st.fixed_template = saved


def _product_template(st) -> str:
    from pathlib import Path

    names = sorted(p.stem for p in Path(st.templates_dir).glob("*.md"))
    return next(n for n in names if n not in st.topic_templates)


@pytest.mark.req("REQ-CNT-013")
def test_a_product_template_is_not_forced_onto_a_topic(templates) -> None:
    templates.fixed_template = _product_template(templates)
    chosen = select_script_template(
        config.llm_settings, product_id="topic-x", is_topic=True
    ).stem
    assert chosen in templates.topic_templates


@pytest.mark.req("REQ-CNT-013")
def test_a_topic_template_is_not_forced_onto_a_product(templates) -> None:
    templates.fixed_template = templates.topic_templates[0]
    chosen = select_script_template(
        config.llm_settings, product_id="B0FIXED001", is_topic=False
    ).stem
    assert chosen not in templates.topic_templates


@pytest.mark.req("REQ-CNT-013")
def test_a_matching_override_still_applies(templates) -> None:
    product = _product_template(templates)
    templates.fixed_template = product
    assert (
        select_script_template(config.llm_settings, product_id="B0FIXED001").stem
        == product
    )
    topic = templates.topic_templates[0]
    templates.fixed_template = topic
    assert (
        select_script_template(
            config.llm_settings, product_id="topic-x", is_topic=True
        ).stem
        == topic
    )
