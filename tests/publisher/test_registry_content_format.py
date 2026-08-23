"""Tests for recording the content-format arm in the published-products registry.

The arm exists so two formats published side by side can be compared later.
A comparison is only fair if the arms are interleaved day by day, and that is
exactly the case where publish date cannot reconstruct which arm a video was in.
"""

import csv
import json
from dataclasses import fields

import pytest

from src.publisher.product_registry import (
    CONTENT_FORMAT_PRODUCT,
    CONTENT_FORMAT_TOPIC,
    RegistryEntry,
    add_to_registry,
    get_registry_path,
    load_registry,
    save_registry,
    summarize_by_content_format,
)


def _write_product(outputs_dir, product_id, **extra):
    d = outputs_dir / product_id
    d.mkdir(parents=True)
    payload = {
        "title": f"Title for {product_id}",
        "url": "https://www.amazon.com/dp/B0TEST0001",
        "affiliate_link": "https://www.amazon.com/dp/B0TEST0001?tag=x",
        "platform": "amazon",
        "price": "$10",
    }
    payload.update(extra)
    (d / "data.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


@pytest.mark.unit
class TestArmIsRecorded:
    def test_a_topic_record_is_labelled_topic(self, tmp_path):
        _write_product(
            tmp_path, "topic-why-wifi-drops-abc12345", topic="Why wifi drops"
        )
        assert add_to_registry("topic-why-wifi-drops-abc12345", tmp_path)
        entry = load_registry(tmp_path)[0]
        assert entry.content_format == CONTENT_FORMAT_TOPIC

    def test_a_scraped_record_is_labelled_product(self, tmp_path):
        _write_product(tmp_path, "B0TEST0001")
        assert add_to_registry("B0TEST0001", tmp_path)
        entry = load_registry(tmp_path)[0]
        assert entry.content_format == CONTENT_FORMAT_PRODUCT

    def test_a_blank_topic_is_not_a_topic(self, tmp_path):
        """An empty string is absence, not an arm."""
        _write_product(tmp_path, "B0TEST0002", topic="   ")
        assert add_to_registry("B0TEST0002", tmp_path)
        assert load_registry(tmp_path)[0].content_format == CONTENT_FORMAT_PRODUCT

    def test_a_refresh_updates_the_arm(self, tmp_path):
        """`--force` republish replaces the row, so the arm has to travel too."""
        _write_product(tmp_path, "B0TEST0003")
        add_to_registry("B0TEST0003", tmp_path)
        (tmp_path / "B0TEST0003" / "data.json").write_text(
            json.dumps(
                {
                    "title": "t",
                    "url": "u",
                    "affiliate_link": "a",
                    "platform": "amazon",
                    "price": "",
                    "topic": "Now a topic",
                }
            ),
            encoding="utf-8",
        )
        add_to_registry("B0TEST0003", tmp_path)
        entries = load_registry(tmp_path)
        assert len(entries) == 1
        assert entries[0].content_format == CONTENT_FORMAT_TOPIC


@pytest.mark.unit
class TestRegistryPersistence:
    def test_the_csv_header_follows_the_dataclass(self, tmp_path):
        """It used to be a hand-written list.

        `DictWriter` raises on a key the header does not name, so a field added
        to the record failed the whole registry write rather than dropping one
        column.
        """
        save_registry([RegistryEntry("id", "t", "u", "a")], tmp_path)
        with get_registry_path(tmp_path, "csv").open(encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert header == [fld.name for fld in fields(RegistryEntry)]

    def test_the_csv_columns_are_pinned(self, tmp_path):
        """The derivation test above cannot fail, by construction.

        Comparing the file to the dataclass proves they agree, not that either
        is right: renaming a field changes both sides together and every
        spreadsheet downstream silently gains a new column name. This states the
        contract, so a rename has to be a deliberate edit here too.
        """
        save_registry([RegistryEntry("id", "t", "u", "a")], tmp_path)
        with get_registry_path(tmp_path, "csv").open(encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert header == [
            "product_id",
            "title",
            "url",
            "affiliate_url",
            "content_format",
        ]

    def test_a_registry_written_before_the_arm_still_loads(self, tmp_path):
        """Existing files have no such key; they must not fail to parse."""
        get_registry_path(tmp_path, "json").write_text(
            json.dumps(
                [
                    {
                        "product_id": "B0OLD",
                        "title": "t",
                        "url": "u",
                        "affiliate_url": "a",
                    }
                ]
            ),
            encoding="utf-8",
        )
        entries = load_registry(tmp_path)
        assert len(entries) == 1
        assert entries[0].content_format == ""


@pytest.mark.unit
class TestSummary:
    def test_counts_each_arm(self):
        entries = [
            RegistryEntry("a", "t", "u", "af", content_format=CONTENT_FORMAT_TOPIC),
            RegistryEntry("b", "t", "u", "af", content_format=CONTENT_FORMAT_TOPIC),
            RegistryEntry("c", "t", "u", "af", content_format=CONTENT_FORMAT_PRODUCT),
        ]
        assert summarize_by_content_format(entries) == {"topic": 2, "product": 1}

    def test_unlabelled_rows_are_shown_not_absorbed(self):
        """Counting unknowns as one arm biases the comparison silently.

        Rows written before the arm existed carry an empty string, and there is
        no way to tell which arm they belonged to.
        """
        entries = [
            RegistryEntry("a", "t", "u", "af", content_format=CONTENT_FORMAT_TOPIC),
            RegistryEntry("b", "t", "u", "af"),
        ]
        counts = summarize_by_content_format(entries)
        assert counts == {"topic": 1, "unlabelled": 1}

    def test_empty_registry_summarises_to_nothing(self):
        assert summarize_by_content_format([]) == {}
