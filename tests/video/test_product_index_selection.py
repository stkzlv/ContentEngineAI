"""`--product-index` renders that one product, and an out-of-range one is an error.

It fell back to every product, so the "out of range" error could not fire
for a non-empty file and a typo rendered the whole file.
"""

from src.video.producer.cli import selected_indices


def test_no_index_is_every_product() -> None:
    assert selected_indices(None, 3) == [0, 1, 2]


def test_an_index_in_range_is_that_product() -> None:
    assert selected_indices(1, 3) == [1]


def test_an_index_out_of_range_is_refused() -> None:
    assert selected_indices(3, 3) is None
    assert selected_indices(-1, 3) is None
