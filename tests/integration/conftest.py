"""Pytest configuration for integration tests.

This conftest provides isolated test environment to avoid circular import issues.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

# Prevent pytest from loading parent conftest fixtures that cause circular imports
collect_ignore_glob = ["../conftest.py"]


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)
