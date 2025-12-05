"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

