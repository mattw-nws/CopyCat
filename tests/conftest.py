"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_datetime_utc():
    """Provide a reference UTC datetime for testing."""
    return datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
