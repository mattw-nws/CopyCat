"""Pytest configuration and shared fixtures."""

from datetime import datetime
import numpy as np
import pytest
import tempfile
from pathlib import Path

import xarray as xr


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def mock_dataset_partial_features():
    """
    Create a mock xarray Dataset that contains only some of the requested feature_ids.
    This will trigger a KeyError when trying to select non-existent feature_ids.
    """
    # Create a dataset with only feature_ids 100 and 101
    feature_ids = [100, 101]
    times = [datetime(2024, 1, 1, 12, 0, 0)]
    
    data = {
        'streamflow': (['feature_id', 'time'], np.array([
            [1.5],  # feature_id 100
            [2.5],  # feature_id 101
        ]))
    }
    
    coords = {
        'feature_id': feature_ids,
        'time': times
    }
    
    ds = xr.Dataset(data, coords=coords)
    return ds


