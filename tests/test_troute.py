"""Tests for the TRouteWarmer class."""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import xarray as xr
import numpy as np
import pandas as pd

from copycatbmi.troute import TRouteWarmer


class TestTRouteWarmer:
    """Tests for TRouteWarmer class."""

    def test_make_channel_restart_file_missing_feature_ids(self, mock_dataset_partial_features, tmp_path):
        """
        Test that make_channel_restart_file does not raise KeyError when feature_ids 
        are missing from the dataset.
        
        The features dict maps flowpath_ids to feature_ids. We pass features where
        some feature_ids (102, 103) do not exist in the dataset, while others (100, 101) do.
        """
        # features maps flowpath_id -> feature_id
        # Some feature_ids exist in the dataset (100, 101), others don't (102, 103)
        features = {
            1: 100,  # exists in dataset
            2: 101,  # exists in dataset
            3: 102,  # DOES NOT exist in dataset - will cause KeyError
            4: 103,  # DOES NOT exist in dataset - will cause KeyError
        }
        
        tm1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dest = tmp_path / "restart.pkl"
        
        warmer = TRouteWarmer(cache_dir=None, source_base=None)
        
        with patch('copycatbmi.troute.SourceManager') as mock_sm_class:
            # Setup the mock SourceManager
            mock_sm_instance = MagicMock()
            mock_sm_class.return_value.__enter__.return_value = mock_sm_instance
            
            # Setup the mock source
            mock_source = Mock()
            mock_sm_instance.derive_source.return_value = mock_source
            
            # Setup the mock dataset to return the actual xarray dataset fixture
            # This will trigger KeyError on .sel() for missing feature_ids
            mock_sm_instance.get_dataset.return_value = mock_dataset_partial_features
            
            # This should raise a KeyError because feature_ids 102 and 103 are not in the dataset
            #with pytest.raises(KeyError, match="not all values found in index 'feature_id'"):
            warmer.make_channel_restart_file(tm1, features, dest)
