"""Tests for the TRouteWarmer class."""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import xarray as xr
import numpy as np
import pandas as pd

from copycatbmi.bmi import CopyCat


class TestCopyCat:
    """Tests for CopyCat class."""

    def test_update_until_missing_feature_id(self, mock_dataset_partial_features, tmp_path):
        """
        Test that update_until does not raise KeyError when feature_id 
        is missing from the dataset.
        """
        
        bmi = CopyCat()
        
        with patch('copycatbmi.troute.SourceManager') as mock_sm_class:
            # Setup the mock SourceManager
            mock_sm_instance = MagicMock()
            mock_sm_class.return_value.__enter__.return_value = mock_sm_instance
            
            # Setup the mock source
            mock_source = Mock()
            #mock_sm_instance.derive_source.return_value = mock_source
            
            # Setup the mock dataset to return the actual xarray dataset fixture
            # This will trigger KeyError on .sel() for missing feature_ids
            mock_sm_instance.get_dataset.return_value = mock_dataset_partial_features
            
            # Set up CopyCat state
            bmi._feature_id = 102
            bmi._area_sqm = 10
            bmi._source_manager = mock_sm_instance
            bmi._source = mock_source
            
            # This should raise a KeyError because feature_ids 102 and 103 are not in the dataset
            #with pytest.raises(KeyError, match="not all values found in index 'feature_id'"):
            bmi.update_until(3600)

        qvar = np.array([-1.0], dtype=np.float32)
        bmi.get_value('Q', qvar)
        assert qvar[0] == 0.0