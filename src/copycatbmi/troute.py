import logging
import sys
from typing import Union, Optional
from datetime import datetime
from pathlib import Path

from .source_manager import SourceManager, Source

logger = logging.getLogger(__name__)

class TRouteWarmer():

    def __init__(self, cache_dir: Union[str, Path, None] = None, source_base: Optional[str] = None):
        # Don't create the SourceManager right now because we'll want to use `with` later...
        self._source_base = source_base
        self._cache_dir = cache_dir
        logger.info('CopyCat TRouteWarmer init - v' + sys.modules[self.__module__.split('.')[0]].__version__)


    def make_channel_restart_file(self, tm1: datetime, features: dict[int,int], dest: Union[str, Path]):
        feature_ids = list(features.values())
        with SourceManager(self._cache_dir) as sm:
            source = sm.derive_source(t0=tm1, tend=tm1, source_base=self._source_base)
            ds = sm.get_dataset(source, 0)

            missing = [feature_id for feature_id in feature_ids if feature_id not in ds.coords['feature_id'].values]
            if len(missing) > 0:
                logger.warning(f"Got {len(missing)} feature IDs not found in datset! These will be omitted!")
                logger.info(f"Missing feature_ids: {missing}")
                # rebuild features
                features = { fpid: fid for fpid, fid in features.items() if fid not in missing }
                feature_ids = list(features.values())
            flowpath_ids = list(features.keys())
            ds = ds.sel(feature_id=feature_ids)

            # Get count of NaNs in the `streamflow` variable and issue a warning if there are more than 0...
            nan_count = int(ds['streamflow'].isnull().sum().values)
            if nan_count > 0:
                logger.warning(f"Dataset has {nan_count} NaN values in `streamflow` variable. Filling with 0.")

            qtm1_out = ds['streamflow'].fillna(0).to_dataframe()
            qtm1_out['flowpath_id'] = flowpath_ids
            qtm1_out = qtm1_out.set_index(['flowpath_id'], drop=True)
            qtm1_out.index.name = None
            qtm1_out = qtm1_out.rename({'streamflow': 'qu0'}, axis=1)
            qtm1_out['qd0'] = qtm1_out['qu0']
            qtm1_out['h0'] = 0
            qtm1_out['time'] = tm1.replace(tzinfo=None)

            qtm1_out.to_pickle(dest)
