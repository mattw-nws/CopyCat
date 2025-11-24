from typing import Union, Optional
from datetime import datetime
from pathlib import Path

from .source_manager import SourceManager, Source

class TRouteWarmer():

    def __init__(self, source_base: str, cache_dir: Union[str, Path, None]):
        # Don't create the SourceManager right now because we'll want to use `with` later...
        self._source_base = source_base
        self._cache_dir = cache_dir

    def make_channel_restart_file(self, tm1: datetime, features: dict[int,int], dest: Union[str, Path]):
        flowpath_ids = list(features.keys())
        feature_ids = list(features.values())
        with SourceManager(self._cache_dir) as sm:
            source = sm.derive_source(source_base=self._source_base, t0=tm1, tend=tm1)
            ds = sm.get_dataset(source, 0)

            qtm1_out = ds.sel(feature_id=feature_ids)['streamflow'].to_dataframe()
            qtm1_out['flowpath_id'] = flowpath_ids
            qtm1_out = qtm1_out.set_index(['flowpath_id'], drop=True)
            qtm1_out.index.name = None
            qtm1_out = qtm1_out.rename({'streamflow': 'qu0'}, axis=1)
            qtm1_out['qd0'] = qtm1_out['qu0']
            qtm1_out['h0'] = 0
            qtm1_out['time'] = tm1.replace(tzinfo=None)

            qtm1_out.to_pickle(dest)
