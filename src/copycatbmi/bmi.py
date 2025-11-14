import fcntl
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePath, PosixPath, PurePosixPath
import time
from urllib.error import HTTPError
from urllib.parse import urlparse, ParseResult
import re
from urllib.request import Request, urlopen, urlretrieve
import uuid
from typing import Any
from typing import Union, Optional
import dbm

import numpy as np
import yaml # Not in standard lib but LSTM uses it... so... allowed?
import xarray as xr

try:
    from numpy.typing import NDArray
except Exception as e:
    from numpy import ndarray as NDArray    

from .base import BmiBase
from .source_manager import SourceManager

logger = logging.getLogger(__name__)


class CopyCat(BmiBase):


    def __init__(self) -> None:
        self._timestep: int = 0
        self._t0: datetime = datetime.now()
        self._feature_id: int = 0
        self._tN: int = 0
        self._source_base: str = ""
        self._base: PurePath = Path('/')
        self._base_url: ParseResult
        self._t0_fnum: int = 0
        self._crosswalk_path: Optional[Path] = None
        self._cache_dir: Optional[Path] = None
        self._source_manager: Optional[SourceManager] = None

    def initialize(self, config_file: str) -> None:
        with open(config_file, "r") as f:
            self._config = yaml.safe_load(f)

        try:
            self._t0 = datetime.fromisoformat(self._config['start_time']).replace(tzinfo=timezone.utc)
        except KeyError as e:
            logger.critical("start_time not found in config file")
            raise e
        except ValueError as e:
            logger.critical("start_time not in valid ISO format")
            raise e
        if 'end_time' in self._config:
            try:
                self._tend = datetime.fromisoformat(self._config['end_time']).replace(tzinfo=timezone.utc)
            except ValueError as e:
                logger.critical("end_time not in valid ISO format")
                raise e
        else:
            self._tend = None
        
        self._source_base = self._config.get('source_base', None)

        crosswalk_path = self._config.get('crosswalk', None)
        if crosswalk_path is not None:
            self._crosswalk_path = Path(crosswalk_path)
        
        cache_dir = self._config.get('cache_dir', None)
        self._source_manager = SourceManager(cache_dir).__enter__()

        self._tN = 0

        self._init_store()
        self._init_crosswalk()

        self._q = 0.0
        self._qvar = np.array([self._q], dtype=np.float32)
        self._fidvar = np.array([self._q], dtype=np.int64)
        self._cnumvar = np.array([self._q], dtype=np.int64)
        self._areavar = np.array([self._q], dtype=np.float32)
        
    def update(self) -> None:
        self.update_until(self._tN + 3600)

    def update_until(self, time: float) -> None:
        self._tN = int(time)
        ds = self._source_manager.get_dataset(self._base, self._base_url, self._t0_fnum, self._tN)
        logger.info(self._feature_id)
        logger.info(ds['streamflow'][(ds['feature_id'] == self._feature_id)].values[0])
        self._q = 3600 * ds['streamflow'][(ds['feature_id'] == self._feature_id)].values[0] / self._area_sqm
        self._qvar = np.array([self._q], dtype=np.float32)

    def get_time_step(self) -> float:
        return 3600

    def get_current_time(self) -> float:
        return self._tN

    def get_time_units(self) -> str:
        return "s"

    def get_value_at_indices(self, name: str, dest: NDArray[Any], inds: NDArray[np.int_]) -> NDArray[Any]:
        return self.get_value(name, dest)
    
    def get_value(self, name: str, dest: NDArray[Any]) -> NDArray[Any]:
        dest[:] = self.get_value_ptr(name).flatten()
        return dest

    def get_value_ptr(self, name: str) -> NDArray[Any]:
        if name == 'Q':
            return self._qvar
        if name == 'feature_id':
            return self._fidvar
        if name == 'catchment_num':
            return self._cnumvar
        if name == 'area':
            return self._areavar
        else:
            raise RuntimeError(f"Unknown variable '{name}'")

    def get_var_units(self, name: str) -> str:
        if name == 'Q':
            return 'm/h'
        if name in 'feature_id' or name == 'catchment_num':
            return 'm/m'
        if name == 'area':
            return 'km^3'
        else:
            raise RuntimeError(f"Unknown variable '{name}'")

    def set_value(self, name: str, src: NDArray[Any]) -> None:
        if name == 'feature_id':
            self._feature_id = int(src[0])
            self._fidvar[0] = self._feature_id
        if name == 'catchment_num':
            self._feature_id = self._get_xw_catchment(int(src[0]))
            self._cnumvar[0] = int(src[0])
        if name == 'area':
            self._area_sqm = float(src[0])*1_000_000 # now in sqm!
            self._areavar[0] = float(src[0]) # remains in sqkm
        else:
            pass #ignore

    def set_value_at_indices(self, name: str, inds: NDArray[np.int_], src: NDArray[Any]) -> None:
        return self.set_value(name, src)

    def _init_store(self) -> None:

        (self._base, self._base_url, self._t0_fnum) = self._source_manager.derive_source(self._source_base, self._t0, self._tend)
            

    def _cleanup_source_manager():
        #TODO: Do we need to call this in more places?
        if self._source_manager is not None:
            self._source_manager.__exit__()

    def _init_crosswalk(self) -> None:
        #FIXME: Implement!
        pass

    def _get_xw_catchment(self, cat_num) -> int:
        raise NotImplementedError("Crosswalk not implemented!")

    def finalize(self):
        self._cleanup_source_manager()
        
    # Boilerplate
    def get_var_type(self, name: str) -> str:
        return str(self.get_value_ptr(name).dtype)

    def get_var_nbytes(self, name: str) -> int:
        return self.get_value_ptr(name).nbytes

    def get_component_name(self) -> str:
        return 'CopyCat'
    
    def get_input_item_count(self) -> int:
        return len(self.get_input_var_names())

    def get_output_item_count(self) -> int:
        return len(self.get_output_var_names())

    def get_input_var_names(self) -> tuple[str, ...]:
        return ()

    def get_output_var_names(self) -> tuple[str, ...]:
        return ('Q',)
        
    def get_var_itemsize(self, name: str) -> int:
        return np.dtype(self.get_var_type(name)).itemsize
    
    def get_var_grid(self, name: str) -> int:
        return 0

    def get_grid_rank(self, grid: int) -> int:
        return 1

    def get_grid_size(self, grid: int) -> int:
        return 1

    def get_grid_type(self, grid: int) -> str:
        return "scalar"
    
    def get_var_location(self, name: str) -> str:
        return "node"

    def get_start_time(self) -> float:
        return 0

    def get_end_time(self) -> float:
        return np.finfo("d").max  # type: ignore
