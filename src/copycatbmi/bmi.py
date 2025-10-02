import fcntl
import logging
from datetime import datetime
from pathlib import Path, PurePath, PosixPath, PurePosixPath
import time
from urllib.parse import urlparse, ParseResult
import re
from urllib.request import urlretrieve
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

logger = logging.getLogger(__name__)


class CopyCat(BmiBase):

    def __init__(self) -> None:
        self._timestep: int = 0
        self._t0: datetime = datetime.now()
        self._uuid: uuid.UUID = uuid.uuid4()
        self._feature_id: int = 0
        self._tN: int = 0
        self._source_base: str = ""
        self._base: PurePath = Path('/')
        self._base_url: ParseResult
        self._t0_fnum: int = 0
        self._crosswalk_path: Optional[Path] = None
        self._cache_dir: Optional[Path] = None

    def initialize(self, config_file: str) -> None:
        with open(config_file, "r") as f:
            self._config = yaml.safe_load(f)

        try:
            self._t0 = datetime.fromisoformat(self._config['start_time'])
        except KeyError as e:
            logger.critical("start_time not found in config file")
            raise e
        except ValueError as e:
            logger.critical("start_time not in valid ISO format")
            raise e
        
        try:
            self._source_base = self._config['source_base']
        except KeyError as e:
            logger.critical("source_base not found in config")
            raise e
        
        crosswalk_path = self._config.get('crosswalk', None)
        if crosswalk_path is not None:
            self._crosswalk_path = Path(crosswalk_path)
        
        cache_dir = self._config.get('cache_dir', None)
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)

        self._tN = 0

        if self._cache_dir:
            self._elect_leader()

        self._init_store()
        self._init_crosswalk()

        self._q = 0.0
        self._qvar = np.array([self._q], dtype=np.float32)
        
    def update(self) -> None:
        self.update_until(self._tN + 3600)

    def update_until(self, time: float) -> None:
        self._tN = int(time)
        ds = self._get_dataset()
        logger.info(self._feature_id)
        logger.info(ds['streamflow'][(ds['feature_id'] == self._feature_id)].values[0])
        self._q = ds['streamflow'][(ds['feature_id'] == self._feature_id)].values[0]
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
        else:
            raise RuntimeError(f"Unknown variable '{name}'")

    def get_var_units(self, name: str) -> str:
        if name == 'Q':
            return 'm^3/s'
        else:
            raise RuntimeError(f"Unknown variable '{name}'")

    def set_value(self, name: str, src: NDArray[Any]) -> None:
        if name == 'feature_id':
            self._feature_id = int(src[0])
        if name == 'catchment_num':
            self._feature_id = self._get_xw_catchment(int(src[0]))
        else:
            pass #ignore

    def set_value_at_indices(self, name: str, inds: NDArray[np.int_], src: NDArray[Any]) -> None:
        return self.set_value(name, src)

    def _elect_leader(self) -> None:
        if not self._cache_dir:
            self._is_leader = False
            return
        try:
            with open(self._cache_dir/'leader.id', "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB) # Try to acquire exclusive lock
                self._is_leader = True
                f.seek(0)
                f.truncate()
                f.write(str(self._uuid))
                f.flush()
                return
        except BlockingIOError:
            pass
        except Exception as e:
            logger.warning(f"Unexpected error during leader election. Possibly no leader will be elected!")

        self._is_leader = False

    def _release_leader(self) -> None:
        if not self._cache_dir or not self._is_leader:
            return
        try:
            with open(self._cache_dir/'leader.id', "a") as f:
                fcntl.lockf(f, fcntl.LOCK_UN | fcntl.LOCK_NB) # Release exclusive lock
                self._is_leader = False
                return
        except Exception as e:
            logger.critical(f"Unexpected error releasing leader lock! Seppuku to ensure lock release!")
            raise e

        self._is_leader = False

    def _init_store(self) -> None:
        #TODO: Eventually, allow just specifying "NOMADS" or "AWS" (like Herbie) and figuring out the right path

        pr = urlparse(self._source_base)
        self._base_url = pr

        is_file = False
        if pr.scheme != '':
            p = PurePosixPath(pr.path)
            if p.suffix == '.nc':
                is_file = True
        else:
            p = Path(pr.path)
            if p.exists() and p.is_file():
                is_file = True

        if not is_file:
            raise NotImplementedError("Directory source_base not yet implemented--specify a filename to use for t0 instead.")
            #TODO: figure out what file to use within the directory

        self._base = p

        self._t0_fnum = int(re.search('f([0-9]{3})',p.stem).group(1))

    
    def _get_dataset(self) -> xr.Dataset:
        p = self._base.with_stem(re.sub('f[0-9]{3}', f"f{str(self._t0_fnum + (self._tN//3600)).zfill(3)}", self._base.stem))
        logger.info(p)
        source = str(p)
        if self._base_url.scheme:
            source = self._base_url.scheme + '://' + self._base_url.netloc + source + '#mode=bytes'
            
        if self._cache_dir:
            p = self._cache_dir / p.name
            if p.exists():
                ds = xr.open_dataset(p)
            else:
                if self._is_leader:
                    ptemp = p.with_name('_'+p.name)
                    urlretrieve(source, ptemp) #FIXME: Make robust to HTTP errors, retry!
                    ptemp.rename(p) # Should be fairly atomic
                else:
                    waitmax = 300 #TODO: Make configurable?
                    waitstep = 2
                    waited = 0
                    while not p.exists() and waited < waitmax:
                        time.sleep(waitstep)
                        waited += waitstep
                    if waited >= waitmax and not p.exists():
                        logger.critical(f"Waited >={waitmax}s for {p.name} to arrive. Timed out!")
                        raise RuntimeError(f"Waited >={waitmax}s for {p.name} to arrive. Timed out!")
                ds = xr.open_dataset(p)
        else:
            ds = xr.open_dataset(source)
        return ds
            
    def _init_crosswalk(self) -> None:
        #FIXME: Implement!
        pass

    def _get_xw_catchment(self, cat_num) -> int:
        raise NotImplementedError("Crosswalk not implemented!")

    def finalize(self):
        self._release_leader()
        
        
    # Boilerplate
    def get_var_type(self, name: str) -> str:
        return str(self.get_value_ptr(name).dtype)

    def get_var_nbytes(self, name: str) -> int:
        return self.get_value_ptr(name).nbytes

    def get_component_name(self) -> str:
        return 'CopyCat'
    
    def get_input_item_count(self) -> int:
        return 2

    def get_output_item_count(self) -> int:
        return 1

    def get_input_var_names(self) -> tuple[str, ...]:
        return ('feature_id','catchment_number')

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
