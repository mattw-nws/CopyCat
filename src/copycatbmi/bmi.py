import fcntl
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePath, PosixPath, PurePosixPath
import time
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

logger = logging.getLogger(__name__)


class CopyCat(BmiBase):

    _source_data_dict = {
        "NODD": {
            "url_base": "https://storage.googleapis.com/national-water-model/",
            "path_template": "nwm.{init_date}/{model_dir}/nwm.t{init_hour}z.{model_name}.channel_rt{var_file_suffix}.f{forecast_hour}.conus.nc"
        },
        "NOMADS": {
            "url_base": "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwm/v3.0/",
            "path_template": "nwm.{init_date}/{model_dir}/nwm.t{init_hour}z.{model_name}.channel_rt{var_file_suffix}.f{forecast_hour}.conus.nc"
        }
    }

    _model_path_data_dict = {
        "medium_range_mem1": {
            "version": "3.0",
            "model_dir": "medium_range_mem1",
            "model_name": "medium_range",
            "var_file_suffix": "_1",
            "hours": 240,
            "run_freq": 6,
            "lag": 5
        },
        "medium_range_blend": {
            "version": "3.0",
            "model_dir": "medium_range_blend",
            "model_name": "medium_range_blend",
            "var_file_suffix": "",
            "hours": 240,
            "run_freq": 6,
            "lag": 5
        },
        "short_range": {
            "version": "3.0",
            "model_dir": "short_range",
            "model_name": "short_range",
            "var_file_suffix": "",
            "hours": 18,
            "run_freq": 1,
            "lag": 1
        }
    }

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
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)

        self._tN = 0

        if self._cache_dir:
            self._elect_leader()

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
        ds = self._get_dataset()
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
                fcntl.flock(f, fcntl.LOCK_UN) # Release exclusive lock
                self._is_leader = False
                return
        except Exception as e:
            logger.critical(f"Unexpected error releasing leader lock! Seppuku to ensure lock release!")
            raise e

        self._is_leader = False

    def _init_store(self) -> None:

        self._derive_source(self._source_base)


    
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
    
    def _derive_source(self, source_base: str):
        # A source_base config entry can be a specific starting FILE, OR a 
        # known source key OR a URL or filesystem path to a NOMADS-style 
        # directory structure leading to model files.
        #FIXME: Allow path_template to be specified in config to enable custom directory structures or even ""/"." for a direct path.

        #TODO: Make configurable!
        source_variant = "medium_range_mem1"
        variant_info = CopyCat._model_path_data_dict[source_variant]

        if source_base is None:
            now = datetime.now(tz=timezone.utc)
            wayback = now - self._t0
            if wayback < timedelta(hours=40):
                source_base = "NOMADS"
            elif self._t0 > datetime(year=2023, month=9, day=20, tzinfo=timezone.utc):
                source_base = "NODD"
            else:
                source_base = "RETRO"

        url_base = None
        path_template = None
        if source_base in CopyCat._source_data_dict:
            url_base = CopyCat._source_data_dict[source_base]['url_base']
            path_template = CopyCat._source_data_dict[source_base]['path_template']
        elif source_base == "RETRO":
            raise NotImplementedError("NWM Retrospective source not yet implemented!")
        else:
            url_base = source_base
        
        if path_template is None:
            # Assume a NOMADS path structure if none other has been derived...
            path_template = CopyCat._source_data_dict["NOMADS"]['path_template']

        pr = urlparse(url_base)
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

        model_freq = variant_info.get('run_freq')
        model_lag = variant_info.get('lag')
        model_hours = variant_info.get('hours')
        model_stride = variant_info.get('stride', 1) # will need if we figure out how to support LR

        quantizer = timedelta(hours=model_freq).seconds
        attempt = min(self._t0, datetime.now(timezone.utc) - timedelta(hours=model_lag)) # no sooner than 5 hours
        attempt = datetime.fromtimestamp(
            (attempt.timestamp()//quantizer)*quantizer, # quantize to 6-hourly
            tz=timezone.utc)
        hour_str = None

        if not is_file:
            while True:
                hour_str = str(attempt.hour).zfill(2)
                t0_delta = self._t0 - attempt
                t0_forecast_hour = int(t0_delta.total_seconds() // 3600)
                logger.debug(f"{t0_delta=}")
                attempt_str = path_template.format(init_date = attempt.strftime('%Y%m%d'), init_hour=hour_str, forecast_hour=str(t0_forecast_hour).zfill(3), **variant_info)
                logger.info(f"Trying {url_base}{attempt_str}")
                if pr.scheme == '':
                    logger.debug("Using filesystem path")
                    if (p / attempt_str).exists():
                        break
                else:
                    # Bypasses parsed URL! Is this best?
                    req = Request(url = (url_base + attempt_str), method='HEAD')
                    max_retries = 3
                    retries = 0
                    with urlopen(req) as response:
                        status_code = response.getcode()
                        logger.debug(f"{status_code=}")
                        if status_code == 200:
                            break
                        if status_code != 404:
                            logger.error(f"Got {status_code} response code for {attempt_str}! Rate-limiting?")
                            retries += 1
                            if retries > max_retries:
                                raise RuntimeError(f"Max retries attempting to get {attempt_str}. Check data and parameters.")
                        # else, must be 404
                if(datetime.now(timezone.utc) - attempt > timedelta(days=1)):
                    logger.error(f"Rolled all the way back to {attempt.isoformat()} looking for {variant_info['model_name']} forecast data!")
                    raise RuntimeError("Unable to retrieve forecast data. Check data and parameters.")
                # else, go around again!
                attempt = attempt - timedelta(hours=model_freq)
            
            pr = urlparse(url_base + attempt_str)
            self._base_url = pr

            if pr.scheme != '':
                p = PurePosixPath(pr.path)
            else:
                p = Path(pr.path)
            
        self._base = p

        self._t0_fnum = int(re.search('f([0-9]{3})',p.stem).group(1))

        if self._tend is not None:
            # Validate if end hour is possible to obtain...
            t0_tend_delta_hours = (self._tend - self._t0).total_seconds() // 3600
            if self._t0_fnum + t0_tend_delta_hours > model_hours:
                raise ValueError(f"Simulation end date {self._tend} exceeds the data available for model {variant_info['model_name']} when starting at forecast hour {self._t0_fnum} (init_time {attempt.strftime('%Y%m%d')})")



            
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
