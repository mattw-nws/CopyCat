import fcntl
import json
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

import numpy as np
import yaml # Not in standard lib but LSTM uses it... so... allowed?
import xarray as xr

try:
    from numpy.typing import NDArray
except Exception as e:
    from numpy import ndarray as NDArray    

logger = logging.getLogger(__name__)

# Adapted from https://refactoring.guru/design-patterns/singleton/python/example#example-1
from threading import Lock, Thread
class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """
    _instances = {}
    _lock: Lock = Lock()
    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]
    
class Source():
    def __init__(self, base: Union[PurePath,str], base_url: Union[ParseResult,str], t0_fnum: int):
        if not isinstance(base_url, ParseResult):
            base_url = urlparse(base_url) 
        if not isinstance(base, PurePath):
            if base_url.scheme != '':
                base = PurePosixPath(base)
            else:
                base = Path(base)
        self._base = base
        self._base_url = base_url
        self._t0_fnum = t0_fnum

    @property
    def base(self):
        return self._base
    
    @property
    def base_url(self):
        return self._base_url

    @property
    def t0_fnum(self):
        return self._t0_fnum


class SourceManager(metaclass=SingletonMeta):
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

    # for same-process re-use of derived sources.
    # Should probably only ever contain one item.
    _source_cache = {} 

    def __init__(self, cache_dir) -> None:
        self._entries = 0
        self._is_leader = False
        self._leader_lock_fd = None
        self._uuid: uuid.UUID = uuid.uuid4()
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
            self._elect_leader()
        else:
            self._cache_dir = None

    def __enter__(self):
        if self._entries == 0:
           self._elect_leader()
        with SourceManager._lock:
            self._entries += 1 #FIXME: Lock this?
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        with SourceManager._lock:
            self._entries -= 1
        if self._entries == 0:
            self._release_leader()
        #TODO: Ignoring exceptions--is this the right thing to do?
        return False

    def derive_source(self, t0: datetime, tend: Optional[datetime] = None, source_base: Optional[str] = None) -> Source:
        cache_key = (t0, tend, source_base)
        if cache_key in SourceManager._source_cache:
            return SourceManager._source_cache[cache_key]
        
        psource = None
        if self._cache_dir:
            psource = self._cache_dir / 'source.json'

        # If we are not the leader, wait for the leader to drop a source file...
        if psource and not self._is_leader:
            waitmax = 300 #TODO: Make configurable?
            waitstep = 2
            waited = 0
            while True:
                logger.info(f"Waiting for leader to drop source.json, waited {waited}s...")
                source = None
                try:
                    if psource.exists():
                        with open(psource, 'r') as fsource:
                            source_dict = json.load(fsource)
                            source = Source(**source_dict)
                            SourceManager._source_cache[cache_key] = source
                            return source # Infinite loop ends here normally
                except:
                    pass
                if not source and waited < waitmax:
                    time.sleep(waitstep)
                    waited += waitstep
                if waited >= waitmax and not psource.exists():
                    logger.critical(f"Waited >={waitmax}s for {psource.name} to arrive. Timed out!")
                    raise RuntimeError(f"Waited >={waitmax}s for {psource.name} to arrive. Timed out!")

        logger.info(f"Leader {self._uuid} deriving source...")
        # A source_base config entry can be a specific starting FILE, OR a 
        # known source key OR a URL or filesystem path to a NOMADS-style 
        # directory structure leading to model files.
        #FIXME: Allow path_template to be specified to enable custom directory structures or even ""/"." for a direct path.

        #TODO: Make configurable!
        source_variant = "medium_range_mem1"
        variant_info = SourceManager._model_path_data_dict[source_variant]

        if source_base is None:
            now = datetime.now(tz=timezone.utc)
            wayback = now - t0
            if wayback < timedelta(hours=40):
                source_base = "NOMADS"
            elif t0 > datetime(year=2023, month=9, day=20, tzinfo=timezone.utc):
                source_base = "NODD"
            else:
                source_base = "RETRO"

        url_base = None
        path_template = None
        if source_base in SourceManager._source_data_dict:
            url_base = SourceManager._source_data_dict[source_base]['url_base']
            path_template = SourceManager._source_data_dict[source_base]['path_template']
        elif source_base == "RETRO":
            raise NotImplementedError("NWM Retrospective source not yet implemented!")
        else:
            url_base = source_base
        
        if path_template is None:
            # Assume a NOMADS path structure if none other has been derived...
            path_template = SourceManager._source_data_dict["NOMADS"]['path_template']

        pr = urlparse(url_base)
        base_url = pr

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
        attempt = min(t0, datetime.now(timezone.utc) - timedelta(hours=model_lag)) # no sooner than 5 hours
        attempt = datetime.fromtimestamp(
            (attempt.timestamp()//quantizer)*quantizer, # quantize to 6-hourly
            tz=timezone.utc)
        hour_str = None

        #TODO: Decompose somewhere around here so that it is possible to test above logic with mocks before making HTTP calls

        if not is_file:
            while True:
                hour_str = str(attempt.hour).zfill(2)
                t0_delta = t0 - attempt
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
                    try:
                        with urlopen(req) as response:
                            status_code = response.getcode()
                            logger.debug(f"{status_code=}")
                    except HTTPError as e:
                        status_code = e.code
                    if status_code == 200:
                        break
                    if status_code != 404:
                        logger.error(f"Got {status_code} response code for {attempt_str}! Rate-limiting?")
                        retries += 1
                        if retries > max_retries:
                            raise RuntimeError(f"Max retries attempting to get {attempt_str}. Check data and parameters.")
                    # else, must be 404
                if(t0 - attempt > timedelta(days=1)): #TODO: Make rollback limit configurable for some use case?
                    logger.error(f"Rolled all the way back to {attempt.isoformat()} looking for {variant_info['model_name']} forecast data!")
                    raise RuntimeError("Unable to retrieve forecast data. Check data and parameters.")
                # else, go around again!
                attempt = attempt - timedelta(hours=model_freq)
            
            pr = urlparse(url_base + attempt_str)
            base_url = pr

            if pr.scheme != '':
                p = PurePosixPath(pr.path)
            else:
                p = Path(pr.path)
            
        base = p

        t0_fnum = int(re.search('f([0-9]{3})',p.stem).group(1))

        if tend is not None:
            # Validate if end hour is possible to obtain...
            t0_tend_delta_hours = (tend - t0).total_seconds() // 3600
            if t0_fnum + t0_tend_delta_hours > model_hours:
                raise ValueError(f"Simulation end date {tend} exceeds the data available for model {variant_info['model_name']} when starting at forecast hour {t0_fnum} (init_time {attempt.strftime('%Y%m%d')})")

        if psource:
            with open(psource, 'w') as fsource:
                json.dump({
                    'base': str(base),
                    'base_url': base_url.geturl(),
                    't0_fnum': t0_fnum
                }, fsource)
            #with open(psource, 'r') as f: print(f.read())

        source = Source(base, base_url, t0_fnum)
        SourceManager._source_cache[cache_key] = source
        return source

    def get_dataset(self, source: Source, tN: int) -> xr.Dataset:
        max_retries = 5
        retry_backoff_start = 5
        p = source.base.with_stem(re.sub('f[0-9]{3}', f"f{str(source.t0_fnum + (tN//3600)).zfill(3)}", source.base.stem))
        logger.info(p)
        source_str = str(p)
        if source.base_url.scheme:
            source_str = source.base_url.scheme + '://' + source.base_url.netloc + source_str + '#mode=bytes'
            
        if self._cache_dir:
            p = self._cache_dir / p.name
            if p.exists():
                ds = xr.open_dataset(p)
            else:
                if self._is_leader:
                    logger.warning(f"Leader {self._uuid} is downloading {source_str}")
                    retries = max_retries
                    retry_backoff = retry_backoff_start
                    while retries > 0:
                        ptemp = p.with_name('_'+p.name)
                        try:
                            urlretrieve(source_str, ptemp)
                            ptemp.rename(p) # Should be fairly atomic
                            break
                        except HTTPError as e:
                            logger.error(f"HTTPError {e.code} ({e.reason}) - retrying, {retries} left")
                            retries -= 1
                            logger.warning(f"Sleeping {retry_backoff} before retry")
                            time.sleep(retry_backoff)
                            retry_backoff = retry_backoff + retry_backoff
                        except FileNotFoundError as e:
                            logger.error(f"FileNotFoundError when trying to finish download (race condition?) - retrying, {retries} left")
                            retries -= 1
                            logger.warning(f"Sleeping {retry_backoff} before retry")
                            time.sleep(retry_backoff)
                            retry_backoff = retry_backoff + retry_backoff
                    else:
                        msg = f"Repeated failures downloading {p.name}. Aborting."
                        logger.critical(msg)
                        raise RuntimeError(msg)
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
            ds = xr.open_dataset(source_str)
        
        #TODO: To copy, nor not to copy? We may get significant memory savings by not copying,
        # but given that we can't use a context manager on the ds, are we playing with fire?
        return xr.Dataset({'streamflow': ds['streamflow']})

    def _elect_leader(self) -> None:
        if not self._cache_dir:
            self._is_leader = False
            return
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._leader_lock_fd = open(self._cache_dir/'leader.id', "a")
            fcntl.flock(self._leader_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB) # Try to acquire exclusive lock
            self._is_leader = True
            self._leader_lock_fd.seek(0)
            self._leader_lock_fd.truncate()
            self._leader_lock_fd.write(str(self._uuid))
            self._leader_lock_fd.flush()
            (self._cache_dir / 'source.json').unlink(missing_ok=True)


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
            fcntl.flock(self._leader_lock_fd, fcntl.LOCK_UN) # Release exclusive lock
            self._leader_lock_fd.close()
            self._is_leader = False
        except Exception as e:
            logger.error(f"Failed to release lock on leader file--this could cause subsequent issues!")
            logger.debug(str(e))

        self._is_leader = False

