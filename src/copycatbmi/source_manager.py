from __future__ import annotations
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
    
# Used to partially-format a path template: known keys (e.g. model_dir, model_name)
# are substituted immediately, while unknown keys (e.g. forecast_hour) are left
# in place--format spec and all--so they can be filled in later per-timestep.
class _KeepPlaceholder:
    def __init__(self, name: str):
        self._name = name

    def __format__(self, spec: str) -> str:
        return '{' + self._name + (':' + spec if spec else '') + '}'


class _PartialFormatDict(dict):
    def __missing__(self, key):
        return _KeepPlaceholder(key)


#TODO: Split into two (or more) classes like SourceTemplate/Source?
class Source():
    def __init__(self, base: Union[PurePath,str], base_url: Union[ParseResult,str], t0_fnum: int, variable_template_map:dict[str,str] = {}, init_datetime: Union[datetime,None] = None, url_root: Union[str,None] = None):
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
        self._vtm = variable_template_map
        self._init_datetime = init_datetime
        # Root (e.g. bucket base URL/dir) that per-variable templates are relative to.
        # Needed to reconstruct an absolute location for each new timestep, since
        # templated (e.g. retrospective) paths are not simple increments of `base`.
        self._url_root = url_root

    @property
    def base(self):
        return self._base
    
    @property
    def base_url(self):
        return self._base_url

    @property
    def t0_fnum(self):
        return self._t0_fnum

    @property
    def url_root(self):
        return self._url_root
    
    @property
    def cache_key(self, variable: str = 'streamflow') -> str:
        # Drop the leading '/' part of an absolute path--otherwise joining this
        # key onto a cache_dir would discard the cache_dir entirely (pathlib
        # treats joining an absolute path as replacing the left-hand side).
        return '_'.join(part for part in self.base.parts if part != '/')
    
    # @property
    # def url(self, variable: str = 'streamflow') -> ParseResult:
    #     return '_'.join(self.base.parts)

    #TODO: Support other-than-hourly timesteps?
    def _evaluate_path_template(self, template, tN, t0_fnum, init_dt):
        fnum = t0_fnum + (tN // 3600) # tN is in seconds; template deals in hours
        forecast_dt = init_dt + timedelta(hours=fnum)
        ftime_info = {
            "init_yyyymmdd": init_dt.strftime('%Y%m%d'),
            "init_hour": init_dt.hour, 
            "forecast_hour": fnum,
            "forecast_year": forecast_dt.year,
            "forecast_month": forecast_dt.month,
            "forecast_day": forecast_dt.day,
            "forecast_hourz": forecast_dt.hour
        }
        return template.format(**ftime_info)

    def get_source_for_t(self, n: int, variable: str = 'streamflow') -> Source:
        if variable not in self._vtm: # Assume legacy/basic function (fnum increments in the filename itself)
            new_fnum = self.t0_fnum + (n // 3600)
            new_base = self.base.with_stem(re.sub('f[0-9]{3}', f"f{new_fnum:03d}", self.base.stem))
            new_base_url = ParseResult(self.base_url.scheme, self.base_url.netloc, str(new_base), self.base_url.params, self.base_url.query, self.base_url.fragment)
            return Source(new_base, new_base_url, 0, variable_template_map=self._vtm)
        
        #TODO: Currently templates supported only in path
        relative_path = self._evaluate_path_template(
            template=self._vtm[variable],
            tN=n, t0_fnum=self.t0_fnum, init_dt=self._init_datetime
        )
        # The template is relative to `url_root`--rebuild the absolute location from scratch,
        # since (unlike the legacy fnum-substitution case) the new path isn't a simple edit of `base`.
        full_str = (self._url_root or '') + relative_path
        new_base_url = urlparse(full_str)
        new_base = PurePosixPath(new_base_url.path) if new_base_url.scheme != '' else Path(new_base_url.path)
        return Source(new_base, new_base_url, 0, variable_template_map=self._vtm, url_root=self._url_root)


class SourceManager(metaclass=SingletonMeta):
    _source_data_dict = {
        "NODD": {
            "url_base": "https://storage.googleapis.com/national-water-model/",
            "path_template": "nwm.{init_yyyymmdd}/{model_dir}/nwm.t{init_hour:02d}z.{model_name}.channel_rt{var_file_suffix}.f{forecast_hour:03d}.conus.nc",
            "terrain_path_template": "nwm.{init_yyyymmdd}/{model_dir}/nwm.t{init_hour:02d}z.{model_name}.terrain{var_file_suffix}.f{forecast_hour:03d}.conus.nc"
        },
        "NOMADS": {
            "url_base": "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwm/v3.0/",
            "path_template": "nwm.{init_yyyymmdd}/{model_dir}/nwm.t{init_hour:02d}z.{model_name}.channel_rt{var_file_suffix}.f{forecast_hour:03d}.conus.nc",
            "terrain_path_template": "nwm.{init_yyyymmdd}/{model_dir}/nwm.t{init_hour:02d}z.{model_name}.terrain{var_file_suffix}.f{forecast_hour:03d}.conus.nc"
        },
        "RETRO": {
            "url_base": "https://s3.amazonaws.com/noaa-nwm-retrospective-3-0-pds/CONUS/netcdf/",
            "path_template": "CHRTOUT/{forecast_year:04d}/{forecast_year:04d}{forecast_month:02d}{forecast_day:02d}{forecast_hourz:02d}00.CHRTOUT_DOMAIN1",
            "gwout_path_template": "GWOUT/{forecast_year:04d}/{forecast_year:04d}{forecast_month:02d}{forecast_day:02d}{forecast_hourz:02d}00.GWOUT_DOMAIN1",
        }
    }

    _model_path_data_dict = {
        "medium_range_mem1": {
            "version": "3.1",
            "model_dir": "medium_range_mem1",
            "model_name": "medium_range",
            "var_file_suffix": "_1",
            "hours": 240,
            "run_freq": 6,
            "lag": 5
        },
        "medium_range_blend": {
            "version": "3.1",
            "model_dir": "medium_range_blend",
            "model_name": "medium_range_blend",
            "var_file_suffix": "",
            "hours": 240,
            "run_freq": 6,
            "terrain_dt_h": 3, # Delta-T hours between forecast files/forecast hour numbers for terrain files
            "lag": 5
        },
        "short_range": {
            "version": "3.1",
            "model_dir": "short_range",
            "model_name": "short_range",
            "var_file_suffix": "",
            "hours": 18,
            "run_freq": 1,
            "terrain_dt_h": 1, # Delta-T hours between forecast files/forecast hour numbers for terrain files
            "lag": 1
        },
        "retrospective": {
            "version": "3.0",
            "model_dir": "",
            "model_name": "",
            "var_file_suffix": "",
            "hours": 1,
            "run_freq": 1,
            "lag": 0
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
                            source_dict['init_datetime'] = datetime.fromisoformat(source_dict['init_datetime'])
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

        if source_base is None:
            now = datetime.now(tz=timezone.utc)
            wayback = now - t0
            if wayback < timedelta(hours=40):
                source_base = "NOMADS"
            elif t0 > datetime(year=2023, month=9, day=20, tzinfo=timezone.utc):
                source_base = "NODD"
            else:
                source_base = "RETRO"

        #TODO: Make configurable!
        source_variant = "retrospective" if source_base == "RETRO" else "medium_range_mem1"
        variant_info = SourceManager._model_path_data_dict[source_variant]

        url_base = None
        path_template = None
        gwout_path_template = None
        terrain_path_template = None
        if source_base in SourceManager._source_data_dict:
            source_data = SourceManager._source_data_dict[source_base]
            url_base = source_data['url_base']
            path_template = source_data['path_template']
            gwout_path_template = source_data.get('gwout_path_template')
            terrain_path_template = source_data.get('terrain_path_template')
        else:
            url_base = source_base
        
        if path_template is None:
            # Assume a NOMADS path structure if none other has been derived...
            path_template = SourceManager._source_data_dict["NOMADS"]['path_template']

        pr = urlparse(url_base)
        base_url = pr

        # Retrospective data is addressed by absolute datetime rather than an
        # incrementing fnum, so it always needs the template/discovery path below
        # (it can't be a single fnum-bearing file to walk forward from).
        is_file = False
        if source_base != "RETRO":
            if pr.scheme != '':
                p = PurePosixPath(pr.path)
                if p.suffix == '.nc':
                    is_file = True
            else:
                p = Path(pr.path)
                if p.exists() and p.is_file():
                    is_file = True
        else:
            p = PurePosixPath(pr.path) if pr.scheme != '' else Path(pr.path)

        model_freq = variant_info.get('run_freq')
        model_lag = variant_info.get('lag')
        model_hours = variant_info.get('hours')
        model_stride = variant_info.get('stride', 1) # will need if we figure out how to support LR

        quantizer = timedelta(hours=model_freq).seconds
        attempt = min(t0, datetime.now(timezone.utc) - timedelta(hours=model_lag)) # e.g. no sooner than 5 hours
        attempt = datetime.fromtimestamp(
            (attempt.timestamp()//quantizer)*quantizer, # quantize to 6-hourly
            tz=timezone.utc)

        #TODO: Decompose somewhere around here so that it is possible to test above logic with mocks before making HTTP calls

        if source_variant == "retrospective":
            # One file per hour, keyed by absolute datetime--no cycle to search
            # backward through like the operational forecasts below.
            ftime_info = {
                "forecast_year": attempt.year,
                "forecast_month": attempt.month,
                "forecast_day": attempt.day,
                "forecast_hourz": attempt.hour,
            }
            attempt_str = path_template.format(**ftime_info, **variant_info)
            logger.info(f"Trying {url_base}{attempt_str}")
            if pr.scheme == '':
                if not (p / attempt_str).exists():
                    raise RuntimeError(f"Retrospective data file not found: {p / attempt_str}")
            else:
                req = Request(url=(url_base + attempt_str), method='HEAD')
                try:
                    with urlopen(req) as response:
                        status_code = response.getcode()
                except HTTPError as e:
                    status_code = e.code
                if status_code != 200:
                    raise RuntimeError(f"Retrospective data file not found (HTTP {status_code}): {url_base}{attempt_str}")

            pr = urlparse(url_base + attempt_str)
            base_url = pr
            p = PurePosixPath(pr.path) if pr.scheme != '' else Path(pr.path)
        elif not is_file:
            while True:
                t0_delta = t0 - attempt
                t0_forecast_hour = int(t0_delta.total_seconds() // 3600)
                logger.debug(f"{t0_delta=}")
                ftime_info = {
                    "init_hour": attempt.hour,
                    "forecast_hour": t0_forecast_hour,
                    "forecast_year": attempt.year,
                    "forecast_month": attempt.month,
                    "forecast_day": attempt.day,
                    "forecast_hourz": attempt.hour
                }
                attempt_str = path_template.format(init_yyyymmdd = attempt.strftime('%Y%m%d'), **ftime_info, **variant_info)
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

        if source_variant == "retrospective":
            # No forecast lead time--every hour is directly addressable.
            t0_fnum = 0
            init_datetime = attempt
        else:
            fnum_match = re.search('f([0-9]{3})', p.stem)
            if fnum_match is None:
                raise RuntimeError(f"Unable to determine forecast hour from source file name: {p.name}")
            t0_fnum = int(fnum_match.group(1))
            init_datetime = t0 - timedelta(hours=t0_fnum)

        if tend is not None:
            # Validate if end hour is possible to obtain...
            t0_tend_delta_hours = (tend - t0).total_seconds() // 3600
            if model_hours > 1 and t0_fnum + t0_tend_delta_hours > model_hours:
                raise ValueError(f"Simulation end date {tend} exceeds the data available for model {variant_info['model_name']} when starting at forecast hour {t0_fnum} (init_time {init_datetime.strftime('%Y%m%d')})")

        vtm = {}
        url_root = None
        if source_variant == "retrospective":
            # Retrospective paths change structurally between timesteps (e.g. year
            # folders), so--unlike operational fnum substitution--they need the
            # full template mechanism, evaluated relative to the bucket root.
            vtm['streamflow'] = path_template.format_map(_PartialFormatDict(variant_info))
            url_root = url_base
        #TODO: terrain_path_template / gwout_path_template support not yet implemented

        if psource:
            with open(psource, 'w') as fsource:
                json.dump({
                    'base': str(base),
                    'base_url': base_url.geturl(),
                    't0_fnum': t0_fnum,
                    'init_datetime': init_datetime.isoformat(),
                    'variable_template_map': vtm,
                    'url_root': url_root
                }, fsource)
            #with open(psource, 'r') as f: print(f.read())

        source = Source(base, base_url, t0_fnum, variable_template_map=vtm, init_datetime=init_datetime, url_root=url_root)
        SourceManager._source_cache[cache_key] = source
        return source

    def get_dataset(self, source: Source, tN: int) -> xr.Dataset:
        max_retries = 5
        retry_backoff_start = 5
        sN = source.get_source_for_t(tN)
        p = sN.base
        source_str = sN.base_url.geturl()

        if self._cache_dir:
            p = self._cache_dir / sN.cache_key
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
            ds = xr.open_dataset(source_str + '#mode=bytes')
        
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
        
        try:
            (self._cache_dir / 'leader.id').unlink()
            (self._cache_dir / 'source.json').unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clean up leader files (check permissions?)")

        self._is_leader = False

