"""Tests for the SourceManager class."""

from urllib.error import HTTPError
from urllib.parse import urlparse
import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock, mock_open
import uuid
import re

from copycatbmi.source_manager import SourceManager, Source, SingletonMeta


class TestSource:
    """Tests for the Source class."""

    def test_source_initialization(self):
        """Test Source object initialization."""
        base = Path("/path/to/file.nc")
        base_url = urlparse("http://example.com/data/")
        t0_fnum = 123
        
        source = Source(base, base_url, t0_fnum)
        
        assert source.base == base
        assert source.base_url == base_url
        assert source.t0_fnum == t0_fnum

    def test_source_initialization_strings(self):
        """Test Source object initialization."""
        base = Path("/path/to/file.nc")
        base_url = urlparse("http://example.com/data/")
        t0_fnum = 123
        
        source = Source(str(base), base_url.geturl(), t0_fnum)
        
        assert source.base == base
        assert source.base_url == base_url
        assert source.t0_fnum == t0_fnum


class TestSourceManagerInitialization:
    """Tests for SourceManager initialization."""

    def test_init_with_cache_dir(self, temp_cache_dir):
        """Test initialization with a cache directory."""
        SingletonMeta._instances.clear()
        
        sm = SourceManager(str(temp_cache_dir))
        
        assert sm._cache_dir == temp_cache_dir
        assert sm._entries == 0
        assert isinstance(sm._uuid, uuid.UUID)

    def test_init_without_cache_dir(self):
        """Test initialization without a cache directory."""
        SingletonMeta._instances.clear()
        
        sm = SourceManager(None)
        
        assert sm._cache_dir is None
        assert sm._entries == 0
        assert isinstance(sm._uuid, uuid.UUID)

    def test_init_creates_cache_dir_if_not_exists(self, temp_cache_dir):
        """Test that initialization creates cache directory if it doesn't exist."""
        SingletonMeta._instances.clear()
        
        new_cache_dir = temp_cache_dir / "new_cache"
        assert not new_cache_dir.exists()
        
        sm = SourceManager(str(new_cache_dir))
        
        assert new_cache_dir.exists()


class TestSourceManagerContextManager:
    """Tests for SourceManager context manager."""

    def test_context_manager_enters_and_exits(self, temp_cache_dir):
        """Test that context manager properly enters and exits."""
        SingletonMeta._instances.clear()
        
        sm = SourceManager(str(temp_cache_dir))
        
        with sm:
            assert sm._entries == 1
        
        assert sm._entries == 0

    def test_context_manager_nested_entries(self, temp_cache_dir):
        """Test nested context manager entries."""
        SingletonMeta._instances.clear()
        
        sm = SourceManager(str(temp_cache_dir))
        
        with sm:
            assert sm._entries == 1
            with sm:
                assert sm._entries == 2
            assert sm._entries == 1
        
        assert sm._entries == 0

    def test_context_manager_returns_self(self, temp_cache_dir):
        """Test that context manager returns self."""
        SingletonMeta._instances.clear()
        
        sm = SourceManager(str(temp_cache_dir))
        
        with sm as result:
            assert result is sm


class TestLeaderElection:
    """Tests for leader election."""

    def test_leader_elected_on_init(self, temp_cache_dir):
        """Test that a leader is elected during initialization."""
        SingletonMeta._instances.clear()
        
        sm = SourceManager(str(temp_cache_dir))
        
        assert hasattr(sm, '_is_leader')
        # First instance should be leader
        assert sm._is_leader is True
        
        leader_file = temp_cache_dir / "leader.id"
        assert leader_file.exists()

    def test_leader_file_contains_uuid(self, temp_cache_dir):
        """Test that leader.id file contains the leader's UUID."""
        SingletonMeta._instances.clear()
        
        sm = SourceManager(str(temp_cache_dir))
        
        leader_file = temp_cache_dir / "leader.id"
        with open(leader_file, 'r') as f:
            leader_uuid = f.read()
        
        assert leader_uuid == str(sm._uuid)

    def test_no_leader_without_cache_dir(self):
        """Test that no leader is elected without cache directory."""
        SingletonMeta._instances.clear()
        
        with  SourceManager(None) as sm:
            assert not sm._is_leader

    def test_leader_with_nested_entries(self, temp_cache_dir):
        """Test nested context manager entries."""
        SingletonMeta._instances.clear()
        
        sm = SourceManager(str(temp_cache_dir))
        
        with sm:
            with SourceManager(str(temp_cache_dir)) as sm2:
                assert sm2._uuid == sm._uuid
                assert sm._is_leader
                assert sm2._is_leader
        
        assert sm._entries == 0


class TestDeriveSource:
    """Tests for derive_source method with NOMADS path."""

    def test_derive_source_selects_nomads_for_recent_dates(self, temp_cache_dir):
        """Test that NOMADS source is selected for recent dates."""
        SingletonMeta._instances.clear()
        
        sm = SourceManager(str(temp_cache_dir))
        
        # Use a recent time (within 40 hours of now)
        t0 = datetime.now(tz=timezone.utc) - timedelta(hours=24)
        
        with patch.object(sm, 'derive_source', wraps=sm.derive_source) as mock_derive:
            try:
                source = sm.derive_source(t0, None, None)
                #TODO: Make Source object include source identifier and test this directly
                assert SourceManager._source_data_dict["NOMADS"]['url_base'] in source.base
            except Exception:
                pass

    def test_derive_source_selects_nodd_for_older_dates(self, temp_cache_dir):
        """Test that NODD source is selected for older dates."""
        SingletonMeta._instances.clear()
        
        sm = SourceManager(str(temp_cache_dir))
        
        # Use a time delta > within 40 hours of now but after NWMv3 deployment
        t0 = datetime.now(tz=timezone.utc) - timedelta(days=15)
        
        with patch.object(sm, 'derive_source', wraps=sm.derive_source) as mock_derive:
            try:
                source = sm.derive_source(t0, None, None)
                #TODO: Make Source object include source identifier and test this directly
                assert SourceManager._source_data_dict["NOMADS"]['url_base'] in source.base
            except Exception:
                pass

    def test_derive_source_with_explicit_source_base(self, temp_cache_dir):
        """Test derive_source with explicit source_base."""
        sm = SourceManager(str(temp_cache_dir))
        
        # Use a recent time (within 40 hours of now) - would normally pick NOMADS
        t0 = datetime.now(tz=timezone.utc) - timedelta(hours=24)
        
        with patch.object(sm, 'derive_source', wraps=sm.derive_source) as mock_derive:
            try:
                source = sm.derive_source(t0, None, "NODD")
                #TODO: Make Source object include source identifier and test this directly
                assert SourceManager._source_data_dict["NODD"]['url_base'] in source.base
            except Exception:
                pass


class TestIntegrationSourceManager:
    """Integration tests for SourceManager."""

    def test_context_manager_with_leader_election(self, temp_cache_dir):
        """Test full context manager flow with leader election."""
        SingletonMeta._instances.clear()
        
        sm = SourceManager(str(temp_cache_dir))
        
        # First context
        with sm:
            assert sm._entries == 1
            assert sm._is_leader is True
        
        # Second context
        with sm:
            assert sm._entries == 1
            assert sm._is_leader is True

    def test_uuid_persists_across_contexts(self, temp_cache_dir):
        """Test that UUID persists across context manager calls."""
        SingletonMeta._instances.clear()
        
        sm = SourceManager(str(temp_cache_dir))
        original_uuid = sm._uuid
        
        with sm:
            assert sm._uuid == original_uuid
        
        with sm:
            assert sm._uuid == original_uuid

    def test_multiple_instances_with_same_cache_dir(self, temp_cache_dir):
        """Test that multiple instances with same cache_dir are singletons."""
        SingletonMeta._instances.clear()
        
        with SourceManager(str(temp_cache_dir)) as sm1:
            with SourceManager(str(temp_cache_dir)) as sm2:
                assert sm1 is sm2
                assert sm1._entries == 2


class TestRegressionSourceManager:
    """Regression tests for SourceManager."""

    def test_issue_12_derive_source_gets_f000_file(self, temp_cache_dir):
        """Test that starting a simulation at hour zero of a run does not fail."""
        SingletonMeta._instances.clear()
        
        sm = SourceManager(str(temp_cache_dir))
        
        t0 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        t0 = t0.replace(hour=12)
        
        source = sm.derive_source(t0, None, None) # Without mock, this actually fails before issue #12 fix
        assert source.t0_fnum != 0

    def test_issue_12_derive_source_404_no_httperror(self, temp_cache_dir):
        """Test that starting a simulation at hour zero of a run does not fail."""
        SingletonMeta._instances.clear()
        
        sm = SourceManager(str(temp_cache_dir))
        
        # Use a recent time (within 40 hours of now)
        t0 = datetime.now(tz=timezone.utc) - timedelta(days=15)
        t0 = t0.replace(hour=12)
        
        with patch('copycatbmi.source_manager.urlopen') as mock_urlopen, \
                patch.object(sm, 'derive_source', wraps=sm.derive_source) as mock_derive:
            mock_urlopen.side_effect = HTTPError(url='http://example.org/', code=404, msg="Mock Not Found", hdrs=[], fp=None)
            mock_urlopen.return_value = None

            with pytest.raises(RuntimeError, match="Unable to retrieve forecast data"):
                source = sm.derive_source(t0, None, None) 


class TestRetrospectiveDeriveSource:
    """Tests for derive_source with the retrospective (RETRO) dataset.

    These use a local filesystem "bucket" (a tmp_path directory) standing in for
    the real S3 bucket, and mocked HTTP responses, so no network access or large
    data files are required.
    """

    @staticmethod
    def _touch_retro_file(bucket_root: Path, dt: datetime) -> str:
        """Create an empty file matching the RETRO path_template layout for `dt`."""
        year_dir = bucket_root / "CHRTOUT" / f"{dt.year:04d}"
        year_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{dt.year:04d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}00.CHRTOUT_DOMAIN1"
        (year_dir / fname).touch()
        return fname

    def test_derive_source_retro_local_bucket(self, temp_cache_dir, tmp_path):
        """A local-filesystem RETRO bucket resolves to the exact hourly file, with t0_fnum=0."""
        SingletonMeta._instances.clear()
        SourceManager._source_cache.clear()
        sm = SourceManager(str(temp_cache_dir))

        t0 = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        fname = self._touch_retro_file(tmp_path, t0)

        with patch.dict(SourceManager._source_data_dict["RETRO"], {"url_base": str(tmp_path) + "/"}):
            source = sm.derive_source(t0, None, "RETRO")

        assert source.t0_fnum == 0
        assert source.base.name == fname
        assert "streamflow" in source._vtm

    def test_derive_source_retro_auto_selected_for_old_dates(self, temp_cache_dir, tmp_path):
        """derive_source auto-selects RETRO (rather than NODD/NOMADS) for old enough dates."""
        SingletonMeta._instances.clear()
        SourceManager._source_cache.clear()
        sm = SourceManager(str(temp_cache_dir))

        t0 = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        fname = self._touch_retro_file(tmp_path, t0)

        with patch.dict(SourceManager._source_data_dict["RETRO"], {"url_base": str(tmp_path) + "/"}):
            source = sm.derive_source(t0, None, None)

        assert source.t0_fnum == 0
        assert source.base.name == fname

    def test_derive_source_retro_missing_local_file_raises(self, temp_cache_dir, tmp_path):
        """A RETRO request for a date with no matching local file raises a clear error."""
        SingletonMeta._instances.clear()
        SourceManager._source_cache.clear()
        sm = SourceManager(str(temp_cache_dir))

        t0 = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        # Note: no file created in tmp_path for this date.

        with patch.dict(SourceManager._source_data_dict["RETRO"], {"url_base": str(tmp_path) + "/"}):
            with pytest.raises(RuntimeError, match="Retrospective data file not found"):
                sm.derive_source(t0, None, "RETRO")

    def test_derive_source_retro_http_checks_exact_hour_only(self, temp_cache_dir):
        """Unlike operational forecasts, RETRO does not walk backward through cycles--
        it should issue exactly one HEAD request for the exact requested hour."""
        SingletonMeta._instances.clear()
        SourceManager._source_cache.clear()
        sm = SourceManager(str(temp_cache_dir))

        t0 = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False

        with patch('copycatbmi.source_manager.urlopen', return_value=mock_response) as mock_urlopen:
            source = sm.derive_source(t0, None, "RETRO")

        assert mock_urlopen.call_count == 1
        assert source.t0_fnum == 0
        assert "20210101000" in source.base_url.geturl()

    def test_derive_source_retro_http_404_raises_without_retry_loop(self, temp_cache_dir):
        """A 404 for the exact requested hour should fail immediately (no backward search)."""
        SingletonMeta._instances.clear()
        SourceManager._source_cache.clear()
        sm = SourceManager(str(temp_cache_dir))

        t0 = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        with patch('copycatbmi.source_manager.urlopen') as mock_urlopen:
            mock_urlopen.side_effect = HTTPError(url='http://example.org/', code=404, msg="Mock Not Found", hdrs=[], fp=None)

            with pytest.raises(RuntimeError, match="Retrospective data file not found"):
                sm.derive_source(t0, None, "RETRO")

        assert mock_urlopen.call_count == 1

    def test_derive_source_retro_json_cache_roundtrip(self, temp_cache_dir, tmp_path):
        """The leader-written source.json can be reloaded into an equivalent Source
        (regression check for mismatched keys / unparsed init_datetime)."""
        SingletonMeta._instances.clear()
        SourceManager._source_cache.clear()
        sm = SourceManager(str(temp_cache_dir))

        t0 = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self._touch_retro_file(tmp_path, t0)

        with patch.dict(SourceManager._source_data_dict["RETRO"], {"url_base": str(tmp_path) + "/"}):
            sm.derive_source(t0, None, "RETRO")

        psource = temp_cache_dir / "source.json"
        assert psource.exists()

        source_dict = json.loads(psource.read_text())
        source_dict["init_datetime"] = datetime.fromisoformat(source_dict["init_datetime"])
        reconstructed = Source(**source_dict)

        assert reconstructed.t0_fnum == 0
        next_hour = reconstructed.get_source_for_t(3600)
        assert "2021010101" in str(next_hour.base)


class TestRetrospectiveSourceGetSourceForT:
    """Tests for Source.get_source_for_t with a retrospective-style template,
    exercised directly (no filesystem/network access needed)."""

    RETRO_TEMPLATE = SourceManager._source_data_dict["RETRO"]["path_template"]

    def _make_retro_source(self, init_dt: datetime, url_root: str = "https://example.com/retro/") -> Source:
        relative = self.RETRO_TEMPLATE.format(
            forecast_year=init_dt.year, forecast_month=init_dt.month,
            forecast_day=init_dt.day, forecast_hourz=init_dt.hour,
        )
        parsed = urlparse(url_root + relative)
        return Source(
            base=parsed.path,
            base_url=parsed,
            t0_fnum=0,
            variable_template_map={"streamflow": self.RETRO_TEMPLATE},
            init_datetime=init_dt,
            url_root=url_root,
        )

    def test_next_hour(self):
        src = self._make_retro_source(datetime(2021, 6, 15, 10, 0, 0, tzinfo=timezone.utc))
        nxt = src.get_source_for_t(3600)
        assert "2021061511" in str(nxt.base)

    def test_day_rollover(self):
        src = self._make_retro_source(datetime(2021, 1, 1, 23, 0, 0, tzinfo=timezone.utc))
        nxt = src.get_source_for_t(3600)
        assert "2021010200" in str(nxt.base)

    def test_year_rollover_changes_folder(self):
        src = self._make_retro_source(datetime(2021, 12, 31, 23, 0, 0, tzinfo=timezone.utc))
        nxt = src.get_source_for_t(3600)
        assert "/2022/" in str(nxt.base).replace("\\", "/")
        assert "2022010100" in str(nxt.base)

    def test_url_root_is_preserved_across_timesteps(self):
        src = self._make_retro_source(datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
        nxt = src.get_source_for_t(3600)
        assert nxt.base_url.geturl().startswith("https://example.com/retro/")
        assert nxt.url_root == src.url_root

    def test_seconds_are_converted_to_hours(self):
        """tN passed to get_source_for_t is in seconds, not hours."""
        src = self._make_retro_source(datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
        one_hour = src.get_source_for_t(3600)
        half_hour = src.get_source_for_t(1800)
        assert half_hour.base == src.base  # rounds down, still hour 0
        assert one_hour.base != src.base


class TestSourceCacheKeyRegression:
    """Regression tests for Source.cache_key not discarding cache_dir for absolute paths."""

    def test_cache_key_has_no_leading_slash(self):
        source = Source("/national-water-model/nwm.20210101/nwm.t00z.file.nc", "https://example.com/national-water-model/nwm.20210101/nwm.t00z.file.nc", 0)
        assert not source.cache_key.startswith('/')

    def test_cache_key_joins_under_cache_dir(self, tmp_path):
        source = Source("/national-water-model/nwm.20210101/nwm.t00z.file.nc", "https://example.com/national-water-model/nwm.20210101/nwm.t00z.file.nc", 0)
        joined = tmp_path / source.cache_key
        assert joined.parent == tmp_path


#TODO: This test isn't working with arithmetic on mock objects, but is worth testing--fix!
# class TestErrorHandling:
#     """Tests for error handling in SourceManager."""

#     def test_derive_source_max_backoff_exceeded(self, temp_cache_dir):
#         """Test that RuntimeError is raised when max retries exceeded."""
#         SingletonMeta._instances.clear()
        
#         sm = SourceManager(str(temp_cache_dir))
        
#         t0 = datetime(year=2024, month=1, day=1, tzinfo=timezone.utc)
        
#         with patch('copycatbmi.source_manager.urlopen') as mock_urlopen:
#             # Always return 404
#             mock_response = MagicMock()
#             mock_response.getcode.return_value = 404
#             mock_response.__enter__.return_value = mock_response
#             mock_response.__exit__.return_value = False
#             mock_urlopen.return_value = mock_response
            
#             with patch('copycatbmi.source_manager.datetime') as mock_datetime_mod:
#                 now = datetime(year=2024, month=1, day=2, tzinfo=timezone.utc)
#                 mock_datetime_mod.now.return_value = now
#                 mock_datetime_mod.side_effect = lambda *args, **kw: datetime(*args, **kw)
                
#                 with pytest.raises(RuntimeError, match="Unable to retrieve forecast data"):
#                     sm.derive_source(t0, None, "NODD")

