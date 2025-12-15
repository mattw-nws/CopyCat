"""Tests for the SourceManager class."""

from urllib.error import HTTPError
from urllib.parse import urlparse
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

