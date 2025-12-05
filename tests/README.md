# SourceManager Tests

Comprehensive pytest test suite for the `SourceManager` class in the copycatbmi package.

## Requirements

- Python 3.9+
- pytest
- copycatbmi (the package being tested)

## Test Coverage

The test suite includes tests for:

### Core Classes
- **SingletonMeta**: Tests for the singleton metaclass implementation
- **Source**: Tests for the Source data class

### SourceManager Initialization
- Cache directory creation and handling
- UUID generation
- Initialization with and without cache directories

### Context Manager
- Entry and exit behavior
- Nested context management
- Reference counting (_entries)

### Leader Election
- Leader election on initialization
- Leader file creation and UUID persistence
- Non-leader behavior without cache directory

### Source Derivation
- NOMADS source selection for recent dates
- NODD source selection for older dates
- Explicit source_base specification
- RETRO source error handling
- End date validation

### Dataset Retrieval
- Loading from cache
- Loading without cache
- NaN value handling in streamflow data

### Utility Functions
- Forecast hour calculation
- Timestamp quantization to 6-hour boundaries
- Path template handling

### Error Handling
- Max retries exceeded scenarios
- HTTP error handling
- File not found scenarios

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run specific test class
pytest tests/test_source_manager.py::TestSourceManagerInitialization

# Run specific test
pytest tests/test_source_manager.py::TestSourceManagerInitialization::test_init_with_cache_dir

# Run with coverage report
pytest --cov=copycatbmi tests/
```

## Test Organization

Tests are organized into test classes by functionality:

- `TestSingletonMeta`: Singleton pattern tests
- `TestSource`: Source class tests
- `TestSourceManagerInitialization`: Initialization tests
- `TestSourceManagerContextManager`: Context manager tests
- `TestLeaderElection`: Leader election tests
- `TestSourceDataDict`: Data dictionary validation
- `TestDeriveSourceNomadsPath`: NOMADS source derivation
- `TestDeriveSourceRetro`: Retrospective source tests
- `TestDeriveSourceEndDate`: End date validation
- `TestGetDataset`: Dataset retrieval tests
- `TestDeriveSourcePathTemplate`: Path template tests
- `TestForecastHourCalculation`: Forecast hour math tests
- `TestQuantizationLogic`: Timestamp quantization tests
- `TestIntegrationSourceManager`: Integration tests
- `TestErrorHandling`: Error scenario tests

## Fixtures

The test suite uses pytest fixtures defined in `conftest.py`:

- `temp_cache_dir`: Creates a temporary directory for cache testing
- `mock_datetime_utc`: Provides a reference UTC datetime

## Mocking Strategy

Tests use `unittest.mock` extensively to:
- Mock network requests (urlopen)
- Mock file system operations
- Mock xarray dataset operations
- Avoid external dependencies
