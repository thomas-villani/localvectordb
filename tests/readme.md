# LocalVectorDB Test Suite

This directory contains a comprehensive test suite for the LocalVectorDB library. The tests are organized by component and include unit tests, integration tests, and performance benchmarks.

## 📁 Test Structure

```
tests/
├── conftest.py              # Common fixtures and utilities
├── pytest.ini              # Pytest configuration
├── test_runner.py          # Test runner script
├── test_core.py            # Core components tests
├── test_embeddings.py      # Embedding providers tests
├── test_chunking.py        # Text chunking tests
├── test_database.py        # Main database functionality tests
├── test_client.py          # Remote client tests
├── test_factory.py         # Factory function tests
├── test_exceptions.py      # Exception handling tests
├── test_utils.py           # Utility functions tests
├── test_init.py            # Package import tests
├── test_integration.py     # Integration tests
├── test_performance.py     # Performance benchmarks
├── ... maybe more?
└── README.md               # This file
```

## 🏷️ Test Categories

Tests are organized using pytest markers:

- **`unit`**: Unit tests for individual components
- **`integration`**: Integration tests that test multiple components together
- **`slow`**: Tests that may take longer to run (>5 seconds)
- **`network`**: Tests that require network access (currently mocked)
- **`database`**: Tests involving database operations
- **`embedding`**: Tests involving embedding operations
- **`chunking`**: Tests involving text chunking
- **`client`**: Tests for remote client functionality
- **`performance`**: Performance and benchmark tests

## 🚀 Quick Start

### Prerequisites

Install testing dependencies:

```bash
pip install pytest pytest-cov pytest-mock pytest-xdist psutil
```

### Running Tests

```bash
# Run all tests
pytest

# Run fast tests only
pytest -m "not slow and not network"

# Run unit tests only
pytest -m unit

# Run with coverage
pytest --cov=localvectordb --cov-report=html

# Run specific test file
pytest test_core.py

# Run tests matching pattern
pytest -k "test_database"

# Run in parallel
pytest -n auto

# Verbose output
pytest -v
```

## 📊 Test Coverage

The test suite aims for high code coverage:

- **Target**: 85%+ overall coverage
- **Core modules**: 90%+ coverage
- **Critical paths**: 95%+ coverage

View coverage reports:

```bash
# Generate coverage report
pytest --cov=localvectordb --cov-report=html

# Open HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## 🧪 Test Types

### Unit Tests

Test individual components in isolation:

- **test_core.py**: Core data structures and utilities
- **test_embeddings.py**: Embedding provider implementations
- **test_chunking.py**: Text chunking algorithms
- **test_exceptions.py**: Exception classes
- **test_utils.py**: Utility functions
- **test_init.py**: Package imports and structure

### Integration Tests

Test component interactions:

- **test_integration.py**: Cross-component functionality
- **test_database.py**: Database operations (with mocked dependencies)
- **test_client.py**: Remote client functionality
- **test_factory.py**: Factory function behavior

### Performance Tests

Benchmark performance characteristics:

- **test_performance.py**: Scalability and performance benchmarks
- Memory usage patterns
- Concurrent operation handling
- Large dataset processing

## 🔧 Configuration

### Pytest Configuration

Key settings in `pytest.ini`:

```ini
[tool:pytest]
testpaths = tests
markers = 
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests
    performance: Performance tests
addopts = 
    --strict-markers
    --cov=localvectordb
    --cov-fail-under=85
```

### Test Fixtures

Common fixtures in `conftest.py`:

- `temp_dir`: Temporary directory for test databases
- `sample_metadata_schema`: Standard metadata schema for testing
- `sample_documents`: Test document content
- `mock_embeddings`: Mock embedding provider
- `mock_faiss_index`: Mock FAISS index

## 🎯 Writing New Tests

### Guidelines

1. **Follow naming conventions**: `test_*.py` files, `test_*` functions
2. **Use appropriate markers**: Mark tests with relevant categories
3. **Mock external dependencies**: Use mocks for databases, APIs, file systems
4. **Test edge cases**: Include tests for error conditions and edge cases
5. **Keep tests fast**: Unit tests should run in milliseconds
6. **Document complex tests**: Add docstrings explaining test purpose

### Example Test

```python
import pytest
from unittest.mock import Mock, patch

from localvectordb.core import MetadataField, MetadataFieldType

class TestMetadataField:
    """Test MetadataField functionality."""
    
    def test_create_text_field(self):
        """Test creating a text metadata field."""
        field = MetadataField(type=MetadataFieldType.TEXT, indexed=True)
        
        assert field.type == MetadataFieldType.TEXT
        assert field.indexed is True
        assert field.required is False
    
    @pytest.mark.parametrize("field_type,expected", [
        (str, MetadataFieldType.TEXT),
        (int, MetadataFieldType.INTEGER),
        (float, MetadataFieldType.REAL),
    ])
    def test_type_conversion(self, field_type, expected):
        """Test automatic type conversion."""
        field = MetadataField(type=field_type)
        assert field.type == expected
```

### Test Organization

```python
class TestComponentName:
    """Test suite for ComponentName."""
    
    @pytest.fixture
    def component_instance(self):
        """Create component instance for testing."""
        return ComponentName()
    
    def test_basic_functionality(self, component_instance):
        """Test basic component functionality."""
        pass
    
    def test_error_handling(self, component_instance):
        """Test error handling."""
        with pytest.raises(ValueError):
            component_instance.invalid_operation()
    
    @pytest.mark.slow
    def test_performance_characteristic(self, component_instance):
        """Test performance characteristic (marked as slow)."""
        pass
```

## 🚨 Troubleshooting

### Common Issues

#### Tests Fail with Import Errors

```bash
# Ensure package is installed in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Coverage Reports Missing

```bash
# Install coverage dependencies
pip install pytest-cov

# Ensure source code is in the right location
pytest --cov=localvectordb --cov-report=term-missing
```

#### Slow Test Performance

```bash
# Run only fast tests
pytest -m "not slow"

# Use parallel execution
pytest -n auto

# Skip performance tests
pytest -m "not performance"
```

#### Mock-related Errors

- Ensure mocks are properly configured
- Check that patch paths are correct
- Verify mock return values match expected types

### Debugging Tests

```bash
# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Show local variables in tracebacks
pytest -l

# Run specific test with output
pytest -s test_file.py::test_function
```

## 📈 Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[test]
    
    - name: Run tests
      run: |
        python test_runner.py fast --coverage
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 🤝 Contributing

### Adding New Tests

1. **Identify test category**: Unit, integration, or performance
2. **Choose appropriate file**: Add to existing or create new test file
3. **Add markers**: Mark tests with appropriate categories
4. **Mock dependencies**: Use mocks for external dependencies
5. **Update documentation**: Document any new test utilities

### Test Review Checklist

- [ ] Tests follow naming conventions
- [ ] Appropriate markers are used
- [ ] External dependencies are mocked
- [ ] Tests cover both success and error cases
- [ ] Tests are reasonably fast (except those marked `slow`)
- [ ] Docstrings explain test purpose
- [ ] No hardcoded paths or credentials

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Python Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

## 🆘 Support

For questions about the test suite:

1. Check this README and test documentation
2. Look at existing test examples
3. Review pytest documentation
4. Ask in project discussions or issues

---

*Happy Testing! 🧪*