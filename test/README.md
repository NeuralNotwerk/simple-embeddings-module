# SEM Test Suite

This directory contains tests for the Simple Embeddings Module (SEM).

## Available Tests

### Smoke Test (`smoke_test.py`)

A comprehensive end-to-end test that verifies all core functionality:

- ✅ Python imports and module loading
- ✅ CLI command availability (`sem-cli`)
- ✅ GPU acceleration detection (MPS/CUDA/CPU)
- ✅ Database initialization
- ✅ Document addition and processing
- ✅ Semantic search functionality
- ✅ Search result quality verification

## Running Tests

### Quick Test
```bash
# Run just the smoke test
python test/smoke_test.py
```

### Full Test Suite
```bash
# Run all tests
./test/run_tests.sh
```

### From Virtual Environment
```bash
# Activate environment first
source .venv/bin/activate

# Then run tests
python test/smoke_test.py
```

## Test Requirements

- SEM package must be installed (`pip install -e .`)
- All dependencies must be available
- Sufficient disk space for temporary test files
- Internet connection for model downloads (first run only)

## Expected Output

Successful test run should show:
```
🧪 Simple Embeddings Module (SEM) Smoke Test
==================================================
🐍 Testing Python imports...
  ✅ Main module import successful
  ✅ SEMConfigBuilder import successful
  ✅ Default config creation successful
  ✅ All base classes import successful
🔧 Testing CLI availability...
  ✅ sem-cli command available and working
🚀 Testing GPU acceleration...
  ✅ Apple Silicon MPS acceleration available
🚀 Testing end-to-end functionality...
  📝 Created test documents
  ✅ Database initialization successful
  ✅ Document addition successful
  ✅ Database info successful
  ✅ Search functionality successful
  ✅ Semantic search quality verified

==================================================
📊 Test Results: 4/4 passed
⏱️  Total time: 15.23s
🎉 All tests passed! SEM is working correctly.
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure SEM is installed (`pip install -e .`)
2. **CLI Not Found**: Check that installation included console scripts
3. **GPU Detection**: Normal to see CPU-only on systems without GPU
4. **Model Download**: First run downloads ~90MB model (cached afterward)
5. **Timeout**: Large models may take longer on slower systems

### Test Failures

If tests fail, check:
- Virtual environment is activated
- All dependencies are installed
- Sufficient disk space available
- Network connectivity for model downloads
