#!/bin/bash
"""
Test runner for Simple Embeddings Module (SEM)

This script runs all available tests for the SEM package.
"""

set -e  # Exit on any error

echo "🧪 Simple Embeddings Module Test Suite"
echo "======================================"

# Check if we're in a virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider running: source .venv/bin/activate"
    echo ""
fi

# Run smoke test
echo "🔥 Running smoke test..."
python test/smoke_test.py

echo ""
echo "✅ All tests completed!"
