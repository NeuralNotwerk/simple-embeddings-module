#!/bin/bash
"""
Test runner for Simple Embeddings Module (SEM)

This script runs all available tests for the SEM package.
Usage: ./run_tests.sh [-v|--verbose]
"""

set -e  # Exit on any error

# Parse command line arguments
VERBOSE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-v|--verbose]"
            exit 1
            ;;
    esac
done

echo "🧪 Simple Embeddings Module Test Suite"
echo "======================================"

if [[ -n "$VERBOSE" ]]; then
    echo "🔍 VERBOSE MODE ENABLED - Tests will show raw inputs and outputs"
    echo ""
fi

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
echo "🌳 Running semantic chunking tests..."
python test/test_semantic_chunking.py $VERBOSE

echo ""
echo "🧩 Running code chunking provider tests..."
python test/test_code_chunking_provider.py $VERBOSE

echo ""
echo "🔗 Running hierarchy grouping tests..."
python test/test_hierarchy_grouping.py $VERBOSE

echo ""
echo "⚡ Running lazy loading tests..."
python test/test_lazy_loading.py

echo ""
echo "✅ All tests completed!"

if [[ -n "$VERBOSE" ]]; then
    echo ""
    echo "📊 Verbose mode showed:"
    echo "  • Raw input content for all tests"
    echo "  • Raw output data structures"
    echo "  • Embedding details and similarity matrices"
    echo "  • Detailed constraint validation"
    echo "  • Full chunk content and metadata"
fi
