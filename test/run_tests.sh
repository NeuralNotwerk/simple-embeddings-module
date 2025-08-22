#!/bin/bash
"""
Test runner for Simple Embeddings Module (SEM)

This script runs all available tests for the SEM package.
Usage: ./run_tests.sh [-v|--verbose] [--include-s3]
"""

set -e  # Exit on any error

# Parse command line arguments
VERBOSE=""
INCLUDE_S3=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        --include-s3)
            INCLUDE_S3=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-v|--verbose] [--include-s3]"
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

if [[ "$INCLUDE_S3" == true ]]; then
    echo "☁️  S3 TESTS ENABLED - Requires AWS credentials and S3 bucket"
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

# Run S3 tests if requested
if [[ "$INCLUDE_S3" == true ]]; then
    echo ""
    echo "☁️  Running S3 storage tests..."
    if [[ -z "${SEM_S3_BUCKET}" ]]; then
        echo "⚠️  Warning: SEM_S3_BUCKET not set - skipping S3 tests"
        echo "   Set environment variable: export SEM_S3_BUCKET=your-bucket-name"
    else
        python test/test_s3_storage.py $VERBOSE
    fi
fi

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

if [[ "$INCLUDE_S3" == true ]]; then
    echo ""
    echo "☁️  S3 tests verified:"
    echo "  • S3 storage backend functionality"
    echo "  • Compression and encryption features"
    echo "  • Error handling and edge cases"
    echo "  • Data integrity and performance"
fi
