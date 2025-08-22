#!/bin/bash
"""
Demo runner for Simple Embeddings Module (SEM)

This script runs all available demos for the SEM package.
Usage: ./run_demos.sh [--include-s3]
"""

set -e  # Exit on any error

# Parse command line arguments
INCLUDE_S3=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --include-s3)
            INCLUDE_S3=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--include-s3]"
            exit 1
            ;;
    esac
done

echo "🎭 Simple Embeddings Module Demo Suite"
echo "======================================"

if [[ "$INCLUDE_S3" == true ]]; then
    echo "☁️  S3 DEMOS ENABLED - Requires AWS credentials and S3 bucket"
    echo ""
fi

# Check if we're in a virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider running: source .venv/bin/activate"
    echo ""
fi

echo "🌳 Running semantic chunking demo..."
python demo/demo_semantic_chunking.py

echo ""
echo "🔍 Running code search example..."
python demo/example_code_search.py

echo ""
echo "🚀 Running ultimate code chunking demo..."
python demo/demo_ultimate_code_chunking.py

echo ""
echo "🧩 Running hierarchy grouping demo..."
python demo/demo_hierarchy_grouping.py

# Run S3 demo if requested
if [[ "$INCLUDE_S3" == true ]]; then
    echo ""
    echo "☁️  Running S3 storage demo..."
    if [[ -z "${SEM_S3_BUCKET}" ]]; then
        echo "⚠️  Warning: SEM_S3_BUCKET not set - skipping S3 demo"
        echo "   Set environment variable: export SEM_S3_BUCKET=your-bucket-name"
    else
        python demo/demo_s3_storage.py
    fi
fi

echo ""
echo "✅ All demos completed!"

if [[ "$INCLUDE_S3" == true ]]; then
    echo ""
    echo "☁️  S3 demos showcased:"
    echo "  • Cloud-native semantic search with S3"
    echo "  • Secure storage with encryption and compression"
    echo "  • Global accessibility and scalability"
    echo "  • Production-ready configuration examples"
fi
