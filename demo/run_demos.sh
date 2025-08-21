#!/bin/bash
"""
Demo runner for Simple Embeddings Module (SEM)

This script runs all available demos for the SEM package.
"""

set -e  # Exit on any error

echo "🎭 Simple Embeddings Module Demo Suite"
echo "======================================"

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

echo ""
echo "✅ All demos completed!"
