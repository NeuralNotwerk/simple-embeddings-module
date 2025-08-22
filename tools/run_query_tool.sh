#!/bin/bash
# Simple runner script for the dev docs query tool
# Loads environment variables from .env and runs the query tool

echo "🚀 Loading environment and starting dev docs query tool..."

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded from .env"
else
    echo "❌ .env file not found - make sure AWS credentials are set"
    exit 1
fi

# Run the query tool
python query_dev_docs.py
