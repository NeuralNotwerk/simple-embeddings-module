# Dev Docs Semantic Search Query Tool

This tool embeds all documentation files from `./dev_docs/` using AWS Bedrock + S3 and provides an interactive command-line interface for semantic search queries.

## Features

- 🚀 **One-command setup**: Automatically embeds all `.md` and `.py` files from `./dev_docs/`
- 🔍 **Interactive search**: Real-time semantic search with colored output
- 📊 **Smart results**: Shows similarity scores, file types, and content previews
- 💾 **Persistent storage**: Uses S3 for storage, reuses existing indexes
- 🎨 **Colored output**: Beautiful terminal interface with icons and colors

## Quick Start

### Option 1: Use the runner script (recommended)
```bash
./run_query_tool.sh
```

### Option 2: Run directly
```bash
# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Run the tool
python query_dev_docs.py
```

## What Gets Embedded

The tool automatically finds and embeds:
- **Markdown files** (`.md`): Documentation, READMEs, guides
- **Python files** (`.py`): Code examples, scripts

Current dev_docs content:
- `pytorch-over-numpy.md` - PyTorch vs NumPy comparison
- `faiss_INSTALL.md` - FAISS installation guide  
- `faiss_README.md` - FAISS library overview
- `orjson_README.md` - orjson JSON library documentation
- `sf_sfr-embedding-code-2b_r.md` - Embedding model documentation
- `faiss_*.py` - FAISS code examples (Flat, IVFFlat, GPU, Multi-GPU)

## Interactive Commands

Once the tool starts, you can use these commands:

- **`help`** - Show available commands
- **`info`** - Display database information (document count, S3 bucket, etc.)
- **`examples`** - Show example queries to try
- **`quit`** - Exit the program

## Example Queries

Try these semantic search queries:

- `GPU acceleration and CUDA`
- `JSON serialization performance`
- `vector similarity search`
- `PyTorch tensor operations`
- `FAISS index types`
- `installation and setup`
- `Python code examples`
- `performance benchmarks`

## Sample Output

```
🔍 Searching for: 'GPU acceleration and CUDA'
   Search completed in 0.245s
   Found 2 result(s):

1. 🐍 faiss_4-GPU.py (Python)
   Similarity: 0.734
   Preview: import faiss import numpy as np # Generate some random data d = 64 # dimension nb = 100000 # database size...

2. 📄 pytorch-over-numpy.md (Markdown)
   Similarity: 0.612
   Preview: # PyTorch vs NumPy Performance Analysis This document compares PyTorch and NumPy for various operations...
```

## How It Works

1. **Document Loading**: Scans `./dev_docs/` for `.md` and `.py` files
2. **Embedding**: Uses AWS Bedrock Titan embeddings to create vector representations
3. **Storage**: Stores embeddings in S3 with automatic bucket creation
4. **Search**: Performs semantic similarity search using cosine similarity
5. **Results**: Returns documents above similarity threshold (0.1) with scores

## Configuration

The tool uses these AWS services:
- **Bedrock**: `amazon.titan-embed-text-v2:0` for embeddings
- **S3**: Auto-generated bucket for persistent storage
- **Region**: `us-east-1` (configurable)

## Reusing Existing Index

If you run the tool multiple times, it will detect existing embedded documents and ask if you want to reindex:

```
📚 Found existing index with 9 documents
Do you want to reindex all documents? (y/N):
```

Choose `N` to reuse the existing index for faster startup.

## Troubleshooting

### No results found
If searches return no results, try:
- Using more general terms
- Checking if the content actually exists in the docs
- Using the `info` command to verify documents were embedded

### AWS credentials error
Make sure your `.env` file contains:
```
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
```

### Low similarity scores
The tool uses a similarity threshold of 0.1. Very low scores (< 0.1) are filtered out. This is normal for semantic search - only semantically related content will appear.

## Technical Details

- **Embedding Model**: Amazon Titan Text Embeddings v2 (1024 dimensions)
- **Chunking**: Automatic text chunking for large documents
- **Similarity**: Cosine similarity with 0.1 threshold
- **Storage**: Compressed JSON in S3 with AES256 encryption
- **Performance**: ~0.25s average search time, ~0.5s per document embedding

## Files Created

- `query_dev_docs.py` - Main interactive tool
- `run_query_tool.sh` - Convenience runner script
- S3 bucket (auto-generated) - Persistent storage for embeddings
