# SEM Tools

Utility scripts and tools for working with the Simple Embeddings Module.

## Available Tools

### query_dev_docs.py

Interactive tool for embedding and querying the development documentation in `./dev_docs/`.

**Features:**
- Embeds all `.md` and `.py` files from `./dev_docs/`
- Interactive command-line query interface
- AWS S3 + Bedrock backend for persistence
- Colored output with similarity scores

**Usage:**
```bash
# Set up AWS credentials first
export $(cat .env | grep -v '^#' | xargs)

# Run the interactive tool
python tools/query_dev_docs.py

# Or use the convenience script
./tools/run_query_tool.sh
```

**Commands in interactive mode:**
- `help` - Show available commands
- `info` - Show database information
- `examples` - Show example queries
- `quit` - Exit the program
- Any text - Search the embedded documentation

**Example queries:**
- "GPU acceleration and CUDA"
- "JSON serialization performance"
- "vector similarity search"
- "PyTorch tensor operations"

### run_query_tool.sh

Convenience script that loads environment variables and runs the query tool.

**Usage:**
```bash
./tools/run_query_tool.sh
```

**Requirements:**
- `.env` file with AWS credentials in the project root
- AWS credentials with Bedrock and S3 access

## Development Tools

These tools are designed for:
- **Documentation exploration** - Semantic search through technical docs
- **Development workflow** - Quick access to implementation details
- **Learning and reference** - Interactive exploration of codebase knowledge

## Setup

**Prerequisites:**
```bash
# Install SEM
pip install -e .

# Set up AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Or create .env file
echo "AWS_ACCESS_KEY_ID=your_key" > .env
echo "AWS_SECRET_ACCESS_KEY=your_secret" >> .env
```

**First run:**
```bash
# The tool will automatically:
# 1. Create an S3 bucket for storage
# 2. Embed all documentation files
# 3. Start interactive query interface

python tools/query_dev_docs.py
```

## Contributing

When adding new tools:

1. **Create the tool** in the `tools/` directory
2. **Add documentation** to this README
3. **Include usage examples** and prerequisites
4. **Test thoroughly** on different platforms
5. **Follow the existing code style** and patterns

## Tool Guidelines

**All tools should:**
- Include comprehensive help and usage information
- Handle errors gracefully with helpful messages
- Use the SEM simple interfaces when possible
- Include example usage in docstrings
- Be self-contained and easy to run
