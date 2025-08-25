# SEM Simple CLI Commands

The `sem-cli simple` command provides easy access to the simple constructs directly from the command line, making it perfect for quick operations and shell scripting.

## Overview

The simple CLI interface maps directly to the simple Python modules (`SEMSimple` and `simple_aws`) and provides three main operations:
- **`index`** - Index text from stdin or arguments
- **`indexfiles`** - Index files from paths (stdin or arguments)
- **`search`** - Search the semantic index

## Command Structure

```bash
sem-cli simple <backend> <operation> [options]
```

### Backends
- **`local`** - Uses local storage with sentence-transformers
- **`aws`** - Uses AWS S3 + Bedrock for cloud storage and embeddings

### Operations
- **`index`** - Index text content
- **`indexfiles`** - Index files by path
- **`search`** - Search the semantic index

## Examples

### Local Backend Examples

#### Index text from stdin
```bash
echo "Machine learning is transforming software development" | sem-cli simple local index
```

#### Index multiple text arguments
```bash
sem-cli simple local index --text "First document" "Second document" "Third document"
```

#### Index files from ls output
```bash
ls -d ./dev_docs/*.md | sem-cli simple local indexfiles
```

#### Index specific files
```bash
sem-cli simple local indexfiles --files doc1.txt doc2.txt doc3.txt
```

#### Search the local index
```bash
sem-cli simple local search --query "machine learning algorithms"
```

#### Custom local settings
```bash
# Use custom index name and storage path
sem-cli simple local index --index my_docs --path ./my_storage --text "Custom document"
sem-cli simple local search --index my_docs --path ./my_storage --query "custom"
```

### AWS Backend Examples

#### Index text to AWS
```bash
echo "Cloud-based machine learning deployment" | sem-cli simple aws index --bucket my-sem-bucket
```

#### Index files to AWS
```bash
ls -d ./documentation/* | sem-cli simple aws indexfiles --bucket my-sem-bucket
```

#### Search AWS index
```bash
sem-cli simple aws search --query "deployment strategies" --bucket my-sem-bucket
```

#### Custom AWS settings
```bash
# Use specific region and model
sem-cli simple aws index --bucket my-bucket --region us-west-2 --model amazon.titan-embed-text-v1 --text "Document content"
```

## Practical Use Cases

### 1. Quick Documentation Search
```bash
# Index all markdown files in a project
find . -name "*.md" | sem-cli simple local indexfiles

# Search for specific topics
sem-cli simple local search --query "installation instructions"
sem-cli simple local search --query "API documentation"
```

### 2. Code Documentation Pipeline
```bash
# Index source code comments and docs
find ./src -name "*.py" -o -name "*.md" | sem-cli simple aws indexfiles --bucket code-docs

# Search for implementation details
sem-cli simple aws search --query "authentication middleware" --bucket code-docs
```

### 3. Log Analysis
```bash
# Index recent log files
ls /var/log/*.log | head -10 | sem-cli simple local indexfiles --path ./log_index

# Search for error patterns
sem-cli simple local search --query "database connection error" --path ./log_index
```

### 4. Research Paper Management
```bash
# Index research papers
ls ~/papers/*.pdf | sem-cli simple aws indexfiles --bucket research-papers

# Find papers on specific topics
sem-cli simple aws search --query "neural network optimization" --bucket research-papers
```

## Options Reference

### Common Options
- `--query QUERY` - Search query (required for search operation)
- `--text TEXT [TEXT ...]` - Text content to index
- `--files FILES [FILES ...]` - Files to index
- `--top-k TOP_K` - Number of search results (default: 5)

### Local Backend Options
- `--index INDEX` - Index name (default: `sem_simple_index`)
- `--path PATH` - Storage path (default: `./sem_indexes`)

### AWS Backend Options
- `--bucket BUCKET` - S3 bucket name (auto-generated if not specified)
- `--region REGION` - AWS region (default: `us-east-1`)
- `--model MODEL` - Bedrock embedding model (default: `amazon.titan-embed-text-v2:0`)

## Pipeline Integration

### Shell Scripting
```bash
#!/bin/bash
# Index all documentation and search for topics

echo "📚 Indexing documentation..."
find ./docs -name "*.md" | sem-cli simple local indexfiles --index docs

echo "🔍 Searching for API references..."
sem-cli simple local search --index docs --query "API reference" --top-k 3

echo "🔍 Searching for installation guides..."
sem-cli simple local search --index docs --query "installation setup" --top-k 3
```

### CI/CD Integration
```bash
# In your CI pipeline
name: Index Documentation
run: |
  # Index updated documentation
  git diff --name-only HEAD~1 HEAD | grep '\.md$' | sem-cli simple aws indexfiles --bucket ci-docs

  # Verify search works
  sem-cli simple aws search --bucket ci-docs --query "getting started" --top-k 1
```

### Data Processing Workflows
```bash
# Process and index data files
for file in data/*.json; do
  jq -r '.description' "$file" | sem-cli simple local index --index data_descriptions
done

# Search processed data
sem-cli simple local search --index data_descriptions --query "user behavior analysis"
```

## Performance Notes

### Local Backend
- **Model Loading**: ~2-3 seconds for sentence-transformers model
- **Indexing Speed**: ~100-500 docs/second depending on content size
- **Search Speed**: ~0.1-0.5 seconds for typical queries
- **Storage**: Compressed JSON files in local directory

### AWS Backend
- **Setup Time**: ~1-2 seconds for AWS service initialization
- **Indexing Speed**: ~3-5 docs/second (limited by Bedrock API)
- **Search Speed**: ~0.3-0.8 seconds including S3 retrieval
- **Storage**: Encrypted, compressed data in S3

## Error Handling

### Common Issues
```bash
# No AWS credentials
❌ AWS credentials not available. Configure with:
   - AWS CLI: aws configure
   - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

# No input provided
❌ No text to index. Provide text via stdin or --text arguments

# File not found
❌ File not found: nonexistent.txt

# Empty search results
🔍 Searching for: 'very specific query'
   No results found
```

### Best Practices
1. **Always specify bucket names for AWS** to ensure data persistence
2. **Use consistent index names** for local operations
3. **Test with small datasets first** before processing large files
4. **Monitor AWS costs** when using Bedrock extensively
5. **Use appropriate top-k values** to balance relevance and performance

## Integration with Existing CLI

The simple commands complement the existing `sem-cli` commands:

```bash
# Traditional approach (more configuration)
sem-cli init --name my_docs --path ./indexes
sem-cli add --files doc1.txt doc2.txt --path ./indexes
sem-cli search "query" --path ./indexes

# Simple approach (minimal configuration)
sem-cli simple local indexfiles --files doc1.txt doc2.txt
sem-cli simple local search --query "query"
```

## Standardized Naming

All simple commands use consistent naming:
- **Index Name**: `sem_simple_index` (default across all backends)
- **Local Storage**: `./sem_indexes/` (default directory)
- **AWS Buckets**: Auto-generated with `sem-` prefix or user-specified

This ensures predictable behavior and easy data location across different operations.
