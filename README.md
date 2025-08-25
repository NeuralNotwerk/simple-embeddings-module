# Simple Embeddings Module (SEM)

🚀 **A modular, cross-platform semantic search engine with intelligent chunking and GPU acceleration.**

## ✨ Features

- 🧠 **Semantic Search**: Find documents by meaning, not just keywords
- ⚡ **GPU Acceleration**: Apple Silicon MPS, NVIDIA CUDA, AMD ROCm support via PyTorch
- 📝 **Intelligent Chunking**: Auto-configured based on embedding model constraints
- 🧩 **Hierarchy-Constrained Grouping**: Groups related code chunks within same scope boundaries
- 🔒 **Secure**: No pickle files - pure JSON serialization with orjson
- 🔧 **Modular**: "Bring your own" embedding models, storage backends, and chunking strategies
- 🎯 **Production Ready**: Atomic writes, backups, compression, validation
- 📊 **Scalable**: Designed for 100K+ documents by default

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git
cd simple-embeddings-module
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Super Simple Usage

```python
# The simplest possible semantic search - just import and go!
from simple_embeddings_module import SEMSimple

sem = SEMSimple()
sem.add_text("Machine learning is transforming software development.")
results = sem.search("AI technology")
print(results[0]['text'])  # Found it!
```

### CLI Usage

```bash
# Index text from stdin
echo "some text" | sem-cli simple local index

# Index files from ls output
ls -d ./docs/* | sem-cli simple aws indexfiles --bucket my-bucket

# Search semantically
sem-cli simple local search --query "machine learning"

# List indexed documents
sem-cli simple local list
```

## 📚 Documentation

**Complete documentation is available in the `docs/` directory:**

- **[Quick Start Guide](docs/quickstart.rst)** - Get running in 5 minutes
- **[Installation Guide](docs/installation.rst)** - Platform-specific installation
- **[CLI Guide](docs/cli-guide.rst)** - Command-line interface and scripting
- **[Simple Interface Guide](docs/simple-interface.rst)** - Zero-config semantic search
- **[Examples](docs/examples/index.rst)** - Complete working examples
- **[CLI Reference](docs/cli-reference.rst)** - Complete command reference

### Choose Your Path

🚀 **New to semantic search?** → Start with [Quick Start Guide](docs/quickstart.rst)

⚙️ **Want command-line power?** → See [CLI Guide](docs/cli-guide.rst)

🐍 **Need programmatic control?** → Check [Simple Interface Guide](docs/simple-interface.rst)

☁️ **Planning cloud deployment?** → Explore [AWS Examples](docs/examples/aws-examples.rst)

## 🌟 Simple Interface

Three ways to get semantic search working immediately:

### Python Simple Interface

```python
from simple_embeddings_module import SEMSimple

# Local semantic search
sem = SEMSimple()
sem.add_text("Machine learning transforms software development.")
results = sem.search("AI technology")

# List what's indexed
docs = sem.list_documents()
for doc in docs:
    print(f"ID: {doc['id']}, Text: {doc['text'][:50]}...")

# AWS cloud semantic search
from simple_embeddings_module import simple_aws
sem = simple_aws(bucket_name="my-semantic-search")
sem.add_text("Cloud-based ML deployment strategies.")
results = sem.search("deployment")

# List AWS documents
docs = sem.list_documents(limit=5)
for doc in docs:
    print(f"ID: {doc['id']}, Created: {doc['created_at']}")
```

### CLI Simple Interface

```bash
# Local operations
echo "Machine learning content" | sem-cli simple local index
sem-cli simple local search --query "AI algorithms"
sem-cli simple local list

# AWS operations
echo "Cloud deployment guide" | sem-cli simple aws index --bucket my-docs
sem-cli simple aws search --query "deployment" --bucket my-docs
sem-cli simple aws list --bucket my-docs
```

### Pipeline Integration

```bash
# Index all documentation
find ./docs -name "*.md" | sem-cli simple local indexfiles

# Search your docs
sem-cli simple local search --query "installation instructions"

# List what's indexed
sem-cli simple local list

# Team knowledge base in the cloud
ls -d ./team_docs/* | sem-cli simple aws indexfiles --bucket team-knowledge
sem-cli simple aws search --query "project requirements" --bucket team-knowledge
sem-cli simple aws list --bucket team-knowledge
```

## 🏗️ Architecture

SEM uses a plugin-based architecture with four main component types:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Embedding     │    │    Chunking     │    │    Storage      │    │ Serialization   │
│   Providers     │    │   Strategies    │    │   Backends      │    │   Providers     │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • sentence-     │    │ • text          │    │ • local_disk    │    │ • orjson        │
│   transformers  │    │ • code          │    │ • s3            │    │ • json (TODO)   │
│ • openai        │    │ • hierarchy     │    │ • gcs (TODO)    │    │                 │
│ • bedrock       │    │   grouping      │    │                 │    │                 │
│ • ollama        │    │ • semantic      │    │                 │    │                 │
│ • llamacpp(TODO)│    │ • chunk_mux     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎛️ Performance

### Benchmarks (Apple Silicon M-series)

- **Model Loading**: ~2.5s (sentence-transformers/all-MiniLM-L6-v2)
- **Document Processing**: ~0.3s per document (chunking + embedding + storage)
- **Search Speed**: 0.4-2.1s depending on index size and query complexity
- **Memory Usage**: ~2GB for model + index size
- **Storage**: 44% smaller files than manual .tolist() with orjson compression

### Supported Hardware

- **Apple Silicon**: MPS (Metal Performance Shaders) acceleration
- **NVIDIA**: CUDA acceleration
- **AMD**: ROCm acceleration (Linux)
- **Intel/AMD**: CPU with optimized BLAS libraries
- **Cross-platform**: Same codebase works everywhere via PyTorch

## 🔒 Security

- **No Pickle Files**: Uses secure orjson serialization exclusively
- **Atomic Writes**: Prevents data corruption during saves
- **Backup Rotation**: Configurable backup copies
- **Input Validation**: Comprehensive parameter validation
- **Safe Deserialization**: Only plain JSON data structures

## 🧪 Testing

SEM includes a comprehensive test suite:

```bash
# Run smoke test (recommended after installation)
python test/smoke_test.py

# Run all tests
./test/run_tests.sh

# Run all demos
./demo/run_demos.sh
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the existing code patterns and ABC base classes
4. Add tests for new functionality
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

Apache 2.0

## 📞 Support

- **Documentation**: See `docs/` directory for comprehensive guides
- **Issues**: [GitHub Issues](https://github.com/NeuralNotwerk/simple-embeddings-module/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NeuralNotwerk/simple-embeddings-module/discussions)
- **Examples**: See `docs/examples/` for complete working examples

---

**Get started in 5 minutes:** [Quick Start Guide](docs/quickstart.rst)