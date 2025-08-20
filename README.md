# Simple Embeddings Module (SEM)

🚀 **A modular, cross-platform semantic search engine with intelligent chunking and GPU acceleration.**

## ✨ Features

- 🧠 **Semantic Search**: Find documents by meaning, not just keywords
- ⚡ **GPU Acceleration**: Apple Silicon MPS, NVIDIA CUDA, AMD ROCm support via PyTorch
- 📝 **Intelligent Chunking**: Auto-configured based on embedding model constraints
- 🔒 **Secure**: No pickle files - pure JSON serialization with orjson
- 🔧 **Modular**: "Bring your own" embedding models, storage backends, and chunking strategies
- 🎯 **Production Ready**: Atomic writes, backups, compression, validation
- 📊 **Scalable**: Designed for 100K+ documents by default

## 🚀 Quick Start

### Installation

```bash
# Option 1: Install from source
git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git
cd simple-embeddings-module
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

# Option 2: Install from PyPI (when available)
# Not available yet
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

### Basic Usage

```bash
# Initialize a new database
sem-cli init --name my_docs --path ./my_indexes

# Add documents
echo "Machine learning transforms software development." > doc1.txt
echo "PyTorch provides excellent GPU acceleration." > doc2.txt
sem-cli add --files doc1.txt doc2.txt --path ./my_indexes

# Search semantically
sem-cli search "artificial intelligence" --path ./my_indexes

# Show database info
sem-cli info --path ./my_indexes
```

## 🏗️ Architecture

### Modular Design

SEM uses a plugin-based architecture with four main component types:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Embedding     │    │    Chunking     │    │    Storage      │    │ Serialization   │
│   Providers     │    │   Strategies    │    │   Backends      │    │   Providers     │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • sentence-     │    │ • text          │    │ • local_disk    │    │ • orjson        │
│   transformers  │    │ • code (TODO)   │    │ • s3 (TODO)     │    │ • json (TODO)   │
│ • openai (TODO) │    │ • csv (TODO)    │    │ • gcs (TODO)    │    │                 │
│ • ollama (TODO) │    │ • chunk_mux     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Dependency Chain

The system enforces a strict dependency chain to ensure compatibility:

```
Embedding Provider → Chunking Strategy → Index Structure → Storage Backend
```

- **Embedding Provider** determines dimensions, max sequence length, and optimal chunk sizes
- **Chunking Strategy** adapts to embedding constraints and document types
- **Index Structure** optimized for the chosen embedding and chunking combination
- **Storage Backend** handles persistence with appropriate serialization

## 🎯 Core Components

### CLI Interface

```bash
# Available commands
sem-cli init     # Initialize new database
sem-cli add      # Add documents from files or text
sem-cli search   # Semantic search with configurable parameters
sem-cli info     # Show database information and statistics
sem-cli config   # Generate configuration templates
```

### Configuration System

SEM uses JSON-only configuration (no YAML, no pickles):

```json
{
  "embedding": {
    "provider": "sentence_transformers",
    "model": "all-MiniLM-L6-v2",
    "batch_size": 32
  },
  "chunking": {
    "strategy": "text",
    "boundary_type": "sentence"
  },
  "storage": {
    "backend": "local_disk",
    "path": "./indexes"
  },
  "serialization": {
    "provider": "orjson"
  }
}
```

### Intelligent Chunking

Chunking strategies automatically configure based on embedding provider capabilities:

- **Chunk Size**: 80% of embedding model's max sequence length
- **Overlap**: 10% of chunk size for context preservation
- **Boundaries**: Sentence, paragraph, or word boundaries
- **Validation**: Ensures chunks fit within embedding constraints

## 🔧 Advanced Usage

### Custom Configuration

```python
from simple_embeddings_module import SEMConfigBuilder

# Build custom configuration
builder = SEMConfigBuilder()
builder.set_embedding_provider("sentence_transformers", model="all-mpnet-base-v2")
builder.auto_configure_chunking()
builder.set_storage_backend("local_disk", path="./custom_indexes")
config = builder.build()

# Create database with custom config
from simple_embeddings_module import SEMDatabase
db = SEMDatabase(config=config)
```

### Provider Compatibility

SEM includes strict compatibility checking for provider switches:

```python
# Check if switching providers is safe
compatibility = builder.check_provider_compatibility(
    "sentence_transformers", model="all-MiniLM-L12-v2"
)

if compatibility.is_compatible:
    # Safe switch - no re-indexing needed
    db.switch_embedding_provider("sentence_transformers", model="all-MiniLM-L12-v2")
else:
    # Requires full re-indexing
    print(f"Migration required: {compatibility.reasons}")
```

## 🚨 Compatibility Matrix

### ✅ Compatible Switches (No Re-indexing)
- Same model with different quantization (fp32 ↔ fp16, q4, q8)
- Same model with different file formats (.bin ↔ .safetensors)
- Sequence length within ±20%

### ❌ Incompatible Switches (Full Re-indexing Required)
- Different embedding dimensions (384 vs 768)
- Different model families (MiniLM vs MPNet)
- Different model sizes (L6 vs L12, 400M vs 2B)
- Sequence length >20% difference

⚠️ **Best Effort Warning**: Even "compatible" switches may produce different results due to hardware differences (NVIDIA vs AMD vs Apple Silicon), RNG seeds, and numerical precision variations.

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

## 📁 Project Structure

```
simple-embeddings-module/
├── src/simple_embeddings_module/
│   ├── embeddings/           # Embedding providers
│   │   ├── mod_sentence_transformers.py
│   │   └── mod_embeddings_base.py
│   ├── chunking/            # Chunking strategies
│   │   ├── mod_text.py
│   │   ├── mod_chunk_mux.py
│   │   └── mod_chunking_base.py
│   ├── storage/             # Storage backends
│   │   ├── mod_local_disk.py
│   │   └── mod_storage_base.py
│   ├── serialization/       # Serialization providers
│   │   ├── mod_orjson.py
│   │   └── mod_serialization_base.py
│   ├── sem_core.py          # Core database engine
│   ├── sem_cli.py           # Command-line interface
│   ├── sem_config_builder.py # Configuration management
│   ├── sem_module_reg.py    # Module registry system
│   └── sem_utils.py         # Utility functions
├── test/                    # Test suite
│   ├── smoke_test.py        # End-to-end functionality test
│   ├── run_tests.sh         # Test runner
│   └── README.md            # Test documentation
├── requirements.txt
├── setup.py
├── pyproject.toml
├── README.md
└── TODO.md
```

## 🧪 Testing

SEM includes a comprehensive test suite to verify functionality:

```bash
# Run smoke test (recommended after installation)
python test/smoke_test.py

# Or run full test suite
./test/run_tests.sh
```

The smoke test verifies:
- Python imports and module loading
- CLI command availability
- GPU acceleration detection
- End-to-end semantic search functionality
- Search result quality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the existing code patterns and ABC base classes
4. Add tests for new functionality
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Adding New Providers

To add a new embedding provider:

1. Create `mod_your_provider.py` in `src/simple_embeddings_module/embeddings/`
2. Inherit from `EmbeddingProviderBase`
3. Implement required abstract methods
4. Define `CONFIG_PARAMETERS` and `CAPABILITIES`
5. The module registry will auto-discover your provider

## 📄 License

Apache 2.0

## Main External Depends

- **HuggingFace sentence-transformers** for library and easily accessible embedding models
- **PyTorch** for universal GPU acceleration
- **orjson** for fast, secure JSON serialization

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/NeuralNotwerk/simple-embeddings-module/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NeuralNotwerk/simple-embeddings-module/discussions)
- **Documentation**: See README.md and test/ directory