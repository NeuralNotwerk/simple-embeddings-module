# Simple Embeddings Module (SEM) - Demos

This directory contains demonstration scripts showcasing the capabilities of the Simple Embeddings Module.

## Available Demos

### 🌳 `demo_semantic_chunking.py`
Demonstrates basic semantic code chunking using tree-sitter parsing.
- Shows how code is broken into semantic units (classes, functions, etc.)
- Displays chunk statistics and previews
- Perfect introduction to SEM's chunking capabilities

### 🔍 `example_code_search.py`
Example of semantic code search using tree-sitter chunking.
- Creates a sample codebase with multiple Python files
- Demonstrates semantic search across code chunks
- Shows search results with relevance scores

### 🚀 `demo_ultimate_code_chunking.py`
Comprehensive demo showcasing chunking across multiple programming languages.
- Processes code in 25+ supported languages
- Shows statistics and analysis of chunking results
- Demonstrates semantic search integration
- Includes detailed chunking examples

### 🧩 `demo_hierarchy_grouping.py`
**NEW!** Comprehensive demo of hierarchy-constrained semantic grouping.
- Shows intelligent grouping of code chunks within hierarchy boundaries
- Demonstrates flat storage with rich metadata
- Includes advanced search with both individual chunks and semantic groups
- Shows JSON serialization and loading capabilities

### 📄 `demo_code_samples.py`
Sample code files used by other demos.
- Contains example code in Python, JavaScript, TypeScript
- Used as input for chunking and search demonstrations

## Running Demos

### Individual Demos
```bash
# Run a specific demo
python demo/demo_semantic_chunking.py
python demo/example_code_search.py
python demo/demo_ultimate_code_chunking.py
python demo/demo_hierarchy_grouping.py
```

### All Demos
```bash
# Run all demos in sequence
./demo/run_demos.sh
```

## Requirements

- Python 3.8+
- Virtual environment activated (recommended)
- All SEM dependencies installed (`pip install -e .`)

## Demo Features

### Real Embeddings
All demos use real sentence-transformers embedding models (no mocks!):
- `all-MiniLM-L6-v2` for fast, efficient embeddings
- Automatic GPU acceleration (Apple Silicon MPS, NVIDIA CUDA, AMD ROCm)
- Authentic performance characteristics

### Supported Languages
Demos showcase semantic chunking across 25+ programming languages:
- **Popular**: Python, JavaScript, TypeScript, Java, C/C++
- **Modern**: Rust, Go, Swift, Kotlin, Scala
- **Scripting**: Bash, Lua, PHP, Ruby
- **Config**: JSON, YAML, TOML, XML
- **And more**: See full list in documentation

### Key Capabilities Demonstrated

1. **Semantic Chunking**: Intelligent code parsing that preserves structure
2. **Hierarchy-Constrained Grouping**: Groups related chunks within same scope
3. **Cross-Language Support**: Works with any tree-sitter supported language
4. **Flat Storage**: Efficient storage with rich metadata
5. **Advanced Search**: Search individual chunks and semantic groups
6. **Real-Time Processing**: Fast chunking and embedding generation

## Output Examples

### Chunking Output
```
📦 Chunk 1: 17 lines, 471 chars
   Starts with: class Calculator:
   Hierarchy: sample.py → Calculator
   Lines: 12-28
```

### Search Results
```
🔎 Query: 'database operations'
   1. [GROUP] my_file_py_MyClass2_data_ops (score: 0.892)
      Scope: my_file.py::MyClass2
      Theme: database
      Chunks: 3
```

### Grouping Analysis
```
🔗 Semantic Groups Analysis:
   related_functions: 5 groups
   related_classes: 2 groups
   mixed_code_units: 1 group
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **Missing Dependencies**: Run `pip install -e .` to install all requirements
3. **GPU Issues**: Demos automatically fall back to CPU if GPU unavailable
4. **Memory Issues**: Large files are automatically truncated for embedding models

### Performance Notes

- First run may be slower due to model downloading
- Subsequent runs use cached models for faster performance
- GPU acceleration significantly improves embedding speed
- Tree-sitter parsing is very fast (~0.3s per document)

## Contributing

To add new demos:
1. Create a new `demo_*.py` file in this directory
2. Add appropriate imports with `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))`
3. Follow existing demo patterns for consistency
4. Update this README with demo description
5. Add to `run_demos.sh` if appropriate

## Next Steps

After running demos, try:
- Modifying demo code to test different scenarios
- Running with your own code files
- Experimenting with different similarity thresholds
- Exploring the hierarchy grouping capabilities
- Integrating SEM into your own projects
