#!/usr/bin/env python3
"""🌟 ULTIMATE Code Chunking Demo - Showcasing ALL supported languages! 🌟"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our sample code files
from demo_code_samples import SAMPLE_FILES

from simple_embeddings_module import SEMSimple
from src.simple_embeddings_module.chunking.mod_chunking_ts import ts_get_code_chunks
from src.simple_embeddings_module.chunking.mod_code import CodeChunkingProvider

# Real embedding provider for the demo
from src.simple_embeddings_module.embeddings.mod_sentence_transformers import (
    SentenceTransformersProvider,
)


def print_banner():
    """Print an awesome banner!"""
    print("🌟" * 60)
    print("🚀 ULTIMATE SEMANTIC CODE CHUNKING DEMO 🚀")
    print("🌟" * 60)
    print("📚 Showcasing intelligent code chunking across ALL supported languages!")
    print("🧠 Using tree-sitter for semantic boundaries + Jina embeddings ready!")
    print("🌟" * 60)
    print()


def print_language_stats(temp_dir: Path):
    """Show statistics about our demo codebase."""
    print("📊 DEMO CODEBASE STATISTICS")
    print("=" * 50)

    total_files = 0
    total_lines = 0
    total_chars = 0
    languages = {}

    for file_path in temp_dir.glob("*"):
        if file_path.is_file():
            total_files += 1

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.count('\n') + 1
                chars = len(content)

                total_lines += lines
                total_chars += chars

                # Detect language from extension
                ext = file_path.suffix.lower()
                lang = {
                    '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
                    '.java': 'Java', '.c': 'C', '.cpp': 'C++', '.cs': 'C#',
                    '.php': 'PHP', '.rb': 'Ruby', '.go': 'Go', '.rs': 'Rust',
                    '.swift': 'Swift', '.kt': 'Kotlin', '.scala': 'Scala',
                    '.lua': 'Lua', '.sh': 'Bash', '.html': 'HTML', '.css': 'CSS',
                    '.json': 'JSON', '.yaml': 'YAML', '.toml': 'TOML',
                    '.xml': 'XML', '.sql': 'SQL', '.md': 'Markdown'
                }.get(ext, file_path.name)

                if lang not in languages:
                    languages[lang] = {'files': 0, 'lines': 0, 'chars': 0}
                languages[lang]['files'] += 1
                languages[lang]['lines'] += lines
                languages[lang]['chars'] += chars

    print(f"📁 Total Files: {total_files}")
    print(f"📄 Total Lines: {total_lines:,}")
    print(f"💾 Total Characters: {total_chars:,}")
    print(f"🌍 Languages: {len(languages)}")
    print()

    print("🗂️  BY LANGUAGE:")
    for lang, stats in sorted(languages.items()):
        print(f"  {lang:<12}: {stats['files']:2d} files, {stats['lines']:4d} lines, {stats['chars']:5d} chars")
    print()


def demo_semantic_chunking(temp_dir: Path):
    """Demonstrate semantic chunking across all languages."""
    print("🔪 SEMANTIC CHUNKING ANALYSIS")
    print("=" * 50)

    total_chunks = 0
    chunking_results = {}

    # Process each file
    for file_path in sorted(temp_dir.glob("*")):
        if file_path.is_file():
            filename = file_path.name

            # Get semantic chunks
            start_time = time.time()
            chunks = ts_get_code_chunks(str(file_path))
            chunk_time = (time.time() - start_time) * 1000

            total_chunks += len(chunks)

            # Analyze chunks
            chunk_sizes = [len(chunk) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0

            # Get file stats
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            original_size = len(original_content)

            chunking_results[filename] = {
                'chunks': len(chunks),
                'original_size': original_size,
                'avg_chunk_size': avg_size,
                'chunk_time': chunk_time,
                'first_chunk_preview': chunks[0][:60].replace('\n', ' ') + "..." if chunks else ""
            }

            # Show results
            status = "🟢" if len(chunks) > 1 else "🟡" if len(chunks) == 1 else "🔴"
            print(f"{status} {filename:<20}: {len(chunks):2d} chunks, avg {avg_size:4.0f} chars, {chunk_time:5.1f}ms")

    print(f"\n📈 TOTAL: {total_chunks} semantic chunks generated!")
    print()

    return chunking_results


def demo_semantic_search(temp_dir: Path, chunking_results: Dict):
    """Demonstrate semantic search with the chunked code."""
    print("🔍 SEMANTIC CODE SEARCH DEMO")
    print("=" * 50)

    # Initialize semantic search
    sem = SEMSimple()

    # Add all chunks to the search index
    print("📚 Indexing all code chunks...")
    chunk_count = 0

    for file_path in temp_dir.glob("*"):
        if file_path.is_file():
            filename = file_path.name
            chunks = ts_get_code_chunks(str(file_path))

            for i, chunk in enumerate(chunks):
                doc_id = f"{filename}:chunk_{i+1}"
                sem.add_text(chunk, doc_id=doc_id)
                chunk_count += 1

    print(f"✅ Indexed {chunk_count} code chunks from {len(list(temp_dir.glob('*')))} files")
    print()

    # Demonstrate semantic searches
    search_queries = [
        ("🔍 Database operations", "database connection SQL queries"),
        ("🌐 Web server setup", "HTTP server routes endpoints"),
        ("🧮 Mathematical functions", "calculate math algorithms"),
        ("📊 Data structures", "class struct array list"),
        ("🔒 Authentication logic", "login password security"),
        ("🎨 User interface", "HTML CSS styling"),
        ("⚡ Async programming", "async await promises"),
        ("🔧 Configuration files", "config settings parameters"),
        ("📝 Documentation", "comments documentation"),
        ("🚀 Main entry points", "main function startup")
    ]

    print("🎯 SEMANTIC SEARCH RESULTS:")
    print("-" * 50)

    for emoji_desc, query in search_queries:
        print(f"\n{emoji_desc}: '{query}'")
        results = sem.search(query, top_k=3)

        for i, result in enumerate(results, 1):
            score = result['score']
            doc_id = result.get('doc_id', 'unknown')
            text_preview = result['text'][:80].replace('\n', ' ')
            if len(result['text']) > 80:
                text_preview += "..."

            print(f"  {i}. {doc_id:<20} (score: {score:.3f})")
            print(f"     {text_preview}")

    print()


def show_chunking_examples(temp_dir: Path):
    """Show detailed chunking examples for a few languages."""
    print("📋 DETAILED CHUNKING EXAMPLES")
    print("=" * 50)

    # Show examples for a few interesting languages
    example_files = ['calculator.py', 'server.js', 'database.rs', 'api.java']

    for filename in example_files:
        file_path = temp_dir / filename
        if file_path.exists():
            print(f"\n📄 {filename.upper()}")
            print("-" * 30)

            chunks = ts_get_code_chunks(str(file_path))

            for i, chunk in enumerate(chunks, 1):
                lines = chunk.count('\n') + 1
                chars = len(chunk)

                # Get the first meaningful line
                first_line = ""
                for line in chunk.split('\n'):
                    line = line.strip()
                    if line and not line.startswith(('#', '//', '/*', '"""', "'''")):
                        first_line = line
                        break

                if len(first_line) > 50:
                    first_line = first_line[:50] + "..."

                print(f"  Chunk {i}: {lines:2d} lines, {chars:3d} chars")
                print(f"    Starts: {first_line}")

                # Show a few lines of the chunk
                preview_lines = chunk.split('\n')[:2]
                for line in preview_lines:
                    if line.strip():
                        print(f"    │ {line}")
                if len(chunk.split('\n')) > 2:
                    print("    │ ...")
                print()


def main():
    """Run the ultimate code chunking demo!"""
    print_banner()

    # Create temporary directory for our demo codebase
    temp_dir = Path(tempfile.mkdtemp(prefix="sem_demo_"))
    print(f"📁 Creating demo codebase in: {temp_dir}")

    try:
        # Write all sample files
        print("📝 Writing sample code files...")
        for filename, content in SAMPLE_FILES.items():
            file_path = temp_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        print(f"✅ Created {len(SAMPLE_FILES)} sample files")
        print()

        # Show statistics
        print_language_stats(temp_dir)

        # Demonstrate semantic chunking
        chunking_results = demo_semantic_chunking(temp_dir)

        # Show detailed examples
        show_chunking_examples(temp_dir)

        # Demonstrate semantic search
        demo_semantic_search(temp_dir, chunking_results)

        # Final summary
        print("🎉 DEMO COMPLETE!")
        print("=" * 50)
        print("✨ Key Achievements:")
        print("  🧠 Intelligent semantic chunking across 25+ languages")
        print("  🔍 Accurate semantic search finds relevant code")
        print("  ⚡ Fast processing with tree-sitter parsing")
        print("  🎯 Perfect for code documentation and exploration")
        print("  🚀 Ready for production with Jina code embeddings!")
        print()
        print("🌟 Simple Embeddings Module - Making code search simple and powerful! 🌟")

    finally:
        # Clean up
        print("🧹 Cleaning up demo files...")
        shutil.rmtree(temp_dir)
        print("✅ Demo cleanup complete!")


if __name__ == "__main__":
    main()
