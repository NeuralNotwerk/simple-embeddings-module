#!/usr/bin/env python3
"""Test the code chunking provider."""

import tempfile
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simple_embeddings_module.chunking.mod_code import CodeChunkingProvider
from src.simple_embeddings_module.embeddings.mod_sentence_transformers import SentenceTransformersProvider

# Global verbose flag
VERBOSE = False

def print_verbose(message: str):
    """Print message only in verbose mode."""
    if VERBOSE:
        print(message)

def show_provider_raw_input(title: str, content: str, file_path: str = None):
    """Show raw input for provider testing in verbose mode."""
    if not VERBOSE:
        return
    
    print(f"\n📥 RAW PROVIDER INPUT: {title}")
    print("=" * 60)
    if file_path:
        print(f"File path: {file_path}")
    print(f"Content length: {len(content)} characters")
    print(f"Content lines: {content.count(chr(10)) + 1}")
    print("\nFull content:")
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        print(f"{i:3d}: {line}")
    print("=" * 60)

def show_provider_raw_output(title: str, chunks: List[str], provider_info: Dict[str, Any]):
    """Show raw provider output in verbose mode."""
    if not VERBOSE:
        return
    
    print(f"\n📤 RAW PROVIDER OUTPUT: {title}")
    print("=" * 60)
    print(f"Provider capabilities: {provider_info}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Total chunked length: {sum(len(chunk) for chunk in chunks)} chars")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Length: {len(chunk)} characters")
        print(f"  Lines: {chunk.count(chr(10)) + 1}")
        print(f"  Content:")
        chunk_lines = chunk.split('\n')
        for j, line in enumerate(chunk_lines, 1):
            print(f"    {j:2d}: {line}")
    
    print("=" * 60)

# Sample Python code for testing
SAMPLE_CODE = '''#!/usr/bin/env python3
"""Sample module for testing code chunking provider."""

import os
import sys
from typing import List, Dict

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def get_history(self) -> List[str]:
        """Get calculation history."""
        return self.history.copy()

def factorial(n: int) -> int:
    """Calculate factorial recursively."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

async def fetch_data(url: str) -> str:
    """Fetch data from URL."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(5, 3))
    print(factorial(5))
'''

def test_code_chunking_provider():
    """Test the code chunking provider functionality."""
    print("🧩 Testing Code Chunking Provider")
    print("=" * 50)
    
    # Initialize the provider with real sentence-transformers embedding provider
    embedding_provider = SentenceTransformersProvider(model="all-MiniLM-L6-v2")
    provider = CodeChunkingProvider(embedding_provider)
    
    print("📋 Provider capabilities:")
    capabilities = provider.get_capabilities()
    for key, value in capabilities.items():
        print(f"  {key}: {value}")
    
    if VERBOSE:
        print(f"\n🔍 RAW PROVIDER CAPABILITIES:")
        print("=" * 60)
        print(f"Full capabilities dict: {capabilities}")
        print("=" * 60)
    
    print(f"\n⚙️  Configuration parameters:")
    config_params = provider.get_config_parameters()
    for param, info in config_params.items():
        print(f"  {param}: {info['default']} ({info['description']})")
    
    if VERBOSE:
        print(f"\n🔍 RAW CONFIGURATION PARAMETERS:")
        print("=" * 60)
        print(f"Full config params dict: {config_params}")
        print("=" * 60)
    
    # Test 1: Chunk text directly
    print(f"\n🔤 Test 1: Chunking text directly")
    
    # Show raw input in verbose mode
    show_provider_raw_input("Direct Text Chunking", SAMPLE_CODE)
    
    chunks = provider.chunk_text(SAMPLE_CODE, file_path="test.py")
    print(f"Generated {len(chunks)} chunks from text:")
    
    # Show raw output in verbose mode
    show_provider_raw_output("Direct Text Chunking", chunks, capabilities)
    
    for i, chunk in enumerate(chunks, 1):
        lines = chunk.count('\n') + 1
        chars = len(chunk)
        first_line = chunk.split('\n')[0].strip()
        if len(first_line) > 60:
            first_line = first_line[:60] + "..."
        print(f"  Chunk {i}: {lines:2d} lines, {chars:4d} chars - {first_line}")
        
        if VERBOSE:
            print(f"    Full chunk:")
            chunk_lines = chunk.split('\n')
            for j, line in enumerate(chunk_lines[:5], 1):  # Show first 5 lines
                print(f"      {j}: {line}")
            if len(chunk_lines) > 5:
                print(f"      ... ({len(chunk_lines) - 5} more lines)")
    
    # Test 2: Chunk from file
    print(f"\n📄 Test 2: Chunking from file")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(SAMPLE_CODE)
        temp_file = f.name
    
    try:
        # Show raw input for file chunking in verbose mode
        show_provider_raw_input("File Chunking", SAMPLE_CODE, temp_file)
        
        file_chunks = provider.chunk_file(temp_file)
        print(f"Generated {len(file_chunks)} chunks from file:")
        
        # Show raw output in verbose mode
        show_provider_raw_output("File Chunking", file_chunks, capabilities)
        
        for i, chunk in enumerate(file_chunks, 1):
            lines = chunk.count('\n') + 1
            chars = len(chunk)
            first_line = chunk.split('\n')[0].strip()
            if len(first_line) > 60:
                first_line = first_line[:60] + "..."
            print(f"  Chunk {i}: {lines:2d} lines, {chars:4d} chars - {first_line}")
    
    finally:
        os.unlink(temp_file)
    
    # Test 3: Non-code text (should fallback)
    print(f"\n📝 Test 3: Non-code text (fallback)")
    non_code_text = """
    This is a regular text document.
    
    It contains multiple paragraphs of text that don't look like code.
    There are no function definitions or class declarations here.
    
    This should trigger the fallback text chunking mechanism
    instead of trying to use tree-sitter semantic parsing.
    
    The provider should detect this and handle it gracefully.
    """
    
    # Show raw input for fallback test in verbose mode
    show_provider_raw_input("Fallback Text Chunking", non_code_text)
    
    fallback_chunks = provider.chunk_text(non_code_text)
    print(f"Generated {len(fallback_chunks)} chunks from non-code text:")
    
    # Show raw output in verbose mode
    show_provider_raw_output("Fallback Text Chunking", fallback_chunks, capabilities)
    
    for i, chunk in enumerate(fallback_chunks, 1):
        chars = len(chunk)
        preview = chunk.strip()[:50].replace('\n', ' ')
        if len(chunk.strip()) > 50:
            preview += "..."
        print(f"  Chunk {i}: {chars:3d} chars - {preview}")
        
        if VERBOSE:
            print(f"    Full fallback chunk:")
            print(f"      {repr(chunk)}")
    
    print(f"\n✅ Code chunking provider test completed!")


if __name__ == "__main__":
    # Parse command line arguments for verbose mode
    parser = argparse.ArgumentParser(description='Test code chunking provider')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Enable verbose output showing raw inputs and outputs')
    args = parser.parse_args()
    
    # Set global verbose flag
    VERBOSE = args.verbose
    
    if VERBOSE:
        print("🔍 VERBOSE MODE ENABLED - Showing raw inputs and outputs")
        print("=" * 60)
    
    test_code_chunking_provider()
