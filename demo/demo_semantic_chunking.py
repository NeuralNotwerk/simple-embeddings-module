#!/usr/bin/env python3
"""Demo script showing semantic code chunking in action."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simple_embeddings_module.chunking.mod_chunking_ts import ts_get_code_chunks

# Sample Python code to demonstrate chunking
SAMPLE_CODE = '''#!/usr/bin/env python3
"""Sample module for demonstration."""

import os
from typing import List

class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

async def fetch_data(url: str) -> str:
    """Async function to fetch data."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(5, 3))
    print(calc.multiply(4, 7))
    print(factorial(5))
'''

def main():
    print("🌳 Semantic Code Chunking Demo")
    print("=" * 50)

    # Write sample code to a temporary file
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(SAMPLE_CODE)
        temp_file = f.name

    try:
        # Get semantic chunks
        chunks = ts_get_code_chunks(temp_file)

        print(f"📄 Original code: {len(SAMPLE_CODE)} characters")
        print(f"🔪 Chunked into: {len(chunks)} semantic units")
        print()

        for i, chunk in enumerate(chunks, 1):
            lines = chunk.count('\n') + 1
            chars = len(chunk)

            # Get the first meaningful line (skip empty lines and comments)
            first_line = ""
            for line in chunk.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('"""'):
                    first_line = line
                    break

            if len(first_line) > 60:
                first_line = first_line[:60] + "..."

            print(f"📦 Chunk {i}: {lines:2d} lines, {chars:3d} chars")
            print(f"   Starts with: {first_line}")
            print(f"   Preview:")

            # Show first few lines of the chunk
            preview_lines = chunk.split('\n')[:3]
            for line in preview_lines:
                if line.strip():
                    print(f"     {line}")

            if len(chunk.split('\n')) > 3:
                print("     ...")
            print()

    finally:
        # Clean up
        os.unlink(temp_file)

    print("✨ Each chunk represents a complete semantic unit:")
    print("   • Classes with all their methods")
    print("   • Standalone functions") 
    print("   • Import statements and module-level code")
    print()
    print("🎯 Perfect for semantic search with code embeddings!")

if __name__ == "__main__":
    main()
