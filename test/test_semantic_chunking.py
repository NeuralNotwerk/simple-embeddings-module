#!/usr/bin/env python3
"""Test script for semantic code chunking with tree-sitter."""

import tempfile
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simple_embeddings_module.chunking.mod_chunking_ts import ts_get_code_chunks, _get_code_line_breaks, test_chunking

# Global verbose flag
VERBOSE = False

def print_verbose(message: str):
    """Print message only in verbose mode."""
    if VERBOSE:
        print(message)

def show_raw_chunking_input(title: str, content: str, file_path: str):
    """Show raw input for chunking in verbose mode."""
    if not VERBOSE:
        return
    
    print(f"\n📥 RAW CHUNKING INPUT: {title}")
    print("=" * 60)
    print(f"File path: {file_path}")
    print(f"Content length: {len(content)} characters")
    print(f"Content lines: {content.count(chr(10)) + 1}")
    print(f"Language detected from extension: {os.path.splitext(file_path)[1]}")
    print("\nFull content:")
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        print(f"{i:3d}: {line}")
    print("=" * 60)

def show_raw_chunking_output(title: str, chunks: list, original_content: str):
    """Show raw chunking output in verbose mode."""
    if not VERBOSE:
        return
    
    print(f"\n📤 RAW CHUNKING OUTPUT: {title}")
    print("=" * 60)
    print(f"Original content length: {len(original_content)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Total chunked length: {sum(len(chunk) for chunk in chunks)} chars")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Length: {len(chunk)} characters")
        print(f"  Lines: {chunk.count(chr(10)) + 1}")
        print(f"  Starts with: {repr(chunk[:50])}...")
        print(f"  Ends with: {repr(chunk[-50:])}...")
        print(f"  Full content:")
        chunk_lines = chunk.split('\n')
        for j, line in enumerate(chunk_lines, 1):
            print(f"    {j:2d}: {line}")
    
    # Show chunk boundaries
    print(f"\nChunk boundaries analysis:")
    current_pos = 0
    for i, chunk in enumerate(chunks, 1):
        chunk_start = original_content.find(chunk, current_pos)
        chunk_end = chunk_start + len(chunk)
        print(f"  Chunk {i}: positions {chunk_start}-{chunk_end}")
        current_pos = chunk_end
    
    print("=" * 60)

def show_line_breaks_analysis(content: str, file_path: str):
    """Show line breaks analysis in verbose mode."""
    if not VERBOSE:
        return
    
    print(f"\n🔍 LINE BREAKS ANALYSIS")
    print("=" * 60)
    
    try:
        line_breaks = _get_code_line_breaks(content, file_path)
        print(f"Detected line breaks: {line_breaks}")
        
        lines = content.split('\n')
        print(f"Total lines: {len(lines)}")
        
        print(f"\nLine break positions in content:")
        for i, pos in enumerate(line_breaks):
            if pos < len(lines):
                print(f"  Break {i+1} at line {pos+1}: {repr(lines[pos][:50])}...")
            else:
                print(f"  Break {i+1} at line {pos+1}: (beyond content)")
        
    except Exception as e:
        print(f"Line breaks analysis failed: {e}")
    
    print("=" * 60)

# Sample Python code for testing
PYTHON_CODE = '''#!/usr/bin/env python3
"""Sample Python module for testing semantic chunking."""

import os
import sys
from typing import List, Optional

# Global variable
GLOBAL_CONSTANT = "test"

class DataProcessor:
    """A sample class for data processing."""
    
    def __init__(self, name: str):
        self.name = name
        self.data = []
    
    def add_data(self, item: str) -> None:
        """Add an item to the data list."""
        self.data.append(item)
        
    def process_data(self) -> List[str]:
        """Process all data items."""
        return [item.upper() for item in self.data]
    
    @staticmethod
    def validate_input(value: str) -> bool:
        """Validate input string."""
        return len(value) > 0

def standalone_function(x: int, y: int) -> int:
    """A standalone function."""
    return x + y

async def async_function(delay: float) -> str:
    """An async function."""
    import asyncio
    await asyncio.sleep(delay)
    return "done"

@property
def decorated_function():
    """A decorated function."""
    return "decorated"

# More global code
if __name__ == "__main__":
    processor = DataProcessor("test")
    processor.add_data("hello")
    result = processor.process_data()
    print(result)
'''

# Sample JavaScript code
JAVASCRIPT_CODE = '''// Sample JavaScript for testing
const express = require('express');
const app = express();

// Global variable
const PORT = 3000;

class UserService {
    constructor(database) {
        this.db = database;
        this.users = [];
    }
    
    async getUser(id) {
        return await this.db.findById(id);
    }
    
    createUser(userData) {
        const user = {
            id: Date.now(),
            ...userData
        };
        this.users.push(user);
        return user;
    }
}

function validateEmail(email) {
    const regex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return regex.test(email);
}

const arrowFunction = (a, b) => {
    return a + b;
};

// Express routes
app.get('/users', async (req, res) => {
    const users = await userService.getUsers();
    res.json(users);
});

app.post('/users', (req, res) => {
    const user = userService.createUser(req.body);
    res.status(201).json(user);
});

// Server startup
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
'''

# Sample Rust code
RUST_CODE = '''//! Sample Rust module for testing semantic chunking

use std::collections::HashMap;
use std::fmt::Display;

/// A trait for data processing
pub trait Processor {
    fn process(&self, data: &str) -> String;
}

/// Main data structure
#[derive(Debug, Clone)]
pub struct DataManager {
    data: HashMap<String, String>,
    name: String,
}

impl DataManager {
    /// Create a new DataManager
    pub fn new(name: String) -> Self {
        Self {
            data: HashMap::new(),
            name,
        }
    }
    
    /// Add data to the manager
    pub fn add_data(&mut self, key: String, value: String) {
        self.data.insert(key, value);
    }
    
    /// Get data by key
    pub fn get_data(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
}

impl Processor for DataManager {
    fn process(&self, data: &str) -> String {
        format!("{}: {}", self.name, data.to_uppercase())
    }
}

impl Display for DataManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DataManager({})", self.name)
    }
}

/// Standalone function
pub fn calculate_hash(input: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    hasher.finish()
}

/// Async function
pub async fn fetch_data(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Simulated async operation
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    Ok(format!("Data from {}", url))
}

/// Module for utilities
pub mod utils {
    pub fn format_string(s: &str) -> String {
        s.trim().to_lowercase()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_manager() {
        let mut manager = DataManager::new("test".to_string());
        manager.add_data("key1".to_string(), "value1".to_string());
        assert_eq!(manager.get_data("key1"), Some(&"value1".to_string()));
    }
}
'''

def test_semantic_chunking():
    """Test semantic chunking with different programming languages."""
    print("🌳 Testing Semantic Code Chunking")
    print("=" * 60)
    
    test_cases = [
        ("test.py", PYTHON_CODE),
        ("test.js", JAVASCRIPT_CODE), 
        ("test.rs", RUST_CODE)
    ]
    
    for filename, code in test_cases:
        print(f"\n📄 Testing {filename}")
        print("-" * 40)
        
        # Show raw input in verbose mode
        show_raw_chunking_input(f"Language Test: {filename}", code, filename)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=filename, delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Test line breaks detection
            if VERBOSE:
                show_line_breaks_analysis(code, filename)
            
            line_breaks = _get_code_line_breaks(code, filename)
            print(f"Line breaks found: {line_breaks}")
            
            # Test chunking
            chunks = ts_get_code_chunks(temp_file)
            print(f"Number of chunks: {len(chunks)}")
            
            # Show raw output in verbose mode
            show_raw_chunking_output(f"Language Test: {filename}", chunks, code)
            
            for i, chunk in enumerate(chunks, 1):
                lines = chunk.count('\n') + 1
                chars = len(chunk)
                # Get first line for preview
                first_line = chunk.split('\n')[0].strip()
                if len(first_line) > 60:
                    first_line = first_line[:60] + "..."
                    
                print(f"  Chunk {i}: {lines:2d} lines, {chars:3d} chars - {first_line}")
                
                if VERBOSE:
                    print(f"    Full chunk content:")
                    chunk_lines = chunk.split('\n')
                    for j, line in enumerate(chunk_lines[:10], 1):  # Show first 10 lines
                        print(f"      {j:2d}: {line}")
                    if len(chunk_lines) > 10:
                        print(f"      ... ({len(chunk_lines) - 10} more lines)")
                
        finally:
            # Clean up temp file
            os.unlink(temp_file)
    
    print(f"\n✅ Semantic chunking test completed!")

def test_with_real_files():
    """Test with actual files in the project."""
    print(f"\n🔍 Testing with real project files")
    print("-" * 40)
    
    # Test with some actual Python files from the project
    test_files = [
        "../src/simple_embeddings_module/sem_core.py",
        "../src/simple_embeddings_module/chunking/mod_chunking_ts.py",
        "../src/simple_embeddings_module/chunking/mod_hierarchy_grouping.py"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n📁 {file_path}")
            
            # Show raw input for real files in verbose mode
            if VERBOSE:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    show_raw_chunking_input(f"Real File: {os.path.basename(file_path)}", content, file_path)
                except Exception as e:
                    print(f"Could not read file for verbose output: {e}")
            
            chunks = ts_get_code_chunks(file_path)
            print(f"  Chunks: {len(chunks)}")
            
            # Show size distribution
            sizes = [len(chunk) for chunk in chunks]
            if sizes:
                avg_size = sum(sizes) / len(sizes)
                min_size = min(sizes)
                max_size = max(sizes)
                print(f"  Size: avg={avg_size:.0f}, min={min_size}, max={max_size} chars")
                
                if VERBOSE:
                    print(f"  Detailed size distribution:")
                    for i, size in enumerate(sizes, 1):
                        print(f"    Chunk {i}: {size} chars")
                    
                    # Show raw output for real files
                    if VERBOSE and len(chunks) <= 5:  # Only for small files to avoid spam
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            show_raw_chunking_output(f"Real File: {os.path.basename(file_path)}", chunks, content)
                        except Exception as e:
                            print(f"Could not show raw output: {e}")


if __name__ == "__main__":
    # Parse command line arguments for verbose mode
    parser = argparse.ArgumentParser(description='Test semantic code chunking with tree-sitter')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Enable verbose output showing raw inputs and outputs')
    args = parser.parse_args()
    
    # Set global verbose flag
    VERBOSE = args.verbose
    
    if VERBOSE:
        print("🔍 VERBOSE MODE ENABLED - Showing raw inputs and outputs")
        print("=" * 60)
    
    test_semantic_chunking()
    test_with_real_files()
