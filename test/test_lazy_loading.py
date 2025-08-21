#!/usr/bin/env python3
"""Test script for lazy tree-sitter language loading."""

import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simple_embeddings_module.chunking.mod_chunking_ts_lang_lazy import (
    get_language_for_file,
    get_language_by_name,
    list_available_languages,
    preload_common_languages
)

def test_lazy_loading():
    print("🌳 Testing Tree-Sitter Lazy Loading")
    print("=" * 50)
    
    # Test file extension detection
    print("\n📁 Testing file extension detection:")
    test_files = [
        "main.py",
        "app.js", 
        "component.tsx",
        "Main.java",
        "program.c",
        "service.go",
        "lib.rs",
        "Dockerfile",
        "config.yaml",
        "unknown.xyz"
    ]
    
    for filename in test_files:
        start_time = time.time()
        language = get_language_for_file(filename)
        load_time = (time.time() - start_time) * 1000
        
        status = "✅" if language else "❌"
        print(f"  {status} {filename:<15} -> {language is not None} ({load_time:.1f}ms)")
    
    # Test language name detection
    print("\n🏷️  Testing language name detection:")
    test_languages = ["python", "javascript", "rust", "go", "nonexistent"]
    
    for lang_name in test_languages:
        start_time = time.time()
        language = get_language_by_name(lang_name)
        load_time = (time.time() - start_time) * 1000
        
        status = "✅" if language else "❌"
        print(f"  {status} {lang_name:<12} -> {language is not None} ({load_time:.1f}ms)")
    
    # List available languages
    print("\n📋 Available languages:")
    available = list_available_languages()
    print(f"  Found {len(available)} languages: {', '.join(available[:10])}{'...' if len(available) > 10 else ''}")
    
    # Test preloading
    print("\n⚡ Testing preloading:")
    start_time = time.time()
    preload_common_languages()
    preload_time = (time.time() - start_time) * 1000
    print(f"  Preloaded common languages in {preload_time:.1f}ms")
    
    # Test that subsequent loads are faster (cached)
    print("\n🚀 Testing cache performance:")
    for filename in ["main.py", "app.js", "lib.rs"]:
        start_time = time.time()
        language = get_language_for_file(filename)
        load_time = (time.time() - start_time) * 1000
        print(f"  {filename:<10} -> cached load in {load_time:.1f}ms")

if __name__ == "__main__":
    test_lazy_loading()
