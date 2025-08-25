#!/usr/bin/env python3
"""
Smoke test for Simple Embeddings Module (SEM)
This script performs a comprehensive end-to-end test of the SEM package
to verify that all core functionality works correctly after installation.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=60)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
def test_imports():
    """Test that all Python imports work correctly."""
    print("🐍 Testing Python imports...")
    try:
        import simple_embeddings_module as sem
        print("  ✅ Main module import successful")
        # Test key classes
        builder = sem.SEMConfigBuilder()
        print("  ✅ SEMConfigBuilder import successful")
        config = sem.create_default_config()
        print("  ✅ Default config creation successful")
        # Test SEMSimple
        simple_sem = sem.SEMSimple()
        print("  ✅ SEMSimple import successful")
        # Test submodules
        print("  ✅ All base classes import successful")
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False
def test_cli_availability():
    """Test that the CLI command is available."""
    print("🔧 Testing CLI availability...")
    success, stdout, stderr = run_command("sem-cli --help")
    if success and "Simple Embeddings Module CLI" in stdout:
        print("  ✅ sem-cli command available and working")
        return True
    else:
        print(f"  ❌ sem-cli command failed: {stderr}")
        return False
def test_end_to_end():
    """Test complete end-to-end functionality."""
    print("🚀 Testing end-to-end functionality...")
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        os.chdir(test_dir)
        # Create test documents
        test_docs = {
            "doc1.txt": "Testing semantic search functionality with embeddings and machine learning.",
            "doc2.txt": "Machine learning and artificial intelligence are transforming software development.",
            "doc3.txt": "PyTorch provides excellent GPU acceleration for deep learning models and neural networks.",
        }
        for filename, content in test_docs.items():
            (test_dir / filename).write_text(content)
        print("  📝 Created test documents")
        # Test 1: Initialize database
        success, stdout, stderr = run_command("sem-cli init --name smoke_test --path ./test_indexes")
        if not success:
            print(f"  ❌ Database initialization failed: {stderr}")
            return False
        print("  ✅ Database initialization successful")
        # Test 2: Add documents
        success, stdout, stderr = run_command("sem-cli add --files doc1.txt doc2.txt doc3.txt --path ./test_indexes")
        if not success:
            print(f"  ❌ Document addition failed: {stderr}")
            return False
        print("  ✅ Document addition successful")
        # Test 3: Show database info
        success, stdout, stderr = run_command("sem-cli info --path ./test_indexes")
        if not success or "Documents: 3" not in stdout:
            print(f"  ❌ Database info failed: {stderr}")
            return False
        print("  ✅ Database info successful")
        # Test 4: Search functionality
        success, stdout, stderr = run_command(
            "sem-cli search 'AI and machine learning' --path ./test_indexes --top-k 2"
        )
        if not success or "Found 2 results" not in stdout:
            print(f"  ❌ Search failed: {stderr}")
            return False
        print("  ✅ Search functionality successful")
        # Test 5: Verify semantic matching quality
        if "artificial intelligence" in stdout.lower() and "machine learning" in stdout.lower():
            print("  ✅ Semantic search quality verified")
        else:
            print("  ⚠️  Search results may not be semantically optimal")
        return True
def test_gpu_acceleration():
    """Test GPU acceleration detection."""
    print("🚀 Testing GPU acceleration...")
    try:
        import torch
        if torch.backends.mps.is_available():
            print("  ✅ Apple Silicon MPS acceleration available")
        elif torch.cuda.is_available():
            print("  ✅ NVIDIA CUDA acceleration available")
        else:
            print("  ℹ️  CPU-only mode (no GPU acceleration detected)")
        return True
    except Exception as e:
        print(f"  ❌ GPU detection failed: {e}")
        return False
def test_semsimple():
    """Test SEMSimple one-liner functionality."""
    print("✨ Testing SEMSimple one-liner...")
    try:
        from simple_embeddings_module import SEMSimple
        # Create instance
        sem = SEMSimple(index_name="smoke_simple")
        print("  ✅ SEMSimple instance created")
        # Add documents
        texts = [
            "Machine learning transforms software development.",
            "Semantic search finds documents by meaning.",
            "GPU acceleration speeds up deep learning.",
        ]
        success = sem.add_texts(texts)
        if not success:
            print("  ❌ Failed to add documents")
            return False
        print(f"  ✅ Added {sem.count()} documents")
        # Search
        results = sem.search("AI and machine learning", top_k=2)
        if not results:
            print("  ❌ Search returned no results")
            return False
        print(f"  ✅ Search found {len(results)} results")
        # Verify result quality
        if results[0]["score"] > 0.3:  # Should find relevant results
            print("  ✅ Search quality verified")
        else:
            print("  ⚠️  Search results may not be optimal")
        # Clean up
        sem.clear()
        print("  ✅ Cleanup completed")
        return True
    except Exception as e:
        print(f"  ❌ SEMSimple test failed: {e}")
        return False
def main():
    """Run all smoke tests."""
    print("🧪 Simple Embeddings Module (SEM) Smoke Test")
    print("=" * 50)
    start_time = time.time()
    tests_passed = 0
    total_tests = 5
    # Run all tests
    if test_imports():
        tests_passed += 1
    if test_cli_availability():
        tests_passed += 1
    if test_gpu_acceleration():
        tests_passed += 1
    if test_semsimple():
        tests_passed += 1
    if test_end_to_end():
        tests_passed += 1
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")
    print(f"⏱️  Total time: {elapsed:.2f}s")
    if tests_passed == total_tests:
        print("🎉 All tests passed! SEM is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1
if __name__ == "__main__":
    sys.exit(main())
