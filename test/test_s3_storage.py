#!/usr/bin/env python3
"""Test script for S3 storage backend."""

import os
import sys
import argparse
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import boto3
    from src.simple_embeddings_module.storage.mod_s3 import S3Storage
    from src.simple_embeddings_module.storage.mod_storage_base import StorageBackendError
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Global verbose flag
VERBOSE = False

def print_verbose(message: str):
    """Print message only in verbose mode."""
    if VERBOSE:
        print(message)

def show_raw_s3_config(title: str, config: Dict[str, Any]):
    """Show raw S3 configuration in verbose mode."""
    if not VERBOSE:
        return
    
    print(f"\n📥 RAW S3 CONFIG: {title}")
    print("=" * 60)
    # Mask sensitive information
    safe_config = config.copy()
    for key in ["aws_secret_access_key", "aws_session_token"]:
        if key in safe_config:
            safe_config[key] = "***MASKED***"
    
    for key, value in safe_config.items():
        print(f"  {key}: {value}")
    print("=" * 60)

def show_raw_s3_operations(title: str, operation: str, details: Dict[str, Any]):
    """Show raw S3 operations in verbose mode."""
    if not VERBOSE:
        return
    
    print(f"\n📤 RAW S3 OPERATION: {title}")
    print("=" * 60)
    print(f"Operation: {operation}")
    for key, value in details.items():
        if key == "data_size" and isinstance(value, int):
            print(f"  {key}: {value:,} bytes")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)

def test_s3_storage_basic():
    """Test basic S3 storage functionality."""
    print("🪣 Testing S3 Storage Backend - Basic Operations")
    print("=" * 60)
    
    if not DEPENDENCIES_AVAILABLE:
        print(f"❌ Dependencies not available: {IMPORT_ERROR}")
        print("   Install with: pip install boto3")
        return False
    
    # Get S3 configuration from environment
    bucket_name = os.environ.get("SEM_S3_BUCKET")
    region = os.environ.get("SEM_S3_REGION", "us-east-1")
    
    if not bucket_name:
        print("❌ S3 bucket name not provided")
        print("   Set environment variable: SEM_S3_BUCKET=your-bucket-name")
        return False
    
    # Build S3 configuration
    s3_config = {
        "bucket_name": bucket_name,
        "region": region,
        "prefix": f"sem-test-{int(time.time())}/",  # Unique prefix for testing
        "compression": True,
        "encryption": "AES256",
        "storage_class": "STANDARD",
    }
    
    # Add AWS credentials if provided via environment
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        s3_config["aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
    if os.environ.get("AWS_SECRET_ACCESS_KEY"):
        s3_config["aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
    if os.environ.get("AWS_SESSION_TOKEN"):
        s3_config["aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]
    
    show_raw_s3_config("S3 Storage Configuration", s3_config)
    
    try:
        # Initialize S3 storage
        print("🔧 Initializing S3 storage backend...")
        storage = S3Storage(**s3_config)
        
        # Show capabilities
        capabilities = storage.get_capabilities()
        print("📋 S3 Storage capabilities:")
        for key, value in capabilities.items():
            print(f"  {key}: {value}")
        
        if VERBOSE:
            print(f"\n🔍 RAW S3 CAPABILITIES:")
            print("=" * 60)
            print(f"Full capabilities dict: {capabilities}")
            print("=" * 60)
        
        # Test 1: Create test data
        print(f"\n📊 Test 1: Creating test data")
        test_vectors = torch.randn(100, 384)  # 100 vectors of 384 dimensions
        test_metadata = {
            "index_name": "test_s3_index",
            "vector_count": 100,
            "embedding_dim": 384,
            "model_name": "test-model",
            "created_at": "2025-08-22T02:25:00Z",
            "test_data": True,
        }
        
        show_raw_s3_operations("Test Data Creation", "create_test_data", {
            "vector_shape": list(test_vectors.shape),
            "vector_dtype": str(test_vectors.dtype),
            "metadata_keys": list(test_metadata.keys()),
            "data_size": test_vectors.numel() * test_vectors.element_size(),
        })
        
        # Test 2: Save index
        print(f"\n💾 Test 2: Saving index to S3")
        index_name = "test_s3_index"
        
        start_time = time.time()
        success = storage.save_index(test_vectors, test_metadata, index_name)
        save_time = time.time() - start_time
        
        if success:
            print(f"✅ Index saved successfully in {save_time:.2f}s")
        else:
            print(f"❌ Failed to save index")
            return False
        
        show_raw_s3_operations("Save Index", "save_index", {
            "index_name": index_name,
            "success": success,
            "duration_seconds": save_time,
            "bucket": bucket_name,
            "prefix": s3_config["prefix"],
        })
        
        # Test 3: Check if index exists
        print(f"\n🔍 Test 3: Checking if index exists")
        exists = storage.index_exists(index_name)
        print(f"Index exists: {exists}")
        
        if not exists:
            print(f"❌ Index should exist but doesn't")
            return False
        
        # Test 4: List indexes
        print(f"\n📋 Test 4: Listing indexes")
        indexes = storage.list_indexes()
        print(f"Found {len(indexes)} indexes: {indexes}")
        
        if index_name not in indexes:
            print(f"❌ Test index not found in list")
            return False
        
        show_raw_s3_operations("List Indexes", "list_indexes", {
            "total_indexes": len(indexes),
            "indexes": indexes,
            "test_index_found": index_name in indexes,
        })
        
        # Test 5: Load index
        print(f"\n📥 Test 5: Loading index from S3")
        
        start_time = time.time()
        loaded_vectors, loaded_metadata = storage.load_index(index_name)
        load_time = time.time() - start_time
        
        print(f"✅ Index loaded successfully in {load_time:.2f}s")
        print(f"Loaded vectors shape: {loaded_vectors.shape}")
        print(f"Loaded metadata keys: {list(loaded_metadata.keys())}")
        
        show_raw_s3_operations("Load Index", "load_index", {
            "index_name": index_name,
            "loaded_shape": list(loaded_vectors.shape),
            "loaded_dtype": str(loaded_vectors.dtype),
            "duration_seconds": load_time,
            "metadata_keys": list(loaded_metadata.keys()),
        })
        
        # Test 6: Verify data integrity
        print(f"\n🔍 Test 6: Verifying data integrity")
        
        # Check vector shapes
        if loaded_vectors.shape != test_vectors.shape:
            print(f"❌ Vector shape mismatch: {loaded_vectors.shape} != {test_vectors.shape}")
            return False
        
        # Check vector data (approximate due to serialization)
        vector_diff = torch.abs(loaded_vectors - test_vectors).max().item()
        print(f"Max vector difference: {vector_diff:.10f}")
        
        if vector_diff > 1e-6:
            print(f"❌ Vector data differs too much: {vector_diff}")
            return False
        
        # Check metadata
        for key, value in test_metadata.items():
            if key not in loaded_metadata:
                print(f"❌ Missing metadata key: {key}")
                return False
            if loaded_metadata[key] != value:
                print(f"❌ Metadata mismatch for {key}: {loaded_metadata[key]} != {value}")
                return False
        
        print(f"✅ Data integrity verified")
        
        # Test 7: Get index info
        print(f"\n📊 Test 7: Getting index information")
        index_info = storage.get_index_info(index_name)
        
        if index_info:
            print(f"Index info:")
            print(f"  Name: {index_info.index_name}")
            print(f"  Size: {index_info.size_bytes:,} bytes")
            print(f"  Created: {index_info.created_at}")
            print(f"  Updated: {index_info.updated_at}")
            print(f"  Backend: {index_info.backend_type}")
        else:
            print(f"❌ Failed to get index info")
            return False
        
        if VERBOSE:
            print(f"\n🔍 RAW INDEX INFO:")
            print("=" * 60)
            print(f"Full index info: {index_info}")
            print("=" * 60)
        
        # Test 8: Delete index
        print(f"\n🗑️  Test 8: Deleting index")
        
        delete_success = storage.delete_index(index_name)
        if delete_success:
            print(f"✅ Index deleted successfully")
        else:
            print(f"❌ Failed to delete index")
            return False
        
        # Verify deletion
        exists_after_delete = storage.index_exists(index_name)
        if exists_after_delete:
            print(f"❌ Index still exists after deletion")
            return False
        
        print(f"✅ Index deletion verified")
        
        show_raw_s3_operations("Delete Index", "delete_index", {
            "index_name": index_name,
            "delete_success": delete_success,
            "exists_after_delete": exists_after_delete,
        })
        
        print(f"\n✅ All S3 storage tests passed!")
        return True
        
    except StorageBackendError as e:
        print(f"❌ Storage backend error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        if VERBOSE:
            traceback.print_exc()
        return False

def test_s3_storage_advanced():
    """Test advanced S3 storage features."""
    print(f"\n🚀 Testing S3 Storage Backend - Advanced Features")
    print("=" * 60)
    
    if not DEPENDENCIES_AVAILABLE:
        print(f"❌ Dependencies not available: {IMPORT_ERROR}")
        return False
    
    bucket_name = os.environ.get("SEM_S3_BUCKET")
    if not bucket_name:
        print("❌ S3 bucket name not provided")
        return False
    
    # Test different configurations
    test_configs = [
        {
            "name": "No Compression",
            "config": {"compression": False, "encryption": "AES256"},
        },
        {
            "name": "Different Storage Class",
            "config": {"compression": True, "storage_class": "STANDARD_IA"},
        },
        {
            "name": "No Encryption",
            "config": {"compression": True, "encryption": None},
        },
    ]
    
    for test_case in test_configs:
        print(f"\n🧪 Testing: {test_case['name']}")
        
        # Build configuration
        s3_config = {
            "bucket_name": bucket_name,
            "region": os.environ.get("SEM_S3_REGION", "us-east-1"),
            "prefix": f"sem-advanced-test-{int(time.time())}/",
            **test_case["config"]
        }
        
        # Add AWS credentials if available
        if os.environ.get("AWS_ACCESS_KEY_ID"):
            s3_config["aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
        if os.environ.get("AWS_SECRET_ACCESS_KEY"):
            s3_config["aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
        
        show_raw_s3_config(f"Advanced Test: {test_case['name']}", s3_config)
        
        try:
            storage = S3Storage(**s3_config)
            
            # Test with small data
            test_vectors = torch.randn(10, 128)
            test_metadata = {"test": test_case["name"], "small_data": True}
            index_name = f"advanced_test_{int(time.time())}"
            
            # Save and load
            save_success = storage.save_index(test_vectors, test_metadata, index_name)
            if not save_success:
                print(f"❌ Failed to save with config: {test_case['name']}")
                continue
            
            loaded_vectors, loaded_metadata = storage.load_index(index_name)
            
            # Verify
            if loaded_vectors.shape != test_vectors.shape:
                print(f"❌ Shape mismatch in {test_case['name']}")
                continue
            
            # Cleanup
            storage.delete_index(index_name)
            
            print(f"✅ {test_case['name']} test passed")
            
        except Exception as e:
            print(f"❌ {test_case['name']} test failed: {e}")
            continue
    
    print(f"\n✅ Advanced S3 storage tests completed!")
    return True

def test_s3_error_handling():
    """Test S3 storage error handling."""
    print(f"\n⚠️  Testing S3 Storage Backend - Error Handling")
    print("=" * 60)
    
    if not DEPENDENCIES_AVAILABLE:
        print(f"❌ Dependencies not available: {IMPORT_ERROR}")
        return False
    
    # Test 1: Invalid bucket name
    print(f"\n🧪 Test 1: Invalid bucket name")
    try:
        invalid_config = {
            "bucket_name": "this-bucket-definitely-does-not-exist-12345",
            "region": "us-east-1",
        }
        
        show_raw_s3_config("Invalid Bucket Test", invalid_config)
        
        storage = S3Storage(**invalid_config)
        print(f"❌ Should have failed with invalid bucket")
        return False
    except StorageBackendError as e:
        print(f"✅ Correctly caught invalid bucket error: {e}")
    
    # Test 2: Invalid credentials (if not using IAM roles)
    print(f"\n🧪 Test 2: Invalid credentials")
    bucket_name = os.environ.get("SEM_S3_BUCKET")
    if bucket_name and not os.environ.get("AWS_ACCESS_KEY_ID"):
        print("⏭️  Skipping credential test (using IAM roles)")
    elif bucket_name:
        try:
            invalid_creds_config = {
                "bucket_name": bucket_name,
                "region": "us-east-1",
                "aws_access_key_id": "invalid_key",
                "aws_secret_access_key": "invalid_secret",
            }
            
            show_raw_s3_config("Invalid Credentials Test", invalid_creds_config)
            
            storage = S3Storage(**invalid_creds_config)
            print(f"❌ Should have failed with invalid credentials")
            return False
        except StorageBackendError as e:
            print(f"✅ Correctly caught invalid credentials error: {e}")
    
    # Test 3: Load non-existent index
    if bucket_name:
        print(f"\n🧪 Test 3: Load non-existent index")
        try:
            valid_config = {
                "bucket_name": bucket_name,
                "region": os.environ.get("SEM_S3_REGION", "us-east-1"),
            }
            
            if os.environ.get("AWS_ACCESS_KEY_ID"):
                valid_config["aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
                valid_config["aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
            
            storage = S3Storage(**valid_config)
            
            # Try to load non-existent index
            vectors, metadata = storage.load_index("non_existent_index_12345")
            print(f"❌ Should have failed loading non-existent index")
            return False
        except StorageBackendError as e:
            print(f"✅ Correctly caught non-existent index error: {e}")
    
    print(f"\n✅ Error handling tests completed!")
    return True

def main():
    """Run S3 storage tests."""
    parser = argparse.ArgumentParser(description='Test S3 storage backend')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Enable verbose output showing raw inputs and outputs')
    parser.add_argument('--basic-only', action='store_true',
                       help='Run only basic tests')
    parser.add_argument('--advanced-only', action='store_true',
                       help='Run only advanced tests')
    parser.add_argument('--errors-only', action='store_true',
                       help='Run only error handling tests')
    args = parser.parse_args()
    
    # Set global verbose flag
    global VERBOSE
    VERBOSE = args.verbose
    
    if VERBOSE:
        print("🔍 VERBOSE MODE ENABLED - Showing raw inputs and outputs")
        print("=" * 60)
    
    print("🧪 S3 Storage Backend Test Suite")
    print("=" * 60)
    print("📋 Prerequisites:")
    print("  • Set SEM_S3_BUCKET environment variable")
    print("  • Configure AWS credentials (CLI, env vars, or IAM roles)")
    print("  • Ensure S3 bucket exists and is accessible")
    print("  • Install boto3: pip install boto3")
    print()
    
    # Check environment
    if not os.environ.get("SEM_S3_BUCKET"):
        print("❌ Missing SEM_S3_BUCKET environment variable")
        print("   Example: export SEM_S3_BUCKET=my-test-bucket")
        return False
    
    success = True
    
    # Run tests based on arguments
    if args.basic_only:
        success &= test_s3_storage_basic()
    elif args.advanced_only:
        success &= test_s3_storage_advanced()
    elif args.errors_only:
        success &= test_s3_error_handling()
    else:
        # Run all tests
        success &= test_s3_storage_basic()
        success &= test_s3_storage_advanced()
        success &= test_s3_error_handling()
    
    if success:
        print(f"\n🎉 All S3 storage tests passed!")
        print("✨ S3 storage backend is ready for production use!")
    else:
        print(f"\n❌ Some S3 storage tests failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
