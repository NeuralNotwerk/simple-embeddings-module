#!/usr/bin/env python3
"""Demo script for S3 storage backend with SEM."""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from simple_embeddings_module import SEMConfigBuilder, SEMDatabase
    from src.simple_embeddings_module.storage.mod_s3 import S3Storage
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)

def print_banner():
    """Print demo banner."""
    print("🌟" * 70)
    print("🪣 S3 STORAGE BACKEND DEMO 🪣")
    print("🌟" * 70)
    print("☁️  Cloud-native semantic search with Amazon S3!")
    print("🔒 Secure, scalable, and globally accessible storage")
    print("⚡ Automatic compression and encryption")
    print("🌟" * 70)
    print()

def demo_s3_configuration():
    """Demo S3 storage configuration options."""
    print("⚙️  S3 STORAGE CONFIGURATION")
    print("=" * 50)
    
    print("📋 Available configuration options:")
    
    # Show configuration parameters
    config_options = {
        "bucket_name": "S3 bucket name (required)",
        "region": "AWS region (default: us-east-1)",
        "prefix": "S3 key prefix for organization (default: sem-indexes/)",
        "compression": "Enable gzip compression (default: True)",
        "encryption": "Server-side encryption (AES256, aws:kms, None)",
        "storage_class": "S3 storage class (STANDARD, STANDARD_IA, GLACIER)",
        "aws_access_key_id": "AWS access key (optional with IAM roles)",
        "aws_secret_access_key": "AWS secret key (optional with IAM roles)",
        "endpoint_url": "Custom S3 endpoint (for S3-compatible services)",
    }
    
    for param, description in config_options.items():
        print(f"  • {param}: {description}")
    
    print(f"\n🔧 Example configuration:")
    example_config = {
        "bucket_name": "my-sem-bucket",
        "region": "us-west-2",
        "prefix": "production/indexes/",
        "compression": True,
        "encryption": "AES256",
        "storage_class": "STANDARD",
    }
    
    for key, value in example_config.items():
        print(f"  {key}: {value}")

def demo_s3_with_sem():
    """Demo using S3 storage with SEM database."""
    print(f"\n🚀 SEM DATABASE WITH S3 STORAGE")
    print("=" * 50)
    
    if not DEPENDENCIES_AVAILABLE:
        print(f"❌ Dependencies not available: {IMPORT_ERROR}")
        print("   Install with: pip install boto3")
        return False
    
    # Check for S3 configuration
    bucket_name = os.environ.get("SEM_S3_BUCKET")
    if not bucket_name:
        print("❌ S3 bucket name not provided")
        print("   Set environment variable: SEM_S3_BUCKET=your-bucket-name")
        print("   Example: export SEM_S3_BUCKET=my-sem-demo-bucket")
        return False
    
    try:
        # Build SEM configuration with S3 storage
        print("🔧 Building SEM configuration with S3 storage...")
        
        builder = SEMConfigBuilder()
        
        # Configure embedding provider
        builder.set_embedding_provider("sentence_transformers", model="all-MiniLM-L6-v2")
        
        # Configure S3 storage
        s3_config = {
            "bucket_name": bucket_name,
            "region": os.environ.get("SEM_S3_REGION", "us-east-1"),
            "prefix": f"sem-demo-{int(time.time())}/",
            "compression": True,
            "encryption": "AES256",
            "storage_class": "STANDARD",
        }
        
        # Add AWS credentials if provided
        if os.environ.get("AWS_ACCESS_KEY_ID"):
            s3_config["aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
        if os.environ.get("AWS_SECRET_ACCESS_KEY"):
            s3_config["aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
        
        builder.set_storage_backend("s3", **s3_config)
        
        # Auto-configure chunking
        builder.auto_configure_chunking()
        
        # Build configuration
        config = builder.build()
        
        print("✅ Configuration built successfully")
        print(f"   Bucket: {bucket_name}")
        print(f"   Region: {s3_config['region']}")
        print(f"   Prefix: {s3_config['prefix']}")
        
        # Create SEM database
        print(f"\n📊 Creating SEM database with S3 storage...")
        db = SEMDatabase(config=config)
        
        # Add sample documents
        print(f"\n📝 Adding sample documents...")
        sample_docs = [
            "Machine learning is transforming how we build software applications.",
            "Cloud storage provides scalable and reliable data persistence solutions.",
            "Semantic search enables finding documents by meaning rather than keywords.",
            "Amazon S3 offers industry-leading durability and availability for object storage.",
            "Vector embeddings capture semantic relationships between text documents.",
        ]
        
        for i, doc in enumerate(sample_docs, 1):
            doc_id = f"demo_doc_{i}"
            db.add_documents({doc_id: doc})
            print(f"   Added document {i}: {doc[:50]}...")
        
        print(f"✅ Added {len(sample_docs)} documents to S3-backed database")
        
        # Perform semantic search
        print(f"\n🔍 Performing semantic search...")
        search_queries = [
            "artificial intelligence and software",
            "cloud data storage solutions",
            "finding documents by meaning",
        ]
        
        for query in search_queries:
            print(f"\n🔎 Query: '{query}'")
            results = db.search(query, top_k=2)
            
            for i, result in enumerate(results, 1):
                score = result.get('score', 0)
                text = result.get('text', '')[:60] + "..."
                print(f"   {i}. Score: {score:.3f} - {text}")
        
        # Show database info
        print(f"\n📊 Database information:")
        info = db.get_info()
        for key, value in info.items():
            if key == "storage_backend_info":
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\n✅ S3 storage demo completed successfully!")
        print(f"💡 Your data is now stored securely in Amazon S3!")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_s3_direct_usage():
    """Demo direct usage of S3 storage backend."""
    print(f"\n🔧 DIRECT S3 STORAGE USAGE")
    print("=" * 50)
    
    if not DEPENDENCIES_AVAILABLE:
        print(f"❌ Dependencies not available: {IMPORT_ERROR}")
        return False
    
    bucket_name = os.environ.get("SEM_S3_BUCKET")
    if not bucket_name:
        print("❌ S3 bucket name not provided")
        return False
    
    try:
        # Create S3 storage directly
        print("🔧 Creating S3 storage backend directly...")
        
        s3_config = {
            "bucket_name": bucket_name,
            "region": os.environ.get("SEM_S3_REGION", "us-east-1"),
            "prefix": f"direct-demo-{int(time.time())}/",
            "compression": True,
            "encryption": "AES256",
        }
        
        if os.environ.get("AWS_ACCESS_KEY_ID"):
            s3_config["aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
            s3_config["aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
        
        storage = S3Storage(**s3_config)
        
        # Show capabilities
        print(f"\n📋 S3 storage capabilities:")
        capabilities = storage.get_capabilities()
        for key, value in capabilities.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
        
        # Create sample data
        print(f"\n📊 Creating sample vector data...")
        import torch
        
        vectors = torch.randn(50, 384)  # 50 vectors of 384 dimensions
        metadata = {
            "index_name": "direct_demo",
            "vector_count": 50,
            "embedding_dim": 384,
            "model_name": "demo-model",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        print(f"   Vectors shape: {vectors.shape}")
        print(f"   Metadata: {metadata}")
        
        # Save to S3
        print(f"\n💾 Saving vectors to S3...")
        index_name = "direct_demo_index"
        
        start_time = time.time()
        success = storage.save_index(vectors, metadata, index_name)
        save_time = time.time() - start_time
        
        if success:
            print(f"✅ Saved successfully in {save_time:.2f} seconds")
        else:
            print(f"❌ Save failed")
            return False
        
        # Load from S3
        print(f"\n📥 Loading vectors from S3...")
        
        start_time = time.time()
        loaded_vectors, loaded_metadata = storage.load_index(index_name)
        load_time = time.time() - start_time
        
        print(f"✅ Loaded successfully in {load_time:.2f} seconds")
        print(f"   Loaded shape: {loaded_vectors.shape}")
        print(f"   Metadata keys: {list(loaded_metadata.keys())}")
        
        # Verify data integrity
        print(f"\n🔍 Verifying data integrity...")
        
        shape_match = loaded_vectors.shape == vectors.shape
        data_diff = torch.abs(loaded_vectors - vectors).max().item()
        
        print(f"   Shape match: {shape_match}")
        print(f"   Max difference: {data_diff:.10f}")
        
        if shape_match and data_diff < 1e-6:
            print(f"✅ Data integrity verified")
        else:
            print(f"❌ Data integrity check failed")
            return False
        
        # Get index info
        print(f"\n📊 Index information:")
        index_info = storage.get_index_info(index_name)
        if index_info:
            print(f"   Name: {index_info.index_name}")
            print(f"   Size: {index_info.size_bytes:,} bytes")
            print(f"   Backend: {index_info.backend_type}")
            print(f"   Created: {index_info.created_at}")
        
        # Clean up
        print(f"\n🧹 Cleaning up...")
        delete_success = storage.delete_index(index_name)
        if delete_success:
            print(f"✅ Index deleted successfully")
        else:
            print(f"❌ Failed to delete index")
        
        print(f"\n✅ Direct S3 usage demo completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct usage demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_s3_best_practices():
    """Demo S3 storage best practices."""
    print(f"\n💡 S3 STORAGE BEST PRACTICES")
    print("=" * 50)
    
    practices = [
        {
            "title": "🔒 Security",
            "tips": [
                "Use IAM roles instead of hardcoded credentials when possible",
                "Enable server-side encryption (AES256 or KMS)",
                "Use bucket policies to restrict access",
                "Enable S3 access logging for audit trails",
            ]
        },
        {
            "title": "💰 Cost Optimization",
            "tips": [
                "Use appropriate storage classes (STANDARD_IA for infrequent access)",
                "Enable compression to reduce storage costs",
                "Set up lifecycle policies for automatic archiving",
                "Monitor usage with AWS Cost Explorer",
            ]
        },
        {
            "title": "⚡ Performance",
            "tips": [
                "Use prefixes to distribute load across S3 partitions",
                "Choose regions close to your compute resources",
                "Enable transfer acceleration for global access",
                "Use multipart uploads for large indexes",
            ]
        },
        {
            "title": "🛡️ Reliability",
            "tips": [
                "Enable versioning for data protection",
                "Set up cross-region replication for disaster recovery",
                "Use CloudWatch metrics to monitor health",
                "Implement retry logic with exponential backoff",
            ]
        },
    ]
    
    for practice in practices:
        print(f"\n{practice['title']}")
        for tip in practice['tips']:
            print(f"   • {tip}")
    
    print(f"\n📋 Example production configuration:")
    production_config = {
        "bucket_name": "my-company-sem-production",
        "region": "us-west-2",
        "prefix": "production/semantic-indexes/",
        "compression": True,
        "encryption": "aws:kms",  # Use KMS for enhanced security
        "storage_class": "STANDARD",
        # Use IAM roles - no hardcoded credentials
    }
    
    for key, value in production_config.items():
        print(f"   {key}: {value}")

def main():
    """Run S3 storage demo."""
    print_banner()
    
    print("📋 Prerequisites:")
    print("  • AWS account with S3 access")
    print("  • S3 bucket created and accessible")
    print("  • AWS credentials configured (CLI, env vars, or IAM roles)")
    print("  • boto3 installed: pip install boto3")
    print()
    
    print("🔧 Environment setup:")
    print("  export SEM_S3_BUCKET=your-bucket-name")
    print("  export SEM_S3_REGION=us-east-1  # optional")
    print("  export AWS_ACCESS_KEY_ID=your-key  # if not using IAM roles")
    print("  export AWS_SECRET_ACCESS_KEY=your-secret  # if not using IAM roles")
    print()
    
    # Run demos
    try:
        demo_s3_configuration()
        demo_s3_with_sem()
        demo_s3_direct_usage()
        demo_s3_best_practices()
        
        print(f"\n🎉 S3 STORAGE DEMO COMPLETE!")
        print("=" * 50)
        print("✨ Key takeaways:")
        print("  🪣 S3 provides scalable, durable storage for SEM indexes")
        print("  🔒 Built-in encryption and compression for security and efficiency")
        print("  ☁️  Global accessibility with regional optimization")
        print("  💰 Cost-effective with multiple storage class options")
        print("  🚀 Production-ready with enterprise features")
        print()
        print("🌟 Your semantic search is now cloud-native! 🌟")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
