Simple Interface Guide
=====================

The Simple Interface is the fastest way to get semantic search working. It provides sensible defaults and automatic configuration while still being powerful enough for production use.

Philosophy
----------

The Simple Interface follows these principles:

🚀 **Zero Configuration** - Works out of the box with sensible defaults
📝 **Standardized Naming** - Consistent index and storage names across all backends
🔍 **Automatic Detection** - Finds existing indexes and provides helpful guidance
⚡ **Performance Optimized** - GPU acceleration and optimal chunking automatically configured

Available Interfaces
--------------------

Python Simple Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~

**SEMSimple (Local)**

.. code-block:: python

   from simple_embeddings_module import SEMSimple
   
   # Create with defaults
   sem = SEMSimple()
   
   # Or customize
   sem = SEMSimple(
       index_name="my_custom_index",
       storage_path="./my_custom_storage"
   )

**simple_aws (Cloud)**

.. code-block:: python

   from simple_embeddings_module import simple_aws
   
   # Create with auto-generated bucket
   sem = simple_aws()
   
   # Or customize
   sem = simple_aws(
       bucket_name="my-semantic-search",
       region="us-west-2",
       embedding_model="amazon.titan-embed-text-v1"
   )

CLI Simple Interface
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Local operations
   sem-cli simple local <operation> [options]
   
   # AWS operations  
   sem-cli simple aws <operation> [options]

Standardized Naming
-------------------

All Simple Interfaces use consistent naming:

**Index Names:**
   - Default: ``sem_simple_index``
   - Consistent across local and AWS backends
   - Automatically detected and reused

**Storage Locations:**
   - Local: ``./sem_indexes/`` (default directory)
   - AWS: Auto-generated bucket names with ``sem-`` prefix
   - User-specified names always respected

**Benefits:**
   - Predictable data locations
   - Easy to find and manage indexes
   - Consistent behavior across different environments
   - Simplified troubleshooting

Automatic Index Detection
-------------------------

The Simple Interface automatically detects existing indexes:

**First Run (No Existing Index):**

.. code-block:: python

   from simple_embeddings_module import SEMSimple
   
   sem = SEMSimple()
   # Output: 📝 Ready to add documents! Use .add_text('your content') to start

**Subsequent Runs (Existing Index Found):**

.. code-block:: python

   from simple_embeddings_module import SEMSimple
   
   sem = SEMSimple()
   # Output: 📚 Found existing semantic search index with 42 documents
   #         🔍 Ready to search! Use .search('your query') to find documents

**CLI Detection:**

.. code-block:: bash

   $ sem-cli simple local search --query "test"
   🌟 SEM Simple Interface
   📝 Using local backend with index: sem_simple_index
   📚 Found existing semantic search index with 42 documents
   🔍 Ready to search! Use .search('your query') to find documents

Local Simple Interface (SEMSimple)
-----------------------------------

Perfect for development, single-machine deployments, and getting started.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from simple_embeddings_module import SEMSimple
   
   # Create instance
   sem = SEMSimple()
   
   # Add documents
   sem.add_text("Machine learning is transforming software development.")
   sem.add_text("Python is excellent for data science applications.")
   
   # Search semantically
   results = sem.search("AI and programming")
   
   # Process results
   for result in results:
       print(f"Score: {result['similarity_score']:.3f}")
       print(f"Text: {result['text']}")

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   # Custom configuration
   sem = SEMSimple(
       index_name="project_docs",
       storage_path="./project_indexes"
   )
   
   # Batch operations
   documents = [
       "Document 1 content...",
       "Document 2 content...", 
       "Document 3 content..."
   ]
   
   # Add multiple documents
   for i, doc in enumerate(documents):
       sem.add_text(doc, doc_id=f"doc_{i}")
   
   # Advanced search
   results = sem.search(
       "your query",
       top_k=10,
       similarity_threshold=0.2
   )

Features
~~~~~~~~

- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Storage**: Local compressed JSON files
- **GPU**: Automatic MPS/CUDA/ROCm detection
- **Performance**: ~100-500 docs/second indexing, ~0.1-0.5s search
- **Cost**: Free (local processing only)

AWS Simple Interface (simple_aws)
----------------------------------

Scalable cloud deployment with AWS S3 + Bedrock integration.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from simple_embeddings_module import simple_aws
   
   # Create instance (auto-generates bucket)
   sem = simple_aws()
   
   # Add documents
   doc_id = sem.add_text("Cloud-based machine learning deployment strategies.")
   
   # Search
   results = sem.search("ML deployment in cloud environments")
   
   # Get information
   info = sem.get_info()
   print(f"Bucket: {info['s3_bucket']}")
   print(f"Documents: {info['document_count']}")

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   # Custom configuration
   sem = simple_aws(
       bucket_name="my-company-semantic-search",
       region="us-west-2",
       embedding_model="amazon.titan-embed-text-v1"
   )
   
   # Batch operations with metadata
   documents = [
       ("Technical documentation for API v2.0", "api_docs_v2"),
       ("User guide for new features", "user_guide_2024"),
       ("Troubleshooting common issues", "troubleshooting")
   ]
   
   for content, doc_id in documents:
       sem.add_text(content, document_id=doc_id)
   
   # Search with custom parameters
   results = sem.search(
       "API documentation",
       top_k=5,
       similarity_threshold=0.3
   )

Features
~~~~~~~~

- **Model**: Amazon Titan Text Embeddings v2 (1024 dimensions)
- **Storage**: S3 with AES256 encryption and compression
- **Performance**: ~3-5 docs/second indexing, ~0.3-0.8s search
- **Scalability**: Unlimited document storage
- **Cost**: Pay-per-use AWS pricing

Configuration Options
---------------------

Local Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   sem = SEMSimple(
       index_name="custom_index",      # Default: "sem_simple_index"
       storage_path="./custom_path"    # Default: "./sem_indexes"
   )

AWS Configuration
~~~~~~~~~~~~~~~~~

.. code-block:: python

   sem = simple_aws(
       bucket_name="my-bucket",                    # Default: auto-generated
       region="us-west-2",                        # Default: "us-east-1"
       embedding_model="amazon.titan-embed-text-v1"  # Default: v2
   )

CLI Configuration
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Local options
   sem-cli simple local index --index my_docs --path ./my_storage
   
   # AWS options
   sem-cli simple aws index --bucket my-bucket --region us-west-2 --model amazon.titan-embed-text-v1

Common Patterns
---------------

Documentation Search
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simple_embeddings_module import SEMSimple
   from pathlib import Path
   
   # Create documentation search
   sem = SEMSimple(index_name="docs", storage_path="./doc_indexes")
   
   # Index all markdown files
   for md_file in Path("./docs").glob("**/*.md"):
       content = md_file.read_text()
       sem.add_text(content, doc_id=md_file.name)
   
   # Search documentation
   results = sem.search("installation instructions")

Code Search
~~~~~~~~~~~

.. code-block:: python

   from simple_embeddings_module import simple_aws
   from pathlib import Path
   
   # Create cloud-based code search
   sem = simple_aws(bucket_name="company-code-search")
   
   # Index Python files
   for py_file in Path("./src").glob("**/*.py"):
       content = py_file.read_text()
       sem.add_text(content, document_id=str(py_file))
   
   # Find code by functionality
   results = sem.search("database connection handling")

Research Paper Management
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simple_embeddings_module import simple_aws
   
   # Create research paper index
   sem = simple_aws(bucket_name="research-papers-index")
   
   # Index paper abstracts
   papers = [
       ("Neural networks for natural language processing", "paper_001"),
       ("Deep learning approaches to computer vision", "paper_002"),
       ("Transformer architectures and attention mechanisms", "paper_003")
   ]
   
   for abstract, paper_id in papers:
       sem.add_text(abstract, document_id=paper_id)
   
   # Find relevant papers
   results = sem.search("attention mechanisms in NLP")

Migration Between Backends
--------------------------

You can easily migrate data between local and AWS backends:

**Local to AWS:**

.. code-block:: python

   from simple_embeddings_module import SEMSimple, simple_aws
   
   # Load from local
   local_sem = SEMSimple()
   local_results = local_sem.search("", top_k=1000)  # Get all documents
   
   # Save to AWS
   aws_sem = simple_aws(bucket_name="migrated-data")
   for result in local_results:
       aws_sem.add_text(result['text'], document_id=result.get('document_id'))

**AWS to Local:**

.. code-block:: python

   from simple_embeddings_module import SEMSimple, simple_aws
   
   # Load from AWS
   aws_sem = simple_aws(bucket_name="source-data")
   aws_results = aws_sem.search("", top_k=1000)  # Get all documents
   
   # Save to local
   local_sem = SEMSimple(storage_path="./migrated_indexes")
   for result in aws_results:
       local_sem.add_text(result['document'], document_id=result.get('document_id'))

Performance Optimization
------------------------

**Local Performance:**
   - First run downloads model (~2-3 seconds)
   - GPU acceleration automatically detected
   - Batch operations for better throughput
   - Compressed storage saves ~44% space

**AWS Performance:**
   - Bedrock API rate limits apply (~3-5 docs/second)
   - S3 operations are optimized with compression
   - Consider batch operations for large datasets
   - Monitor AWS costs for large-scale usage

**General Tips:**
   - Use appropriate ``top_k`` values (5-10 for most use cases)
   - Adjust similarity thresholds based on your data
   - Index documents in batches when possible
   - Reuse existing indexes when available

Troubleshooting
---------------

**Local Issues:**

.. code-block:: python

   # Model download issues
   try:
       sem = SEMSimple()
   except Exception as e:
       print(f"Model download failed: {e}")
       # Check internet connection and retry

**AWS Issues:**

.. code-block:: python

   # Credentials issues
   try:
       sem = simple_aws()
   except Exception as e:
       print(f"AWS setup failed: {e}")
       # Check AWS credentials: aws sts get-caller-identity

**Search Issues:**

.. code-block:: python

   # No results found
   results = sem.search("very specific query")
   if not results:
       # Try broader terms
       results = sem.search("broader query", top_k=10)

**Next: Learn about** :doc:`backends` **for more deployment options**
