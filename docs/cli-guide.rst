CLI User Guide
==============

The SEM CLI provides powerful command-line tools for semantic search, from simple one-liners to complex automation workflows.

Overview
--------

SEM offers two CLI approaches:

**🌟 Simple Interface** - Quick operations with minimal configuration
   Perfect for getting started, scripting, and common workflows

**⚙️ Traditional Interface** - Full control with explicit configuration
   Ideal for complex setups, custom configurations, and advanced use cases

Simple Interface
----------------

The simple interface maps directly to the Python simple modules with easy command-line access.

Command Structure
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   sem-cli simple <backend> <operation> [options]

**Backends:**
   - ``local`` - Local storage with sentence-transformers
   - ``aws`` - AWS S3 + Bedrock for cloud operations

**Operations:**
   - ``index`` - Index text from stdin or arguments
   - ``indexfiles`` - Index files from paths
   - ``search`` - Search the semantic index

Basic Examples
~~~~~~~~~~~~~~

**Local Operations:**

.. code-block:: bash

   # Index text from stdin
   echo "Machine learning is transforming software development" | sem-cli simple local index
   
   # Index multiple text arguments
   sem-cli simple local index --text "First document" "Second document"
   
   # Index files from ls output
   ls -d ./docs/*.md | sem-cli simple local indexfiles
   
   # Index specific files
   sem-cli simple local indexfiles --files doc1.txt doc2.txt doc3.txt
   
   # Search the index
   sem-cli simple local search --query "machine learning algorithms"
   
   # Custom local settings
   sem-cli simple local index --index my_docs --path ./my_storage --text "Custom document"

**AWS Operations:**

.. code-block:: bash

   # Index text to AWS
   echo "Cloud-based machine learning deployment" | sem-cli simple aws index --bucket my-sem-bucket
   
   # Index files to AWS
   ls -d ./documentation/* | sem-cli simple aws indexfiles --bucket my-sem-bucket
   
   # Search AWS index
   sem-cli simple aws search --query "deployment strategies" --bucket my-sem-bucket
   
   # Custom AWS settings
   sem-cli simple aws index --bucket my-bucket --region us-west-2 --model amazon.titan-embed-text-v1

Pipeline Integration
~~~~~~~~~~~~~~~~~~~~

The simple interface excels at shell pipeline integration:

**Documentation Indexing:**

.. code-block:: bash

   #!/bin/bash
   # Index all documentation and search for topics
   
   echo "📚 Indexing documentation..."
   find ./docs -name "*.md" | sem-cli simple local indexfiles --index docs
   
   echo "🔍 Searching for API references..."
   sem-cli simple local search --index docs --query "API reference" --top-k 3

**CI/CD Integration:**

.. code-block:: bash

   # In your CI pipeline
   name: Index Documentation
   run: |
     # Index updated documentation
     git diff --name-only HEAD~1 HEAD | grep '\.md$' | sem-cli simple aws indexfiles --bucket ci-docs
     
     # Verify search works
     sem-cli simple aws search --bucket ci-docs --query "getting started" --top-k 1

**Data Processing Workflows:**

.. code-block:: bash

   # Process and index data files
   for file in data/*.json; do
     jq -r '.description' "$file" | sem-cli simple local index --index data_descriptions
   done
   
   # Search processed data
   sem-cli simple local search --index data_descriptions --query "user behavior analysis"

Options Reference
~~~~~~~~~~~~~~~~~

**Common Options:**
   - ``--query QUERY`` - Search query (required for search operation)
   - ``--text TEXT [TEXT ...]`` - Text content to index
   - ``--files FILES [FILES ...]`` - Files to index
   - ``--top-k TOP_K`` - Number of search results (default: 5)

**Local Backend Options:**
   - ``--index INDEX`` - Index name (default: ``sem_simple_index``)
   - ``--path PATH`` - Storage path (default: ``./sem_indexes``)

**AWS Backend Options:**
   - ``--bucket BUCKET`` - S3 bucket name (auto-generated if not specified)
   - ``--region REGION`` - AWS region (default: ``us-east-1``)
   - ``--model MODEL`` - Bedrock embedding model (default: ``amazon.titan-embed-text-v2:0``)

Traditional Interface
---------------------

The traditional interface provides full control over SEM configuration and operations.

Available Commands
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   sem-cli init        # Initialize new database
   sem-cli add         # Add documents
   sem-cli search      # Search documents
   sem-cli info        # Show database info
   sem-cli config      # Generate config template

Workflow Example
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Initialize a database
   sem-cli init --name my_docs --path ./my_indexes --model all-mpnet-base-v2
   
   # 2. Add documents
   sem-cli add --files doc1.txt doc2.txt doc3.txt --path ./my_indexes
   sem-cli add --text "Additional document content" --path ./my_indexes
   
   # 3. Search documents
   sem-cli search "machine learning" --path ./my_indexes --top-k 5 --threshold 0.1
   
   # 4. Show database information
   sem-cli info --path ./my_indexes

Configuration-Based Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Generate configuration template
   sem-cli config --output my_config.json --model all-mpnet-base-v2 --storage local_disk
   
   # 2. Edit configuration as needed
   # (modify my_config.json)
   
   # 3. Initialize with configuration
   sem-cli init --config my_config.json
   
   # 4. Use configuration for operations
   sem-cli add --files *.txt --config my_config.json
   sem-cli search "query" --config my_config.json

Help System
-----------

SEM provides comprehensive help at multiple levels:

Discovery Help
~~~~~~~~~~~~~~

.. code-block:: bash

   # Main CLI overview
   sem-cli --help
   sem-cli -h
   
   # Interactive help overview
   sem-cli help

Command-Specific Help
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Detailed help for any command
   sem-cli <command> --help
   
   # Examples:
   sem-cli init --help
   sem-cli simple --help
   sem-cli search --help

Contextual Help
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Workflow-focused help
   sem-cli help simple
   sem-cli help <command>

Error Recovery
~~~~~~~~~~~~~~

When commands fail, SEM provides helpful error messages with examples:

.. code-block:: bash

   # Missing search query
   $ sem-cli simple local search
   ❌ Search operation requires --query argument
   Example: sem-cli simple local search --query 'your search terms'
   
   # Missing text input
   $ sem-cli simple local index
   ❌ No text to index. Provide text via stdin or --text arguments
   Examples:
     echo 'some text' | sem-cli simple local index
     sem-cli simple local index --text 'document 1' 'document 2'

Performance Considerations
--------------------------

**Local Backend:**
   - Model loading: ~2-3 seconds (first run only)
   - Indexing speed: ~100-500 docs/second
   - Search speed: ~0.1-0.5 seconds
   - Storage: Compressed JSON files

**AWS Backend:**
   - Setup time: ~1-2 seconds
   - Indexing speed: ~3-5 docs/second (Bedrock API limited)
   - Search speed: ~0.3-0.8 seconds (including S3 retrieval)
   - Storage: Encrypted, compressed data in S3

Best Practices
--------------

**Bucket Management (AWS):**
   - Always specify bucket names for data persistence
   - Use descriptive bucket names for team collaboration
   - Consider bucket lifecycle policies for cost optimization

**Index Organization (Local):**
   - Use consistent index names across projects
   - Organize indexes by project or domain
   - Use descriptive storage paths

**Pipeline Integration:**
   - Test with small datasets first
   - Monitor AWS costs when using Bedrock extensively
   - Use appropriate ``--top-k`` values for performance
   - Implement error handling in scripts

**Security:**
   - Store AWS credentials securely
   - Use IAM roles when possible
   - Consider S3 bucket policies for team access
   - Regularly rotate access keys

Troubleshooting
---------------

**Common Issues:**

.. code-block:: bash

   # AWS credentials not found
   ❌ AWS credentials not available. Configure with:
      - AWS CLI: aws configure
      - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
   
   # File not found
   ❌ File not found: nonexistent.txt
   
   # Empty search results
   🔍 Searching for: 'very specific query'
      No results found

**Solutions:**

.. code-block:: bash

   # Check AWS credentials
   aws sts get-caller-identity
   
   # Verify file paths
   ls -la your_file.txt
   
   # Try broader search terms
   sem-cli simple local search --query "broader terms" --top-k 10

Advanced Usage
--------------

**Custom Embedding Models:**

.. code-block:: bash

   # Use different sentence-transformers model
   sem-cli init --model all-mpnet-base-v2 --path ./custom_index
   
   # Use AWS Bedrock models
   sem-cli simple aws index --model amazon.titan-embed-text-v1 --bucket my-bucket

**Batch Processing:**

.. code-block:: bash

   # Process multiple directories
   for dir in docs/ src/ examples/; do
     find "$dir" -name "*.md" -o -name "*.py" | \
       sem-cli simple local indexfiles --index "${dir%/}_index"
   done

**Integration with Other Tools:**

.. code-block:: bash

   # Combine with ripgrep for code search
   rg -l "function.*search" --type py | \
     sem-cli simple local indexfiles --index code_search
   
   # Use with jq for JSON processing
   find . -name "*.json" -exec jq -r '.description // empty' {} \; | \
     sem-cli simple local index --index json_descriptions

**Next: Explore the** :doc:`python-api` **for programmatic control**
