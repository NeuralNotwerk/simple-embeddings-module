CLI Reference
=============

Complete reference for all SEM CLI commands and options.

Command Overview
----------------

.. code-block:: bash

   sem-cli <command> [options]

**Available Commands:**

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Command
     - Description
   * - ``simple``
     - Simple interface for quick operations with minimal configuration
   * - ``init``
     - Initialize a new semantic search database
   * - ``add``
     - Add documents to an existing database
   * - ``search``
     - Search documents in a database
   * - ``info``
     - Show information about a database
   * - ``config``
     - Generate configuration templates
   * - ``help``
     - Show detailed help for commands

Global Options
--------------

.. option:: --verbose, -v

   Enable verbose output for debugging

.. option:: --help, -h

   Show help message and exit

Simple Command
--------------

The simple command provides easy access to semantic search with minimal configuration.

Synopsis
~~~~~~~~

.. code-block:: bash

   sem-cli simple <backend> <operation> [options]

Backends
~~~~~~~~

.. option:: local

   Use local storage with sentence-transformers embedding model

.. option:: aws

   Use AWS S3 storage with Bedrock embedding service

Operations
~~~~~~~~~~

.. option:: index

   Index text content from stdin or command arguments

.. option:: indexfiles

   Index files specified by paths from stdin or command arguments

.. option:: search

   Search the semantic index for similar content

Options
~~~~~~~

**Common Options:**

.. option:: --query QUERY

   Search query string (required for search operation)

.. option:: --text TEXT [TEXT ...]

   One or more text strings to index

.. option:: --files FILES [FILES ...]

   One or more file paths to index

.. option:: --top-k TOP_K

   Number of search results to return (default: 5)

**Local Backend Options:**

.. option:: --index INDEX

   Index name for local backend (default: ``sem_simple_index``)

.. option:: --path PATH

   Storage directory path for local backend (default: ``./sem_indexes``)

**AWS Backend Options:**

.. option:: --bucket BUCKET

   S3 bucket name for AWS backend (auto-generated if not specified)

.. option:: --region REGION

   AWS region for AWS backend (default: ``us-east-1``)

.. option:: --model MODEL

   Bedrock embedding model for AWS backend (default: ``amazon.titan-embed-text-v2:0``)

Examples
~~~~~~~~

**Local Operations:**

.. code-block:: bash

   # Index text from stdin
   echo "Machine learning transforms software" | sem-cli simple local index
   
   # Index multiple text arguments
   sem-cli simple local index --text "Document 1" "Document 2"
   
   # Index files from ls output
   ls *.md | sem-cli simple local indexfiles
   
   # Index specific files
   sem-cli simple local indexfiles --files doc1.txt doc2.txt
   
   # Search with custom settings
   sem-cli simple local search --query "AI" --index my_docs --top-k 10

**AWS Operations:**

.. code-block:: bash

   # Index text to AWS
   echo "Cloud deployment strategies" | sem-cli simple aws index --bucket my-bucket
   
   # Index files to AWS
   find ./docs -name "*.md" | sem-cli simple aws indexfiles --bucket docs-search
   
   # Search AWS index
   sem-cli simple aws search --query "deployment" --bucket my-bucket --top-k 3
   
   # Custom AWS configuration
   sem-cli simple aws index --bucket my-bucket --region us-west-2 --model amazon.titan-embed-text-v1

**Pipeline Examples:**

.. code-block:: bash

   # Documentation pipeline
   find ./docs -name "*.rst" -o -name "*.md" | sem-cli simple local indexfiles --index docs
   sem-cli simple local search --query "installation" --index docs
   
   # Code search pipeline
   find ./src -name "*.py" | sem-cli simple aws indexfiles --bucket code-search
   sem-cli simple aws search --query "database connection" --bucket code-search

Init Command
------------

Initialize a new semantic search database with custom configuration.

Synopsis
~~~~~~~~

.. code-block:: bash

   sem-cli init [options]

Options
~~~~~~~

.. option:: --name NAME

   Index name (default: ``default``)

.. option:: --path PATH

   Storage directory path (default: ``./indexes``)

.. option:: --model MODEL

   Embedding model name (default: ``all-MiniLM-L6-v2``)

.. option:: --config CONFIG

   Use existing configuration file instead of creating new one

Examples
~~~~~~~~

.. code-block:: bash

   # Initialize with defaults
   sem-cli init
   
   # Initialize with custom settings
   sem-cli init --name my_docs --path ./my_indexes --model all-mpnet-base-v2
   
   # Initialize from configuration file
   sem-cli init --config my_config.json

Add Command
-----------

Add documents to an existing semantic search database.

Synopsis
~~~~~~~~

.. code-block:: bash

   sem-cli add [options]

Options
~~~~~~~

.. option:: --files FILES [FILES ...]

   One or more file paths to add to the index

.. option:: --text TEXT [TEXT ...]

   One or more text strings to add to the index

.. option:: --path PATH

   Storage directory path (required unless using --config)

.. option:: --config CONFIG

   Configuration file to use for database settings

Examples
~~~~~~~~

.. code-block:: bash

   # Add files to index
   sem-cli add --files doc1.txt doc2.txt doc3.txt --path ./my_indexes
   
   # Add text directly
   sem-cli add --text "Document content 1" "Document content 2" --path ./my_indexes
   
   # Add using configuration file
   sem-cli add --files *.txt --config my_config.json

Search Command
--------------

Search for documents in a semantic search database.

Synopsis
~~~~~~~~

.. code-block:: bash

   sem-cli search <query> [options]

Arguments
~~~~~~~~~

.. option:: query

   Search query string (required)

Options
~~~~~~~

.. option:: --top-k TOP_K

   Number of search results to return (default: 10)

.. option:: --threshold THRESHOLD

   Minimum similarity threshold for results (default: 0.1)

.. option:: --path PATH

   Storage directory path (required unless using --config)

.. option:: --config CONFIG

   Configuration file to use for database settings

Examples
~~~~~~~~

.. code-block:: bash

   # Basic search
   sem-cli search "machine learning" --path ./my_indexes
   
   # Search with custom parameters
   sem-cli search "AI algorithms" --path ./my_indexes --top-k 5 --threshold 0.2
   
   # Search using configuration file
   sem-cli search "neural networks" --config my_config.json

Info Command
------------

Display information about a semantic search database.

Synopsis
~~~~~~~~

.. code-block:: bash

   sem-cli info [options]

Options
~~~~~~~

.. option:: --path PATH

   Storage directory path (required unless using --config)

.. option:: --config CONFIG

   Configuration file to use for database settings

Examples
~~~~~~~~

.. code-block:: bash

   # Show database info
   sem-cli info --path ./my_indexes
   
   # Show info using configuration file
   sem-cli info --config my_config.json

Config Command
--------------

Generate configuration file templates for custom setups.

Synopsis
~~~~~~~~

.. code-block:: bash

   sem-cli config --output <file> [options]

Options
~~~~~~~

.. option:: --output OUTPUT

   Output configuration file path (required)

.. option:: --provider PROVIDER

   Embedding provider (default: ``sentence_transformers``)

.. option:: --model MODEL

   Embedding model name (default: ``all-MiniLM-L6-v2``)

.. option:: --storage STORAGE

   Storage backend (default: ``local_disk``)

.. option:: --path PATH

   Storage directory path (default: ``./indexes``)

Examples
~~~~~~~~

.. code-block:: bash

   # Generate basic configuration
   sem-cli config --output my_config.json
   
   # Generate configuration with custom settings
   sem-cli config --output advanced_config.json --model all-mpnet-base-v2 --storage s3

Help Command
------------

Show detailed help information for commands.

Synopsis
~~~~~~~~

.. code-block:: bash

   sem-cli help [command]

Arguments
~~~~~~~~~

.. option:: command

   Command name to get help for (optional)

Examples
~~~~~~~~

.. code-block:: bash

   # General help overview
   sem-cli help
   
   # Specific command help
   sem-cli help simple
   sem-cli help init

Exit Codes
----------

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Code
     - Description
   * - 0
     - Success
   * - 1
     - General error (invalid arguments, operation failed)
   * - 2
     - Command line parsing error (missing required arguments)

Environment Variables
---------------------

**AWS Operations:**

.. envvar:: AWS_ACCESS_KEY_ID

   AWS access key for authentication

.. envvar:: AWS_SECRET_ACCESS_KEY

   AWS secret key for authentication

.. envvar:: AWS_REGION

   Default AWS region (overridden by --region option)

.. envvar:: AWS_PROFILE

   AWS profile name to use from ~/.aws/credentials

**General:**

.. envvar:: SEM_LOG_LEVEL

   Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

Configuration Files
-------------------

Configuration files use JSON format:

.. code-block:: json

   {
     "embedding": {
       "provider": "sentence_transformers",
       "model": "all-MiniLM-L6-v2",
       "batch_size": 32
     },
     "chunking": {
       "strategy": "text",
       "boundary_type": "sentence"
     },
     "storage": {
       "backend": "local_disk",
       "path": "./indexes"
     },
     "serialization": {
       "provider": "orjson"
     },
     "index": {
       "name": "sem_simple_index",
       "max_documents": 100000,
       "similarity_threshold": 0.1
     }
   }

Error Messages
--------------

**Common Error Patterns:**

.. code-block:: bash

   # Missing required arguments
   sem-cli simple: error: the following arguments are required: backend, operation
   
   # AWS credentials not found
   ❌ AWS credentials not available. Configure with:
      - AWS CLI: aws configure
      - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
   
   # File not found
   ❌ File not found: nonexistent.txt
   
   # Missing search query
   ❌ Search operation requires --query argument
   Example: sem-cli simple local search --query 'your search terms'

**Error Recovery:**

All error messages include:
- Clear description of the problem
- Specific examples showing correct usage
- Alternative approaches when applicable

Performance Notes
-----------------

**Local Operations:**
   - First run: ~2-3 seconds (model download)
   - Indexing: ~100-500 documents/second
   - Search: ~0.1-0.5 seconds per query

**AWS Operations:**
   - Setup: ~1-2 seconds (service initialization)
   - Indexing: ~3-5 documents/second (API rate limited)
   - Search: ~0.3-0.8 seconds per query (including S3 retrieval)

**Optimization Tips:**
   - Use batch operations for multiple documents
   - Specify appropriate --top-k values (5-10 for most use cases)
   - Consider similarity thresholds based on your data quality
   - Reuse existing indexes when possible

**Next: See** :doc:`examples/index` **for complete working examples**
