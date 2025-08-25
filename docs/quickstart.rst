Quick Start Guide
=================

Get up and running with SEM in under 5 minutes! This guide shows you the fastest path to semantic search.

Installation
------------

.. code-block:: bash

   # Install from source (recommended)
   git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git
   cd simple-embeddings-module
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .

Your First Semantic Search
---------------------------

**Option 1: Python Simple Interface (Easiest)**

.. code-block:: python

   from simple_embeddings_module import SEMSimple

   # Create semantic search instance
   sem = SEMSimple()

   # Add some documents
   sem.add_text("Machine learning is transforming software development.")
   sem.add_text("Python is a popular programming language for AI.")
   sem.add_text("Docker containers help with application deployment.")

   # Search semantically (not just keywords!)
   results = sem.search("artificial intelligence")

   # Print results
   for result in results:
       print(f"Score: {result['similarity_score']:.3f}")
       print(f"Text: {result['text']}")
       print("---")

**Option 2: CLI Simple Interface (Great for scripting)**

.. code-block:: bash

   # Index some text
   echo "Machine learning transforms software development" | sem-cli simple local index
   echo "Python is popular for AI applications" | sem-cli simple local index
   echo "Docker helps with deployment" | sem-cli simple local index

   # Search semantically
   sem-cli simple local search --query "artificial intelligence"

**Option 3: AWS Cloud (Scalable)**

.. code-block:: bash

   # Set up AWS credentials first
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret

   # Index to AWS
   echo "Cloud-based machine learning deployment" | sem-cli simple aws index --bucket my-sem-bucket

   # Search AWS index
   sem-cli simple aws search --query "ML deployment" --bucket my-sem-bucket

What Just Happened?
-------------------

🎯 **Semantic Understanding**: The search found documents about "AI" and "machine learning" even though you searched for "artificial intelligence" - that's semantic search in action!

🚀 **Zero Configuration**: SEM automatically:
   - Downloaded the embedding model (first run only)
   - Configured optimal chunk sizes
   - Set up local storage
   - Enabled GPU acceleration (if available)

📊 **Standardized Naming**: All your data is stored with consistent naming:
   - Index name: ``sem_simple_index``
   - Local storage: ``./sem_indexes/``
   - AWS buckets: Auto-generated or user-specified

Next Steps
----------

Now that you have semantic search working, explore these paths:

**🔍 Index Real Documents**

.. code-block:: bash

   # Index all markdown files in your project
   find . -name "*.md" | sem-cli simple local indexfiles

   # Search your documentation
   sem-cli simple local search --query "installation instructions"

**☁️ Scale to the Cloud**

.. code-block:: bash

   # Index files to AWS for team sharing
   ls -d ./docs/* | sem-cli simple aws indexfiles --bucket team-docs

   # Team members can search the same index
   sem-cli simple aws search --query "API documentation" --bucket team-docs

**🐍 Build Custom Applications**

.. code-block:: python

   from simple_embeddings_module import SEMSimple

   # Custom storage location
   sem = SEMSimple(storage_path="./my_custom_index")

   # Batch add documents
   documents = [
       "Document 1 content...",
       "Document 2 content...",
       "Document 3 content..."
   ]

   for doc in documents:
       sem.add_text(doc)

   # Advanced search with more results
   results = sem.search("your query", top_k=10)

**⚙️ Explore Advanced Features**

- :doc:`cli-guide` - Full CLI capabilities
- :doc:`python-api` - Complete Python API
- :doc:`configuration` - Custom configurations
- :doc:`backends` - Different storage and embedding options

Common Patterns
---------------

**Documentation Search**

.. code-block:: bash

   # Index all documentation
   find ./docs -name "*.md" -o -name "*.rst" | sem-cli simple local indexfiles

   # Search for topics
   sem-cli simple local search --query "getting started"
   sem-cli simple local search --query "API reference"
   sem-cli simple local search --query "troubleshooting"

**Code Search**

.. code-block:: bash

   # Index Python files
   find ./src -name "*.py" | sem-cli simple local indexfiles

   # Find code by functionality
   sem-cli simple local search --query "database connection"
   sem-cli simple local search --query "error handling"

**Research Papers**

.. code-block:: bash

   # Index research papers (AWS for large collections)
   ls ~/papers/*.pdf | sem-cli simple aws indexfiles --bucket research-papers

   # Find papers by topic
   sem-cli simple aws search --query "neural networks" --bucket research-papers

Troubleshooting
---------------

**Model Download Issues**

If the embedding model download fails:

.. code-block:: bash

   # Check internet connection and try again
   python -c "from simple_embeddings_module import SEMSimple; SEMSimple()"

**AWS Credentials**

If AWS operations fail:

.. code-block:: bash

   # Check credentials are set
   aws sts get-caller-identity

   # Or set them explicitly
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret

**No Results Found**

If searches return no results:

- Try broader search terms
- Check that documents were actually indexed
- Use ``sem-cli simple local search --query "test" --top-k 10`` to see more results

**Performance Issues**

- First run is slower (model download)
- Subsequent runs are much faster
- GPU acceleration automatically detected and used

Getting Help
------------

.. code-block:: bash

   # Get help at any level
   sem-cli --help                    # Overview of all commands
   sem-cli simple --help             # Complete simple interface help
   sem-cli help simple               # Contextual simple interface help

   # Specific command help
   sem-cli init --help
   sem-cli search --help

**Next: Dive deeper with the** :doc:`cli-guide` **or** :doc:`python-api`
