Examples
========

Complete working examples demonstrating SEM capabilities across different use cases and interfaces.

.. toctree::
   :maxdepth: 1

   quickstart-examples
   cli-examples
   python-examples
   pipeline-examples
   aws-examples
   advanced-examples

Overview
--------

These examples are organized by complexity and use case:

**🚀 Quickstart Examples** - Get started in minutes
   Simple, copy-paste examples for immediate results

**⚙️ CLI Examples** - Command-line workflows
   Shell scripts and pipeline integration examples

**🐍 Python Examples** - Programmatic usage
   Complete Python applications and integrations

**🔄 Pipeline Examples** - Automation workflows
   CI/CD, data processing, and batch operation examples

**☁️ AWS Examples** - Cloud deployment
   Scalable cloud-based semantic search implementations

**🎯 Advanced Examples** - Complex use cases
   Custom configurations, multi-backend setups, and optimization

Quick Reference
---------------

**Most Common Use Cases:**

.. code-block:: bash

   # Documentation search (CLI)
   find ./docs -name "*.md" | sem-cli simple local indexfiles
   sem-cli simple local search --query "installation guide"

.. code-block:: python

   # Documentation search (Python)
   from simple_embeddings_module import SEMSimple
   
   sem = SEMSimple()
   sem.add_text("Installation instructions for the application...")
   results = sem.search("how to install")

.. code-block:: bash

   # Cloud-based team search (CLI)
   ls -d ./team_docs/* | sem-cli simple aws indexfiles --bucket team-knowledge
   sem-cli simple aws search --query "project requirements" --bucket team-knowledge

.. code-block:: python

   # Cloud-based team search (Python)
   from simple_embeddings_module import simple_aws
   
   sem = simple_aws(bucket_name="team-knowledge")
   sem.add_text("Project requirements and specifications...")
   results = sem.search("requirements")

Example Categories
------------------

By Interface
~~~~~~~~~~~~

**Simple Interface Examples:**
   - :doc:`quickstart-examples` - Immediate results with zero configuration
   - :doc:`python-examples` - SEMSimple and simple_aws usage

**CLI Interface Examples:**
   - :doc:`cli-examples` - Command-line operations and scripting
   - :doc:`pipeline-examples` - Shell integration and automation

**Advanced Interface Examples:**
   - :doc:`advanced-examples` - Custom configurations and complex setups

By Use Case
~~~~~~~~~~~

**Documentation & Knowledge Management:**
   - Technical documentation search
   - API reference indexing
   - Team knowledge bases
   - Research paper management

**Code & Development:**
   - Source code semantic search
   - Function and class discovery
   - Code documentation generation
   - Development workflow integration

**Data Processing:**
   - Log file analysis
   - Content classification
   - Data pipeline integration
   - Batch processing workflows

**Cloud & Enterprise:**
   - Multi-team deployments
   - Scalable cloud architectures
   - Cost optimization strategies
   - Security and compliance patterns

By Complexity
~~~~~~~~~~~~~

**Beginner (5 minutes):**
   - Single-command operations
   - Default configurations
   - Basic search and indexing

**Intermediate (15 minutes):**
   - Custom configurations
   - Pipeline integration
   - Multi-file operations

**Advanced (30+ minutes):**
   - Multi-backend setups
   - Custom embedding models
   - Performance optimization
   - Production deployments

Running the Examples
--------------------

**Prerequisites:**

.. code-block:: bash

   # Install SEM
   git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git
   cd simple-embeddings-module
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .

**For AWS Examples:**

.. code-block:: bash

   # Set up AWS credentials
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   
   # Or use AWS CLI
   aws configure

**Test Your Setup:**

.. code-block:: bash

   # Test local functionality
   echo "test document" | sem-cli simple local index
   sem-cli simple local search --query "test"
   
   # Test AWS functionality (if configured)
   echo "test document" | sem-cli simple aws index --bucket test-sem-bucket
   sem-cli simple aws search --query "test" --bucket test-sem-bucket

Example Data
------------

Many examples use sample data included in the repository:

**Sample Documents:**
   - ``./dev_docs/`` - Technical documentation
   - ``./demo/demo_code_samples.py`` - Code examples
   - ``./README.md`` - Project documentation

**Creating Test Data:**

.. code-block:: bash

   # Create sample documents for testing
   mkdir -p ./test_docs
   
   echo "Machine learning transforms software development" > ./test_docs/ml.txt
   echo "Python is excellent for data science applications" > ./test_docs/python.txt
   echo "Docker containers simplify application deployment" > ./test_docs/docker.txt
   
   # Index the test documents
   sem-cli simple local indexfiles --files ./test_docs/*.txt

Performance Benchmarks
----------------------

**Local Performance (Apple Silicon M-series):**
   - Model loading: ~2.5s (first run only)
   - Document processing: ~0.3s per document
   - Search speed: 0.4-2.1s depending on index size
   - Memory usage: ~2GB for model + index size

**AWS Performance:**
   - Setup time: ~1-2s for service initialization
   - Document processing: ~0.5s per document (Bedrock API)
   - Search speed: ~0.5s including S3 retrieval
   - Storage: 44% smaller than manual serialization

**Scaling Characteristics:**
   - Local: Optimized for up to 100K documents
   - AWS: Unlimited scaling with S3 + Bedrock
   - Memory usage scales linearly with index size
   - Search performance remains consistent across index sizes

Contributing Examples
---------------------

We welcome community examples! To contribute:

1. **Create a new example file** in the appropriate category
2. **Follow the existing format** with clear explanations
3. **Test thoroughly** on different platforms
4. **Include performance notes** and expected outputs
5. **Submit a pull request** with your example

**Example Template:**

.. code-block:: python

   """
   Example: [Brief Description]
   
   This example demonstrates [specific functionality].
   
   Prerequisites:
   - [List any requirements]
   
   Expected Output:
   - [Describe what users should see]
   
   Performance Notes:
   - [Any relevant performance information]
   """
   
   # Your example code here

**Next: Start with** :doc:`quickstart-examples` **for immediate results**
