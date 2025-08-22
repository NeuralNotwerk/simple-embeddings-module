Simple Embeddings Module (SEM) Documentation
===========================================

🚀 **A modular, cross-platform semantic search engine with intelligent chunking and GPU acceleration.**

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   installation
   simple-interface

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   cli-guide
   python-api
   configuration
   backends

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   architecture
   chunking-strategies
   embedding-providers
   storage-backends
   performance

.. toctree::
   :maxdepth: 2
   :caption: Reference

   cli-reference
   api-reference
   configuration-reference
   troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/index

Overview
--------

SEM provides semantic search capabilities through multiple interfaces:

**🌟 Simple Interface** - Get started in seconds
   One-line semantic search with sensible defaults

**⚙️ CLI Interface** - Command-line power
   Full-featured CLI for scripting and automation

**🐍 Python API** - Programmatic control
   Complete API for custom applications

**☁️ Cloud Ready** - AWS Integration
   S3 + Bedrock for scalable cloud deployment

Key Features
------------

- 🧠 **Semantic Search**: Find documents by meaning, not just keywords
- ⚡ **GPU Acceleration**: Apple Silicon MPS, NVIDIA CUDA, AMD ROCm support
- 📝 **Intelligent Chunking**: Auto-configured based on embedding model constraints
- 🧩 **Hierarchy-Constrained Grouping**: Groups related code chunks within scope boundaries
- 🔒 **Secure**: No pickle files - pure JSON serialization with orjson
- 🔧 **Modular**: "Bring your own" embedding models, storage backends, and chunking strategies
- 🎯 **Production Ready**: Atomic writes, backups, compression, validation
- 📊 **Scalable**: Designed for 100K+ documents by default

Quick Start
-----------

**Simple Interface (Recommended for new users):**

.. code-block:: python

   from simple_embeddings_module import SEMSimple
   
   sem = SEMSimple()
   sem.add_text("Machine learning is transforming software development.")
   results = sem.search("AI technology")
   print(results[0]['text'])

**CLI Interface:**

.. code-block:: bash

   # Index text from stdin
   echo "some text" | sem-cli simple local index
   
   # Index files from ls output
   ls -d ./docs/* | sem-cli simple aws indexfiles --bucket my-bucket
   
   # Search for content
   sem-cli simple local search --query "machine learning"

Choose Your Path
----------------

.. grid:: 2

   .. grid-item-card:: 🚀 I want to get started quickly
      :link: quickstart
      :link-type: doc

      Jump right in with the simple interface and start searching in minutes.

   .. grid-item-card:: ⚙️ I want command-line power
      :link: cli-guide
      :link-type: doc

      Learn the full CLI interface for scripting and automation.

   .. grid-item-card:: 🐍 I want programmatic control
      :link: python-api
      :link-type: doc

      Explore the complete Python API for custom applications.

   .. grid-item-card:: ☁️ I want cloud deployment
      :link: backends
      :link-type: doc

      Set up AWS S3 + Bedrock for scalable cloud deployment.

Support
-------

- **Issues**: `GitHub Issues <https://github.com/NeuralNotwerk/simple-embeddings-module/issues>`_
- **Discussions**: `GitHub Discussions <https://github.com/NeuralNotwerk/simple-embeddings-module/discussions>`_
- **Documentation**: This documentation site
- **Examples**: See the examples directory for complete working examples
