Installation Guide
==================

Complete installation instructions for all platforms and deployment scenarios.

Quick Installation
------------------

**Recommended Method (Source):**

.. code-block:: bash

   git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git
   cd simple-embeddings-module
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .

**Verify Installation:**

.. code-block:: bash

   # Test CLI
   sem-cli --help
   
   # Test Python import
   python -c "from simple_embeddings_module import SEMSimple; print('✅ Installation successful!')"

System Requirements
-------------------

**Minimum Requirements:**
   - Python 3.8 or higher
   - 4GB RAM (8GB recommended)
   - 2GB free disk space
   - Internet connection (for model downloads)

**Recommended Requirements:**
   - Python 3.10 or higher
   - 8GB RAM or more
   - GPU with CUDA/MPS/ROCm support
   - SSD storage for better performance

**Supported Platforms:**
   - macOS (Intel and Apple Silicon)
   - Linux (Ubuntu, CentOS, RHEL, etc.)
   - Windows 10/11
   - Docker containers
   - Cloud platforms (AWS, GCP, Azure)

Platform-Specific Installation
-------------------------------

macOS
~~~~~

**Using Homebrew (Recommended):**

.. code-block:: bash

   # Install Python if needed
   brew install python@3.11
   
   # Clone and install SEM
   git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git
   cd simple-embeddings-module
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -e .

**Apple Silicon Notes:**
   - MPS (Metal Performance Shaders) acceleration automatically detected
   - PyTorch with MPS support included
   - Optimized for M1/M2/M3 processors

Linux
~~~~~

**Ubuntu/Debian:**

.. code-block:: bash

   # Install dependencies
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3.10-dev git
   
   # Clone and install SEM
   git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git
   cd simple-embeddings-module
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install -e .

**CentOS/RHEL:**

.. code-block:: bash

   # Install dependencies
   sudo yum install python3 python3-venv python3-devel git
   
   # Clone and install SEM
   git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git
   cd simple-embeddings-module
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .

**GPU Support (NVIDIA):**

.. code-block:: bash

   # Install CUDA toolkit (if not already installed)
   # Follow NVIDIA's official installation guide
   
   # PyTorch with CUDA support is automatically installed
   # Verify GPU detection:
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

Windows
~~~~~~~

**Using Command Prompt:**

.. code-block:: batch

   REM Install Git if needed (download from git-scm.com)
   
   REM Clone and install SEM
   git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git
   cd simple-embeddings-module
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e .

**Using PowerShell:**

.. code-block:: powershell

   # Clone and install SEM
   git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git
   cd simple-embeddings-module
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -e .

**Windows Notes:**
   - Requires Python 3.8+ from python.org
   - Git for Windows recommended
   - CUDA support available for NVIDIA GPUs

Docker Installation
-------------------

**Using Docker (Recommended for Production):**

.. code-block:: dockerfile

   FROM python:3.11-slim
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       git \
       && rm -rf /var/lib/apt/lists/*
   
   # Clone and install SEM
   WORKDIR /app
   RUN git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git .
   RUN pip install -e .
   
   # Set up working directory
   WORKDIR /workspace
   
   # Default command
   CMD ["sem-cli", "--help"]

**Build and Run:**

.. code-block:: bash

   # Build image
   docker build -t sem:latest .
   
   # Run interactively
   docker run -it --rm -v $(pwd):/workspace sem:latest bash
   
   # Run specific command
   docker run --rm -v $(pwd):/workspace sem:latest sem-cli simple local search --query "test"

**Docker Compose Example:**

.. code-block:: yaml

   version: '3.8'
   services:
     sem:
       build: .
       volumes:
         - ./data:/workspace/data
         - ./indexes:/workspace/indexes
       environment:
         - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
         - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
       command: sem-cli simple aws search --query "deployment" --bucket my-bucket

Cloud Platform Installation
----------------------------

AWS EC2
~~~~~~~

.. code-block:: bash

   # Launch EC2 instance with Amazon Linux 2
   # Connect via SSH
   
   # Install dependencies
   sudo yum update -y
   sudo yum install python3 python3-pip git -y
   
   # Clone and install SEM
   git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git
   cd simple-embeddings-module
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   
   # Configure AWS credentials (if using AWS features)
   aws configure

Google Cloud Platform
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create Compute Engine instance
   # Connect via SSH
   
   # Install dependencies
   sudo apt update
   sudo apt install python3.10 python3.10-venv git -y
   
   # Clone and install SEM
   git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git
   cd simple-embeddings-module
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install -e .

Azure
~~~~~

.. code-block:: bash

   # Create Azure VM
   # Connect via SSH
   
   # Install dependencies
   sudo apt update
   sudo apt install python3 python3-venv git -y
   
   # Clone and install SEM
   git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git
   cd simple-embeddings-module
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .

Development Installation
------------------------

**For Contributors:**

.. code-block:: bash

   # Clone with development dependencies
   git clone https://github.com/NeuralNotwerk/simple-embeddings-module.git
   cd simple-embeddings-module
   python -m venv .venv
   source .venv/bin/activate
   
   # Install in development mode with all dependencies
   pip install -e ".[dev,test,docs]"
   
   # Install pre-commit hooks
   pre-commit install
   
   # Run tests to verify installation
   python -m pytest test/

**Development Dependencies:**
   - pytest (testing framework)
   - black (code formatting)
   - ruff (linting)
   - mypy (type checking)
   - sphinx (documentation)
   - pre-commit (git hooks)

Optional Dependencies
---------------------

**AWS Support:**

.. code-block:: bash

   # AWS dependencies are included by default
   # Verify AWS functionality:
   python -c "from simple_embeddings_module import simple_aws; print('✅ AWS support available')"

**Additional Embedding Models:**

.. code-block:: bash

   # Install additional sentence-transformers models
   pip install sentence-transformers[all]

**GPU Acceleration:**

.. code-block:: bash

   # NVIDIA CUDA (Linux/Windows)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # AMD ROCm (Linux)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

Verification
------------

**Test Basic Functionality:**

.. code-block:: bash

   # Test CLI
   sem-cli --help
   sem-cli simple --help
   
   # Test local functionality
   echo "test document" | sem-cli simple local index
   sem-cli simple local search --query "test"

**Test Python API:**

.. code-block:: python

   from simple_embeddings_module import SEMSimple
   
   # Create instance
   sem = SEMSimple()
   
   # Add test document
   sem.add_text("This is a test document for verification.")
   
   # Search
   results = sem.search("test verification")
   
   print(f"✅ Found {len(results)} results")
   if results:
       print(f"Score: {results[0]['similarity_score']:.3f}")

**Test GPU Acceleration:**

.. code-block:: python

   import torch
   
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"MPS available: {torch.backends.mps.is_available()}")
   
   if torch.cuda.is_available():
       print(f"CUDA device: {torch.cuda.get_device_name()}")
   elif torch.backends.mps.is_available():
       print("MPS (Apple Silicon) acceleration available")

**Test AWS Functionality (Optional):**

.. code-block:: bash

   # Set up AWS credentials first
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   
   # Test AWS functionality
   echo "test aws document" | sem-cli simple aws index --bucket test-sem-installation
   sem-cli simple aws search --query "test" --bucket test-sem-installation

Troubleshooting
---------------

**Common Issues:**

**Python Version Issues:**

.. code-block:: bash

   # Check Python version
   python --version
   
   # Use specific Python version if needed
   python3.10 -m venv .venv

**Permission Issues (Linux/macOS):**

.. code-block:: bash

   # Fix permission issues
   sudo chown -R $USER:$USER ~/.cache/huggingface/
   
   # Or use user installation
   pip install --user -e .

**Network Issues:**

.. code-block:: bash

   # Test internet connectivity
   curl -I https://huggingface.co
   
   # Use proxy if needed
   pip install -e . --proxy http://proxy.company.com:8080

**GPU Issues:**

.. code-block:: bash

   # Check NVIDIA driver
   nvidia-smi
   
   # Reinstall PyTorch with correct CUDA version
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

**Model Download Issues:**

.. code-block:: bash

   # Clear cache and retry
   rm -rf ~/.cache/huggingface/
   python -c "from simple_embeddings_module import SEMSimple; SEMSimple()"

**AWS Issues:**

.. code-block:: bash

   # Check AWS credentials
   aws sts get-caller-identity
   
   # Check AWS region
   aws configure get region

Getting Help
------------

If you encounter issues:

1. **Check the troubleshooting section above**
2. **Search existing issues**: `GitHub Issues <https://github.com/NeuralNotwerk/simple-embeddings-module/issues>`_
3. **Create a new issue** with:
   - Your operating system and Python version
   - Complete error messages
   - Steps to reproduce the problem
4. **Join discussions**: `GitHub Discussions <https://github.com/NeuralNotwerk/simple-embeddings-module/discussions>`_

**Next: Get started with the** :doc:`quickstart` **guide**
