#!/usr/bin/env python3
"""Setup script for Simple Embeddings Module (SEM)."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="simple-embeddings-module",
    version="0.1.0",
    author="NeuralNotwerk",
    author_email="",
    description="A modular, cross-platform semantic search engine with intelligent chunking and GPU acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeuralNotwerk/simple-embeddings-module",
    project_urls={
        "Bug Reports": "https://github.com/NeuralNotwerk/simple-embeddings-module/issues",
        "Source": "https://github.com/NeuralNotwerk/simple-embeddings-module",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sem-cli=simple_embeddings_module.sem_cli:main",
        ],
    },
    keywords="semantic search, embeddings, nlp, machine learning, pytorch, gpu acceleration",
    include_package_data=True,
    zip_safe=False,
)
