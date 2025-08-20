#!/usr/bin/env python3
"""
Simple Embeddings Module CLI

Command-line interface for testing and using the Simple Embeddings Module.
Provides basic operations for MVP demonstration.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .sem_config_builder import SEMConfigBuilder
from .sem_core import SEMDatabase, create_database
from .sem_utils import (
    create_quick_config,
    generate_config_template,
    get_config_info,
    load_config,
    save_config,
    setup_logging,
)


def setup_cli_logging(verbose: bool = False):
    """Setup logging for CLI"""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level)


def cmd_init(args):
    """Initialize a new SEM database"""
    print("🚀 Initializing Simple Embeddings Module database...")

    try:
        if args.config:
            # Use existing config file
            config = load_config(args.config)
            print(f"📄 Using configuration: {args.config}")
        else:
            # Create quick config
            config = create_quick_config(
                embedding_model=args.model, storage_path=args.path, index_name=args.name
            )
            print("⚙️  Created quick configuration:")
            print(f"   Model: {args.model}")
            print(f"   Storage: {args.path}")
            print(f"   Index: {args.name}")

        # Create database
        db = SEMDatabase(config=config.to_dict())

        # Save config if not provided
        if not args.config:
            config_path = Path(args.path) / "config.json"
            save_config(config, str(config_path))
            print(f"💾 Saved configuration to: {config_path}")

        print("✅ Database initialized successfully!")
        print("📊 Configuration summary:")
        info = get_config_info(config)
        for key, value in info.items():
            print(f"   {key}: {value}")

    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return 1

    return 0


def cmd_add(args):
    """Add documents to the database"""
    print("📝 Adding documents to database...")

    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            # Try to find config in storage path
            config_path = Path(args.path or "./indexes") / "config.json"
            if config_path.exists():
                config = load_config(str(config_path))
            else:
                print("❌ No configuration found. Run 'sem-cli init' first.")
                return 1

        # Create database
        db = SEMDatabase(config=config.to_dict())

        # Collect documents
        documents = []
        document_ids = []

        if args.files:
            # Read from files
            for file_path in args.files:
                path = Path(file_path)
                if not path.exists():
                    print(f"⚠️  File not found: {file_path}")
                    continue

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()

                    documents.append(content)
                    document_ids.append(path.stem)
                    print(f"📄 Loaded: {file_path}")

                except Exception as e:
                    print(f"⚠️  Failed to read {file_path}: {e}")

        if args.text:
            # Add text directly
            for i, text in enumerate(args.text):
                documents.append(text)
                document_ids.append(f"text_{i}")

        if not documents:
            print("❌ No documents to add. Specify --files or --text.")
            return 1

        # Add documents to database
        print(f"🔄 Processing {len(documents)} documents...")
        result = db.add_documents(documents, document_ids)

        print("✅ Documents added successfully!")
        print("📊 Results:")
        print(f"   Documents added: {result['documents_added']}")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Embedding dimension: {result['embedding_dimension']}")

    except Exception as e:
        print(f"❌ Failed to add documents: {e}")
        return 1

    return 0


def cmd_search(args):
    """Search the database"""
    print(f"🔍 Searching for: '{args.query}'")

    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            # Try to find config in storage path
            config_path = Path(args.path or "./indexes") / "config.json"
            if config_path.exists():
                config = load_config(str(config_path))
            else:
                print("❌ No configuration found. Run 'sem-cli init' first.")
                return 1

        # Create database
        db = SEMDatabase(config=config.to_dict())

        # Perform search
        results = db.search(
            args.query, top_k=args.top_k, similarity_threshold=args.threshold
        )

        if not results:
            print("🤷 No results found.")
            return 0

        print(f"📋 Found {len(results)} results:")
        print()

        for i, result in enumerate(results, 1):
            print(f"🏆 Result {i}:")
            print(f"   ID: {result['document_id']}")
            print(f"   Score: {result['similarity_score']:.4f}")
            print(
                f"   Text: {result['document'][:200]}{'...' if len(result['document']) > 200 else ''}"
            )
            print()

    except Exception as e:
        print(f"❌ Search failed: {e}")
        return 1

    return 0


def cmd_info(args):
    """Show database information"""
    print("📊 Database Information")

    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            config_path = Path(args.path or "./indexes") / "config.json"
            if config_path.exists():
                config = load_config(str(config_path))
            else:
                print("❌ No configuration found.")
                return 1

        # Show config info
        print("\n⚙️  Configuration:")
        info = get_config_info(config)
        for key, value in info.items():
            print(f"   {key}: {value}")

        # Create database and get index info
        try:
            db = SEMDatabase(config=config.to_dict())
            index_info = db.get_index_info()

            if index_info:
                print("\n📈 Index Information:")
                print(f"   Documents: {index_info.get('document_count', 'unknown')}")
                print(
                    f"   Embedding dimension: {index_info.get('embedding_dim', 'unknown')}"
                )
                print(f"   Model: {index_info.get('model_name', 'unknown')}")
                print(f"   Created: {index_info.get('created_at', 'unknown')}")
                print(f"   Updated: {index_info.get('updated_at', 'unknown')}")

                if "size_bytes" in index_info and index_info["size_bytes"]:
                    size_mb = index_info["size_bytes"] / (1024 * 1024)
                    print(f"   Size: {size_mb:.1f} MB")
            else:
                print("\n📭 No index found (database not initialized)")

        except Exception as e:
            print(f"\n⚠️  Could not access database: {e}")

    except Exception as e:
        print(f"❌ Failed to get info: {e}")
        return 1

    return 0


def cmd_config(args):
    """Generate configuration template"""
    print("⚙️  Generating configuration template...")

    try:
        success = generate_config_template(
            args.output,
            embedding_provider=args.provider,
            embedding_model=args.model,
            storage_backend=args.storage,
            storage_path=args.path,
        )

        if success:
            print(f"✅ Configuration template saved to: {args.output}")
        else:
            print("❌ Failed to generate configuration template")
            return 1

    except Exception as e:
        print(f"❌ Failed to generate config: {e}")
        return 1

    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Simple Embeddings Module CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize database with default settings
  sem-cli init --name my_docs --path ./my_indexes

  # Add documents from files
  sem-cli add --files doc1.txt doc2.txt --path ./my_indexes

  # Add text directly
  sem-cli add --text "Hello world" "Another document" --path ./my_indexes

  # Search documents
  sem-cli search "machine learning" --path ./my_indexes --top-k 5

  # Show database info
  sem-cli info --path ./my_indexes

  # Generate config template
  sem-cli config --output config.json --model all-mpnet-base-v2
        """,
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize new database")
    init_parser.add_argument("--name", default="default", help="Index name")
    init_parser.add_argument("--path", default="./indexes", help="Storage path")
    init_parser.add_argument(
        "--model", default="all-MiniLM-L6-v2", help="Embedding model"
    )
    init_parser.add_argument("--config", help="Use existing config file")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add documents")
    add_parser.add_argument("--files", nargs="+", help="Document files to add")
    add_parser.add_argument("--text", nargs="+", help="Text content to add")
    add_parser.add_argument("--path", help="Storage path (or use --config)")
    add_parser.add_argument("--config", help="Configuration file")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--top-k", type=int, default=10, help="Number of results"
    )
    search_parser.add_argument("--threshold", type=float, help="Similarity threshold")
    search_parser.add_argument("--path", help="Storage path (or use --config)")
    search_parser.add_argument("--config", help="Configuration file")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show database info")
    info_parser.add_argument("--path", help="Storage path (or use --config)")
    info_parser.add_argument("--config", help="Configuration file")

    # Config command
    config_parser = subparsers.add_parser("config", help="Generate config template")
    config_parser.add_argument("--output", required=True, help="Output config file")
    config_parser.add_argument(
        "--provider", default="sentence_transformers", help="Embedding provider"
    )
    config_parser.add_argument(
        "--model", default="all-MiniLM-L6-v2", help="Embedding model"
    )
    config_parser.add_argument(
        "--storage", default="local_disk", help="Storage backend"
    )
    config_parser.add_argument("--path", default="./indexes", help="Storage path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    setup_cli_logging(args.verbose)

    # Route to command handlers
    if args.command == "init":
        return cmd_init(args)
    elif args.command == "add":
        return cmd_add(args)
    elif args.command == "search":
        return cmd_search(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "config":
        return cmd_config(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
