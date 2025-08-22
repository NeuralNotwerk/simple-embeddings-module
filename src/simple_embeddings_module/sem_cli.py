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


def cmd_simple(args):
    """Simple command for easy access to simple constructs"""
    print("🌟 SEM Simple Interface")
    
    # Check for help-like arguments in the backend/operation
    if hasattr(args, 'backend') and args.backend in ['-h', '--help']:
        print_simple_help()
        return 0
    
    if hasattr(args, 'operation') and args.operation in ['-h', '--help']:
        print_simple_help()
        return 0
    
    try:
        if args.backend == "local":
            return cmd_simple_local(args)
        elif args.backend == "aws":
            return cmd_simple_aws(args)
        else:
            print(f"❌ Unknown backend: {args.backend}")
            print("Available backends: local, aws")
            print("Use 'sem-cli simple --help' for detailed usage information")
            return 1
    except Exception as e:
        print(f"❌ Simple command failed: {e}")
        return 1


def cmd_help(args):
    """Show detailed help for commands"""
    if not args.help_command:
        print("🌟 SEM CLI Help")
        print("=" * 50)
        print("Available commands:")
        print("  init        Initialize a new semantic search database")
        print("  add         Add documents to an existing database")
        print("  search      Search documents in a database")
        print("  info        Show information about a database")
        print("  config      Generate configuration templates")
        print("  simple      Simple interface for quick operations")
        print("  help        Show this help or help for specific commands")
        print()
        print("For detailed help on any command:")
        print("  sem-cli <command> --help")
        print("  sem-cli help <command>")
        print()
        print("Quick start with simple interface:")
        print("  sem-cli help simple")
        return 0
    
    elif args.help_command == "simple":
        print_simple_help()
        return 0
    
    elif args.help_command in ["init", "add", "search", "info", "config"]:
        print(f"For detailed help on '{args.help_command}' command:")
        print(f"  sem-cli {args.help_command} --help")
        return 0
    
    else:
        print(f"❌ Unknown command: {args.help_command}")
        print("Available commands: init, add, search, info, config, simple, help")
        return 1


def print_simple_help():
    """Print contextual help for simple commands"""
    print("""
🌟 SEM Simple Interface Help

USAGE:
  sem-cli simple <backend> <operation> [options]

BACKENDS:
  local    Use local storage with sentence-transformers
  aws      Use AWS S3 + Bedrock for cloud operations

OPERATIONS:
  index        Index text from stdin or --text arguments
  indexfiles   Index files from stdin paths or --files arguments  
  search       Search the semantic index (requires --query)

QUICK EXAMPLES:
  # Index text from stdin (local)
  echo "some text" | sem-cli simple local index
  
  # Index files from ls output (AWS)
  ls -d ./docs/* | sem-cli simple aws indexfiles --bucket my-bucket
  
  # Search for content
  sem-cli simple local search --query "machine learning"
  sem-cli simple aws search --query "deployment" --bucket my-bucket

For detailed help with all options:
  sem-cli simple --help
    """)


def cmd_simple_local(args):
    """Handle local simple operations"""
    from .sem_simple import SEMSimple
    
    print(f"📝 Using local backend with index: {args.index or 'sem_simple_index'}")
    
    # Validate operation-specific requirements
    if args.operation == "search" and not args.query:
        print("❌ Search operation requires --query argument")
        print("Example: sem-cli simple local search --query 'your search terms'")
        return 1
    
    # Create simple instance
    sem = SEMSimple(
        index_name=args.index or "sem_simple_index",
        storage_path=args.path or "./sem_indexes"
    )
    
    if args.operation == "index":
        # Index text from stdin or arguments
        texts_to_index = []
        
        # Read from stdin if available
        if not sys.stdin.isatty():
            stdin_text = sys.stdin.read().strip()
            if stdin_text:
                texts_to_index.append(stdin_text)
        
        # Add text arguments
        if args.text:
            texts_to_index.extend(args.text)
        
        if not texts_to_index:
            print("❌ No text to index. Provide text via stdin or --text arguments")
            print("Examples:")
            print("  echo 'some text' | sem-cli simple local index")
            print("  sem-cli simple local index --text 'document 1' 'document 2'")
            return 1
        
        print(f"📝 Indexing {len(texts_to_index)} text(s)...")
        for i, text in enumerate(texts_to_index):
            success = sem.add_text(text)
            if success:
                print(f"   ✅ Indexed text {i+1}: {text[:50]}...")
            else:
                print(f"   ❌ Failed to index text {i+1}")
        
        print("✅ Indexing complete!")
        return 0
    
    elif args.operation == "indexfiles":
        # Index files from arguments or stdin
        files_to_index = []
        
        # Read file paths from stdin if available
        if not sys.stdin.isatty():
            stdin_lines = sys.stdin.read().strip().split('\n')
            files_to_index.extend([line.strip() for line in stdin_lines if line.strip()])
        
        # Add file arguments
        if args.files:
            files_to_index.extend(args.files)
        
        if not files_to_index:
            print("❌ No files to index. Provide file paths via stdin or --files arguments")
            return 1
        
        print(f"📁 Indexing {len(files_to_index)} file(s)...")
        for file_path in files_to_index:
            try:
                path = Path(file_path)
                if not path.exists():
                    print(f"   ❌ File not found: {file_path}")
                    continue
                
                content = path.read_text(encoding='utf-8')
                success = sem.add_text(content, doc_id=path.name)
                if success:
                    print(f"   ✅ Indexed: {path.name} ({len(content)} chars)")
                else:
                    print(f"   ❌ Failed to index: {path.name}")
            except Exception as e:
                print(f"   ❌ Error indexing {file_path}: {e}")
        
        print("✅ File indexing complete!")
        return 0
    
    elif args.operation == "search":
        # Search operation
        if not args.query:
            print("❌ No search query provided. Use --query argument")
            return 1
        
        print(f"🔍 Searching for: '{args.query}'")
        results = sem.search(args.query, top_k=args.top_k or 5)
        
        if not results:
            print("   No results found")
            return 0
        
        print(f"   Found {len(results)} result(s):")
        for i, result in enumerate(results, 1):
            score = result.get('similarity_score', 0)
            text = result.get('text', result.get('document', ''))[:100] + "..."
            doc_id = result.get('document_id', f'doc_{i}')
            print(f"   {i}. {doc_id} (score: {score:.3f})")
            print(f"      {text}")
        
        return 0
    
    else:
        print(f"❌ Unknown operation: {args.operation}")
        print("Available operations: index, indexfiles, search")
        return 1


def cmd_simple_aws(args):
    """Handle AWS simple operations"""
    try:
        from .sem_simple_aws import simple_aws
    except ImportError:
        print("❌ AWS dependencies not available. Install with: pip install boto3")
        return 1
    
    print(f"☁️  Using AWS backend with bucket: {args.bucket or 'auto-generated'}")
    
    # Validate operation-specific requirements
    if args.operation == "search" and not args.query:
        print("❌ Search operation requires --query argument")
        print("Example: sem-cli simple aws search --query 'your search terms' --bucket my-bucket")
        return 1
    
    # Create AWS simple instance
    kwargs = {}
    if args.bucket:
        kwargs['bucket_name'] = args.bucket
    if args.region:
        kwargs['region'] = args.region
    if args.model:
        kwargs['embedding_model'] = args.model
    
    sem = simple_aws(**kwargs)
    
    if args.operation == "index":
        # Index text from stdin or arguments
        texts_to_index = []
        
        # Read from stdin if available
        if not sys.stdin.isatty():
            stdin_text = sys.stdin.read().strip()
            if stdin_text:
                texts_to_index.append(stdin_text)
        
        # Add text arguments
        if args.text:
            texts_to_index.extend(args.text)
        
        if not texts_to_index:
            print("❌ No text to index. Provide text via stdin or --text arguments")
            print("Examples:")
            print("  echo 'cloud document' | sem-cli simple aws index --bucket my-bucket")
            print("  sem-cli simple aws index --text 'doc 1' 'doc 2' --bucket my-bucket")
            return 1
        
        print(f"📝 Indexing {len(texts_to_index)} text(s) to AWS...")
        for i, text in enumerate(texts_to_index):
            doc_id = sem.add_text(text)
            print(f"   ✅ Indexed text {i+1} as {doc_id}: {text[:50]}...")
        
        print("✅ AWS indexing complete!")
        return 0
    
    elif args.operation == "indexfiles":
        # Index files from arguments or stdin
        files_to_index = []
        
        # Read file paths from stdin if available
        if not sys.stdin.isatty():
            stdin_lines = sys.stdin.read().strip().split('\n')
            files_to_index.extend([line.strip() for line in stdin_lines if line.strip()])
        
        # Add file arguments
        if args.files:
            files_to_index.extend(args.files)
        
        if not files_to_index:
            print("❌ No files to index. Provide file paths via stdin or --files arguments")
            return 1
        
        print(f"📁 Indexing {len(files_to_index)} file(s) to AWS...")
        for file_path in files_to_index:
            try:
                path = Path(file_path)
                if not path.exists():
                    print(f"   ❌ File not found: {file_path}")
                    continue
                
                content = path.read_text(encoding='utf-8')
                doc_id = sem.add_text(content, document_id=path.name)
                print(f"   ✅ Indexed: {path.name} as {doc_id} ({len(content)} chars)")
            except Exception as e:
                print(f"   ❌ Error indexing {file_path}: {e}")
        
        print("✅ AWS file indexing complete!")
        return 0
    
    elif args.operation == "search":
        # Search operation
        if not args.query:
            print("❌ No search query provided. Use --query argument")
            return 1
        
        print(f"🔍 Searching AWS index for: '{args.query}'")
        results = sem.search(args.query, top_k=args.top_k or 5)
        
        if not results:
            print("   No results found")
            return 0
        
        print(f"   Found {len(results)} result(s):")
        for i, result in enumerate(results, 1):
            score = result.get('similarity_score', 0)
            text = result.get('document', '')[:100] + "..."
            doc_id = result.get('document_id', f'doc_{i}')
            print(f"   {i}. {doc_id} (score: {score:.3f})")
            print(f"      {text}")
        
        return 0
    
    else:
        print(f"❌ Unknown operation: {args.operation}")
        print("Available operations: index, indexfiles, search")
        return 1


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

  # Simple interface examples:
  # Index text from stdin (local)
  echo "some text" | sem-cli simple local index

  # Index files from ls output (AWS)
  ls -d ./dev_docs/* | sem-cli simple aws indexfiles

  # Index specific files (local)
  sem-cli simple local indexfiles --files doc1.txt doc2.txt

  # Search with simple interface
  sem-cli simple aws search --query "machine learning"
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

    # Simple command
    simple_parser = subparsers.add_parser("simple", 
        help="Simple interface for easy operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Backend and Operation Combinations:

LOCAL BACKEND:
  sem-cli simple local index [options]
    Index text from stdin or --text arguments
    Examples:
      echo "some text" | sem-cli simple local index
      sem-cli simple local index --text "doc1" "doc2"
      
  sem-cli simple local indexfiles [options]  
    Index files from stdin paths or --files arguments
    Examples:
      ls *.md | sem-cli simple local indexfiles
      sem-cli simple local indexfiles --files doc1.txt doc2.txt
      
  sem-cli simple local search --query "search terms" [options]
    Search the local semantic index
    Examples:
      sem-cli simple local search --query "machine learning"
      sem-cli simple local search --query "API docs" --top-k 3

AWS BACKEND:
  sem-cli simple aws index [options]
    Index text to AWS S3 + Bedrock
    Examples:
      echo "cloud document" | sem-cli simple aws index --bucket my-bucket
      sem-cli simple aws index --text "doc1" --bucket my-bucket
      
  sem-cli simple aws indexfiles [options]
    Index files to AWS S3 + Bedrock  
    Examples:
      ls *.md | sem-cli simple aws indexfiles --bucket my-bucket
      sem-cli simple aws indexfiles --files *.py --bucket my-bucket
      
  sem-cli simple aws search --query "search terms" [options]
    Search the AWS semantic index
    Examples:
      sem-cli simple aws search --query "deployment" --bucket my-bucket
      sem-cli simple aws search --query "config" --bucket my-bucket --top-k 5

COMMON PATTERNS:
  # Pipeline from ls to indexing
  ls -d ./docs/* | sem-cli simple local indexfiles
  
  # Find and index specific file types
  find . -name "*.py" | sem-cli simple aws indexfiles --bucket code-search
  
  # Index and immediately search
  echo "test document" | sem-cli simple local index
  sem-cli simple local search --query "test"
  
  # Use custom settings
  sem-cli simple local index --index my_docs --path ./my_storage --text "content"
  sem-cli simple aws index --bucket my-bucket --region us-west-2 --model amazon.titan-embed-text-v1
        """
    )
    simple_parser.add_argument("backend", choices=["local", "aws"], help="Backend to use")
    simple_parser.add_argument("operation", choices=["index", "indexfiles", "search"], help="Operation to perform")
    
    # Common simple arguments
    simple_parser.add_argument("--query", help="Search query (required for search operation)")
    simple_parser.add_argument("--text", nargs="+", help="Text content to index")
    simple_parser.add_argument("--files", nargs="+", help="Files to index")
    simple_parser.add_argument("--top-k", type=int, default=5, help="Number of search results (default: 5)")
    
    # Local backend arguments
    simple_parser.add_argument("--index", help="Index name for local backend (default: sem_simple_index)")
    simple_parser.add_argument("--path", help="Storage path for local backend (default: ./sem_indexes)")
    
    # AWS backend arguments
    simple_parser.add_argument("--bucket", help="S3 bucket name for AWS backend (auto-generated if not specified)")
    simple_parser.add_argument("--region", help="AWS region for AWS backend (default: us-east-1)")
    simple_parser.add_argument("--model", help="Bedrock embedding model for AWS backend (default: amazon.titan-embed-text-v2:0)")

    # Help command
    help_parser = subparsers.add_parser("help", help="Show detailed help for commands")
    help_parser.add_argument("help_command", nargs="?", help="Command to get help for (optional)")

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
    elif args.command == "simple":
        return cmd_simple(args)
    elif args.command == "help":
        return cmd_help(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
