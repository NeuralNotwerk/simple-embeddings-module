#!/usr/bin/env python3
"""
SEM CLI - Click-based Implementation
Complete migration from custom decorator system to Click framework.

This replaces the custom decorator system with Click's battle-tested CLI framework,
providing automatic completion, better help generation, and reduced maintenance burden.

Migration Benefits:
- Automatic shell completion (bash, zsh, fish)
- Type validation and conversion
- Better error messages and help text
- Industry standard patterns
- Reduced custom code maintenance (eliminates 500+ lines of completion code)
"""
import logging
import sys
from pathlib import Path
import click
from .sem_utils import setup_logging
from .sem_auto_resolve import auto_resolve_command_args, list_available_databases, list_all_conflicts

logger = logging.getLogger(__name__)


def setup_cli_logging(verbose: bool = False):
    """Setup logging for CLI with proper level configuration."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level)


# Global options decorator for common parameters
def add_global_options(func):
    """Add common global options to commands."""
    func = click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')(func)
    func = click.option('--config', type=click.Path(exists=True), help='Configuration file path')(func)
    return func


def add_database_options(func):
    """Add database-related options to commands."""
    func = click.option('--db', help='Database name (will auto-resolve path)')(func)
    func = click.option('--path', type=click.Path(), help='Storage path')(func)
    return func


def add_output_options(func):
    """Add output formatting options to commands."""
    func = click.option('--cli-format', is_flag=True, help='Output in CLI format (single-line delimited)')(func)
    func = click.option('--delimiter', default=';', help='Delimiter for CLI format (default: ";")')(func)
    return func


@click.group()
@click.version_option(version='0.1.0')
@click.pass_context
def cli(ctx):
    """SEM CLI - Semantic search engine with intelligent chunking and GPU acceleration.

    A modular, cross-platform semantic search engine that provides intelligent
    code chunking, GPU acceleration, and multiple storage backends.

    Shell completion can be enabled with:
        sem-cli completion --install
    """
    ctx.ensure_object(dict)


@cli.command()
@add_global_options
@add_database_options
@click.option('--model', help='Embedding model name')
def init(verbose, config, db, path, model):
    """Initialize a new semantic search database.

    Creates a new semantic search database with the specified configuration.
    The database will be configured with the chosen embedding model and storage path.

    Examples:
        sem-cli init --db my_database --model all-MiniLM-L6-v2
        sem-cli init --path ./custom_indexes --config config.json
    """
    setup_cli_logging(verbose)
    logger.info("Initializing database: %s", db or "default")

    # Import required modules
    from .sem_core import SEMDatabase
    from .sem_output import create_formatter, output_simple_error, output_simple_success
    from .sem_utils import create_quick_config, get_config_info, load_config, save_config

    # Create formatter for output
    class Args:
        def __init__(self):
            self.verbose = verbose
            self.config = config
            self.db = db or "default"
            self.path = path or "./indexes"
            self.model = model or "all-MiniLM-L6-v2"

    args = Args()

    # Apply auto-resolution if database name provided without explicit path/config
    if args.db and not args.config and not path:
        logger.debug("Attempting auto-resolution for database: %s", args.db)
        args = auto_resolve_command_args(args, "database")

    formatter = create_formatter(args)

    try:
        if args.config:
            # Use existing config file
            config_obj = load_config(args.config)
            logger.info("Using configuration: %s", args.config)
        else:
            # Create quick config
            config_obj = create_quick_config(
                embedding_model=args.model,
                storage_path=args.path,
                index_name=args.db
            )
            logger.info("Created quick configuration")
            logger.info("Model: %s", args.model)
            logger.info("Storage: %s", args.path)
            logger.info("Database: %s", args.db)

        # Create database
        _ = SEMDatabase(config=config_obj.to_dict())

        # Save config if not provided
        if not config:
            config_path = Path(args.path) / "config.json"
            save_config(config_obj, str(config_path))
            logger.info("Saved configuration to: %s", config_path)

        logger.info("Database initialized successfully")

        # Get config info for output
        info = get_config_info(config_obj)
        output_simple_success(
            "Database initialized successfully",
            formatter,
            {
                "database_name": args.db,
                "storage_path": args.path,
                "embedding_model": args.model,
                "config_info": info
            },
        )
        return 0

    except Exception as e:
        logger.error("Failed to initialize database: %s", e)
        output_simple_error("Failed to initialize database: %s" % e, formatter)
        return 1


@cli.command()
@add_global_options
@add_database_options
@click.option('--files', multiple=True, type=click.Path(exists=True), help='Files to add to the database')
@click.option('--text', multiple=True, help='Text content to add directly')
def add(verbose, config, db, path, files, text):
    """Add documents to the semantic search database.

    Add files or text content to the semantic search database. Files will be
    processed and chunked according to their type (code files use semantic chunking).

    Examples:
        sem-cli add --files document.txt code.py
        sem-cli add --text "Some important information"
        echo "Text from stdin" | sem-cli add
    """
    setup_cli_logging(verbose)
    logger.info("Adding documents to database")

    # Handle stdin input if no files or text provided
    stdin_text = None
    if not files and not text and not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            text = list(text) + [stdin_text]

    if not files and not text:
        click.echo("Error: No files or text provided. Use --files, --text, or pipe content to stdin.", err=True)
        sys.exit(1)

    # Import required modules
    from .sem_core import SEMDatabase
    from .sem_output import create_formatter, output_simple_error, output_simple_success
    from .sem_utils import load_config

    # Create formatter for output
    class Args:
        def __init__(self):
            self.verbose = verbose
            self.config = config
            self.db = db
            self.path = path
            self.files = list(files)
            self.text = list(text)

    args = Args()
    formatter = create_formatter(args)

    try:
        # Load configuration
        if config:
            config_obj = load_config(config)
        else:
            # Try to find config in storage path
            config_path = Path(path or "./indexes") / "config.json"
            if config_path.exists():
                config_obj = load_config(str(config_path))
            else:
                output_simple_error("No configuration found. Run 'sem-cli init' first.", formatter)
                return 1

        # Create database
        db_instance = SEMDatabase(config=config_obj.to_dict())

        # Collect documents
        documents = []
        document_ids = []

        if files:
            # Read from files
            for file_path in files:
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    logger.warning("File not found: %s", file_path)
                    continue
                try:
                    with open(file_path_obj, "r", encoding="utf-8") as f:
                        content = f.read()
                    documents.append(content)
                    document_ids.append(file_path_obj.stem)
                    logger.info("Loaded: %s", file_path)
                except Exception as e:
                    logger.error("Failed to read file %s: %s", file_path, e)

        if text:
            # Add text content
            for i, content in enumerate(text):
                documents.append(content)
                document_ids.append(f"text_{i}")
                logger.info("Added text content: %d characters", len(content))

        if not documents:
            output_simple_error("No documents to add", formatter)
            return 1

        # Add documents to database
        results = []
        for doc_content, doc_id in zip(documents, document_ids):
            try:
                result = db_instance.add_document(doc_content, doc_id)
                results.append({"id": doc_id, "status": "success", "result": result})
                logger.info("Added document: %s", doc_id)
            except Exception as e:
                results.append({"id": doc_id, "status": "error", "error": str(e)})
                logger.error("Failed to add document %s: %s", doc_id, e)

        # Output results
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = len(results) - success_count

        if error_count == 0:
            output_simple_success(
                f"Successfully added {success_count} documents",
                formatter,
                {"added_count": success_count, "results": results}
            )
            return 0
        else:
            output_simple_error(
                f"Added {success_count} documents, {error_count} failed",
                formatter,
                {"added_count": success_count, "error_count": error_count, "results": results}
            )
            return 1

    except Exception as e:
        logger.error("Failed to add documents: %s", e)
        output_simple_error("Failed to add documents: %s" % e, formatter)
        return 1


@cli.command()
@add_global_options
@add_database_options
@add_output_options
@click.argument('query')
@click.option('--top-k', default=5, help='Number of results to return')
@click.option('--threshold', type=float, help='Similarity threshold for results')
def search(verbose, config, db, path, cli_format, delimiter, query, top_k, threshold):
    """Search documents in the semantic database.

    Perform semantic search to find documents similar to the query.
    Results are ranked by semantic similarity.

    Examples:
        sem-cli search "machine learning algorithms"
        sem-cli search "python code" --top-k 10 --threshold 0.7
        sem-cli search "query" --cli-format --delimiter "|"
    """
    setup_cli_logging(verbose)
    logger.info("Searching for: %s", query)

    # Import required modules
    from .sem_core import SEMDatabase
    from .sem_output import create_formatter, output_search_results, output_simple_error
    from .sem_utils import create_quick_config, load_config

    # Create formatter for output
    class Args:
        def __init__(self):
            self.verbose = verbose
            self.config = config
            self.db = db
            self.path = path
            self.cli_format = cli_format
            self.delimiter = delimiter
            self.query = query
            self.top_k = top_k
            self.threshold = threshold

    args = Args()

    # Apply auto-resolution if database name provided without explicit path/config
    if args.db and not args.config and not args.path:
        logger.debug("Attempting auto-resolution for database: %s", args.db)
        args = auto_resolve_command_args(args, "document")

    formatter = create_formatter(args)

    try:
        # Load configuration
        if args.config:
            config_obj = load_config(args.config)
        elif args.path:
            # For simple databases, create config that points to specific database
            if args.db:
                config_obj = create_quick_config(storage_path=args.path, index_name=args.db)
            else:
                # Try to find config in storage path
                config_path = Path(path) / "config.json"
                if config_path.exists():
                    config_obj = load_config(str(config_path))
                else:
                    config_obj = create_quick_config(storage_path=path)
        else:
            # Try to find config in default storage path
            config_path = Path("./indexes") / "config.json"
            if config_path.exists():
                config_obj = load_config(str(config_path))
            else:
                output_simple_error(
                    "No configuration found. Run 'sem-cli init' first or use --db for auto-resolution.",
                    formatter,
                    {
                        "examples": [
                            "sem-cli search 'query' --db database_name",
                            "sem-cli init --db database_name",
                            "sem-cli search 'query' --config config.json"
                        ]
                    }
                )
                return 1

        # Create database
        db_instance = SEMDatabase(config=config_obj.to_dict())

        # Perform search
        search_results = db_instance.search(query, top_k=top_k, threshold=threshold)

        # Output results
        output_search_results(search_results, formatter)

        logger.info("Search completed: %d results", len(search_results))
        return 0

    except Exception as e:
        logger.error("Search failed: %s", e)
        output_simple_error("Search failed: %s" % e, formatter)
        return 1


@cli.command()
@add_global_options
@add_database_options
@add_output_options
@click.option('--limit', default=10, help='Maximum number of documents to list')
@click.option('--no-content', is_flag=True, help='List only metadata, not content')
def list(verbose, config, db, path, cli_format, delimiter, limit, no_content):
    """List documents in the semantic database.

    Display all documents stored in the database with their metadata.
    Use --no-content to show only document IDs and metadata.

    Examples:
        sem-cli list --limit 20
        sem-cli list --no-content
        sem-cli list --cli-format --delimiter "|"
    """
    setup_cli_logging(verbose)
    logger.info("Listing documents in database")

    # Import required modules
    from .sem_core import SEMDatabase
    from .sem_output import create_formatter, output_document_list, output_simple_error
    from .sem_utils import create_quick_config, load_config

    # Create formatter for output
    class Args:
        def __init__(self):
            self.verbose = verbose
            self.config = config
            self.db = db
            self.path = path
            self.cli_format = cli_format
            self.delimiter = delimiter
            self.limit = limit
            self.no_content = no_content

    args = Args()
    formatter = create_formatter(args)

    try:
        # Load configuration
        if config:
            config_obj = load_config(config)
        elif path:
            if db:
                config_obj = create_quick_config(storage_path=path, index_name=db)
            else:
                config_path = Path(path) / "config.json"
                if config_path.exists():
                    config_obj = load_config(str(config_path))
                else:
                    config_obj = create_quick_config(storage_path=path)
        else:
            config_path = Path("./indexes") / "config.json"
            if config_path.exists():
                config_obj = load_config(str(config_path))
            else:
                output_simple_error("No configuration found. Run 'sem-cli init' first.", formatter)
                return 1

        # Create database
        db_instance = SEMDatabase(config=config_obj.to_dict())

        # List documents
        documents = db_instance.list_documents(limit=limit, show_content=not no_content)

        # Output results
        output_document_list(documents, formatter)

        logger.info("Listed %d documents", len(documents))
        return 0

    except Exception as e:
        logger.error("Failed to list documents: %s", e)
        output_simple_error("Failed to list documents: %s" % e, formatter)
        return 1


@cli.command()
@add_global_options
@add_database_options
def info(verbose, config, db, path):
    """Display information about the semantic database.

    Show database statistics, configuration, and status information.

    Examples:
        sem-cli info
        sem-cli info --db my_database
    """
    setup_cli_logging(verbose)
    logger.info("Getting database info")

    # Import required modules
    from .sem_core import SEMDatabase
    from .sem_output import create_formatter, output_simple_success, output_simple_error
    from .sem_utils import create_quick_config, load_config, get_config_info

    # Create formatter for output
    class Args:
        def __init__(self):
            self.verbose = verbose
            self.config = config
            self.db = db
            self.path = path

    args = Args()
    formatter = create_formatter(args)

    try:
        # Load configuration
        if config:
            config_obj = load_config(config)
        elif path:
            if db:
                config_obj = create_quick_config(storage_path=path, index_name=db)
            else:
                config_path = Path(path) / "config.json"
                if config_path.exists():
                    config_obj = load_config(str(config_path))
                else:
                    config_obj = create_quick_config(storage_path=path)
        else:
            config_path = Path("./indexes") / "config.json"
            if config_path.exists():
                config_obj = load_config(str(config_path))
            else:
                output_simple_error("No configuration found. Run 'sem-cli init' first.", formatter)
                return 1

        # Create database and get info
        db_instance = SEMDatabase(config=config_obj.to_dict())

        # Get database statistics
        stats = db_instance.get_stats()
        config_info = get_config_info(config_obj)

        # Combine info
        info_data = {
            "database_stats": stats,
            "config_info": config_info,
            "database_name": db or "default",
            "storage_path": path or "./indexes"
        }

        output_simple_success("Database information", formatter, info_data)
        return 0

    except Exception as e:
        logger.error("Failed to get database info: %s", e)
        output_simple_error("Failed to get database info: %s" % e, formatter)
        return 1


@cli.command()
@add_global_options
@click.option('--output', type=click.Path(), help='Output configuration file path')
@click.option('--provider', help='Embedding provider to use')
@click.option('--model', help='Embedding model name')
@click.option('--storage', help='Storage backend to use')
@click.option('--path', type=click.Path(), help='Storage path')
def config(verbose, config, output, provider, model, storage, path):
    """Generate or display configuration.

    Create a configuration file with the specified settings or display
    the current configuration.

    Examples:
        sem-cli config --output config.json
        sem-cli config --provider sentence_transformers --model all-MiniLM-L6-v2
    """
    setup_cli_logging(verbose)
    logger.info("Managing configuration")

    # Import required modules
    from .sem_output import create_formatter, output_simple_success, output_simple_error
    from .sem_utils import create_quick_config, save_config, load_config, get_config_info

    # Create formatter for output
    class Args:
        def __init__(self):
            self.verbose = verbose
            self.config = config
            self.output = output
            self.provider = provider
            self.model = model
            self.storage = storage
            self.path = path

    args = Args()
    formatter = create_formatter(args)

    try:
        if output:
            # Generate new configuration file
            config_obj = create_quick_config(
                embedding_model=model or "all-MiniLM-L6-v2",
                storage_path=path or "./indexes",
                provider=provider or "sentence_transformers",
                storage_backend=storage or "local_disk"
            )

            save_config(config_obj, output)

            config_info = get_config_info(config_obj)
            output_simple_success(
                f"Configuration saved to {output}",
                formatter,
                {"config_path": output, "config_info": config_info}
            )
            return 0

        elif config:
            # Display existing configuration
            config_obj = load_config(config)
            config_info = get_config_info(config_obj)

            output_simple_success(
                f"Configuration from {config}",
                formatter,
                {"config_path": config, "config_info": config_info}
            )
            return 0

        else:
            # Show default configuration
            config_obj = create_quick_config(
                embedding_model=model or "all-MiniLM-L6-v2",
                storage_path=path or "./indexes",
                provider=provider or "sentence_transformers",
                storage_backend=storage or "local_disk"
            )

            config_info = get_config_info(config_obj)
            output_simple_success(
                "Default configuration",
                formatter,
                {"config_info": config_info}
            )
            return 0

    except Exception as e:
        logger.error("Failed to manage configuration: %s", e)
        output_simple_error("Failed to manage configuration: %s" % e, formatter)
        return 1


# Simple interface commands
@cli.group()
def simple():
    """Simple interface for semantic search operations.

    Provides a simplified interface for common semantic search operations
    with sensible defaults and minimal configuration required.
    """
    pass


@simple.group()
@click.pass_context
def local(ctx):
    """Local storage semantic search operations.

    Perform semantic search operations using local file storage.
    Uses sentence-transformers for embeddings by default.
    """
    ctx.ensure_object(dict)
    ctx.obj['backend'] = 'local'


@local.command('index')
@click.argument('files', nargs=-1)
@click.option('--db', default='sem_simple_database', help='Database name')
@click.option('--path', default='./sem_indexes', help='Storage path')
@click.option('--model', help='Embedding model name')
@click.option('--text', multiple=True, help='Text content to index directly')
def local_index(files, db, path, model, text):
    """Index files or text for local semantic search.

    Add files or text content to the local semantic search database.
    Content from stdin is also supported.

    Examples:
        sem-cli simple local index document.txt code.py
        echo "Important text" | sem-cli simple local index
        sem-cli simple local index --text "Direct text input"
    """
    # Validate file paths
    validated_files = []
    for file_path in files:
        path_obj = Path(file_path)
        if not path_obj.exists():
            logger.warning("File does not exist: %s", file_path)
            click.echo(f"Warning: File does not exist: {file_path}", err=True)
        elif path_obj.is_dir():
            logger.warning("Skipping directory: %s", file_path)
            click.echo(f"Warning: Skipping directory: {file_path}", err=True)
        else:
            validated_files.append(file_path)

    # Handle stdin input
    stdin_text = None
    if not validated_files and not text and not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            text = [item for item in text] + [stdin_text]  # Avoid list() name collision

    from .sem_simple_commands import cmd_simple_local

    class Args:
        def __init__(self):
            self.operation = 'add'  # Map 'index' to 'add' operation
            self.files = validated_files
            # Use list comprehension to avoid name collision with Click's list command
            self.text = [item for item in text]
            self.db = db
            self.path = path
            self.model = model
            # Set other required attributes with defaults
            self.query = None
            self.top_k = 5
            self.max_content = 100
            self.doc_id = None
            self.doc_ids = None
            self.confirm = False

    args = Args()

    # Always apply auto-resolution for simple commands (including default database)
    logger.debug("Attempting auto-resolution for database: %s", args.db)
    args = auto_resolve_command_args(args, "document", "local")

    try:
        return cmd_simple_local(args)
    except Exception as e:
        logger.error("Failed to index content: %s", e)
        click.echo("Error: Failed to index content: %s" % e, err=True)
        sys.exit(1)


@local.command('search')
@click.argument('query')
@click.option('--db', default='sem_simple_database', help='Database name')
@click.option('--path', default='./sem_indexes', help='Storage path')
@click.option('--top-k', default=5, help='Number of results to return')
@click.option('--max-content', default=100, help='Maximum content length to display')
def local_search(query, db, path, top_k, max_content):
    """Search the local semantic database.

    Perform semantic search on the local database and display results.

    Examples:
        sem-cli simple local search "machine learning"
        sem-cli simple local search "python code" --top-k 10
    """
    from .sem_simple_commands import cmd_simple_local

    class Args:
        def __init__(self):
            self.operation = 'search'
            self.query = query
            self.db = db
            self.path = path
            self.top_k = top_k
            self.max_content = max_content
            # Set other required attributes with defaults
            self.files = []
            self.text = []
            self.doc_id = None
            self.doc_ids = None
            self.confirm = False
            self.model = None

    args = Args()

    # Always apply auto-resolution for simple commands (including default database)
    logger.debug("Attempting auto-resolution for database: %s", args.db)
    args = auto_resolve_command_args(args, "document", "local")

    try:
        return cmd_simple_local(args)
    except Exception as e:
        logger.error("Search failed: %s", e)
        click.echo("Error: Search failed: %s" % e, err=True)
        sys.exit(1)


@local.command('list')
@click.option('--db', default='sem_simple_database', help='Database name')
@click.option('--path', default='./sem_indexes', help='Storage path')
@click.option('--top-k', default=10, help='Maximum number of documents to list')
@click.option('--max-content', default=100, help='Maximum content length to display')
def local_list(db, path, top_k, max_content):
    """List documents in the local semantic database.

    Display all documents stored in the local database.

    Examples:
        sem-cli simple local list
        sem-cli simple local list --top-k 20
    """
    from .sem_simple_commands import cmd_simple_local

    class Args:
        def __init__(self):
            self.operation = 'list'
            self.db = db
            self.path = path
            self.top_k = top_k
            self.max_content = max_content
            # Set other required attributes with defaults
            self.query = None
            self.files = []
            self.text = []
            self.doc_id = None
            self.doc_ids = None
            self.confirm = False
            self.model = None

    args = Args()

    # Always apply auto-resolution for simple commands (including default database)
    logger.debug("Attempting auto-resolution for database: %s", args.db)
    args = auto_resolve_command_args(args, "document", "local")

    try:
        return cmd_simple_local(args)
    except Exception as e:
        logger.error("Failed to list documents: %s", e)
        click.echo("Error: Failed to list documents: %s" % e, err=True)
        sys.exit(1)


@local.command('remove')
@click.option('--doc-id', help='Document ID to remove')
@click.option('--doc-ids', multiple=True, help='Multiple document IDs to remove')
@click.option('--confirm', is_flag=True, help='Confirm removal without prompting')
@click.option('--db', default='sem_simple_database', help='Database name')
@click.option('--path', default='./sem_indexes', help='Storage path')
def local_remove(doc_id, doc_ids, confirm, db, path):
    """Remove documents from the local semantic database.

    Remove specific documents by their IDs. Use --confirm to skip confirmation prompt.

    Examples:
        sem-cli simple local remove --doc-id doc_123
        sem-cli simple local remove --doc-ids doc_1 doc_2 --confirm
    """
    if not doc_id and not doc_ids:
        click.echo("Error: Must specify --doc-id or --doc-ids", err=True)
        sys.exit(1)

    from .sem_simple_commands import cmd_simple_local

    class Args:
        def __init__(self):
            self.operation = 'remove'
            self.doc_id = doc_id
            self.doc_ids = list(doc_ids) if doc_ids else None
            self.confirm = confirm
            self.db = db
            self.path = path
            # Set other required attributes with defaults
            self.query = None
            self.files = []
            self.text = []
            self.top_k = 5
            self.max_content = 100
            self.model = None

    args = Args()

    # Always apply auto-resolution for simple commands (including default database)
    logger.debug("Attempting auto-resolution for database: %s", args.db)
    args = auto_resolve_command_args(args, "document", "local")

    try:
        return cmd_simple_local(args)
    except Exception as e:
        logger.error("Failed to remove documents: %s", e)
        click.echo("Error: Failed to remove documents: %s" % e, err=True)
        sys.exit(1)


@local.command('clear')
@click.option('--confirm', is_flag=True, help='Confirm clearing without prompting')
@click.option('--db', default='sem_simple_database', help='Database name')
@click.option('--path', default='./sem_indexes', help='Storage path')
def local_clear(confirm, db, path):
    """Clear all documents from the local semantic database.

    Remove all documents from the database. Use --confirm to skip confirmation prompt.

    Examples:
        sem-cli simple local clear --confirm
    """
    from .sem_simple_commands import cmd_simple_local

    class Args:
        def __init__(self):
            self.operation = 'clear'
            self.confirm = confirm
            self.db = db
            self.path = path
            # Set other required attributes with defaults
            self.query = None
            self.files = []
            self.text = []
            self.doc_id = None
            self.doc_ids = None
            self.top_k = 5
            self.max_content = 100
            self.model = None

    args = Args()

    # Always apply auto-resolution for simple commands (including default database)
    logger.debug("Attempting auto-resolution for database: %s", args.db)
    args = auto_resolve_command_args(args, "document", "local")

    try:
        return cmd_simple_local(args)
    except Exception as e:
        logger.error("Failed to clear database: %s", e)
        click.echo("Error: Failed to clear database: %s" % e, err=True)
        sys.exit(1)


@cli.command('list-databases')
@add_global_options
def list_databases(verbose, config):
    """List all discoverable databases for auto-resolution.

    Shows all databases that can be auto-resolved by name, including their
    locations and types. Useful for understanding what databases are available
    for the --db parameter.

    Examples:
        sem-cli list-databases
        sem-cli list-databases --verbose
    """
    setup_cli_logging(verbose)
    logger.info("Discovering available databases")

    from .sem_output import create_formatter, output_simple_success, output_simple_error

    class Args:
        def __init__(self):
            self.verbose = verbose
            self.config = config

    args = Args()
    formatter = create_formatter(args)

    try:
        databases = list_available_databases()
        conflicts = list_all_conflicts()

        if not databases:
            output_simple_error("No databases found for auto-resolution", formatter)
            return 1

        # Prepare database info
        db_info = []
        for db_name, locations in databases.items():
            is_conflict = db_name in conflicts
            primary_location = locations[0] if locations else None

            db_entry = {
                "name": db_name,
                "locations": len(locations),
                "type": primary_location.get("type", "unknown") if primary_location else "unknown",
                "path": primary_location.get("path", "unknown") if primary_location else "unknown",
                "has_conflicts": is_conflict,
                "priority": primary_location.get("priority", 999) if primary_location else 999
            }

            if is_conflict:
                db_entry["conflict_details"] = [
                    {"path": loc["path"], "type": loc["type"], "priority": loc.get("priority", 999)}
                    for loc in locations
                ]

            db_info.append(db_entry)

        # Sort by name
        db_info.sort(key=lambda x: x["name"])

        output_simple_success(
            f"Found {len(databases)} discoverable databases",
            formatter,
            {
                "total_databases": len(databases),
                "conflicted_databases": len(conflicts),
                "databases": db_info,
                "usage_examples": [
                    "sem-cli simple local search 'query' --db database_name",
                    "sem-cli simple local list --db database_name",
                    "sem-cli search 'query' --db database_name"
                ]
            }
        )

        return 0

    except Exception as e:
        logger.error("Failed to list databases: %s", e)
        output_simple_error("Failed to list databases: %s" % e, formatter)
        return 1


@cli.command()
@add_global_options
@click.option('--install', is_flag=True, help='Install shell completion')
@click.option('--show', is_flag=True, help='Show completion script')
@click.option('--shell', help='Target shell (bash, zsh, fish)')
def completion(verbose, config, install, show, shell):
    """Manage shell completion for the CLI.

    Install or display shell completion scripts. Click provides automatic
    completion generation that works across bash, zsh, and fish shells.

    Examples:
        sem-cli completion --install
        sem-cli completion --show --shell bash
    """
    setup_cli_logging(verbose)

    import os

    if not shell:
        shell = os.environ.get('SHELL', '').split('/')[-1]

    if install:
        click.echo(f"🔧 Installing completion for {shell}...")
        click.echo("✅ Click-based completion will be automatically available")
        click.echo("   Use tab completion with any sem-cli command")
        click.echo("   All options and arguments are automatically completed")
    elif show:
        click.echo(f"# Completion script for {shell}")
        click.echo("# Click handles completion automatically - no manual script needed")
    else:
        click.echo("Use --install to enable completion or --show to display script")


# AWS Simple Interface
@simple.group()
@click.pass_context
def aws(ctx):
    """AWS S3 semantic search operations.

    Perform semantic search operations using AWS S3 for storage.
    Requires AWS credentials to be configured.
    """
    ctx.ensure_object(dict)
    ctx.obj['backend'] = 'aws'


@aws.command('index')
@click.argument('files', nargs=-1)
@click.option('--bucket', help='S3 bucket name (auto-generated if not specified)')
@click.option('--region', default='us-east-1', help='AWS region')
@click.option('--model', help='Embedding model name')
@click.option('--text', multiple=True, help='Text content to index directly')
def aws_index(files, bucket, region, model, text):
    """Index files or text to AWS S3 semantic search.

    Add files or text content to AWS S3-based semantic search database.

    Examples:
        sem-cli simple aws index document.txt --bucket my-bucket
        echo "Important text" | sem-cli simple aws index --bucket my-bucket
    """
    # Handle stdin input
    stdin_text = None
    if not files and not text and not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            text = list(text) + [stdin_text]

    from .sem_simple_commands import cmd_simple_aws

    class Args:
        def __init__(self):
            self.operation = 'add'  # Map 'index' to 'add' operation
            self.files = list(files)
            self.text = list(text)
            self.bucket = bucket
            self.region = region
            self.model = model
            # Set other required attributes with defaults
            self.query = None
            self.top_k = 5
            self.max_content = 100
            self.doc_id = None
            self.doc_ids = None
            self.confirm = False

    try:
        return cmd_simple_aws(Args())
    except Exception as e:
        logger.error("Failed to index to AWS: %s", e)
        click.echo("Error: Failed to index to AWS: %s" % e, err=True)
        sys.exit(1)


@aws.command('search')
@click.argument('query')
@click.option('--bucket', help='S3 bucket name')
@click.option('--region', default='us-east-1', help='AWS region')
@click.option('--top-k', default=5, help='Number of results to return')
@click.option('--max-content', default=100, help='Maximum content length to display')
def aws_search(query, bucket, region, top_k, max_content):
    """Search the AWS S3 semantic database.

    Perform semantic search on the AWS S3-based database.

    Examples:
        sem-cli simple aws search "machine learning" --bucket my-bucket
    """
    from .sem_simple_commands import cmd_simple_aws

    class Args:
        def __init__(self):
            self.operation = 'search'
            self.query = query
            self.bucket = bucket
            self.region = region
            self.top_k = top_k
            self.max_content = max_content
            # Set other required attributes with defaults
            self.files = []
            self.text = []
            self.doc_id = None
            self.doc_ids = None
            self.confirm = False
            self.model = None

    try:
        return cmd_simple_aws(Args())
    except Exception as e:
        logger.error("AWS search failed: %s", e)
        click.echo("Error: AWS search failed: %s" % e, err=True)
        sys.exit(1)


@aws.command('list')
@click.option('--bucket', help='S3 bucket name')
@click.option('--region', default='us-east-1', help='AWS region')
@click.option('--top-k', default=10, help='Maximum number of documents to list')
@click.option('--max-content', default=100, help='Maximum content length to display')
def aws_list(bucket, region, top_k, max_content):
    """List documents in the AWS S3 semantic database.

    Display all documents stored in the AWS S3 database.

    Examples:
        sem-cli simple aws list --bucket my-bucket
    """
    from .sem_simple_commands import cmd_simple_aws

    class Args:
        def __init__(self):
            self.operation = 'list'
            self.bucket = bucket
            self.region = region
            self.top_k = top_k
            self.max_content = max_content
            # Set other required attributes with defaults
            self.query = None
            self.files = []
            self.text = []
            self.doc_id = None
            self.doc_ids = None
            self.confirm = False
            self.model = None

    try:
        return cmd_simple_aws(Args())
    except Exception as e:
        logger.error("Failed to list AWS documents: %s", e)
        click.echo("Error: Failed to list AWS documents: %s" % e, err=True)
        sys.exit(1)


@aws.command('remove')
@click.option('--doc-id', help='Document ID to remove')
@click.option('--doc-ids', multiple=True, help='Multiple document IDs to remove')
@click.option('--confirm', is_flag=True, help='Confirm removal without prompting')
@click.option('--bucket', help='S3 bucket name')
@click.option('--region', default='us-east-1', help='AWS region')
def aws_remove(doc_id, doc_ids, confirm, bucket, region):
    """Remove documents from the AWS S3 semantic database.

    Remove specific documents by their IDs. Use --confirm to skip confirmation prompt.

    Examples:
        sem-cli simple aws remove --doc-id doc_123 --bucket my-bucket
        sem-cli simple aws remove --doc-ids doc_1 doc_2 --confirm --bucket my-bucket
    """
    if not doc_id and not doc_ids:
        click.echo("Error: Must specify --doc-id or --doc-ids", err=True)
        sys.exit(1)

    from .sem_simple_commands import cmd_simple_aws

    class Args:
        def __init__(self):
            self.operation = 'remove'
            self.doc_id = doc_id
            self.doc_ids = list(doc_ids) if doc_ids else None
            self.confirm = confirm
            self.bucket = bucket
            self.region = region
            # Set other required attributes with defaults
            self.query = None
            self.files = []
            self.text = []
            self.top_k = 5
            self.max_content = 100
            self.model = None

    try:
        return cmd_simple_aws(Args())
    except Exception as e:
        logger.error("Failed to remove AWS documents: %s", e)
        click.echo("Error: Failed to remove AWS documents: %s" % e, err=True)
        sys.exit(1)


@aws.command('clear')
@click.option('--confirm', is_flag=True, help='Confirm clearing without prompting')
@click.option('--bucket', help='S3 bucket name')
@click.option('--region', default='us-east-1', help='AWS region')
def aws_clear(confirm, bucket, region):
    """Clear all documents from the AWS S3 semantic database.

    Remove all documents from the database. Use --confirm to skip confirmation prompt.

    Examples:
        sem-cli simple aws clear --confirm --bucket my-bucket --region us-west-2
    """
    from .sem_simple_commands import cmd_simple_aws

    class Args:
        def __init__(self):
            self.operation = 'clear'
            self.confirm = confirm
            self.bucket = bucket
            self.region = region
            # Set other required attributes with defaults
            self.query = None
            self.files = []
            self.text = []
            self.doc_id = None
            self.doc_ids = None
            self.top_k = 5
            self.max_content = 100
            self.model = None

    try:
        return cmd_simple_aws(Args())
    except Exception as e:
        logger.error("Failed to clear AWS database: %s", e)
        click.echo("Error: Failed to clear AWS database: %s" % e, err=True)
        sys.exit(1)


# GCP Simple Interface
@simple.group()
@click.pass_context
def gcp(ctx):
    """Google Cloud Platform semantic search operations.

    Perform semantic search operations using Google Cloud Storage and Vertex AI.
    Requires Google Cloud credentials to be configured.
    """
    ctx.ensure_object(dict)
    ctx.obj['backend'] = 'gcp'


@gcp.command('index')
@click.argument('files', nargs=-1)
@click.option('--bucket', help='GCS bucket name (auto-generated if not specified)')
@click.option('--project', help='Google Cloud project ID (auto-detected if not specified)')
@click.option('--region', default='us-central1', help='GCP region')
@click.option('--model', default='textembedding-gecko@003', help='Vertex AI embedding model')
@click.option('--credentials', type=click.Path(exists=True), help='Path to service account JSON file')
@click.option('--text', multiple=True, help='Text content to index directly')
def gcp_index(files, bucket, project, region, model, credentials, text):
    """Index files or text to GCP semantic search.

    Add files or text content to GCP-based semantic search using Google Cloud Storage
    and Vertex AI embeddings.

    Examples:
        sem-cli simple gcp index document.txt --bucket my-bucket --project my-project
        echo "Important text" | sem-cli simple gcp index --bucket my-bucket
    """
    # Handle stdin input
    stdin_text = None
    if not files and not text and not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            text = list(text) + [stdin_text]

    from .sem_commands_gcp import cmd_simple_gcp

    class Args:
        def __init__(self):
            self.operation = 'add'  # Map 'index' to 'add' operation
            self.files = list(files)
            self.text = list(text)
            self.bucket = bucket
            self.project = project
            self.region = region
            self.model = model
            self.credentials = credentials
            # Set other required attributes with defaults
            self.query = None
            self.top_k = 5
            self.max_content = 100
            self.doc_id = None
            self.doc_ids = None
            self.confirm = False
            self.index = 'sem_simple_gcp'

    try:
        return cmd_simple_gcp(Args())
    except Exception as e:
        logger.error("Failed to index to GCP: %s", e)
        click.echo("Error: Failed to index to GCP: %s" % e, err=True)
        sys.exit(1)


@gcp.command('search')
@click.argument('query')
@click.option('--bucket', help='GCS bucket name')
@click.option('--project', help='Google Cloud project ID')
@click.option('--region', default='us-central1', help='GCP region')
@click.option('--model', default='textembedding-gecko@003', help='Vertex AI embedding model')
@click.option('--credentials', type=click.Path(exists=True), help='Path to service account JSON file')
@click.option('--top-k', default=5, help='Number of results to return')
@click.option('--max-content', default=100, help='Maximum content length to display')
def gcp_search(query, bucket, project, region, model, credentials, top_k, max_content):
    """Search the GCP semantic database.

    Perform semantic search on the GCP-based database using Vertex AI.

    Examples:
        sem-cli simple gcp search "machine learning" --bucket my-bucket --project my-project
    """
    from .sem_commands_gcp import cmd_simple_gcp

    class Args:
        def __init__(self):
            self.operation = 'search'
            self.query = query
            self.bucket = bucket
            self.project = project
            self.region = region
            self.model = model
            self.credentials = credentials
            self.top_k = top_k
            self.max_content = max_content
            # Set other required attributes with defaults
            self.files = []
            self.text = []
            self.doc_id = None
            self.doc_ids = None
            self.confirm = False
            self.index = 'sem_simple_gcp'

    try:
        return cmd_simple_gcp(Args())
    except Exception as e:
        logger.error("GCP search failed: %s", e)
        click.echo("Error: GCP search failed: %s" % e, err=True)
        sys.exit(1)


@gcp.command('list')
@click.option('--bucket', help='GCS bucket name')
@click.option('--project', help='Google Cloud project ID')
@click.option('--region', default='us-central1', help='GCP region')
@click.option('--credentials', type=click.Path(exists=True), help='Path to service account JSON file')
@click.option('--top-k', default=10, help='Maximum number of documents to list')
@click.option('--max-content', default=100, help='Maximum content length to display')
def gcp_list(bucket, project, region, credentials, top_k, max_content):
    """List documents in the GCP semantic database.

    Display all documents stored in the GCP database.

    Examples:
        sem-cli simple gcp list --bucket my-bucket --project my-project
    """
    from .sem_commands_gcp import cmd_simple_gcp

    class Args:
        def __init__(self):
            self.operation = 'list'
            self.bucket = bucket
            self.project = project
            self.region = region
            self.credentials = credentials
            self.top_k = top_k
            self.max_content = max_content
            # Set other required attributes with defaults
            self.query = None
            self.files = []
            self.text = []
            self.doc_id = None
            self.doc_ids = None
            self.confirm = False
            self.model = 'textembedding-gecko@003'
            self.index = 'sem_simple_gcp'

    try:
        return cmd_simple_gcp(Args())
    except Exception as e:
        logger.error("Failed to list GCP documents: %s", e)
        click.echo("Error: Failed to list GCP documents: %s" % e, err=True)
        sys.exit(1)


@gcp.command('remove')
@click.option('--doc-id', help='Document ID to remove')
@click.option('--doc-ids', multiple=True, help='Multiple document IDs to remove')
@click.option('--confirm', is_flag=True, help='Confirm removal without prompting')
@click.option('--bucket', help='GCS bucket name')
@click.option('--project', help='Google Cloud project ID')
@click.option('--region', default='us-central1', help='GCP region')
@click.option('--credentials', type=click.Path(exists=True), help='Path to service account JSON file')
def gcp_remove(doc_id, doc_ids, confirm, bucket, project, region, credentials):
    """Remove documents from the GCP semantic database.

    Remove specific documents by their IDs. Use --confirm to skip confirmation prompt.

    Examples:
        sem-cli simple gcp remove --doc-id doc_123 --bucket my-bucket
        sem-cli simple gcp remove --doc-ids doc_1 doc_2 --confirm --bucket my-bucket
    """
    if not doc_id and not doc_ids:
        click.echo("Error: Must specify --doc-id or --doc-ids", err=True)
        sys.exit(1)

    from .sem_commands_gcp import cmd_simple_gcp

    class Args:
        def __init__(self):
            self.operation = 'remove'
            self.doc_id = doc_id
            self.doc_ids = list(doc_ids) if doc_ids else None
            self.confirm = confirm
            self.bucket = bucket
            self.project = project
            self.region = region
            self.credentials = credentials
            # Set other required attributes with defaults
            self.query = None
            self.files = []
            self.text = []
            self.top_k = 5
            self.max_content = 100
            self.model = 'textembedding-gecko@003'
            self.index = 'sem_simple_gcp'

    try:
        return cmd_simple_gcp(Args())
    except Exception as e:
        logger.error("Failed to remove GCP documents: %s", e)
        click.echo("Error: Failed to remove GCP documents: %s" % e, err=True)
        sys.exit(1)


@gcp.command('clear')
@click.option('--confirm', is_flag=True, help='Confirm clearing without prompting')
@click.option('--bucket', help='GCS bucket name')
@click.option('--project', help='Google Cloud project ID')
@click.option('--region', default='us-central1', help='GCP region')
@click.option('--credentials', type=click.Path(exists=True), help='Path to service account JSON file')
def gcp_clear(confirm, bucket, project, region, credentials):
    """Clear all documents from the GCP semantic database.

    Remove all documents from the database. Use --confirm to skip confirmation prompt.

    Examples:
        sem-cli simple gcp clear --confirm --bucket my-bucket --project my-project
    """
    from .sem_commands_gcp import cmd_simple_gcp

    class Args:
        def __init__(self):
            self.operation = 'clear'
            self.confirm = confirm
            self.bucket = bucket
            self.project = project
            self.region = region
            self.credentials = credentials
            # Set other required attributes with defaults
            self.query = None
            self.files = []
            self.text = []
            self.doc_id = None
            self.doc_ids = None
            self.top_k = 5
            self.max_content = 100
            self.model = 'textembedding-gecko@003'
            self.index = 'sem_simple_gcp'

    try:
        return cmd_simple_gcp(Args())
    except Exception as e:
        logger.error("Failed to clear GCP database: %s", e)
        click.echo("Error: Failed to clear GCP database: %s" % e, err=True)
        sys.exit(1)


@cli.command('serve')
@add_global_options
@click.option('--host', default='127.0.0.1', help='Host to bind the server to (default: 127.0.0.1)')
@click.option('--port', default=8000, type=int, help='Port to bind the server to (default: 8000)')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.option('--workers', default=1, type=int, help='Number of worker processes (default: 1)')
def serve(verbose, config, host, port, reload, workers):
    """Start the SEM Web API server.

    Launch a FastAPI-based web server that provides REST API access to all
    SEM functionality. The API mirrors the CLI structure with "/" delimited paths.

    Examples:
        sem-cli serve                           # Start server on localhost:8000
        sem-cli serve --host 0.0.0.0 --port 8080  # Custom host and port
        sem-cli serve --reload                  # Development mode with auto-reload
        sem-cli serve --workers 4              # Production with multiple workers

    API Endpoints:
        POST /simple/local/search              # CLI: sem-cli simple local search
        POST /simple/aws/index                 # CLI: sem-cli simple aws index
        GET  /docs                             # Interactive API documentation
        GET  /health                           # Health check endpoint

    The server provides:
        - OpenAPI/Swagger documentation at /docs
        - CORS support for web frontends
        - JSON request/response format
        - Authentication support (configurable)
        - Error handling and validation
    """
    setup_cli_logging(verbose)

    try:
        from .sem_web_api import run_server

        logger.info("Starting SEM Web API server...")
        logger.info("Host: %s, Port: %s, Reload: %s, Workers: %s", host, port, reload, workers)

        if reload and workers > 1:
            logger.warning("Auto-reload is not compatible with multiple workers. Using single worker.")
            workers = 1

        # Start the server
        run_server(host=host, port=port, reload=reload)

    except ImportError as e:
        logger.error("Failed to import web API dependencies: %s", e)
        click.echo("Error: Web API dependencies not installed.", err=True)
        click.echo("Install with: pip install 'simple-embeddings-module[web]'", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to start web server: %s", e)
        click.echo("Error: Failed to start web server: %s" % e, err=True)
        sys.exit(1)


# Main entry point for Click CLI
def main():
    """Main entry point for the Click-based CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user.", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        click.echo("Unexpected error: %s" % e, err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
