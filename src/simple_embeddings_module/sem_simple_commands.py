#!/usr/bin/env python3
"""
SEM Simple Interface Commands - Decorator-based Implementation
Converts the simple interface commands to use the decorator system.
Much cleaner and more maintainable than the original approach.
"""
import logging
import sys
from pathlib import Path

from .sem_decorators_stub import sem_simple_command

logger = logging.getLogger(__name__)


@sem_simple_command(
    backend="local",
    description="Local backend operations with sentence-transformers",
    params={
        "operation": {
            "type": "str",
            "nargs": "?",
            "choices": ["add", "search", "list", "remove", "clear", "delete", "info", "update"],
            "help": "Operation to perform",
        },
        "query": {"type": "str", "nargs": "?", "help": "Search query (for search operation)"},
        "files": {"nargs": "*", "help": "Files to add"},
        "--text": {"action": "append", "help": "Text content to add (can be used multiple times)"},
        "--top-k": {"type": "int", "default": 5, "help": "Number of results (default: 5)"},
        "--max-content": {"type": "int", "default": 100, "help": "Maximum content length to display"},
        "--doc-id": {"type": "str", "help": "Document ID to remove/update"},
        "--doc-ids": {"nargs": "+", "help": "Multiple document IDs to remove"},
        "--confirm": {"action": "store_true", "help": "Confirm destructive operations"},
        "--db": {"type": "str", "help": "Database name (default: sem_simple_database)"},
        "--path": {"type": "str", "help": "Storage path (default: ./sem_indexes)"},
        "--model": {"type": "str", "help": "Embedding model (default: auto-detect)"},
    },
    examples=[
        "sem-cli simple local search 'machine learning'",
        "sem-cli simple local list --top-k 10",
        "echo 'text' | sem-cli simple local add",
        "sem-cli simple local add file1.txt file2.txt",
        "sem-cli simple local remove --doc-id doc_123",
        "sem-cli simple local clear --confirm",
    ],
)
def cmd_simple_local(args):
    """Handle local simple operations with enhanced modularity."""
    from .sem_output import (
        create_formatter,
        output_document_list,
        output_search_results,
        output_simple_error,
        output_simple_success,
    )
    from .sem_simple import SEMSimple
    formatter = getattr(args, "_formatter", None)
    if not formatter:
        formatter = create_formatter(args)
    try:
        # Create simple instance
        sem = SEMSimple(
            index_name=getattr(args, "db", None) or "sem_simple_database",
            storage_path=getattr(args, "path", None) or "./sem_indexes",
            embedding_model=getattr(args, "model", None),
        )
        operation = getattr(args, "operation", None)
        logger.info("cmd_simple_local called with operation: %s", operation)
        if not operation:
            # Initialize empty index
            try:
                docs = sem.list_documents(limit=1)
                doc_count = len(sem.list_documents()) if docs else 0
                output_simple_success(
                    f"Index ready with {doc_count} documents" if doc_count > 0 else "Empty index initialized",
                    formatter,
                    {"location": str(sem.storage_path), "index_name": sem.index_name, "document_count": doc_count},
                )
            except Exception:
                output_simple_success(
                    "Empty index initialized",
                    formatter,
                    {"location": str(sem.storage_path), "index_name": sem.index_name, "document_count": 0},
                )
            return 0
        elif operation == "search":
            if not getattr(args, "query", None):
                output_simple_error(
                    "Search requires a query", formatter, {"example": "sem-cli simple local search 'your query'"}
                )
                return 1
            results = sem.search(args.query, top_k=getattr(args, "top_k", 5))
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "id": result.get("id", "unknown"),
                        "score": result.get("score", 0),
                        "text": result.get("text", ""),
                        "metadata": result.get("metadata", {}),
                    }
                )
            output_search_results(formatted_results, formatter, args.query)
            return 0
        elif operation == "list":
            limit = getattr(args, "top_k", None) if getattr(args, "top_k", 5) != 5 else None
            documents = sem.list_documents(
                limit=limit, show_content=True, max_content_length=getattr(args, "max_content", 100)
            )
            formatted_docs = []
            for doc in documents:
                formatted_docs.append(
                    {
                        "id": doc.get("id", "unknown"),
                        "created_at": doc.get("created_at", "unknown"),
                        "text": doc.get("text", "No content available"),
                        "metadata": doc.get("metadata", {}),
                    }
                )
            output_document_list(formatted_docs, formatter, sem.index_name)
            return 0
        elif operation == "add":
            # Handle add operation
            content_added = 0
            # Add from stdin
            if not sys.stdin.isatty():
                stdin_content = sys.stdin.read().strip()
                if stdin_content:
                    sem.add_text(stdin_content)
                    content_added += 1
                    print("   ✅ Added from stdin", file=sys.stderr)
            # Add from --text arguments
            if getattr(args, "text", None):
                for i, text in enumerate(args.text):
                    sem.add_text(text)
                    content_added += 1
                    print(f"   ✅ Added text {i + 1}", file=sys.stderr)
            # Add from files
            if getattr(args, "files", None):
                for file_path in args.files:
                    try:
                        path = Path(file_path)
                        if not path.exists():
                            logger.warning("File not found: %s", file_path)
                        elif path.is_dir():
                            logger.warning("Skipping directory: %s", file_path)
                            print(f"   ⚠️  Skipped directory: {path.name}", file=sys.stderr)
                        else:
                            content = path.read_text(encoding="utf-8")
                            sem.add_text(content, doc_id=path.name)
                            content_added += 1
                            print(f"   ✅ Added: {path.name}", file=sys.stderr)
                    except Exception as e:
                        logger.error("Error adding file %s: %s", file_path, e)
            if content_added > 0:
                output_simple_success(
                    f"Added {content_added} document(s)",
                    formatter,
                    {"documents_added": content_added, "database": sem.index_name},
                )
            else:
                output_simple_error(
                    "No content to add",
                    formatter,
                    {
                        "suggestions": [
                            "echo 'text' | sem-cli simple local add",
                            "sem-cli simple local add --text 'content'",
                            "sem-cli simple local add file1.txt file2.txt",
                        ]
                    },
                )
                return 1
            return 0
        elif operation == "info":
            # Show index information
            info_data = sem.info()
            doc_count = sem.count()
            enhanced_info = {
                "index_name": sem.index_name,
                "storage_path": str(sem.storage_path),
                "document_count": doc_count,
                "backend": "local",
                **info_data,
            }
            from .sem_output import output_info
            output_info(enhanced_info, formatter, "Local Index")
            return 0
        elif operation == "remove":
            # Remove documents operation
            if not getattr(args, "doc_id", None) and not getattr(args, "doc_ids", None):
                output_simple_error(
                    "Remove operation requires --doc-id or --doc-ids",
                    formatter,
                    {
                        "examples": [
                            "sem-cli simple local remove --doc-id doc_abc123",
                            "sem-cli simple local remove --doc-ids doc_1 doc_2 doc_3",
                        ]
                    },
                )
                return 1
            removed_count = 0
            if getattr(args, "doc_id", None):
                success = sem.remove_document(args.doc_id)
                if success:
                    removed_count = 1
                    print(f"   ✅ Removed document: {args.doc_id}", file=sys.stderr)
                else:
                    print(f"   ❌ Document not found: {args.doc_id}", file=sys.stderr)
            if getattr(args, "doc_ids", None):
                for doc_id in args.doc_ids:
                    success = sem.remove_document(doc_id)
                    if success:
                        removed_count += 1
                        print(f"   ✅ Removed document: {doc_id}", file=sys.stderr)
                    else:
                        print(f"   ❌ Document not found: {doc_id}", file=sys.stderr)
            output_simple_success(
                f"Removed {removed_count} document(s)",
                formatter,
                {"documents_removed": removed_count, "database": sem.index_name},
            )
            return 0
        elif operation == "clear":
            # Clear all documents operation
            if not getattr(args, "confirm", False):
                output_simple_error(
                    "Clear operation requires --confirm flag",
                    formatter,
                    {"example": "sem-cli simple local clear --confirm"},
                )
                return 1
            success = sem.clear()
            if success:
                output_simple_success("Cleared all documents from index", formatter, {"database": sem.index_name})
            else:
                output_simple_error("Failed to clear index", formatter)
                return 1
            return 0
        elif operation == "delete":
            # Delete entire index operation
            if not getattr(args, "confirm", False):
                output_simple_error(
                    "Delete operation requires --confirm flag",
                    formatter,
                    {"example": "sem-cli simple local delete --confirm"},
                )
                return 1
            success = sem.delete_index()
            if success:
                output_simple_success("Deleted entire index", formatter, {"database": sem.index_name})
            else:
                output_simple_error("Failed to delete index", formatter)
                return 1
            return 0
        elif operation == "update":
            # Update document operation
            if not getattr(args, "doc_id", None):
                output_simple_error(
                    "Update operation requires --doc-id",
                    formatter,
                    {
                        "examples": [
                            "sem-cli simple local update --doc-id doc_abc123 --text 'New content'",
                            "sem-cli simple local update --doc-id doc_abc123 --files ./updated_file.txt",
                        ]
                    },
                )
                return 1
            if getattr(args, "text", None):
                new_text = " ".join(args.text)
                success = sem.update_document(args.doc_id, new_text)
                if success:
                    output_simple_success("Updated document: %s" % args.doc_id, formatter)
                else:
                    output_simple_error("Failed to update document: %s" % args.doc_id, formatter)
                    return 1
            elif getattr(args, "files", None):
                if len(args.files) != 1:
                    output_simple_error("Update operation accepts only one file", formatter)
                    return 1
                try:
                    file_path = Path(args.files[0])
                    if not file_path.exists():
                        output_simple_error("File not found: %s" % file_path, formatter)
                        return 1
                    new_text = file_path.read_text(encoding="utf-8")
                    success = sem.update_document(args.doc_id, new_text)
                    if success:
                        output_simple_success(
                            "Updated document %s with content from %s" % (args.doc_id, file_path), formatter
                        )
                    else:
                        output_simple_error("Failed to update document: %s" % args.doc_id, formatter)
                        return 1
                except Exception as e:
                    output_simple_error("Failed to read file %s: %s" % (args.files[0], e), formatter)
                    return 1
            else:
                output_simple_error("Update operation requires --text or --files", formatter)
                return 1
            return 0
        else:
            output_simple_error(
                "Unknown operation: %s" % operation,
                formatter,
                {"available_operations": ["add", "search", "list", "remove", "clear", "delete", "info", "update"]},
            )
            return 1
    except Exception as e:
        output_simple_error("Simple local command failed: %s" % e, formatter)
        return 1
@sem_simple_command(
    backend="aws",
    description="AWS backend operations with S3 + Bedrock",
    params={
        "operation": {
            "type": "str",
            "nargs": "?",
            "choices": ["add", "search", "list", "info"],
            "help": "Operation to perform",
        },
        "query": {"type": "str", "nargs": "?", "help": "Search query (for search operation)"},
        "files": {"nargs": "*", "help": "Files to add"},
        "--text": {"action": "append", "help": "Text content to add (can be used multiple times)"},
        "--top-k": {"type": "int", "default": 5, "help": "Number of results (default: 5)"},
        "--max-content": {"type": "int", "default": 100, "help": "Maximum content length to display"},
        "--bucket": {"type": "str", "help": "S3 bucket name (auto-generated if not specified)"},
        "--region": {"type": "str", "help": "AWS region (default: us-east-1)"},
        "--model": {"type": "str", "help": "Bedrock embedding model (default: amazon.titan-embed-text-v2:0)"},
    },
    examples=[
        "sem-cli simple aws search 'deployment' --bucket my-bucket",
        "sem-cli simple aws list --bucket my-bucket --top-k 10",
        "echo 'cloud doc' | sem-cli simple aws add --bucket my-bucket",
        "sem-cli simple aws add file1.txt --bucket my-bucket",
    ],
)
def cmd_simple_aws(args):
    """Handle AWS simple operations with S3 + Bedrock."""
    from .sem_output import (
        create_formatter,
        output_document_list,
        output_search_results,
        output_simple_error,
        output_simple_success,
    )
    formatter = getattr(args, "_formatter", None)
    if not formatter:
        formatter = create_formatter(args)
    try:
        from .sem_simple_aws import simple_aws
    except ImportError:
        output_simple_error("AWS dependencies not available. Install with: pip install boto3", formatter)
        return 1
    try:
        # Create AWS simple instance
        kwargs = {}
        if getattr(args, "bucket", None):
            kwargs["bucket_name"] = args.bucket
        if getattr(args, "region", None):
            kwargs["region"] = args.region
        if getattr(args, "model", None):
            kwargs["embedding_model"] = args.model
        sem = simple_aws(**kwargs)
        operation = getattr(args, "operation", None)
        if not operation:
            # Initialize empty index
            try:
                docs = sem.list_documents(limit=1)
                doc_count = len(sem.list_documents()) if docs else 0
                output_simple_success(
                    f"AWS index ready with {doc_count} documents" if doc_count > 0 else "Empty AWS index initialized",
                    formatter,
                    {"bucket_name": sem.bucket_name, "region": sem.region, "document_count": doc_count},
                )
            except Exception:
                output_simple_success(
                    "Empty AWS index initialized",
                    formatter,
                    {"bucket_name": sem.bucket_name, "region": sem.region, "document_count": 0},
                )
            return 0
        elif operation == "search":
            if not getattr(args, "query", None):
                output_simple_error(
                    "Search requires a query",
                    formatter,
                    {"example": "sem-cli simple aws search 'your query' --bucket my-bucket"},
                )
                return 1
            results = sem.search(args.query, top_k=getattr(args, "top_k", 5))
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "id": result.get("id", "unknown"),
                        "score": result.get("score", 0),
                        "text": result.get("text", ""),
                        "metadata": result.get("metadata", {}),
                    }
                )
            output_search_results(formatted_results, formatter, args.query)
            return 0
        elif operation == "list":
            limit = getattr(args, "top_k", None) if getattr(args, "top_k", 5) != 5 else None
            documents = sem.list_documents(
                limit=limit, show_content=True, max_content_length=getattr(args, "max_content", 100)
            )
            formatted_docs = []
            for doc in documents:
                formatted_docs.append(
                    {
                        "id": doc.get("id", "unknown"),
                        "created_at": doc.get("created_at", "unknown"),
                        "text": doc.get("text", "No content available"),
                        "metadata": doc.get("metadata", {}),
                    }
                )
            output_document_list(formatted_docs, formatter, f"{sem.bucket_name} (AWS)")
            return 0
        elif operation == "add":
            # Handle add operation
            content_added = 0
            # Add from stdin
            if not sys.stdin.isatty():
                stdin_content = sys.stdin.read().strip()
                if stdin_content:
                    doc_id = sem.add_text(stdin_content)
                    content_added += 1
                    print(f"   ✅ Added from stdin as {doc_id}", file=sys.stderr)
            # Add from --text arguments
            if getattr(args, "text", None):
                for i, text in enumerate(args.text):
                    doc_id = sem.add_text(text)
                    content_added += 1
                    print(f"   ✅ Added text {i + 1} as {doc_id}", file=sys.stderr)
            # Add from files
            if getattr(args, "files", None):
                for file_path in args.files:
                    try:
                        path = Path(file_path)
                        if not path.exists():
                            logger.warning("File not found: %s", file_path)
                        elif path.is_dir():
                            logger.warning("Skipping directory: %s", file_path)
                            print(f"   ⚠️  Skipped directory: {path.name}", file=sys.stderr)
                        else:
                            content = path.read_text(encoding="utf-8")
                            doc_id = sem.add_text(content, doc_id=path.name)
                            content_added += 1
                            print(f"   ✅ Added: {path.name} as {doc_id}", file=sys.stderr)
                    except Exception as e:
                        logger.error("Error adding file %s: %s", file_path, e)
            if content_added > 0:
                output_simple_success(
                    f"Added {content_added} document(s) to AWS",
                    formatter,
                    {"documents_added": content_added, "bucket_name": sem.bucket_name},
                )
            else:
                output_simple_error(
                    "No content to add",
                    formatter,
                    {
                        "suggestions": [
                            "echo 'text' | sem-cli simple aws add --bucket my-bucket",
                            "sem-cli simple aws add --text 'content' --bucket my-bucket",
                            "sem-cli simple aws add file1.txt file2.txt --bucket my-bucket",
                        ]
                    },
                )
                return 1
            return 0
        elif operation == "info":
            # Show AWS index information
            info_data = sem.info()
            doc_count = sem.count()
            enhanced_info = {
                "bucket_name": sem.bucket_name,
                "region": sem.region,
                "document_count": doc_count,
                "backend": "aws",
                **info_data,
            }
            from .sem_output import output_info
            output_info(enhanced_info, formatter, "AWS Index")
            return 0
        else:
            output_simple_error(
                "Unknown operation: %s" % operation,
                formatter,
                {"available_operations": ["add", "search", "list", "info"]},
            )
            return 1
    except Exception as e:
        output_simple_error(
            "AWS command failed: %s" % e,
            formatter,
            {
                "troubleshooting": [
                    "Ensure AWS credentials are configured",
                    "Verify S3 bucket permissions",
                    "Check that Bedrock API is enabled",
                ]
            },
        )
        return 1
