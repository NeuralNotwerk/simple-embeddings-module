#!/usr/bin/env python3
"""
SEM GCP Commands - Implementation Functions
Pure implementation functions for GCP operations, called by Click CLI.
No decorators - just the implementation logic.
"""
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_simple_gcp(args):
    """Handle GCP simple operations with Vertex AI + GCS."""
    from .sem_output import (
        create_formatter,
        output_document_list,
        output_search_results,
        output_simple_error,
        output_simple_success,
    )
    from .sem_simple_gcp import SEMSimpleGCP

    formatter = getattr(args, "_formatter", None)
    if not formatter:
        formatter = create_formatter(args)

    try:
        # Create GCP simple instance
        sem = SEMSimpleGCP(
            bucket_name=getattr(args, "bucket", None),
            project_id=getattr(args, "project", None),
            region=getattr(args, "region", "us-central1"),
            embedding_model=getattr(args, "model", "textembedding-gecko@003"),
            index_name=getattr(args, "index", None) or "sem_simple_gcp",
            credentials_path=getattr(args, "credentials", None),
        )

        operation = getattr(args, "operation", None)

        if not operation:
            # Initialize empty index
            try:
                docs = sem.list_documents(limit=1)
                doc_count = sem.count()
                output_simple_success(
                    f"GCP index ready with {doc_count} documents" if doc_count > 0 else "Empty GCP index initialized",
                    formatter,
                    {
                        "bucket_name": sem.bucket_name,
                        "project_id": sem.project_id,
                        "region": sem.region,
                        "index_name": sem.index_name,
                        "document_count": doc_count,
                        "embedding_model": sem.embedding_model,
                    },
                )
            except Exception:
                output_simple_success(
                    "Empty GCP index initialized",
                    formatter,
                    {
                        "bucket_name": sem.bucket_name,
                        "project_id": sem.project_id,
                        "region": sem.region,
                        "index_name": sem.index_name,
                        "document_count": 0,
                        "embedding_model": sem.embedding_model,
                    },
                )
            return 0

        # Handle different operations
        if operation == "add":
            # Collect documents to add
            documents = []
            document_ids = []

            # Handle files
            if hasattr(args, "files") and args.files:
                for file_path in args.files:
                    path = Path(file_path)
                    if not path.exists():
                        logger.warning("File not found: %s", file_path)
                        continue
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                        documents.append(content)
                        document_ids.append(path.stem)
                        logger.info("Loaded: %s", file_path)
                    except Exception as e:
                        logger.error("Failed to read file %s: %s", file_path, e)

            # Handle text content
            if hasattr(args, "text") and args.text:
                for i, content in enumerate(args.text):
                    documents.append(content)
                    document_ids.append(f"text_{i}")
                    logger.info("Added text content: %d characters", len(content))

            # Handle stdin
            if not documents and not sys.stdin.isatty():
                stdin_content = sys.stdin.read().strip()
                if stdin_content:
                    documents.append(stdin_content)
                    document_ids.append("stdin")
                    logger.info("Added stdin content: %d characters", len(stdin_content))

            if not documents:
                output_simple_error("No documents to add", formatter)
                return 1

            # Add documents
            results = []
            for doc_content, doc_id in zip(documents, document_ids):
                try:
                    result = sem.add_text(doc_content, doc_id)
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

        elif operation == "search":
            query = getattr(args, "query", None)
            if not query:
                output_simple_error("Search query is required", formatter)
                return 1

            top_k = getattr(args, "top_k", 5)
            results = sem.search(query, top_k=top_k)

            # Limit content display if requested
            max_content = getattr(args, "max_content", 100)
            if max_content > 0:
                for result in results:
                    if len(result.get("text", "")) > max_content:
                        result["text"] = result["text"][:max_content] + "..."

            output_search_results(results, formatter)
            return 0

        elif operation == "list":
            top_k = getattr(args, "top_k", 10)
            documents = sem.list_documents(limit=top_k)

            # Limit content display if requested
            max_content = getattr(args, "max_content", 100)
            if max_content > 0:
                for doc in documents:
                    if len(doc.get("text", "")) > max_content:
                        doc["text"] = doc["text"][:max_content] + "..."

            output_document_list(documents, formatter)
            return 0

        elif operation == "info":
            doc_count = sem.count()
            output_simple_success(
                f"GCP index information",
                formatter,
                {
                    "bucket_name": sem.bucket_name,
                    "project_id": sem.project_id,
                    "region": sem.region,
                    "index_name": sem.index_name,
                    "document_count": doc_count,
                    "embedding_model": sem.embedding_model,
                },
            )
            return 0

        elif operation in ["remove", "delete"]:
            doc_id = getattr(args, "doc_id", None)
            doc_ids = getattr(args, "doc_ids", None)

            if not doc_id and not doc_ids:
                output_simple_error("Document ID(s) required for removal", formatter)
                return 1

            ids_to_remove = []
            if doc_id:
                ids_to_remove.append(doc_id)
            if doc_ids:
                ids_to_remove.extend(doc_ids)

            # Confirm removal unless --confirm flag is set
            confirm = getattr(args, "confirm", False)
            if not confirm:
                response = input(f"Remove {len(ids_to_remove)} document(s)? [y/N]: ")
                if response.lower() not in ["y", "yes"]:
                    output_simple_error("Removal cancelled", formatter)
                    return 1

            # Remove documents
            results = []
            for doc_id in ids_to_remove:
                try:
                    sem.remove_document(doc_id)
                    results.append({"id": doc_id, "status": "success"})
                    logger.info("Removed document: %s", doc_id)
                except Exception as e:
                    results.append({"id": doc_id, "status": "error", "error": str(e)})
                    logger.error("Failed to remove document %s: %s", doc_id, e)

            success_count = sum(1 for r in results if r["status"] == "success")
            error_count = len(results) - success_count

            if error_count == 0:
                output_simple_success(
                    f"Successfully removed {success_count} documents",
                    formatter,
                    {"removed_count": success_count, "results": results}
                )
                return 0
            else:
                output_simple_error(
                    f"Removed {success_count} documents, {error_count} failed",
                    formatter,
                    {"removed_count": success_count, "error_count": error_count, "results": results}
                )
                return 1

        elif operation == "clear":
            # Confirm clearing unless --confirm flag is set
            confirm = getattr(args, "confirm", False)
            if not confirm:
                response = input("Clear all documents from GCP index? [y/N]: ")
                if response.lower() not in ["y", "yes"]:
                    output_simple_error("Clear operation cancelled", formatter)
                    return 1

            try:
                sem.clear()
                output_simple_success("Successfully cleared all documents", formatter)
                return 0
            except Exception as e:
                output_simple_error("Failed to clear documents: %s" % e, formatter)
                return 1

        else:
            output_simple_error("Unknown operation: %s" % operation, formatter)
            return 1

    except Exception as e:
        logger.error("GCP operation failed: %s", e)
        output_simple_error("GCP operation failed: %s" % e, formatter)
        return 1
