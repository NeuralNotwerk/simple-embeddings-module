#!/usr/bin/env python3
"""
SEM CLI Output Formatting System
Provides standardized output formatting for all CLI commands:
- JSON to stdout (default)
- Human-readable to stderr
- CLI format (single-line delimited) option
"""
import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional


class SEMOutputFormatter:
    """Standardized output formatter for SEM CLI commands."""
    def __init__(self, cli_format: bool = False, delimiter: str = ";"):
        """
        Initialize output formatter.
        Args:
            cli_format: If True, output single-line delimited format
            delimiter: Delimiter for CLI format (default: ';')
        """
        self.cli_format = cli_format
        self.delimiter = delimiter

    def output_result(self, data: Any, human_message: str = "", success: bool = True) -> None:
        """
        Output result in standardized format.
        Args:
            data: Data to output (will be JSON serialized)
            human_message: Human-readable message for stderr
            success: Whether operation was successful
        """
        if self.cli_format:
            self._output_cli_format(data, success)
        else:
            self._output_json_format(data, human_message, success)

    def _output_json_format(self, data: Any, human_message: str, success: bool) -> None:
        """Output JSON to stdout, human message to stderr."""
        # JSON to stdout
        result = {"success": success, "timestamp": datetime.now().isoformat(), "data": data}
        try:
            json_output = json.dumps(result, indent=2, default=str)
            print(json_output, file=sys.stdout)
        except Exception as e:
            # Fallback for non-serializable data
            fallback_result = {
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "data": str(data),
                "serialization_error": str(e),
            }
            print(json.dumps(fallback_result, indent=2), file=sys.stdout)
        # Human message to stderr
        if human_message:
            print(human_message, file=sys.stderr)

    def _output_cli_format(self, data: Any, success: bool) -> None:
        """Output single-line delimited format to stdout."""
        if isinstance(data, list):
            for item in data:
                self._output_single_item_cli(item, success)
        else:
            self._output_single_item_cli(data, success)

    def _output_single_item_cli(self, item: Any, success: bool) -> None:
        """Output single item in CLI format."""
        if isinstance(item, dict):
            # Convert dict to delimited string
            parts = []
            for key, value in item.items():
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, separators=(",", ":"))
                parts.append(f"{key}={value}")
            line = self.delimiter.join(parts)
        else:
            line = str(item)
        # Prefix with success status
        status = "SUCCESS" if success else "ERROR"
        print(f"{status}{self.delimiter}{line}", file=sys.stdout)

    def output_error(self, error_message: str, error_data: Optional[Dict] = None) -> None:
        """
        Output error in standardized format.
        Args:
            error_message: Human-readable error message
            error_data: Optional error data
        """
        data = error_data or {"error": error_message}
        if self.cli_format:
            self._output_cli_format(data, success=False)
        else:
            self._output_json_format(data, f"❌ {error_message}", success=False)
def create_formatter(args) -> SEMOutputFormatter:
    """
    Create output formatter from CLI arguments.
    Args:
        args: Parsed CLI arguments
    Returns:
        SEMOutputFormatter instance
    """
    cli_format = getattr(args, "cli_format", False)
    delimiter = getattr(args, "delimiter", ";")
    return SEMOutputFormatter(cli_format=cli_format, delimiter=delimiter)
# Convenience functions for common output patterns
def output_search_results(results: List[Dict], formatter: SEMOutputFormatter, query: str) -> None:
    """Output search results in standardized format."""
    data = {"query": query, "result_count": len(results), "results": results}
    human_message = f"🔍 Found {len(results)} result(s) for: '{query}'"
    if results:
        human_message += "\n"
        for i, result in enumerate(results, 1):
            score = result.get("score", result.get("similarity_score", 0))
            text = result.get("text", result.get("document", ""))[:100]
            doc_id = result.get("id", result.get("document_id", f"doc_{i}"))
            human_message += f"   {i}. {doc_id} (score: {score:.3f})\n"
            human_message += f"      {text}...\n"
    else:
        human_message += "\n   No results found"
    formatter.output_result(data, human_message)
def output_database_list(databases: List[Dict], formatter: SEMOutputFormatter, db_type: str = "databases") -> None:
    """Output database list in standardized format."""
    data = {"database_type": db_type, "database_count": len(databases), "databases": databases}
    human_message = f"📚 Found {len(databases)} {db_type}:"
    if databases:
        human_message += "\n"
        for db in databases:
            name = db.get("name", db.get("id", "unknown"))
            location = db.get("location", db.get("path", "unknown"))
            doc_count = db.get("document_count", 0)
            model = db.get("model_name", db.get("model", "unknown"))
            human_message += f"   • {name}\n"
            human_message += f"     📁 Location: {location}\n"
            human_message += f"     📄 Documents: {doc_count}\n"
            human_message += f"     🤖 Model: {model}\n\n"
    else:
        human_message += "\n   No databases found"
    formatter.output_result(data, human_message)
def output_document_list(documents: List[Dict], formatter: SEMOutputFormatter, database_name: str = "") -> None:
    """Output document list in standardized format."""
    data = {"database_name": database_name, "document_count": len(documents), "documents": documents}
    human_message = f"📋 Found {len(documents)} document(s)"
    if database_name:
        human_message += f" in {database_name}"
    human_message += ":"
    if documents:
        human_message += "\n"
        for i, doc in enumerate(documents, 1):
            doc_id = doc.get("id", f"doc_{i}")
            created = doc.get("created_at", "unknown")
            text = doc.get("text", "No content available")
            human_message += f"   {i}. {doc_id}\n"
            human_message += f"      Created: {created}\n"
            human_message += f"      Content: {text}\n\n"
    else:
        human_message += "\n   No documents found"
    formatter.output_result(data, human_message)
def output_info(info_data: Dict, formatter: SEMOutputFormatter, info_type: str = "info") -> None:
    """Output info data in standardized format."""
    data = {"info_type": info_type, "info": info_data}
    human_message = f"ℹ️  {info_type.title()} Information:\n"
    for key, value in info_data.items():
        human_message += f"   {key}: {value}\n"
    formatter.output_result(data, human_message)

def output_simple_success(message: str, formatter: SEMOutputFormatter, operation_data: Optional[Dict] = None) -> None:
    """Output simple success message."""
    data = operation_data or {"message": message}
    formatter.output_result(data, f"✅ {message}")

def output_simple_error(message: str, formatter: SEMOutputFormatter, error_data: Optional[Dict] = None) -> None:
    """Output simple error message."""
    formatter.output_error(message, error_data)
