"""
SEMSimple - One-line semantic search
The simplest possible interface to SEM. Just import and go!
"""
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from .sem_config_builder import SEMConfigBuilder
from .sem_core import SEMDatabase
from .sem_utils import generate_config_template, save_config, load_config
from .sem_auto_resolve import list_available_databases

logger = logging.getLogger(__name__)
class SEMSimple:
    """
    Ultra-simple semantic search interface.
    Perfect for getting started quickly with sensible defaults:
    - Uses sentence-transformers/all-MiniLM-L6-v2 model
    - Stores indexes in ./sem_indexes/
    - Automatic chunking and GPU acceleration
    - No configuration needed!
    Example:
        >>> from simple_embeddings_module import SEMSimple
        >>> sem = SEMSimple()
        >>> sem.add_text("Machine learning is transforming software.")
        >>> results = sem.search("AI technology")
        >>> print(results[0]['text'])
    """
    def __init__(
        self, index_name: str = "sem_simple_database", storage_path: str = "./sem_indexes", embedding_model: str = None
    ):
        """
        Initialize SEMSimple with sensible defaults.
        Args:
            index_name: Name for the search index (default: "sem_simple_database")
            storage_path: Where to store the index files (default: "./sem_indexes")
            embedding_model: Sentence-transformers model to use (None = auto-detect from existing DB or use default)
        """
        self.index_name = index_name
        self._db = None
        self._initialized = False
        # Resolve storage path (check multiple locations for default path)
        self.storage_path = self._resolve_storage_path(storage_path)
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        # Determine which model to use: saved config > specified model > default
        self.embedding_model = self._determine_model(embedding_model)
        logger.info(
            "SEMSimple initialized: index='%s', path='%s', model='%s'", index_name, self.storage_path, self.embedding_model
        )

    def _resolve_storage_path(self, requested_path: str) -> Path:
        """
        Resolve storage path by checking multiple locations for existing databases.
        For the default path "./sem_indexes", checks:
        1. ./sem_indexes (current directory)
        2. ~/sem_indexes (home directory)
        For custom paths, uses the path as-is.
        Args:
            requested_path: The requested storage path
        Returns:
            Resolved Path object
        """
        requested_path_obj = Path(requested_path)
        # If it's not the default path, use it as-is
        if requested_path != "./sem_indexes":
            logger.info("Using custom storage path: %s", requested_path_obj)
            return requested_path_obj
        # For default path, check multiple locations
        search_paths = [
            Path("./sem_indexes"),  # Current directory
            Path.home() / "sem_indexes",  # Home directory
        ]
        # Check each path for existing database
        for path in search_paths:
            if path.exists() and path.is_dir():
                # Check if it contains our specific database
                db_path = path / self.index_name
                if db_path.exists() and db_path.is_dir():
                    logger.info("Found existing database at: %s", path)
                    return path
                # Also check if it contains any databases
                elif any(p.is_dir() for p in path.iterdir() if not p.name.startswith(".")):
                    logger.info("Found existing sem_indexes directory at: %s", path)
                    return path
        # No existing databases found, use current directory (default)
        logger.info("No existing databases found, using default: %s", search_paths[0])
        return search_paths[0]

    def _determine_model(self, requested_model: str = None) -> str:
        """
        Determine which model to use based on priority:
        1. Saved model from existing database (highest priority)
        2. Explicitly requested model
        3. Default model (lowest priority)
        Args:
            requested_model: The model explicitly requested by user (can be None)
        Returns:
            The model to actually use
        """
        try:
            # Check if database exists and has a saved model
            from .storage.mod_local_disk import LocalDiskStorage
            storage = LocalDiskStorage(path=str(self.storage_path))
            if storage.index_exists(self.index_name):
                index_info = storage.get_index_info(self.index_name)
                if index_info and index_info.get("model_name"):
                    saved_model = index_info["model_name"]
                    if requested_model and saved_model != requested_model:
                        logger.warning(
                            "Database has saved model '%s' but '%s' was requested. Using saved model.", saved_model, requested_model
                        )
                    else:
                        logger.info("Using saved model '%s' from existing database", saved_model)
                    return saved_model
            # No existing database with saved model
            if requested_model:
                logger.info("Using explicitly requested model: '%s'", requested_model)
                return requested_model
            # No existing database, no requested model - use default
            default_model = "all-MiniLM-L6-v2"
            logger.info("No existing database or requested model, using default: '%s'", default_model)
            return default_model
        except Exception as e:
            logger.warning("Could not check existing database model: %s", e)
            # Fallback logic
            if requested_model:
                return requested_model
            return "all-MiniLM-L6-v2"

    def _ensure_initialized(self):
        """Lazy initialization of the database."""
        if not self._initialized:
            try:
                # Create default configuration
                builder = SEMConfigBuilder()
                builder.set_embedding_provider("sentence_transformers", model=self.embedding_model)
                builder.auto_configure_chunking()
                builder.set_storage_backend("local_disk", path=str(self.storage_path))
                builder.set_serialization_provider("orjson")
                builder.set_index_config(self.index_name)
                config = builder.build()
                # Create database
                self._db = SEMDatabase(config=config)
                # Check for existing index
                existing_info = self._db.get_index_info()
                if existing_info:
                    # Handle both dict and object returns
                    doc_count = getattr(existing_info, "document_count", existing_info.get("document_count", 0))
                    existing_dim = getattr(existing_info, "embedding_dim", existing_info.get("embedding_dim", None))
                    if doc_count > 0:
                        logger.info("Found existing index with %s documents", doc_count)
                        print("📚 Found existing semantic search index with %s documents" % doc_count, file=sys.stderr)
                        # Validate embedding dimensions if we have the info
                        if existing_dim:
                            try:
                                # Get current model's embedding dimension
                                current_dim = None
                                if hasattr(self._db, "_embedding_provider") and hasattr(
                                    self._db._embedding_provider, "_embedding_dim"
                                ):
                                    current_dim = self._db._embedding_provider._embedding_dim
                                if current_dim and existing_dim != current_dim:
                                    error_msg = (
                                        "Embedding dimension mismatch!\n"
                                        "  Existing database: %s dimensions\n"
                                        "  Current model: %s dimensions\n"
                                        "  Model: %s\n\n"
                                        "Solutions:\n"
                                        "  1. Use a compatible model: --model sentence-transformers/all-mpnet-base-v2\n"
                                        "  2. Clear database: sem-cli simple local clear --confirm --db %s\n"
                                        "  3. Use different database: --db new_database_name"
                                    ) % (existing_dim, current_dim, self.embedding_model, self.index_name)
                                    logger.error(
                                        "Embedding dimension mismatch: existing=%s, current=%s", existing_dim, current_dim
                                    )
                                    raise ValueError(error_msg)
                            except AttributeError:
                                # If we can't get the current dimension, just warn
                                logger.warning(
                                    "Could not validate embedding dimensions. Existing database has %s dimensions.", existing_dim
                                )
                        print("🔍 Ready to search! Use .search('your query') to find documents", file=sys.stderr)
                    else:
                        logger.info("No existing index found - ready to add documents")
                        print("📝 Ready to add documents! Use .add_text('your content') to start", file=sys.stderr)
                else:
                    logger.info("No existing index found - ready to add documents")
                    print("📝 Ready to add documents! Use .add_text('your content') to start", file=sys.stderr)
                self._initialized = True
                logger.info("SEMSimple database initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize SEMSimple: %s", e)
                raise RuntimeError("SEMSimple initialization failed: %s" % e)

    def add_text(self, text: str, doc_id: Optional[str] = None) -> bool:
        """
        Add a single text document to the search index.
        Args:
            text: The text content to add
            doc_id: Optional document ID (auto-generated if not provided)
        Returns:
            True if successful
        Example:
            >>> sem.add_text("Machine learning is amazing!")
            True
        """
        return self.add_texts([text], [doc_id] if doc_id else None)

    def add_texts(self, texts: List[str], doc_ids: Optional[List[str]] = None) -> bool:
        """
        Add multiple text documents to the search index.
        Args:
            texts: List of text contents to add
            doc_ids: Optional list of document IDs (auto-generated if not provided)
        Returns:
            True if successful
        Example:
            >>> texts = ["First document", "Second document"]
            >>> sem.add_texts(texts)
            True
        """
        self._ensure_initialized()
        try:
            # Generate doc_ids if not provided
            if doc_ids is None:
                import uuid
                doc_ids = ["doc_%s" % uuid.uuid4().hex[:8] for _ in texts]
            elif len([d for d in doc_ids if d is not None]) != len(texts):
                # Handle case where some doc_ids are None
                import uuid
                doc_ids = [doc_id if doc_id is not None else "doc_%s" % uuid.uuid4().hex[:8] for doc_id in doc_ids]
            if len(texts) != len(doc_ids):
                raise ValueError("Number of texts and doc_ids must match")
            # Add documents - this should append to existing index
            success = self._db.add_documents(texts, document_ids=doc_ids)
            if success:
                logger.info("Added %s text documents", len(texts))
                return True
            else:
                logger.warning("Failed to add %s text documents", len(texts))
                return False
        except Exception as e:
            logger.error("Error adding texts: %s", e)
            return False

    def add_file(self, file_path: str) -> bool:
        """
        Add a text file to the search index.
        Args:
            file_path: Path to the text file
        Returns:
            True if successful
        Example:
            >>> sem.add_file("document.txt")
            True
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error("File not found: %s", file_path)
                return False
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Use filename as doc_id
            doc_id = file_path.stem
            return self.add_text(content, doc_id=doc_id)
        except Exception as e:
            logger.error("Error adding file %s: %s", file_path, e)
            return False

    def add_files(self, file_paths: List[str]) -> bool:
        """
        Add multiple text files to the search index.
        Args:
            file_paths: List of paths to text files
        Returns:
            True if successful
        Example:
            >>> sem.add_files(["doc1.txt", "doc2.txt"])
            True
        """
        try:
            texts = []
            doc_ids = []
            for file_path in file_paths:
                file_path = Path(file_path)
                if not file_path.exists():
                    logger.warning("File not found, skipping: %s", file_path)
                    continue
                # Read file content
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                texts.append(content)
                doc_ids.append(file_path.stem)
            if not texts:
                logger.error("No valid files found")
                return False
            return self.add_texts(texts, doc_ids=doc_ids)
        except Exception as e:
            logger.error("Error adding files: %s", e)
            return False

    def search(self, query: str, top_k: int = 5, output_format: str = "dict", delimiter: str = ";") -> Union[List[dict], str]:
        """
        Search for documents similar to the query.

        Args:
            query: Search query text
            top_k: Number of results to return (default: 5)
            output_format: Output format ("dict", "cli", "json", "csv")
            delimiter: Delimiter for CLI format (default: ";")

        Returns:
            List of result dictionaries or formatted string

        Example:
            >>> results = sem.search("machine learning", top_k=3)
            >>> for result in results:
            ...     print(f"Score: {result['score']:.3f} - {result['text']}")

            >>> cli_results = sem.search("AI", output_format="cli", delimiter="|")
            >>> print(cli_results)
            'doc_1|0.85|Machine learning transforms...|doc_2|0.78|AI technology...'
        """
        self._ensure_initialized()
        # Check for dimension mismatch before attempting search
        existing_info = self._db.get_index_info()
        if existing_info:
            existing_dim = getattr(existing_info, "embedding_dim", existing_info.get("embedding_dim", None))
            if (existing_dim and hasattr(self._db, "_embedding_provider") and hasattr(self._db._embedding_provider, "_embedding_dim")):
                current_dim = self._db._embedding_provider._embedding_dim
                if existing_dim != current_dim:
                    error_msg = (
                        "Cannot search: Embedding dimension mismatch!\n"
                        "  Existing database: %s dimensions\n"
                        "  Current model: %s dimensions\n"
                        "  Model: %s\n\n"
                        "Solutions:\n"
                        "  1. Use a compatible model: --model sentence-transformers/all-mpnet-base-v2\n"
                        "  2. Clear database: sem-cli simple local clear --confirm --db %s\n"
                        "  3. Use different database: --db new_database_name"
                    ) % (existing_dim, current_dim, self.embedding_model, self.index_name)
                    logger.error(
                        "Search blocked due to dimension mismatch: existing=%s, current=%s", existing_dim, current_dim
                    )
                    raise ValueError(error_msg)
        try:
            results = self._db.search(query, top_k=top_k)
            # Convert to simple dictionary format
            simple_results = []
            for result in results:
                simple_results.append(
                    {
                        "id": result.get("document_id", "unknown"),
                        "text": result.get("document", ""),
                        "score": result.get("similarity_score", 0.0),
                    }
                )
            logger.info("Search completed: found %s results for '%s'", len(simple_results), query)

            # Format output based on requested format
            if output_format == "dict":
                return simple_results
            elif output_format == "cli":
                return self._format_search_results_cli(simple_results, delimiter)
            elif output_format == "json":
                import json
                return json.dumps(simple_results, indent=2)
            elif output_format == "csv":
                return self._format_search_results_csv(simple_results, delimiter)
            else:
                logger.warning("Unknown output format '%s', using 'dict'", output_format)
                return simple_results
        except Exception as e:
            logger.error("Search error: %s", e)
            return [] if output_format == "dict" else ""

    def _format_search_results_cli(self, results: List[dict], delimiter: str) -> str:
        """Format search results for CLI output."""
        if not results:
            return ""

        formatted_parts = []
        for result in results:
            # Truncate text for CLI display
            text = result.get("text", "")
            if len(text) > 100:
                text = text[:97] + "..."

            formatted_parts.extend([
                result.get("id", "unknown"),
                f"{result.get('score', 0.0):.3f}",
                text.replace(delimiter, " "),  # Remove delimiter from text
            ])

        return delimiter.join(formatted_parts)

    def _format_search_results_csv(self, results: List[dict], delimiter: str) -> str:
        """Format search results as CSV."""
        if not results:
            return "id,score,text\n"

        lines = ["id,score,text"]
        for result in results:
            text = result.get("text", "").replace('"', '""')  # Escape quotes
            lines.append(f'"{result.get("id", "unknown")}",{result.get("score", 0.0):.3f},"{text}"')

        return "\n".join(lines)

    def count(self) -> int:
        """
        Get the number of documents in the index.
        Returns:
            Number of documents
        Example:
            >>> sem.count()
            42
        """
        self._ensure_initialized()
        try:
            # Get index info
            info = self._db.get_index_info()
            return info.get("document_count", 0) if info else 0
        except Exception as e:
            logger.error("Error getting document count: %s", e)
            return 0

    def clear(self) -> bool:
        """
        Clear all documents from the index.
        Returns:
            True if successful
        Example:
            >>> sem.clear()
            True
        """
        try:
            if self._db and self._initialized:
                # Delete the index
                storage = self._db._storage_backend
                if storage.index_exists(self.index_name):
                    success = storage.delete_index(self.index_name)
                    if success:
                        # Reset initialization to force recreation
                        self._initialized = False
                        self._db = None
                        logger.info("Index cleared successfully")
                        return True
            return True  # Nothing to clear
        except Exception as e:
            logger.error("Error clearing index: %s", e)
            return False

    def info(self) -> dict:
        """
        Get information about the search index.
        Returns:
            Dictionary with index information
        Example:
            >>> info = sem.info()
            >>> print(f"Documents: {info['document_count']}")
        """
        self._ensure_initialized()
        try:
            info = self._db.get_index_info()
            return info if info else {}
        except Exception as e:
            logger.error("Error getting index info: %s", e)
            return {}

    def list_documents(
        self,
        limit: Optional[int] = None,
        show_content: bool = True,
        max_content_length: int = 100,
        output_format: str = "dict",
        delimiter: str = ";"
    ) -> Union[List[dict], str]:
        """
        List documents in the search index.

        Args:
            limit: Maximum number of documents to return (None for all)
            show_content: Whether to include document content snippets
            max_content_length: Maximum length of content to show
            output_format: Output format ("dict", "cli", "json", "csv")
            delimiter: Delimiter for CLI format (default: ";")

        Returns:
            List of document dictionaries or formatted string

        Example:
            >>> docs = sem.list_documents(limit=5)
            >>> for doc in docs:
            ...     print(f"ID: {doc['id']}, Text: {doc['text'][:50]}...")

            >>> cli_docs = sem.list_documents(limit=3, output_format="cli")
            >>> print(cli_docs)
            'doc_1;Machine learning content...;doc_2;AI algorithms...'
        """
        self._ensure_initialized()
        try:
            documents = self._db.list_documents(
                limit=limit, show_content=show_content, max_content_length=max_content_length
            )

            # Format output based on requested format
            if output_format == "dict":
                return documents
            elif output_format == "cli":
                return self._format_documents_cli(documents, delimiter, show_content)
            elif output_format == "json":
                import json
                return json.dumps(documents, indent=2)
            elif output_format == "csv":
                return self._format_documents_csv(documents, delimiter, show_content)
            else:
                logger.warning("Unknown output format '%s', using 'dict'", output_format)
                return documents
        except Exception as e:
            logger.error("Error listing documents: %s", e)
            return [] if output_format == "dict" else ""

    def _format_documents_cli(self, documents: List[dict], delimiter: str, show_content: bool) -> str:
        """Format documents for CLI output."""
        if not documents:
            return ""

        formatted_parts = []
        for doc in documents:
            doc_id = doc.get("id", doc.get("document_id", "unknown"))
            formatted_parts.append(doc_id)

            if show_content:
                text = doc.get("text", doc.get("content", ""))
                # Clean delimiter from text
                text = text.replace(delimiter, " ")
                formatted_parts.append(text)

        return delimiter.join(formatted_parts)

    def _format_documents_csv(self, documents: List[dict], delimiter: str, show_content: bool) -> str:
        """Format documents as CSV."""
        if not documents:
            header = "id,text" if show_content else "id"
            return header + "\n"

        lines = []
        if show_content:
            lines.append("id,text")
            for doc in documents:
                doc_id = doc.get("id", doc.get("document_id", "unknown"))
                text = doc.get("text", doc.get("content", "")).replace('"', '""')
                lines.append(f'"{doc_id}","{text}"')
        else:
            lines.append("id")
            for doc in documents:
                doc_id = doc.get("id", doc.get("document_id", "unknown"))
                lines.append(f'"{doc_id}"')

        return "\n".join(lines)

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a single document from the index.
        Args:
            doc_id: Document ID to remove
        Returns:
            True if document was removed successfully
        Example:
            >>> sem.remove_document("doc_abc123")
            True
        """
        self._ensure_initialized()
        try:
            success = self._db.remove_document(doc_id)
            if success:
                logger.info("Removed document: %s", doc_id)
            return success
        except Exception as e:
            logger.error("Failed to remove document %s: %s", doc_id, e)
            return False

    def remove_documents(self, doc_ids: List[str]) -> int:
        """
        Remove multiple documents from the index.
        Args:
            doc_ids: List of document IDs to remove
        Returns:
            Number of documents successfully removed
        Example:
            >>> removed = sem.remove_documents(["doc_1", "doc_2", "doc_3"])
            >>> print(f"Removed {removed} documents")
        """
        self._ensure_initialized()
        try:
            removed_count = self._db.remove_documents(doc_ids)
            logger.info("Removed %s of %s documents", removed_count, len(doc_ids))
            return removed_count
        except Exception as e:
            logger.error("Failed to remove documents: %s", e)
            return 0

    def remove_by_query(self, query: str, max_results: int = 10) -> int:
        """
        Remove documents that match a search query.
        Args:
            query: Search query to find documents to remove
            max_results: Maximum number of documents to remove
        Returns:
            Number of documents removed
        Example:
            >>> removed = sem.remove_by_query("outdated documentation", max_results=5)
            >>> print(f"Removed {removed} outdated documents")
        """
        self._ensure_initialized()
        try:
            # Search for matching documents
            results = self.search(query, top_k=max_results)
            if not results:
                logger.info("No documents found matching query: %s", query)
                return 0
            # Extract document IDs
            doc_ids = [result.get("document_id", result.get("id")) for result in results]
            doc_ids = [doc_id for doc_id in doc_ids if doc_id]  # Filter out None values
            if not doc_ids:
                logger.warning("No valid document IDs found in search results")
                return 0
            # Remove the documents
            removed_count = self.remove_documents(doc_ids)
            logger.info("Removed %s documents matching query: %s", removed_count, query)
            return removed_count
        except Exception as e:
            logger.error("Failed to remove documents by query '%s': %s", query, e)
            return 0

    def update_document(self, doc_id: str, new_text: str) -> bool:
        """
        Update an existing document's content.
        Args:
            doc_id: Document ID to update
            new_text: New text content
        Returns:
            True if document was updated successfully
        Example:
            >>> sem.update_document("doc_abc123", "Updated content here")
            True
        """
        self._ensure_initialized()
        try:
            success = self._db.update_document(doc_id, new_text)
            if success:
                logger.info("Updated document: %s", doc_id)
            return success
        except Exception as e:
            logger.error("Failed to update document %s: %s", doc_id, e)
            return False

    def delete_index(self) -> bool:
        """
        Delete the entire index and its files.
        Returns:
            True if index was deleted successfully
        Example:
            >>> sem.delete_index()
            True
        """
        self._ensure_initialized()
        try:
            success = self._db.delete_index()
            if success:
                logger.info("Deleted index: %s", self.index_name)
                self._initialized = False  # Mark as uninitialized
            return success
        except Exception as e:
            logger.error("Failed to delete index %s: %s", self.index_name, e)
            return False

    def generate_config_template(self, output_path: Optional[str] = None) -> Union[Dict[str, Any], bool]:
        """
        Generate a configuration template with current settings.

        Args:
            output_path: Optional path to save config file

        Returns:
            Configuration dictionary if no output_path, True if file saved successfully

        Example:
            >>> config = sem.generate_config_template()
            >>> print(config['embedding']['model'])
            'all-MiniLM-L6-v2'

            >>> sem.generate_config_template("my_config.json")
            True
        """
        try:
            # Generate template with current settings
            config_kwargs = {
                "embedding.model": self.embedding_model,
                "storage.path": str(self.storage_path),
                "index.name": self.index_name,
            }

            if output_path:
                # Save to file and return success status
                success = generate_config_template(output_path, **config_kwargs)
                logger.info("Generated config template: %s", output_path)
                return success
            else:
                # Return config object
                config_obj = generate_config_template(**config_kwargs)
                logger.info("Generated config template object")
                return config_obj.to_dict() if hasattr(config_obj, 'to_dict') else dict(config_obj)
        except Exception as e:
            logger.error("Failed to generate config template: %s", e)
            if output_path:
                return False
            else:
                return {}

    def save_config(self, config_path: str) -> bool:
        """
        Save current configuration to file.

        Args:
            config_path: Path to save configuration

        Returns:
            True if successful

        Example:
            >>> sem.save_config("current_config.json")
            True
        """
        try:
            # Get current configuration from the database if initialized
            if self._initialized and self._db:
                config = self._db._config
                save_config(config, config_path)
            else:
                # Generate template with current settings
                self.generate_config_template(config_path)

            logger.info("Saved configuration to: %s", config_path)
            return True
        except Exception as e:
            logger.error("Failed to save config to %s: %s", config_path, e)
            return False

    def load_config_from_file(self, config_path: str) -> bool:
        """
        Load configuration from file and reinitialize.

        Args:
            config_path: Path to configuration file

        Returns:
            True if successful

        Example:
            >>> sem.load_config_from_file("production_config.json")
            True
        """
        try:
            # Load configuration
            config = load_config(config_path)

            # Extract key settings
            if hasattr(config, 'embedding') and hasattr(config.embedding, 'model'):
                self.embedding_model = config.embedding.model
            if hasattr(config, 'storage') and hasattr(config.storage, 'path'):
                self.storage_path = Path(config.storage.path)
            if hasattr(config, 'index') and hasattr(config.index, 'name'):
                self.index_name = config.index.name

            # Force reinitialization with new config
            self._initialized = False
            self._db = None

            logger.info("Loaded configuration from: %s", config_path)
            return True
        except Exception as e:
            logger.error("Failed to load config from %s: %s", config_path, e)
            return False

    @staticmethod
    def discover_databases(search_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Discover available databases in search paths.

        Args:
            search_paths: Optional list of paths to search (uses defaults if None)

        Returns:
            List of database info dictionaries

        Example:
            >>> databases = SEMSimple.discover_databases()
            >>> for db in databases:
            ...     print(f"Name: {db['name']}, Path: {db['path']}")
        """
        try:
            # Use the existing auto-resolve functionality
            all_databases = list_available_databases()

            # Flatten the structure for easier consumption
            discovered = []
            for db_name, locations in all_databases.items():
                for location in locations:
                    discovered.append({
                        "name": db_name,
                        "path": location.get("path", ""),
                        "type": location.get("type", "unknown"),
                        "document_count": location.get("document_count", 0),
                        "last_modified": location.get("last_modified", ""),
                    })

            logger.info("Discovered %s databases", len(discovered))
            return discovered
        except Exception as e:
            logger.error("Failed to discover databases: %s", e)
            return []

    @staticmethod
    def list_available_databases() -> List[str]:
        """
        Get list of discoverable database names.

        Returns:
            List of database names

        Example:
            >>> names = SEMSimple.list_available_databases()
            >>> print(names)
            ['my_project', 'documentation', 'code_search']
        """
        try:
            all_databases = list_available_databases()
            names = list(all_databases.keys())
            logger.info("Found %s available databases", len(names))
            return names
        except Exception as e:
            logger.error("Failed to list available databases: %s", e)
            return []

    def auto_resolve_database(self, db_name: str) -> bool:
        """
        Auto-resolve and switch to a discovered database.

        Args:
            db_name: Name of database to resolve and switch to

        Returns:
            True if successful

        Example:
            >>> sem.auto_resolve_database("my_project")
            True
        """
        try:
            # Discover databases
            databases = self.discover_databases()

            # Find matching database
            matching_db = None
            for db in databases:
                if db["name"] == db_name:
                    matching_db = db
                    break

            if not matching_db:
                logger.error("Database not found: %s", db_name)
                return False

            # Update settings to point to discovered database
            self.index_name = matching_db["name"]
            self.storage_path = Path(matching_db["path"]).parent

            # Force reinitialization
            self._initialized = False
            self._db = None

            logger.info("Auto-resolved to database: %s at %s", db_name, self.storage_path)
            return True
        except Exception as e:
            logger.error("Failed to auto-resolve database %s: %s", db_name, e)
            return False

    def __repr__(self) -> str:
        """String representation of SEMSimple."""
        doc_count = self.count() if self._initialized else "?"
        return f"SEMSimple(index='{self.index_name}', documents={doc_count}, path='{self.storage_path}')"
