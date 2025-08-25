#!/usr/bin/env python3
"""
SEM Simple GCP Interface
Simple, one-line interface for Google Cloud Platform integration.
Uses GCS for storage and Vertex AI for embeddings.
"""
import logging
import os
import sys
from typing import Any, Dict, List
import uuid

logger = logging.getLogger(__name__)
class SEMSimpleGCP:
    """Simple GCP interface for semantic search with GCS + Vertex AI."""
    def __init__(
        self,
        bucket_name: str = None,
        project_id: str = None,
        region: str = "us-central1",
        embedding_model: str = "textembedding-gecko@003",
        index_name: str = "sem_simple_gcp",
        credentials_path: str = None,
    ):
        """
        Initialize simple GCP semantic search.
        Args:
            bucket_name: GCS bucket name (auto-generated if None)
            project_id: Google Cloud project ID (auto-detected if None)
            region: GCP region for services
            embedding_model: Vertex AI embedding model
            index_name: Index name for organization
            credentials_path: Path to service account JSON
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.region = region
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.credentials_path = credentials_path
        # Auto-generate bucket name if not provided
        if not bucket_name:
            safe_project = (self.project_id or "sem").replace("_", "-").lower()
            self.bucket_name = f"{safe_project}-sem-simple-{uuid.uuid4().hex[:8]}"
        else:
            self.bucket_name = bucket_name
        self._db = None
        self._initialized = False
        logger.info("SEMSimpleGCP initialized: bucket='%s', model='%s'", self.bucket_name, embedding_model)

    def _ensure_initialized(self):
        """Ensure the database is initialized."""
        if self._initialized:
            return
        try:
            from .sem_config_builder import SEMConfigBuilder
            from .sem_core import SEMDatabase
            # Build configuration
            config_builder = SEMConfigBuilder()
            # Set up Vertex AI embeddings
            config_builder.set_embedding_provider(
                "vertex_ai",
                {
                    "model": self.embedding_model,
                    "project_id": self.project_id,
                    "region": self.region,
                    "credentials_path": self.credentials_path,
                },
            )
            # Set up GCS storage
            config_builder.set_storage_backend(
                "gcs",
                {
                    "bucket_name": self.bucket_name,
                    "project_id": self.project_id,
                    "region": self.region,
                    "credentials_path": self.credentials_path,
                },
            )
            # Set up text chunking
            config_builder.set_chunking_strategy("text")
            # Set up orjson serialization
            config_builder.set_serialization_provider("orjson")
            # Set index name
            config_builder.set_index_config(self.index_name)
            # Create database
            self._db = SEMDatabase(config=config_builder.to_dict())
            # Check if index exists
            if self._db.storage.index_exists(self.index_name):
                doc_count = len(self._db.list_documents())
                if doc_count > 0:
                    print(f"📚 Found existing GCP semantic search index with {doc_count} documents", file=sys.stderr)
                    print("🔍 Ready to search! Use .search('your query') to find documents", file=sys.stderr)
                else:
                    print("📝 Ready to add documents! Use .add_text('your content') to start", file=sys.stderr)
            else:
                print("📝 Ready to add documents! Use .add_text('your content') to start", file=sys.stderr)
            self._initialized = True
            logger.info("SEMSimpleGCP database initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize SEMSimpleGCP: %s", e)
            raise RuntimeError("SEMSimpleGCP initialization failed: %s" % e)

    def add_text(self, text: str, doc_id: str = None) -> str:
        """
        Add text to the semantic search index.
        Args:
            text: Text content to add
            doc_id: Optional document ID (auto-generated if None)
        Returns:
            Document ID of added text
        """
        self._ensure_initialized()
        if not doc_id:
            doc_id = "doc_%s" % uuid.uuid4().hex[:8]
        try:
            self._db.add_documents([text], [doc_id])
            logger.info("Added document '%s' to GCP index", doc_id)
            return doc_id
        except Exception as e:
            logger.error("Failed to add text to GCP index: %s", e)
            raise RuntimeError("Failed to add text: %s" % e)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the semantic index.
        Args:
            query: Search query
            top_k: Number of results to return
        Returns:
            List of search results with id, text, score, metadata
        """
        self._ensure_initialized()
        try:
            results = self._db.search(query, top_k=top_k)
            # Convert to simple format
            simple_results = []
            for result in results:
                simple_results.append(
                    {
                        "id": result.get("document_id", "unknown"),
                        "text": result.get("document", ""),
                        "score": result.get("similarity_score", 0.0),
                        "metadata": result.get("metadata", {}),
                    }
                )
            logger.info("Search completed: found %d results for '%s'", len(simple_results), query)
            return simple_results
        except Exception as e:
            logger.error("Search failed: %s", e)
            raise RuntimeError("Search failed: %s" % e)

    def list_documents(
        self, limit: int = None, show_content: bool = True, max_content_length: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List documents in the index.
        Args:
            limit: Maximum number of documents to return
            show_content: Whether to include document content
            max_content_length: Maximum content length to return
        Returns:
            List of documents with metadata
        """
        self._ensure_initialized()
        try:
            documents = self._db.list_documents(
                limit=limit, show_content=show_content, max_content_length=max_content_length
            )
            logger.info("Listed %d documents from GCP index", len(documents))
            return documents
        except Exception as e:
            logger.error("Failed to list documents: %s", e)
            raise RuntimeError("Failed to list documents: %s" % e)

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the index.
        Args:
            doc_id: Document ID to remove
        Returns:
            True if successful
        """
        self._ensure_initialized()
        try:
            success = self._db.remove_document(doc_id)
            if success:
                logger.info("Removed document '%s' from GCP index", doc_id)
            else:
                logger.warning("Document '%s' not found in GCP index", doc_id)
            return success
        except Exception as e:
            logger.error("Failed to remove document '%s': %s", doc_id, e)
            return False

    def clear(self) -> bool:
        """
        Clear all documents from the index.
        Returns:
            True if successful
        """
        self._ensure_initialized()
        try:
            success = self._db.clear_index()
            if success:
                logger.info("Cleared all documents from GCP index")
            return success
        except Exception as e:
            logger.error("Failed to clear GCP index: %s", e)
            return False

    def delete_index(self) -> bool:
        """
        Delete the entire index.
        Returns:
            True if successful
        """
        self._ensure_initialized()
        try:
            success = self._db.storage.delete_index(self.index_name)
            if success:
                logger.info("Deleted GCP index '%s'", self.index_name)
                self._initialized = False
            return success
        except Exception as e:
            logger.error("Failed to delete GCP index: %s", e)
            return False

    def info(self) -> Dict[str, Any]:
        """
        Get information about the index.
        Returns:
            Dictionary with index information
        """
        self._ensure_initialized()
        try:
            info = self._db.get_index_info()
            if info:
                info.update(
                    {
                        "backend": "gcp",
                        "bucket_name": self.bucket_name,
                        "project_id": self.project_id,
                        "region": self.region,
                        "embedding_model": self.embedding_model,
                    }
                )
            return info or {}
        except Exception as e:
            logger.error("Failed to get GCP index info: %s", e)
            return {"error": str(e)}

    def count(self) -> int:
        """
        Get the number of documents in the index.
        Returns:
            Number of documents
        """
        try:
            documents = self.list_documents(show_content=False)
            return len(documents)
        except Exception:
            return 0
# Convenience function for one-line usage
def simple_gcp(bucket_name: str = None, **kwargs) -> SEMSimpleGCP:
    """
    Create a simple GCP semantic search instance.
    Args:
        bucket_name: GCS bucket name (auto-generated if None)
        **kwargs: Additional arguments for SEMSimpleGCP
    Returns:
        Configured SEMSimpleGCP instance
    Example:
        >>> from simple_embeddings_module import simple_gcp
        >>> sem = simple_gcp(bucket_name="my-semantic-search")
        >>> sem.add_text("Machine learning is transforming software development.")
        >>> results = sem.search("AI technology")
        >>> print(results[0]['text'])
    """
    return SEMSimpleGCP(bucket_name=bucket_name, **kwargs)
