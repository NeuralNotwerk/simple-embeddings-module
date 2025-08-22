"""
Simple Embeddings Module - Core Database Class

Main interface for the Simple Embeddings Module.
Manages module registry, configuration, and provides unified API.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from .chunking.mod_chunking_base import ChunkingProviderBase
from .embeddings.mod_embeddings_base import EmbeddingProviderBase
from .sem_module_reg import ModuleRegistry, create_module_registry, discover_all_modules
from .serialization.mod_serialization_base import SerializationProviderBase
from .storage.mod_storage_base import StorageBackendBase

logger = logging.getLogger(__name__)


class SEMDatabase:
    """
    Simple Embeddings Module Database

    Main interface for semantic search functionality.
    Each instance has isolated module registry and configuration.
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None
    ):
        """Initialize SEMDatabase with configuration

        Args:
            config: Configuration dictionary
            config_path: Path to JSON configuration file
        """
        # Generate unique instance ID
        self.instance_id = str(uuid.uuid4())

        # Load configuration
        self.config = self._load_configuration(config, config_path)

        # Create isolated module registry
        self.registry = create_module_registry(self.instance_id)

        # Initialize components
        self._embedding_provider: Optional[EmbeddingProviderBase] = None
        self._storage_backend: Optional[StorageBackendBase] = None
        self._serialization_provider: Optional[SerializationProviderBase] = None
        self._chunking_provider: Optional[ChunkingProviderBase] = None

        # Initialize configured modules
        self._initialize_modules()

        logger.info(f"SEMDatabase initialized with instance ID: {self.instance_id}")

    def _load_configuration(
        self, config: Optional[Dict[str, Any]], config_path: Optional[str]
    ) -> Dict[str, Any]:
        """Load configuration from dict or file"""
        if config is not None and config_path is not None:
            raise ValueError("Cannot specify both config and config_path")

        if config_path is not None:
            from .sem_utils import load_config

            return load_config(config_path)

        if config is not None:
            return config.copy()

        # Return default configuration
        return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "embedding": {
                "provider": "sentence_transformers",
                "model": "all-MiniLM-L6-v2",
                "batch_size": 32,
            },
            "chunking": {"strategy": "text", "boundary_type": "sentence"},
            "storage": {"backend": "local_disk", "path": "./indexes"},
            "serialization": {"provider": "orjson"},
            "index": {
                "name": "sem_simple_index",
                "max_documents": 100000,
                "similarity_threshold": 0.1,
            },
        }

    def _initialize_modules(self) -> None:
        """Initialize modules based on configuration"""
        # Initialize embedding provider
        embedding_config = self.config.get("embedding", {})
        embedding_provider = embedding_config.get("provider", "sentence_transformers")

        self._embedding_provider = self.registry.instantiate_module(
            "embeddings", embedding_provider, embedding_config
        )

        # Unload other embedding providers
        self.registry.unload_all_modules_except("embeddings", embedding_provider)

        # Initialize storage backend
        storage_config = self.config.get("storage", {})
        storage_backend = storage_config.get("backend", "local_disk")

        self._storage_backend = self.registry.instantiate_module(
            "storage", storage_backend, storage_config
        )

        # Unload other storage backends
        self.registry.unload_all_modules_except("storage", storage_backend)

        # Initialize serialization provider
        serialization_config = self.config.get("serialization", {})
        serialization_provider = serialization_config.get("provider", "orjson")

        self._serialization_provider = self.registry.instantiate_module(
            "serialization", serialization_provider, serialization_config
        )

        # Initialize chunking provider
        chunking_config = self.config.get("chunking", {})
        chunking_strategy = chunking_config.get("strategy", "text")

        # Chunking provider needs embedding provider as dependency
        chunking_config_with_embedding = chunking_config.copy()
        chunking_config_with_embedding["embedding_provider"] = self._embedding_provider

        self._chunking_provider = self.registry.instantiate_module(
            "chunking", chunking_strategy, chunking_config_with_embedding
        )

        # Unload other chunking providers
        self.registry.unload_all_modules_except("chunking", chunking_strategy)

    def add_documents(
        self,
        documents: List[str],
        document_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Add documents to the index

        Args:
            documents: List of document texts
            document_ids: Optional list of document IDs (auto-generated if None)
            metadata: Optional list of metadata dicts per document

        Returns:
            Dict with operation results
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        if document_ids is not None and len(document_ids) != len(documents):
            raise ValueError("document_ids length must match documents length")

        if metadata is not None and len(metadata) != len(documents):
            raise ValueError("metadata length must match documents length")

        start_time = time.time()

        # Generate document IDs if not provided
        if document_ids is None:
            document_ids = [
                f"doc_{i}_{int(time.time())}" for i in range(len(documents))
            ]

        # Chunk documents using chunking provider
        logger.info(f"Chunking {len(documents)} documents")
        all_chunks = []
        chunk_to_doc_mapping = []

        for i, (doc_id, doc_text) in enumerate(zip(document_ids, documents)):
            doc_metadata = metadata[i] if metadata else {}

            # Chunk the document
            chunked_doc = self._chunking_provider.chunk_document(
                doc_id, doc_text, doc_metadata
            )

            # Collect chunks and maintain mapping
            for chunk_idx, chunk_text in enumerate(chunked_doc.chunks):
                all_chunks.append(chunk_text)
                chunk_to_doc_mapping.append(
                    {
                        "original_doc_id": doc_id,
                        "original_doc_index": i,
                        "chunk_metadata": chunked_doc.chunk_metadata[chunk_idx],
                    }
                )

        logger.info(
            f"Generated {len(all_chunks)} chunks from {len(documents)} documents"
        )

        # Generate embeddings for all chunks
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
        embeddings = self._embedding_provider.batch_embed_documents(all_chunks)

        # Prepare index metadata
        index_metadata = {
            "document_ids": document_ids,
            "documents": dict(zip(document_ids, documents)),  # Store as dict for easy lookup
            "document_metadata": metadata or [{} for _ in documents],
            "chunks": all_chunks,
            "chunk_to_doc_mapping": chunk_to_doc_mapping,
            "chunking_strategy": self._chunking_provider.__class__.__name__,
            "chunking_config": self.config.get("chunking", {}),
            "embedding_provider": self._embedding_provider.__class__.__name__,
            "embedding_config": self.config.get("embedding", {}),
            "index_config": self.config.get("index", {}),
        }

        # Save to storage
        index_name = self.config.get("index", {}).get("name", "sem_simple_index")
        success = self._storage_backend.save_index(
            embeddings, index_metadata, index_name
        )

        processing_time = time.time() - start_time

        result = {
            "success": success,
            "documents_added": len(documents),
            "chunks_generated": len(all_chunks),
            "processing_time": processing_time,
            "index_name": index_name,
            "embedding_dimension": embeddings.shape[1],
        }

        logger.info(
            f"Added {len(documents)} documents ({len(all_chunks)} chunks) in {processing_time:.2f}s"
        )
        return result

    def search(
        self, query: str, top_k: int = 10, similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents

        Args:
            query: Search query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (uses config default if None)

        Returns:
            List of search results with scores and metadata
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if similarity_threshold is None:
            similarity_threshold = self.config.get("index", {}).get(
                "similarity_threshold", 0.0
            )

        start_time = time.time()

        # Load index
        index_name = self.config.get("index", {}).get("name", "sem_simple_index")

        if not self._storage_backend.index_exists(index_name):
            raise ValueError(f"Index '{index_name}' does not exist")

        # Get the device from embedding provider
        embedding_device = getattr(
            self._embedding_provider, "_device", torch.device("cpu")
        )

        vectors, metadata = self._storage_backend.load_index(
            index_name, device=embedding_device
        )

        # Generate query embedding on same device
        query_embedding = self._embedding_provider.embed_query(
            query, device=embedding_device
        )

        # Compute similarities
        similarities = self._compute_similarities(query_embedding, vectors)

        # Get top-k results above threshold
        results = self._rank_results(
            similarities, metadata, top_k, similarity_threshold
        )

        search_time = time.time() - start_time

        logger.info(
            f"Search completed in {search_time:.3f}s, found {len(results)} results"
        )

        # Add search metadata to results
        for result in results:
            result["search_time"] = search_time
            result["query"] = query

        return results

    def _compute_similarities(
        self, query_embedding: torch.Tensor, document_vectors: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarities between query and documents"""
        # Normalize vectors
        query_norm = query_embedding / (query_embedding.norm() + 1e-12)
        doc_norms = document_vectors / (
            document_vectors.norm(dim=1, keepdim=True) + 1e-12
        )

        # Compute cosine similarities using dot product
        similarities = torch.matmul(doc_norms, query_norm)

        return similarities

    def _rank_results(
        self,
        similarities: torch.Tensor,
        metadata: Dict[str, Any],
        top_k: int,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """Rank and filter search results"""
        # Filter by threshold
        valid_indices = similarities >= threshold

        if not valid_indices.any():
            return []

        # Get valid similarities and indices
        valid_similarities = similarities[valid_indices]
        valid_doc_indices = torch.nonzero(valid_indices).squeeze()
        
        # Handle case where squeeze() returns 0-dimensional tensor (single result)
        if valid_doc_indices.dim() == 0:
            valid_doc_indices = valid_doc_indices.unsqueeze(0)

        # Sort by similarity (descending)
        sorted_indices = torch.argsort(valid_similarities, descending=True)

        # Take top-k
        top_indices = sorted_indices[:top_k]

        # Build results
        results = []
        document_ids = metadata.get("document_ids", [])
        documents = metadata.get("documents", [])
        document_metadata = metadata.get("document_metadata", [])
        chunks = metadata.get("chunks", [])
        chunk_to_doc_mapping = metadata.get("chunk_to_doc_mapping", [])

        for i in top_indices:
            chunk_idx = valid_doc_indices[i].item()
            similarity = valid_similarities[i].item()

            # Get the chunk text
            chunk_text = chunks[chunk_idx] if chunk_idx < len(chunks) else ""

            # Map chunk back to original document
            if chunk_idx < len(chunk_to_doc_mapping):
                mapping = chunk_to_doc_mapping[chunk_idx]
                original_doc_id = mapping.get("original_doc_id", f"doc_{chunk_idx}")
                original_doc_index = mapping.get("original_doc_index", 0)
                chunk_metadata = mapping.get("chunk_metadata", {})

                # Get original document text using the document ID
                original_doc_text = documents.get(original_doc_id, "")
                
                # If documents is stored as a dict, we need to use the ID directly
                # If it's empty, try to get it from the document_ids list
                if not original_doc_text and original_doc_index < len(document_ids):
                    doc_id_from_list = document_ids[original_doc_index]
                    original_doc_text = documents.get(doc_id_from_list, "")

                # Get original document metadata
                original_metadata = (
                    document_metadata[original_doc_index]
                    if isinstance(document_metadata, list) and original_doc_index < len(document_metadata)
                    else document_metadata.get(original_doc_id, {}) if isinstance(document_metadata, dict)
                    else {}
                )
            else:
                # Fallback if mapping is missing
                original_doc_id = f"doc_{chunk_idx}"
                original_doc_text = chunk_text
                original_metadata = {}
                chunk_metadata = {}

            result = {
                "document_id": original_doc_id,
                "document": chunk_text,  # Show the matching chunk text
                "original_document": original_doc_text,  # Include full document for reference
                "similarity_score": similarity,
                "metadata": original_metadata,
                "chunk_metadata": chunk_metadata,
                "chunk_index": chunk_idx,
            }

            results.append(result)

        return results

    def get_index_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current index"""
        index_name = self.config.get("index", {}).get("name", "sem_simple_index")
        return self._storage_backend.get_index_info(index_name)

    def list_available_modules(self) -> Dict[str, List[str]]:
        """Get list of available modules by type"""
        return {
            "embeddings": self.registry.get_available_modules("embeddings"),
            "storage": self.registry.get_available_modules("storage"),
            "serialization": self.registry.get_available_modules("serialization"),
            "chunking": self.registry.get_available_modules("chunking"),
        }

    def get_module_capabilities(
        self, module_type: str, module_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get capabilities for a specific module"""
        capabilities = self.registry.get_module_capabilities(module_type, module_name)
        return capabilities.capabilities if capabilities else None

    def switch_embedding_provider(
        self, provider_name: str, config: Dict[str, Any]
    ) -> None:
        """Switch to a different embedding provider"""
        # Update configuration
        self.config["embedding"] = {"provider": provider_name, **config}

        # Instantiate new provider
        self._embedding_provider = self.registry.instantiate_module(
            "embeddings", provider_name, config
        )

        # Unload other providers
        self.registry.unload_all_modules_except("embeddings", provider_name)

        logger.info(f"Switched to embedding provider: {provider_name}")

    def switch_storage_backend(self, backend_name: str, config: Dict[str, Any]) -> None:
        """Switch to a different storage backend"""
        # Update configuration
        self.config["storage"] = {"backend": backend_name, **config}

        # Instantiate new backend
        self._storage_backend = self.registry.instantiate_module(
            "storage", backend_name, config
        )

        # Unload other backends
        self.registry.unload_all_modules_except("storage", backend_name)

        logger.info(f"Switched to storage backend: {backend_name}")

    def __repr__(self) -> str:
        embedding_name = (
            self._embedding_provider.__class__.__name__
            if self._embedding_provider
            else "None"
        )
        storage_name = (
            self._storage_backend.__class__.__name__
            if self._storage_backend
            else "None"
        )

        return (
            f"SEMDatabase(id={self.instance_id[:8]}, "
            f"embedding={embedding_name}, "
            f"storage={storage_name})"
        )

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        # Cleanup can be added here if needed
        pass


# Convenience functions
def create_database(
    config_path: Optional[str] = None, **config_overrides
) -> SEMDatabase:
    """Create a new SEMDatabase instance

    Args:
        config_path: Path to configuration file
        **config_overrides: Configuration overrides

    Returns:
        SEMDatabase instance
    """
    if config_path:
        from .sem_utils import load_config

        config = load_config(config_path)
        config.update(config_overrides)
    else:
        config = config_overrides

    return SEMDatabase(config=config)


def discover_modules() -> Dict[str, List[str]]:
    """Discover all available modules

    Returns:
        Dict mapping module types to available module names
    """
    discover_all_modules()
    from .sem_module_reg import get_available_modules

    return get_available_modules()
