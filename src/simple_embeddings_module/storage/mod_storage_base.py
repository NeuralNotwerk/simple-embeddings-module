from abc import ABC, abstractmethod

"""
Base classes for storage backends
Defines the interface that all storage backends must implement.
Provides capability negotiation and configuration validation.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..sem_module_reg import ConfigParameter


class StorageBackendBase(ABC):
    """Abstract base class for all storage backends"""
    # Subclasses should define these class attributes
    CONFIG_PARAMETERS: List[ConfigParameter] = []
    CAPABILITIES: Dict[str, Any] = {}

    def __init__(self, **config):
        """Initialize the storage backend with validated configuration"""
        self.config = config

    @abstractmethod
    def save_index(self, vectors: torch.Tensor, metadata: Dict[str, Any], index_name: str) -> bool:
        """Save vector index and metadata
        Args:
            vectors: torch.Tensor of embeddings (will be moved to CPU for storage)
            metadata: document metadata and index configuration
            index_name: unique identifier for this index
        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def load_index(self, index_name: str, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load vector index and metadata
        Args:
            index_name: unique identifier for index
            device: target device for loaded vectors (CPU if None)
        Returns:
            Tuple of (vectors on device, metadata)
        """
        pass

    @abstractmethod
    def list_indexes(self) -> List[str]:
        """List available indexes"""
        pass

    @abstractmethod
    def delete_index(self, index_name: str) -> bool:
        """Delete an index
        Args:
            index_name: unique identifier for index to delete
        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists
        Args:
            index_name: unique identifier for index
        Returns:
            True if index exists
        """
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Return storage backend capabilities
        Returns:
            Dict containing:
            - max_index_size: int (bytes, -1 for unlimited)
            - supports_streaming: bool
            - supports_partial_updates: bool
            - supports_concurrent_access: bool
            - connection_info: Dict (for cloud storage)
            - compression_supported: bool
            - encryption_supported: bool
        """
        base_capabilities = {
            "max_index_size": getattr(self, "max_index_size", -1),
            "supports_streaming": getattr(self, "supports_streaming", False),
            "supports_partial_updates": getattr(self, "supports_partial_updates", False),
            "supports_concurrent_access": getattr(self, "supports_concurrent_access", False),
            "connection_info": getattr(self, "connection_info", {}),
            "compression_supported": getattr(self, "compression_supported", True),
            "encryption_supported": getattr(self, "encryption_supported", False),
        }
        # Merge with class-level capabilities
        base_capabilities.update(self.CAPABILITIES)
        return base_capabilities

    def get_index_info(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an index without loading it
        Args:
            index_name: unique identifier for index
        Returns:
            Dict with index information or None if not found
        """
        if not self.index_exists(index_name):
            return None
        try:
            # Load just metadata without vectors
            _, metadata = self.load_index(index_name)
            # Extract model name from embedding config if not directly available
            model_name = metadata.get("model_name")
            if not model_name:
                embedding_config = metadata.get("embedding_config", {})
                model_name = embedding_config.get("model", embedding_config.get("model_name"))
            return {
                "index_name": index_name,
                "created_at": metadata.get("created_at"),
                "updated_at": metadata.get("updated_at"),
                "document_count": metadata.get("document_count", 0),
                "embedding_dim": metadata.get("embedding_dim"),
                "model_name": model_name,
                "size_bytes": self._get_index_size(index_name),
            }
        except Exception:
            return None

    def _get_index_size(self, index_name: str) -> Optional[int]:
        """Get the size of an index in bytes (implementation specific)"""
        return None  # Override in subclasses

    def validate_index_name(self, index_name: str) -> None:
        """Validate index name format"""
        if not index_name:
            raise ValueError("Index name cannot be empty")
        if not isinstance(index_name, str):
            raise ValueError("Index name must be a string")
        # Check for valid characters (alphanumeric, underscore, hyphen)
        import re
        if not re.match(r"^[a-zA-Z0-9_-]+$", index_name):
            raise ValueError("Index name can only contain alphanumeric characters, underscores, and hyphens")
        if len(index_name) > 255:
            raise ValueError("Index name cannot exceed 255 characters")

    def prepare_metadata(self, metadata: Dict[str, Any], vectors: torch.Tensor) -> Dict[str, Any]:
        """Prepare metadata for storage with standard fields"""
        prepared = metadata.copy()
        # Add standard metadata fields
        now = datetime.utcnow().isoformat()
        prepared.update(
            {
                "created_at": prepared.get("created_at", now),
                "updated_at": now,
                "embedding_dim": (vectors.shape[1] if len(vectors.shape) > 1 else vectors.shape[0]),
                "document_count": vectors.shape[0] if len(vectors.shape) > 1 else 1,
                "storage_backend": self.__class__.__name__,
                "format_version": "1.0",
            }
        )
        return prepared

    def __repr__(self) -> str:
        capabilities = self.get_capabilities() # is this duplication of hardcoded values below?
        return (
            f"{self.__class__.__name__}("
            f"streaming={capabilities.get('supports_streaming', False)}, "
            f"concurrent={capabilities.get('supports_concurrent_access', False)})"
        )


@dataclass
class IndexInfo:
    """Information about a stored index"""
    index_name: str
    created_at: Optional[str]
    updated_at: Optional[str]
    document_count: int
    embedding_dim: int
    model_name: Optional[str]
    size_bytes: Optional[int]
    backend_type: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "index_name": self.index_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "document_count": self.document_count,
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "size_bytes": self.size_bytes,
            "backend_type": self.backend_type,
        }

class StorageBackendError(Exception):
    """Base exception for storage backend errors"""
    pass

class StorageConfigurationError(StorageBackendError):
    """Raised when storage backend configuration is invalid"""
    pass

class StorageConnectionError(StorageBackendError):
    """Raised when connection to storage backend fails"""
    pass

class StorageIndexError(StorageBackendError):
    """Raised when index operations fail"""
    pass

class StoragePermissionError(StorageBackendError):
    """Raised when storage permissions are insufficient"""
    pass
