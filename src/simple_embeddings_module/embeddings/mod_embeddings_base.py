from abc import ABC, abstractmethod
"""
Base classes for embedding providers

Defines the interface that all embedding providers must implement.
Provides capability negotiation and configuration validation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from ..sem_module_reg import ConfigParameter


class EmbeddingProviderBase(ABC):
    """Abstract base class for all embedding providers"""

    # Subclasses should define these class attributes
    CONFIG_PARAMETERS: List[ConfigParameter] = []
    CAPABILITIES: Dict[str, Any] = {}

    def __init__(self, **config):
        """Initialize the embedding provider with validated configuration"""
        self.config = config
        self._device = None
        self._model = None

    @abstractmethod
    def embed_documents(
        self, documents: List[str], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generate embeddings for a list of documents

        Args:
            documents: List of document strings to embed
            device: Target device for computation (auto-detected if None)

        Returns:
            torch.Tensor of shape (n_docs, embedding_dim) on specified device
        """
        pass

    @abstractmethod
    def embed_query(
        self, query: str, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generate embedding for a single query

        Args:
            query: Query string to embed
            device: Target device for computation (auto-detected if None)

        Returns:
            torch.Tensor of shape (embedding_dim,) on specified device
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this provider"""
        pass

    @abstractmethod
    def get_max_sequence_length(self) -> int:
        """Get the maximum sequence length supported by this provider"""
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities and configuration

        Returns:
            Dict containing:
            - embedding_dim: int
            - max_sequence_length: int
            - model_name: str
            - supports_batch: bool
            - preferred_batch_size: int
            - device_preferences: List[str] (['cuda', 'mps', 'cpu'])
            - memory_requirements_gb: float
            - supports_fp16: bool
        """
        base_capabilities = {
            "embedding_dim": self.get_embedding_dimension(),
            "max_sequence_length": self.get_max_sequence_length(),
            "model_name": getattr(self, "model_name", "unknown"),
            "supports_batch": True,
            "preferred_batch_size": getattr(self, "preferred_batch_size", 32),
            "device_preferences": getattr(
                self, "device_preferences", ["cuda", "mps", "cpu"]
            ),
            "memory_requirements_gb": getattr(self, "memory_requirements_gb", 2.0),
            "supports_fp16": getattr(self, "supports_fp16", False),
        }

        # Merge with class-level capabilities
        base_capabilities.update(self.CAPABILITIES)
        return base_capabilities

    def get_preferred_device(self, available_device: torch.device) -> torch.device:
        """Get preferred device given available hardware"""
        preferences = self.get_capabilities().get("device_preferences", ["cpu"])

        if available_device.type in preferences:
            return available_device
        elif "cpu" in preferences:
            return torch.device("cpu")
        else:
            return available_device  # Fallback

    def validate_inputs(self, texts: List[str]) -> None:
        """Validate input texts before processing"""
        if not texts:
            raise ValueError("Input text list cannot be empty")

        max_length = self.get_max_sequence_length()
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Input {i} must be a string, got {type(text)}")
            if len(text) > max_length:
                raise ValueError(
                    f"Input {i} exceeds max length {max_length}: {len(text)} characters"
                )

    def batch_embed_documents(
        self,
        documents: List[str],
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Embed documents in batches for memory efficiency

        Args:
            documents: List of documents to embed
            batch_size: Batch size (uses preferred_batch_size if None)
            device: Target device for computation

        Returns:
            torch.Tensor of shape (n_docs, embedding_dim)
        """
        if batch_size is None:
            batch_size = self.get_capabilities().get("preferred_batch_size", 32)

        self.validate_inputs(documents)

        if len(documents) <= batch_size:
            return self.embed_documents(documents, device)

        # Process in batches
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_embeddings = self.embed_documents(batch, device)
            all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)

    def __repr__(self) -> str:
        capabilities = self.get_capabilities()
        return (
            f"{self.__class__.__name__}("
            f"model={capabilities.get('model_name', 'unknown')}, "
            f"dim={capabilities.get('embedding_dim', 'unknown')}, "
            f"max_len={capabilities.get('max_sequence_length', 'unknown')})"
        )


@dataclass
class EmbeddingResult:
    """Result container for embedding operations"""

    embeddings: torch.Tensor
    metadata: Dict[str, Any]
    processing_time: float
    device_used: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "embeddings_shape": list(self.embeddings.shape),
            "embeddings_dtype": str(self.embeddings.dtype),
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "device_used": self.device_used,
        }


class EmbeddingProviderError(Exception):
    """Base exception for embedding provider errors"""

    pass


class EmbeddingConfigurationError(EmbeddingProviderError):
    """Raised when embedding provider configuration is invalid"""

    pass


class EmbeddingModelError(EmbeddingProviderError):
    """Raised when there's an error with the embedding model"""

    pass


class EmbeddingInputError(EmbeddingProviderError):
    """Raised when input validation fails"""

    pass
