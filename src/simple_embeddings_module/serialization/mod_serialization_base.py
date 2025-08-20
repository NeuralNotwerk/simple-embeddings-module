from abc import ABC, abstractmethod
"""
Base classes for serialization providers

Defines the interface that all serialization providers must implement.
Provides secure serialization with validation and error handling.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from ..sem_module_reg import ConfigParameter


class SerializationProviderBase(ABC):
    """Abstract base class for all serialization providers"""

    # Subclasses should define these class attributes
    CONFIG_PARAMETERS: List[ConfigParameter] = []
    CAPABILITIES: Dict[str, Any] = {}

    def __init__(self, **config):
        """Initialize the serialization provider with validated configuration"""
        self.config = config

    @abstractmethod
    def serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """Serialize PyTorch tensor to bytes

        Args:
            tensor: PyTorch tensor to serialize

        Returns:
            Serialized tensor as bytes
        """
        pass

    @abstractmethod
    def deserialize_tensor(
        self, data: bytes, target_device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Deserialize bytes back to PyTorch tensor

        Args:
            data: Serialized tensor data
            target_device: Target device for deserialized tensor

        Returns:
            PyTorch tensor on specified device
        """
        pass

    @abstractmethod
    def serialize_metadata(self, metadata: Dict[str, Any]) -> bytes:
        """Serialize metadata dictionary to bytes

        Args:
            metadata: Metadata dictionary to serialize

        Returns:
            Serialized metadata as bytes
        """
        pass

    @abstractmethod
    def deserialize_metadata(self, data: bytes) -> Dict[str, Any]:
        """Deserialize bytes back to metadata dictionary

        Args:
            data: Serialized metadata data

        Returns:
            Metadata dictionary
        """
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Return serialization provider capabilities

        Returns:
            Dict containing:
            - format_name: str (e.g., "orjson", "json", "torch_native")
            - is_secure: bool (no binary deserialization)
            - supports_compression: bool
            - human_readable: bool
            - cross_platform: bool
            - max_tensor_size: int (bytes, -1 for unlimited)
        """
        base_capabilities = {
            "format_name": getattr(self, "format_name", "unknown"),
            "is_secure": getattr(self, "is_secure", True),
            "supports_compression": getattr(self, "supports_compression", False),
            "human_readable": getattr(self, "human_readable", False),
            "cross_platform": getattr(self, "cross_platform", True),
            "max_tensor_size": getattr(self, "max_tensor_size", -1),
        }

        # Merge with class-level capabilities
        base_capabilities.update(self.CAPABILITIES)
        return base_capabilities

    def validate_tensor(self, tensor: torch.Tensor) -> None:
        """Validate tensor before serialization"""
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(tensor)}")

        if tensor.numel() == 0:
            raise ValueError("Cannot serialize empty tensor")

        max_size = self.get_capabilities().get("max_tensor_size", -1)
        if max_size > 0:
            tensor_bytes = tensor.numel() * tensor.element_size()
            if tensor_bytes > max_size:
                raise ValueError(
                    f"Tensor size {tensor_bytes} exceeds maximum {max_size}"
                )

    def validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """Validate metadata before serialization"""
        if not isinstance(metadata, dict):
            raise ValueError(f"Expected dict, got {type(metadata)}")

        # Check for JSON-serializable types
        self._validate_json_serializable(metadata)

    def _validate_json_serializable(self, obj: Any, path: str = "root") -> None:
        """Recursively validate that object is JSON serializable"""
        allowed_types = (str, int, float, bool, type(None))

        if isinstance(obj, allowed_types):
            return
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if not isinstance(key, str):
                    raise ValueError(f"Non-string key at {path}.{key}")
                self._validate_json_serializable(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._validate_json_serializable(item, f"{path}[{i}]")
        else:
            raise ValueError(f"Non-JSON-serializable type {type(obj)} at {path}")

    def serialize_index(self, vectors: torch.Tensor, metadata: Dict[str, Any]) -> bytes:
        """Serialize complete index (vectors + metadata) to bytes

        Args:
            vectors: PyTorch tensor of embeddings
            metadata: Index metadata

        Returns:
            Serialized index as bytes
        """
        self.validate_tensor(vectors)
        self.validate_metadata(metadata)

        # Create combined data structure
        index_data = {
            "vectors": vectors,
            "metadata": metadata,
            "serializer": self.__class__.__name__,
            "format_version": "1.0",
        }

        return self._serialize_combined_data(index_data)

    def deserialize_index(
        self, data: bytes, target_device: Optional[torch.device] = None
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Deserialize complete index from bytes

        Args:
            data: Serialized index data
            target_device: Target device for vectors

        Returns:
            Tuple of (vectors, metadata)
        """
        index_data = self._deserialize_combined_data(data)

        # Validate structure
        if not isinstance(index_data, dict):
            raise ValueError("Invalid index data structure")

        if "vectors" not in index_data or "metadata" not in index_data:
            raise ValueError("Missing vectors or metadata in index data")

        # Extract and validate components
        vectors = index_data["vectors"]
        metadata = index_data["metadata"]

        if not isinstance(vectors, torch.Tensor):
            raise ValueError("Vectors must be a torch.Tensor")

        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        # Move vectors to target device if specified
        if target_device is not None:
            vectors = vectors.to(target_device)

        return vectors, metadata

    @abstractmethod
    def _serialize_combined_data(self, data: Dict[str, Any]) -> bytes:
        """Serialize combined data structure (implementation specific)"""
        pass

    @abstractmethod
    def _deserialize_combined_data(self, data: bytes) -> Dict[str, Any]:
        """Deserialize combined data structure (implementation specific)"""
        pass

    def __repr__(self) -> str:
        capabilities = self.get_capabilities()
        return (
            f"{self.__class__.__name__}("
            f"format={capabilities.get('format_name', 'unknown')}, "
            f"secure={capabilities.get('is_secure', False)}, "
            f"readable={capabilities.get('human_readable', False)})"
        )


@dataclass
class SerializationResult:
    """Result container for serialization operations"""

    data: bytes
    size_bytes: int
    format_name: str
    compression_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "size_bytes": self.size_bytes,
            "format_name": self.format_name,
            "compression_ratio": self.compression_ratio,
        }


class SerializationProviderError(Exception):
    """Base exception for serialization provider errors"""

    pass


class SerializationFormatError(SerializationProviderError):
    """Raised when serialization format is invalid or unsupported"""

    pass


class SerializationSecurityError(SerializationProviderError):
    """Raised when serialization poses security risks"""

    pass


class SerializationSizeError(SerializationProviderError):
    """Raised when data exceeds size limits"""

    pass
