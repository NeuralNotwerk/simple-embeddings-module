from abc import ABC, abstractmethod
"""
Base classes for chunking providers

Chunking strategies must be configured based on embedding provider capabilities.
This creates a dependency chain: Embedding Provider → Chunking Strategy → Index Structure
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..embeddings.mod_embeddings_base import EmbeddingProviderBase
from ..sem_module_reg import ConfigParameter


class ChunkBoundary(Enum):
    """Types of chunk boundaries"""

    CHARACTER = "character"
    WORD = "word"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SECTION = "section"
    SEMANTIC = "semantic"


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk"""

    chunk_id: str
    document_id: str
    chunk_index: int
    start_position: int
    end_position: int
    chunk_type: str
    parent_chunk_id: Optional[str] = None
    overlap_with: List[str] = None

    def __post_init__(self):
        if self.overlap_with is None:
            self.overlap_with = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "chunk_type": self.chunk_type,
            "parent_chunk_id": self.parent_chunk_id,
            "overlap_with": self.overlap_with,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        """Create from dictionary for JSON deserialization"""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            chunk_index=data["chunk_index"],
            start_position=data["start_position"],
            end_position=data["end_position"],
            chunk_type=data["chunk_type"],
            parent_chunk_id=data.get("parent_chunk_id"),
            overlap_with=data.get("overlap_with", []),
        )


@dataclass
class ChunkedDocument:
    """Result of chunking a document"""

    document_id: str
    original_text: str
    chunks: List[str]
    chunk_metadata: List[ChunkMetadata]
    chunking_strategy: str
    embedding_constraints: Dict[str, Any]

    def validate(self) -> bool:
        """Validate that chunks and metadata are consistent"""
        return len(self.chunks) == len(self.chunk_metadata)


class ChunkingProviderBase(ABC):
    """Abstract base class for all chunking providers"""

    # Subclasses should define these class attributes
    CONFIG_PARAMETERS: List[ConfigParameter] = []
    CAPABILITIES: Dict[str, Any] = {}

    def __init__(self, embedding_provider: EmbeddingProviderBase, **config):
        """Initialize chunking provider with embedding provider constraints

        Args:
            embedding_provider: The embedding provider that will process chunks
            **config: Validated configuration parameters
        """
        self.embedding_provider = embedding_provider
        self.config = config

        # Get embedding constraints
        self.embedding_capabilities = embedding_provider.get_capabilities()
        self.max_sequence_length = self.embedding_capabilities["max_sequence_length"]
        self.embedding_dim = self.embedding_capabilities["embedding_dim"]

        # Configure chunking based on embedding constraints
        self._configure_for_embedding_provider()

    @abstractmethod
    def _configure_for_embedding_provider(self) -> None:
        """Configure chunking parameters based on embedding provider capabilities"""
        pass

    @abstractmethod
    def chunk_document(
        self, document_id: str, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ChunkedDocument:
        """Chunk a single document into embedding-ready pieces

        Args:
            document_id: Unique identifier for the document
            text: Document text to chunk
            metadata: Optional document metadata

        Returns:
            ChunkedDocument with chunks and metadata
        """
        pass

    @abstractmethod
    def chunk_query(self, query: str) -> List[str]:
        """Chunk a query if it exceeds embedding provider limits

        Args:
            query: Query text to potentially chunk

        Returns:
            List of query chunks (single item if no chunking needed)
        """
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Get chunking provider capabilities

        Returns:
            Dict containing:
            - max_chunk_size: int (characters)
            - min_chunk_size: int (characters)
            - overlap_size: int (characters)
            - boundary_type: str
            - supports_metadata: bool
            - preserves_structure: bool
            - embedding_constraints: Dict
        """
        base_capabilities = {
            "max_chunk_size": getattr(self, "max_chunk_size", self.max_sequence_length),
            "min_chunk_size": getattr(self, "min_chunk_size", 100),
            "overlap_size": getattr(self, "overlap_size", 0),
            "boundary_type": getattr(
                self, "boundary_type", ChunkBoundary.SENTENCE.value
            ),
            "supports_metadata": getattr(self, "supports_metadata", True),
            "preserves_structure": getattr(self, "preserves_structure", False),
            "embedding_constraints": self.embedding_capabilities,
        }

        # Merge with class-level capabilities
        base_capabilities.update(self.CAPABILITIES)
        return base_capabilities

    def validate_chunk_size(self, chunk: str) -> bool:
        """Validate that a chunk meets embedding provider constraints"""
        # Check length constraints
        if len(chunk) > self.max_sequence_length:
            return False

        capabilities = self.get_capabilities()
        min_size = capabilities.get("min_chunk_size", 0)
        if len(chunk) < min_size:
            return False

        return True

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        # Simple heuristic: ~4 characters per token for most languages
        return len(text) // 4

    def is_compatible_with_embedding_provider(
        self, other_provider: EmbeddingProviderBase
    ) -> Tuple[bool, List[str]]:
        """Check if this chunking strategy is compatible with another embedding provider

        ⚠️  BEST EFFORT COMPATIBILITY: Even 'compatible' providers may produce different results!

        STRICT COMPATIBILITY RULES:
        ✅ Compatible: Same embedding dimensions, similar max_length (±20%), IDENTICAL MODEL (except quantization)
        ❌ Incompatible: Different dimensions, different limits, different model family, different model size

        IMPORTANT: Hardware differences (NVIDIA vs AMD vs Apple Silicon), RNG seeds, numerical precision,
        and implementation differences can cause result variations even with 'compatible' providers.
        Always validate results after switching!

        Args:
            other_provider: Another embedding provider to check compatibility with

        Returns:
            Tuple of (is_compatible, list_of_reasons_and_warnings)
        """
        reasons = []
        other_capabilities = other_provider.get_capabilities()
        current_capabilities = self.embedding_capabilities

        # Add best effort warning upfront
        reasons.append(
            "⚠️  BEST EFFORT: 'Compatible' providers may still produce different results due to:"
        )
        reasons.append("   • Hardware differences (NVIDIA vs AMD vs Apple Silicon)")
        reasons.append("   • RNG seed variations and numerical precision differences")
        reasons.append("   • Different CUDA/ROCm/MPS kernel implementations")
        reasons.append("   • Quantization method variations and rounding differences")
        reasons.append("   • Model loading and initialization differences")
        reasons.append("   ➤ ALWAYS validate search results after switching providers!")
        reasons.append("")  # Blank line for readability

        # 1. Check embedding dimensions (MUST be identical)
        if other_capabilities["embedding_dim"] != current_capabilities["embedding_dim"]:
            reasons.append(
                f"Embedding dimensions differ: {current_capabilities['embedding_dim']} vs {other_capabilities['embedding_dim']}"
            )

        # 2. Check sequence length compatibility (allow 20% variance)
        other_max_length = other_capabilities["max_sequence_length"]
        current_max_length = current_capabilities["max_sequence_length"]
        length_ratio = abs(other_max_length - current_max_length) / current_max_length
        if length_ratio > 0.2:
            reasons.append(
                f"Max sequence length differs significantly: {current_max_length} vs {other_max_length} ({length_ratio:.1%} difference)"
            )

        # 3. STRICT MODEL COMPATIBILITY CHECK
        current_model = current_capabilities.get("model_name", "").lower()
        other_model = other_capabilities.get("model_name", "").lower()

        if not self._are_models_compatible(current_model, other_model):
            reasons.append(
                f"Models are not compatible: '{current_capabilities.get('model_name')}' vs '{other_capabilities.get('model_name')}'"
            )

        # 4. Check if current chunks would fit in new provider
        current_chunk_capabilities = self.get_capabilities()
        if current_chunk_capabilities["max_chunk_size"] > other_max_length:
            reasons.append(
                f"Current chunks too large for new provider: {current_chunk_capabilities['max_chunk_size']} > {other_max_length}"
            )

        # 5. Check tokenization compatibility (if available)
        current_tokenizer = current_capabilities.get("tokenizer_type")
        other_tokenizer = other_capabilities.get("tokenizer_type")
        if (
            current_tokenizer
            and other_tokenizer
            and current_tokenizer != other_tokenizer
        ):
            reasons.append(
                f"Tokenizer types differ: {current_tokenizer} vs {other_tokenizer}"
            )

        return len(reasons) == 0, reasons

    def _are_models_compatible(self, model1: str, model2: str) -> bool:
        """Check if two models are compatible (identical model, possibly different quantization)

        Args:
            model1: First model name (normalized to lowercase)
            model2: Second model name (normalized to lowercase)

        Returns:
            True if models are compatible
        """
        if model1 == model2:
            return True

        # Normalize model names for comparison
        normalized1 = self._normalize_model_name(model1)
        normalized2 = self._normalize_model_name(model2)

        if normalized1 == normalized2:
            return True

        # Check for quantization variants of the same model
        return self._are_quantization_variants(normalized1, normalized2)

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name for compatibility checking"""
        # Remove common prefixes/suffixes
        normalized = model_name.lower().strip()

        # Remove organization prefixes (huggingface style)
        if "/" in normalized:
            normalized = normalized.split("/")[-1]

        # Remove common suffixes that don't affect compatibility
        suffixes_to_remove = [
            "-uncased",
            "-cased",
            "-base",
            "-large",
            "-small",
            "-v1",
            "-v2",
            "-v3",
            ".bin",
            ".safetensors",
        ]

        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]

        return normalized

    def _are_quantization_variants(self, model1: str, model2: str) -> bool:
        """Check if models are quantization variants of each other

        Quantization variants are considered compatible as they represent
        the same model with different precision/compression.
        """
        # Common quantization indicators
        quant_indicators = [
            "q4",
            "q8",
            "q16",
            "int4",
            "int8",
            "int16",
            "fp16",
            "fp32",
            "bf16",
            "ggml",
            "ggu",
            "awq",
            "gptq",
            "bnb",
            "bitsandbytes",
        ]

        # Remove quantization indicators from both models
        clean1 = model1
        clean2 = model2

        for indicator in quant_indicators:
            clean1 = (
                clean1.replace(f"-{indicator}", "")
                .replace(f"_{indicator}", "")
                .replace(indicator, "")
            )
            clean2 = (
                clean2.replace(f"-{indicator}", "")
                .replace(f"_{indicator}", "")
                .replace(indicator, "")
            )

        # Clean up any double separators
        clean1 = clean1.replace("--", "-").replace("__", "_").strip("-_")
        clean2 = clean2.replace("--", "-").replace("__", "_").strip("-_")

        return clean1 == clean2

    def batch_chunk_documents(
        self,
        documents: List[Tuple[str, str]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ChunkedDocument]:
        """Chunk multiple documents efficiently

        Args:
            documents: List of (document_id, text) tuples
            metadata_list: Optional list of metadata dicts per document

        Returns:
            List of ChunkedDocument objects
        """
        if metadata_list is None:
            metadata_list = [{}] * len(documents)

        if len(metadata_list) != len(documents):
            raise ValueError("metadata_list length must match documents length")

        results = []
        for (doc_id, text), metadata in zip(documents, metadata_list):
            chunked_doc = self.chunk_document(doc_id, text, metadata)
            results.append(chunked_doc)

        return results

    def __repr__(self) -> str:
        capabilities = self.get_capabilities()
        return (
            f"{self.__class__.__name__}("
            f"max_chunk={capabilities.get('max_chunk_size', 'unknown')}, "
            f"boundary={capabilities.get('boundary_type', 'unknown')}, "
            f"embedding_dim={self.embedding_dim})"
        )


class ChunkingProviderError(Exception):
    """Base exception for chunking provider errors"""

    pass


class ChunkingConfigurationError(ChunkingProviderError):
    """Raised when chunking provider configuration is invalid"""

    pass


class ChunkingCompatibilityError(ChunkingProviderError):
    """Raised when chunking strategy is incompatible with embedding provider"""

    pass


class ChunkingSizeError(ChunkingProviderError):
    """Raised when chunks exceed embedding provider limits"""

    pass
