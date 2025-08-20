"""
Sentence Transformers Embedding Provider

Implements embedding provider using the sentence-transformers library.
Supports a wide range of pre-trained models with automatic device detection.
"""

import logging
from typing import Any, Dict, List, Optional

import torch

from ..sem_module_reg import ConfigParameter
from .mod_embeddings_base import EmbeddingProviderBase, EmbeddingProviderError

logger = logging.getLogger(__name__)


class SentenceTransformersProvider(EmbeddingProviderBase):
    """Embedding provider using sentence-transformers library"""

    CONFIG_PARAMETERS = [
        ConfigParameter(
            key_name="model",
            value_type="str",
            config_description="HuggingFace model name or local path",
            required=True,
            value_opt_default="all-MiniLM-L6-v2",
        ),
        ConfigParameter(
            key_name="device",
            value_type="str",
            config_description="Device to use (auto, cpu, cuda, mps)",
            required=False,
            value_opt_default="auto",
            value_opt_regex=r"^(auto|cpu|cuda|mps)$",
        ),
        ConfigParameter(
            key_name="batch_size",
            value_type="numeric",
            config_description="Batch size for embedding generation",
            required=False,
            value_opt_default=32,
            value_opt_regex=r"^[1-9]\d*$",
        ),
        ConfigParameter(
            key_name="normalize_embeddings",
            value_type="bool",
            config_description="Whether to L2 normalize embeddings",
            required=False,
            value_opt_default=True,
        ),
        ConfigParameter(
            key_name="trust_remote_code",
            value_type="bool",
            config_description="Whether to trust remote code in models",
            required=False,
            value_opt_default=False,
        ),
    ]

    CAPABILITIES = {
        "supports_batch": True,
        "supports_fp16": True,
        "cross_platform": True,
        "requires_internet": True,  # For downloading models
    }

    def __init__(self, **config):
        """Initialize sentence-transformers provider"""
        super().__init__(**config)

        self.model_name = config.get("model", "all-MiniLM-L6-v2")
        self.device_preference = config.get("device", "auto")
        self.batch_size = config.get("batch_size", 32)
        self.normalize_embeddings = config.get("normalize_embeddings", True)
        self.trust_remote_code = config.get("trust_remote_code", False)

        # Initialize model
        self._model = None
        self._device = None
        self._embedding_dim = None
        self._max_seq_length = None

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise EmbeddingProviderError(
                "sentence-transformers library not installed. "
                "Install with: pip install sentence-transformers"
            )

        # Determine device
        self._device = self._get_device()

        try:
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=str(self._device),
                trust_remote_code=self.trust_remote_code,
            )

            # Get model capabilities
            self._embedding_dim = self._model.get_sentence_embedding_dimension()

            # Get max sequence length (try different methods)
            try:
                self._max_seq_length = self._model.get_max_seq_length()
            except Exception:
                # Fallback for older versions
                try:
                    self._max_seq_length = self._model.max_seq_length
                except Exception:
                    # Default fallback
                    self._max_seq_length = 512

            logger.info(f"Model loaded: {self.model_name}")
            logger.info(f"  Device: {self._device}")
            logger.info(f"  Embedding dimension: {self._embedding_dim}")
            logger.info(f"  Max sequence length: {self._max_seq_length}")

        except Exception as e:
            raise EmbeddingProviderError(
                f"Failed to load model '{self.model_name}': {e}"
            )

    def _get_device(self) -> torch.device:
        """Get the appropriate device for computation"""
        if self.device_preference == "auto":
            # Auto-detect best available device
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.device_preference)

    def embed_documents(
        self, documents: List[str], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generate embeddings for documents"""
        if not documents:
            raise ValueError("Documents list cannot be empty")

        self.validate_inputs(documents)

        try:
            # Generate embeddings
            embeddings = self._model.encode(
                documents,
                batch_size=self.batch_size,
                show_progress_bar=len(documents) > 100,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize_embeddings,
            )

            # Move to target device if specified
            if device is not None and device != embeddings.device:
                embeddings = embeddings.to(device)

            return embeddings

        except Exception as e:
            raise EmbeddingProviderError(f"Failed to generate embeddings: {e}")

    def embed_query(
        self, query: str, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generate embedding for a single query"""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            # Generate embedding
            embedding = self._model.encode(
                [query],
                batch_size=1,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize_embeddings,
            )

            # Return as 1D tensor
            embedding = embedding.squeeze(0)

            # Move to target device if specified
            if device is not None and device != embedding.device:
                embedding = embedding.to(device)

            return embedding

        except Exception as e:
            raise EmbeddingProviderError(f"Failed to generate query embedding: {e}")

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        if self._embedding_dim is None:
            raise EmbeddingProviderError("Model not initialized")
        return self._embedding_dim

    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length"""
        if self._max_seq_length is None:
            raise EmbeddingProviderError("Model not initialized")
        return self._max_seq_length

    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities"""
        base_capabilities = super().get_capabilities()

        # Update with sentence-transformers specific info
        base_capabilities.update(
            {
                "model_name": self.model_name,
                "device_used": str(self._device) if self._device else "unknown",
                "normalization": "l2" if self.normalize_embeddings else "none",
                "tokenizer_type": "sentence_transformers",
                "preferred_batch_size": self.batch_size,
                "memory_requirements_gb": self._estimate_memory_requirements(),
            }
        )

        return base_capabilities

    def _estimate_memory_requirements(self) -> float:
        """Estimate memory requirements in GB"""
        if not self._model:
            return 2.0  # Default estimate

        try:
            # Rough estimate based on model parameters
            param_count = sum(p.numel() for p in self._model.parameters())
            # Assume 4 bytes per parameter (float32) + overhead
            memory_bytes = param_count * 4 * 1.5  # 50% overhead
            return memory_bytes / (1024**3)  # Convert to GB
        except Exception:
            return 2.0  # Fallback estimate

    def validate_inputs(self, texts: List[str]) -> None:
        """Validate input texts"""
        super().validate_inputs(texts)

        # Additional sentence-transformers specific validation
        for i, text in enumerate(texts):
            if len(text.strip()) == 0:
                raise ValueError(f"Document {i} is empty or whitespace only")

    def __repr__(self) -> str:
        return (
            "SentenceTransformersProvider("
            f"model={self.model_name}, "
            f"device={self._device}, "
            f"dim={self._embedding_dim})"
        )


# Register common model configurations
COMMON_MODELS = {
    "all-MiniLM-L6-v2": {
        "embedding_dim": 384,
        "max_seq_length": 256,
        "description": "Fast and efficient, good for most tasks",
    },
    "all-mpnet-base-v2": {
        "embedding_dim": 768,
        "max_seq_length": 384,
        "description": "Higher quality, slower",
    },
    "all-distilroberta-v1": {
        "embedding_dim": 768,
        "max_seq_length": 512,
        "description": "Good balance of speed and quality",
    },
}
