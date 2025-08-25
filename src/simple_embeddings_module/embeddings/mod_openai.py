"""
OpenAI Embeddings Provider
Provides access to OpenAI's embedding models via their API.
Supports text-embedding-3-small, text-embedding-3-large, and text-embedding-ada-002.
"""
import logging
import time
from typing import Any, Dict, List, Optional

import torch

from ..sem_module_reg import ConfigParameter
from .mod_embeddings_base import EmbeddingProviderBase, EmbeddingProviderError

logger = logging.getLogger(__name__)
class OpenAIEmbeddingProvider(EmbeddingProviderBase):
    """OpenAI embeddings provider with API rate limiting and retry logic"""
    CONFIG_PARAMETERS = [
        ConfigParameter(
            key_name="api_key",
            value_type="str",
            config_description="OpenAI API key (or set OPENAI_API_KEY env var)",
            required=False,  # Can use env var
            value_opt_default=None,
        ),
        ConfigParameter(
            key_name="model",
            value_type="str",
            config_description="OpenAI embedding model name",
            required=False,
            value_opt_default="text-embedding-3-small",
            value_opt_regex=r"^(text-embedding-3-small|text-embedding-3-large|text-embedding-ada-002)$",
        ),
        ConfigParameter(
            key_name="batch_size",
            value_type="numeric",
            config_description="Number of texts to embed in one API call",
            required=False,
            value_opt_default=100,
            value_opt_regex=r"^[1-9][0-9]*$",
        ),
        ConfigParameter(
            key_name="max_retries",
            value_type="numeric",
            config_description="Maximum number of API retry attempts",
            required=False,
            value_opt_default=3,
            value_opt_regex=r"^[0-9]$",
        ),
        ConfigParameter(
            key_name="timeout",
            value_type="numeric",
            config_description="API request timeout in seconds",
            required=False,
            value_opt_default=30,
            value_opt_regex=r"^[1-9][0-9]*$",
        ),
        ConfigParameter(
            key_name="dimensions",
            value_type="numeric",
            config_description="Output dimensions (only for text-embedding-3 models)",
            required=False,
            value_opt_default=None,
            value_opt_regex=r"^[1-9][0-9]*$",
        ),
    ]
    CAPABILITIES = {
        "embedding_dimension": None,  # Set dynamically based on model
        "max_sequence_length": 8192,  # OpenAI's current limit
        "supports_batching": True,
        "requires_api_key": True,
        "supports_custom_dimensions": True,  # For text-embedding-3 models
        "rate_limited": True,
        "tokenizer_type": "openai",
    }
    def __init__(self, **config):
        """Initialize OpenAI embedding provider"""
        super().__init__(**config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "text-embedding-3-small")
        self.batch_size = config.get("batch_size", 100)
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 30)
        self.dimensions = config.get("dimensions")
        # Initialize OpenAI client
        self._init_client()
        # Set model-specific capabilities
        self._set_model_capabilities()
        logger.info("OpenAI embedding provider initialized: %s", self.model)
    def _init_client(self):
        """Initialize OpenAI client with API key"""
        try:
            import openai
        except ImportError as exc:
            raise EmbeddingProviderError("OpenAI library not installed. Install with: pip install openai") from exc
        # Get API key from config or environment
        if self.api_key:
            api_key = self.api_key
        else:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EmbeddingProviderError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                    "or provide api_key in configuration."
                )
        # Initialize client
        self.client = openai.OpenAI(api_key=api_key, timeout=self.timeout)
        logger.info("OpenAI client initialized successfully")
    def _set_model_capabilities(self):
        """Set capabilities based on the selected model"""
        model_specs = {
            "text-embedding-3-small": {
                "embedding_dimension": 1536,
                "max_sequence_length": 8192,
                "supports_custom_dimensions": True,
            },
            "text-embedding-3-large": {
                "embedding_dimension": 3072,
                "max_sequence_length": 8192,
                "supports_custom_dimensions": True,
            },
            "text-embedding-ada-002": {
                "embedding_dimension": 1536,
                "max_sequence_length": 8192,
                "supports_custom_dimensions": False,
            },
        }
        if self.model not in model_specs:
            raise EmbeddingProviderError("Unsupported OpenAI model: %s" % self.model)
        specs = model_specs[self.model]
        # Override default dimensions if custom dimensions specified
        if self.dimensions and specs["supports_custom_dimensions"]:
            self.CAPABILITIES["embedding_dimension"] = self.dimensions
        else:
            self.CAPABILITIES["embedding_dimension"] = specs["embedding_dimension"]
        self.CAPABILITIES["max_sequence_length"] = specs["max_sequence_length"]
        self.CAPABILITIES["supports_custom_dimensions"] = specs["supports_custom_dimensions"]
        logger.info(
            "Model capabilities set: %s dimensions, %s max tokens",
            self.CAPABILITIES['embedding_dimension'],
            self.CAPABILITIES['max_sequence_length']
        )
    def embed_documents(self, documents: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate embeddings for multiple documents"""
        if not documents:
            raise ValueError("No documents provided for embedding")
        logger.info("Generating embeddings for %d documents", len(documents))
        start_time = time.time()
        all_embeddings = []
        # Process in batches to respect API limits
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
            # Add small delay between batches to respect rate limits
            if i + self.batch_size < len(documents):
                time.sleep(0.1)
        # Convert to tensor
        embeddings_tensor = torch.tensor(all_embeddings, dtype=torch.float32)
        # Move to specified device
        if device is not None:
            embeddings_tensor = embeddings_tensor.to(device)
        elapsed = time.time() - start_time
        logger.info("Generated %d embeddings in %.2fs (%.1f docs/sec)",
                   len(documents), elapsed, len(documents)/elapsed)
        return embeddings_tensor
    def embed_query(self, query: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate embedding for a single query"""
        if not query.strip():
            raise ValueError("Query cannot be empty")
        embeddings = self._embed_batch([query])
        embedding_tensor = torch.tensor(embeddings[0], dtype=torch.float32)
        if device is not None:
            embedding_tensor = embedding_tensor.to(device)
        return embedding_tensor
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts with retry logic"""
        for attempt in range(self.max_retries + 1):
            try:
                # Prepare API call parameters
                params = {
                    "input": texts,
                    "model": self.model,
                }
                # Add dimensions parameter for text-embedding-3 models
                if self.dimensions and self.CAPABILITIES.get("supports_custom_dimensions", False):
                    params["dimensions"] = self.dimensions
                # Make API call
                response = self.client.embeddings.create(**params)
                # Extract embeddings
                embeddings = [item.embedding for item in response.data]
                logger.debug("Successfully embedded batch of %d texts", len(texts))
                return embeddings
            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        "OpenAI API error (attempt %s/%s): %s. Retrying in %ss...",
                        attempt + 1, self.max_retries + 1, e, wait_time
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("OpenAI API failed after %d attempts: %s", self.max_retries + 1, e)
                    raise EmbeddingProviderError("OpenAI API error: %s" % e) from e
        # This should never be reached, but added for pylint
        raise EmbeddingProviderError("Unexpected error in _embed_batch")
    def get_capabilities(self) -> Dict[str, Any]:
        """Return provider capabilities"""
        capabilities = super().get_capabilities()
        capabilities.update(
            {
                "model_name": self.model,
                "api_based": True,
                "requires_internet": True,
                "cost_per_1k_tokens": self._get_cost_estimate(),
                "supports_batch_processing": True,
            }
        )
        return capabilities
    def _get_cost_estimate(self) -> float:
        """Get estimated cost per 1K tokens (as of 2024)"""
        costs = {
            "text-embedding-3-small": 0.00002,  # $0.00002 per 1K tokens
            "text-embedding-3-large": 0.00013,  # $0.00013 per 1K tokens
            "text-embedding-ada-002": 0.0001,  # $0.0001 per 1K tokens
        }
        return costs.get(self.model, 0.0001)
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        # Rough approximation: ~4 characters per token for English
        return len(text) // 4
    def validate_text_length(self, text: str) -> bool:
        """Validate that text is within model limits"""
        estimated_tokens = self.estimate_tokens(text)
        max_tokens = self.CAPABILITIES.get("max_sequence_length", 8192)
        return estimated_tokens <= max_tokens
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"OpenAIEmbeddingProvider("
            f"model={self.model}, "
            f"dimensions={self.CAPABILITIES.get('embedding_dimension')}, "
            f"batch_size={self.batch_size})"
        )
