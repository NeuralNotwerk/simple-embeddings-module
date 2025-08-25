#!/usr/bin/env python3
"""
AWS Bedrock embedding provider for Simple Embeddings Module.

This module provides dynamic model capability detection and supports
all Bedrock embedding models with automatic dimension discovery.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import boto3
import torch
from botocore.exceptions import ClientError

from .mod_embeddings_base import EmbeddingProviderBase, EmbeddingProviderError

logger = logging.getLogger(__name__)

@dataclass
class BedrockModelInfo:
    """Information about a Bedrock embedding model."""
    model_id: str
    embedding_dimensions: int
    max_input_tokens: int
    max_batch_size: int
    provider: str
    model_name: str

class BedrockEmbeddingProvider(EmbeddingProviderBase):
    """
    AWS Bedrock embedding provider with dynamic model capability detection.

    Supports all Bedrock embedding models:
    - Amazon Titan Text Embeddings v1/v2
    - Cohere Embed models
    - Future embedding models automatically
    """

    # Known model configurations (fallback if API detection fails)
    KNOWN_MODELS = {
        "amazon.titan-embed-text-v1": BedrockModelInfo(
            model_id="amazon.titan-embed-text-v1",
            embedding_dimensions=1536,
            max_input_tokens=8000,
            max_batch_size=25,
            provider="amazon",
            model_name="Titan Text Embeddings v1"
        ),
        "amazon.titan-embed-text-v2:0": BedrockModelInfo(
            model_id="amazon.titan-embed-text-v2:0",
            embedding_dimensions=1024,
            max_input_tokens=8192,
            max_batch_size=25,
            provider="amazon",
            model_name="Titan Text Embeddings v2"
        ),
        "cohere.embed-english-v3": BedrockModelInfo(
            model_id="cohere.embed-english-v3",
            embedding_dimensions=1024,
            max_input_tokens=512,
            max_batch_size=96,
            provider="cohere",
            model_name="Cohere Embed English v3"
        ),
        "cohere.embed-multilingual-v3": BedrockModelInfo(
            model_id="cohere.embed-multilingual-v3",
            embedding_dimensions=1024,
            max_input_tokens=512,
            max_batch_size=96,
            provider="cohere",
            model_name="Cohere Embed Multilingual v3"
        )
    }

    def __init__(self,
                 model_id: str = "amazon.titan-embed-text-v2:0",
                 region: str = "us-east-1",
                 aws_profile: Optional[str] = None,
                 max_retries: int = 3,
                 **config):
        """
        Initialize Bedrock embedding provider with dynamic model detection.

        Args:
            model_id: Bedrock model identifier
            region: AWS region for Bedrock service
            aws_profile: AWS profile name (optional)
            max_retries: Maximum retry attempts for API calls
            **config: Additional configuration parameters
        """
        super().__init__(**config)

        self.model_id = model_id
        self.region = region
        self.aws_profile = aws_profile
        self.max_retries = max_retries

        # Initialize AWS clients
        try:
            session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
            self.bedrock_client = session.client('bedrock', region_name=region)
            self.bedrock_runtime = session.client('bedrock-runtime', region_name=region)
        except Exception as e:
            raise EmbeddingProviderError("Failed to initialize AWS clients: %s" % e) from e

        # Detect model capabilities
        self.model_info = self._detect_model_capabilities()

        # Set base class attributes
        self.model_name = self.model_info.model_name
        self.preferred_batch_size = min(self.model_info.max_batch_size, 32)
        self.device_preferences = ["cpu"]  # Bedrock is cloud-based
        self.memory_requirements_gb = 0.1  # Minimal local memory
        self.supports_fp16 = False  # Cloud service handles precision

        logger.info("Initialized Bedrock provider: %s", self.model_info.model_name)
        logger.info("  • Model ID: %s", self.model_info.model_id)
        logger.info("  • Dimensions: %s", self.model_info.embedding_dimensions)
        logger.info("  • Max tokens: %s", self.model_info.max_input_tokens)
        logger.info("  • Provider: %s", self.model_info.provider)

    def _detect_model_capabilities(self) -> BedrockModelInfo:
        """
        Dynamically detect model capabilities from Bedrock API.

        Returns:
            BedrockModelInfo with detected or fallback capabilities
        """
        try:
            # First, try to get model info from Bedrock API
            logger.info("Detecting capabilities for model: %s", self.model_id)

            # Method 1: Try to get model details from foundation models API
            model_info = self._get_model_info_from_api()
            if model_info:
                return model_info

            # Method 2: Try dynamic detection via test embedding
            model_info = self._detect_via_test_embedding()
            if model_info:
                return model_info

            # Method 3: Fall back to known model configurations
            if self.model_id in self.KNOWN_MODELS:
                logger.info("Using known configuration for %s", self.model_id)
                return self.KNOWN_MODELS[self.model_id]

            # Method 4: Last resort - try to infer from model ID
            return self._infer_from_model_id()

        except Exception as e:
            logger.warning("Model capability detection failed: %s", e)
            logger.info("Falling back to default Titan v2 configuration")

            # Ultimate fallback
            return BedrockModelInfo(
                model_id=self.model_id,
                embedding_dimensions=1024,
                max_input_tokens=8192,
                max_batch_size=25,
                provider="unknown",
                model_name=f"Unknown Model ({self.model_id})"
            )

    def _get_model_info_from_api(self) -> Optional[BedrockModelInfo]:
        """Try to get model information from Bedrock foundation models API."""
        try:
            # List available foundation models
            response = self.bedrock_client.list_foundation_models()

            for model in response.get('modelSummaries', []):
                if model['modelId'] == self.model_id:
                    # Extract model information
                    model_name = model.get('modelName', 'Unknown')
                    provider = model.get('providerName', 'unknown').lower()

                    # Get detailed model info if available
                    try:
                        detail_response = self.bedrock_client.get_foundation_model(
                            modelIdentifier=self.model_id
                        )
                        model_details = detail_response.get('modelDetails', {})

                        # Extract capabilities from model details
                        input_modalities = model_details.get('inputModalities', [])
                        output_modalities = model_details.get('outputModalities', [])

                        if 'TEXT' in input_modalities and 'EMBEDDING' in output_modalities:
                            # This is an embedding model - use fallback detection
                            # Note: API doesn't currently expose embedding dimensions directly
                            logger.info("Found embedding model via API: %s", self.model_id)
                            return BedrockModelInfo(
                                model_id=self.model_id,
                                embedding_dimensions=self._estimate_dimensions(provider),
                                max_input_tokens=self._extract_max_tokens(model_details),
                                    max_batch_size=self._extract_batch_size(provider),
                                    provider=provider,
                                    model_name=model_name
                                )

                    except ClientError as e:
                        logger.debug("Could not get detailed model info: %s", e)

            return None

        except Exception as e:
            logger.debug("API model detection failed: %s", e)
            return None

    def _detect_via_test_embedding(self) -> Optional[BedrockModelInfo]:
        """Detect model capabilities by making a test embedding call."""
        try:
            logger.info("Attempting dynamic detection via test embedding...")

            # Make a test embedding call with minimal text
            test_text = "test"
            body = self._prepare_request_body(test_text)

            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                body=body
            )

            # Parse response to get embedding dimensions
            response_body = json.loads(response['body'].read())
            embedding = self._extract_embedding_from_response(response_body)

            if embedding:
                dimensions = len(embedding)
                provider = self._infer_provider_from_model_id()

                logger.info("Dynamically detected %s dimensions via test embedding", dimensions)

                return BedrockModelInfo(
                    model_id=self.model_id,
                    embedding_dimensions=dimensions,
                    max_input_tokens=self._estimate_max_tokens(provider),
                    max_batch_size=self._estimate_batch_size(provider),
                    provider=provider,
                    model_name=f"Detected Model ({self.model_id})"
                )

            return None

        except Exception as e:
            logger.debug("Test embedding detection failed: %s", e)
            return None

    def _infer_from_model_id(self) -> BedrockModelInfo:
        """Infer model capabilities from model ID patterns."""
        provider = self._infer_provider_from_model_id()

        # Make educated guesses based on model ID patterns
        if "titan-embed-text-v2" in self.model_id:
            dimensions = 1024
            max_tokens = 8192
        elif "titan-embed-text-v1" in self.model_id:
            dimensions = 1536
            max_tokens = 8000
        elif "cohere.embed" in self.model_id:
            dimensions = 1024
            max_tokens = 512
        else:
            # Conservative defaults
            dimensions = 1024
            max_tokens = 4096

        logger.info("Inferred capabilities from model ID: %s dimensions", dimensions)

        return BedrockModelInfo(
            model_id=self.model_id,
            embedding_dimensions=dimensions,
            max_input_tokens=max_tokens,
            max_batch_size=self._estimate_batch_size(provider),
            provider=provider,
            model_name=f"Inferred Model ({self.model_id})"
        )

    def _prepare_request_body(self, text: str) -> str:
        """Prepare request body based on model provider."""
        provider = self._infer_provider_from_model_id()

        if provider == "amazon":
            # Titan models
            return json.dumps({"inputText": text})
        if provider == "cohere":
            # Cohere models
            return json.dumps({
                "texts": [text],
                "input_type": "search_document"
            })
        # Default to Titan format
        return json.dumps({"inputText": text})

    def _extract_embedding_from_response(self, response_body: Dict[str, Any]) -> Optional[List[float]]:
        """Extract embedding vector from response based on model provider."""
        try:
            # Titan format
            if "embedding" in response_body:
                return response_body["embedding"]

            # Cohere format
            if "embeddings" in response_body and len(response_body["embeddings"]) > 0:
                return response_body["embeddings"][0]

            return None

        except Exception as e:
            logger.debug("Could not extract embedding from response: %s", e)
            return None

    def _infer_provider_from_model_id(self) -> str:
        """Infer provider from model ID."""
        if self.model_id.startswith("amazon."):
            return "amazon"
        if self.model_id.startswith("cohere."):
            return "cohere"
        if self.model_id.startswith("anthropic."):
            return "anthropic"
        return "unknown"

    def _extract_dimensions_from_details(self, _model_details: Dict[str, Any]) -> Optional[int]:
        """Try to extract embedding dimensions from model details."""
        # This would need to be implemented based on actual API response structure
        # Currently, Bedrock API doesn't expose embedding dimensions directly
        return None

    def _extract_max_tokens(self, _model_details: Dict[str, Any]) -> int:
        """Extract maximum token limit from model details."""
        # Try to find token limits in model details
        # Fallback to provider-based estimates
        provider = self._infer_provider_from_model_id()
        return self._estimate_max_tokens(provider)

    def _estimate_dimensions(self, provider: str) -> int:
        """Estimate embedding dimensions based on provider and model."""
        if provider == "amazon":
            if "v2" in self.model_id:
                return 1024  # Titan v2 models
            return 1536  # Titan v1 models
        if provider == "cohere":
            return 1024  # Cohere embed models
        return 1536  # Default fallback

    def _estimate_max_tokens(self, provider: str) -> int:
        """Estimate maximum tokens based on provider."""
        if provider == "amazon":
            return 8192 if "v2" in self.model_id else 8000
        if provider == "cohere":
            return 512
        return 4096

    def _extract_batch_size(self, provider: str) -> int:
        """Extract or estimate batch size based on provider."""
        return self._estimate_batch_size(provider)

    def _estimate_batch_size(self, provider: str) -> int:
        """Estimate batch size based on provider."""
        if provider == "amazon":
            return 25
        if provider == "cohere":
            return 96
        return 10

    # EmbeddingProviderBase implementation

    def embed_documents(self, documents: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate embeddings for multiple documents."""
        if not documents:
            raise ValueError("No documents provided for embedding")

        embeddings = self.embed_batch(documents)
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

        if device is not None:
            embeddings_tensor = embeddings_tensor.to(device)

        return embeddings_tensor

    def embed_query(self, query: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate embedding for a single query."""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        embedding = self.embed_single(query)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)

        if device is not None:
            embedding_tensor = embedding_tensor.to(device)

        return embedding_tensor

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model_info.embedding_dimensions

    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length."""
        return self.model_info.max_input_tokens

    # Bedrock-specific methods

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            raise ValueError("Text cannot be empty")

        for attempt in range(self.max_retries):
            try:
                # Prepare request
                body = self._prepare_request_body(text)

                # Make API call
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=body,
                    contentType='application/json',
                    accept='application/json'
                )

                # Parse response
                response_body = json.loads(response['body'].read())
                embedding = self._extract_embedding_from_response(response_body)

                if embedding is None:
                    raise EmbeddingProviderError("Could not extract embedding from response: %s" % response_body)

                return embedding

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                if error_code == 'ThrottlingException' and attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) * 0.1  # Exponential backoff
                    logger.warning("Rate limited, retrying in %.1fs...", wait_time)
                    time.sleep(wait_time)
                    continue
                logger.error("Bedrock API error: %s", e)
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning("Embedding attempt %d failed: %s", attempt + 1, e)
                    time.sleep(0.1)
                    continue
                logger.error("All embedding attempts failed: %s", e)
                raise

        raise RuntimeError("Failed to generate embedding after %s attempts" % self.max_retries)

    def get_embedding_dim(self) -> int:
        """Get embedding dimensions for this model."""
        return self.model_info.embedding_dimensions

    def get_max_tokens(self) -> int:
        """Get maximum token limit for this model."""
        return self.model_info.max_input_tokens

    def get_model_name(self) -> str:
        """Get human-readable model name."""
        return self.model_info.model_name

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        embeddings = []
        batch_size = self.model_info.max_batch_size

        # Process in batches to respect API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug("Processing batch %s: %s texts", i//batch_size + 1, len(batch))

            batch_embeddings = []
            for text in batch:
                embedding = self.embed_single(text)
                batch_embeddings.append(embedding)

                # Small delay to avoid rate limiting
                time.sleep(0.02)

            embeddings.extend(batch_embeddings)

        return embeddings

    def get_model_info(self) -> BedrockModelInfo:
        """Get complete model information."""
        return self.model_info

    def list_available_models(self) -> List[str]:
        """List all available Bedrock embedding models."""
        try:
            response = self.bedrock_client.list_foundation_models()

            embedding_models = []
            for model in response.get('modelSummaries', []):
                # Check if this is an embedding model
                input_modalities = model.get('inputModalities', [])
                output_modalities = model.get('outputModalities', [])

                if 'TEXT' in input_modalities and 'EMBEDDING' in output_modalities:
                    embedding_models.append(model['modelId'])

            return sorted(embedding_models)

        except Exception as e:
            logger.error("Could not list available models: %s", e)
            return list(self.KNOWN_MODELS.keys())

def create_bedrock_provider(model_id: str = "amazon.titan-embed-text-v2:0",
                           region: str = "us-east-1",
                           **kwargs) -> BedrockEmbeddingProvider:
    """
    Factory function to create a Bedrock embedding provider.

    Args:
        model_id: Bedrock model identifier
        region: AWS region
        **kwargs: Additional arguments for BedrockEmbeddingProvider

    Returns:
        Configured BedrockEmbeddingProvider instance
    """
    return BedrockEmbeddingProvider(model_id=model_id, region=region, **kwargs)
