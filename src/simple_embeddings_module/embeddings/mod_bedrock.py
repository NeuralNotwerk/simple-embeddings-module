"""
AWS Bedrock Embeddings Provider

Provides access to AWS Bedrock embedding models including:
- Amazon Titan Text Embeddings
- Cohere Embed models
- Anthropic Claude embeddings (when available)
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import torch

from ..sem_module_reg import ConfigParameter
from .mod_embeddings_base import EmbeddingProviderBase, EmbeddingProviderError

logger = logging.getLogger(__name__)


class BedrockEmbeddingProvider(EmbeddingProviderBase):
    """AWS Bedrock embeddings provider with multiple model support"""

    CONFIG_PARAMETERS = [
        ConfigParameter(
            key_name="model_id",
            value_type="str",
            config_description="Bedrock model ID (e.g., amazon.titan-embed-text-v1)",
            required=False,
            value_opt_default="amazon.titan-embed-text-v1",
        ),
        ConfigParameter(
            key_name="region",
            value_type="str",
            config_description="AWS region for Bedrock service",
            required=False,
            value_opt_default="us-east-1",
        ),
        ConfigParameter(
            key_name="aws_access_key_id",
            value_type="str",
            config_description="AWS access key ID (or use AWS credentials)",
            required=False,
            value_opt_default=None,
        ),
        ConfigParameter(
            key_name="aws_secret_access_key",
            value_type="str",
            config_description="AWS secret access key (or use AWS credentials)",
            required=False,
            value_opt_default=None,
        ),
        ConfigParameter(
            key_name="aws_session_token",
            value_type="str",
            config_description="AWS session token (for temporary credentials)",
            required=False,
            value_opt_default=None,
        ),
        ConfigParameter(
            key_name="profile_name",
            value_type="str",
            config_description="AWS profile name from ~/.aws/credentials",
            required=False,
            value_opt_default=None,
        ),
        ConfigParameter(
            key_name="batch_size",
            value_type="numeric",
            config_description="Number of texts to process in one batch",
            required=False,
            value_opt_default=25,  # Conservative for Bedrock
            value_opt_regex=r"^[1-9][0-9]*$",
        ),
        ConfigParameter(
            key_name="max_retries",
            value_type="numeric",
            config_description="Maximum number of retry attempts",
            required=False,
            value_opt_default=3,
            value_opt_regex=r"^[0-9]$",
        ),
        ConfigParameter(
            key_name="timeout",
            value_type="numeric",
            config_description="Request timeout in seconds",
            required=False,
            value_opt_default=60,
            value_opt_regex=r"^[1-9][0-9]*$",
        ),
    ]

    CAPABILITIES = {
        "embedding_dimension": None,  # Set based on model
        "max_sequence_length": None,  # Set based on model
        "supports_batching": True,
        "requires_aws_credentials": True,
        "supports_multiple_models": True,
        "tokenizer_type": "bedrock",
    }

    def __init__(self, **config):
        """Initialize Bedrock embedding provider"""
        super().__init__(**config)

        self.model_id = config.get("model_id", "amazon.titan-embed-text-v1")
        self.region = config.get("region", "us-east-1")
        self.batch_size = config.get("batch_size", 25)
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 60)

        # AWS credentials
        self.aws_access_key_id = config.get("aws_access_key_id")
        self.aws_secret_access_key = config.get("aws_secret_access_key")
        self.aws_session_token = config.get("aws_session_token")
        self.profile_name = config.get("profile_name")

        # Initialize Bedrock client
        self._init_client()

        # Set model-specific capabilities
        self._set_model_capabilities()

        logger.info(f"Bedrock embedding provider initialized: {self.model_id}")

    def _init_client(self):
        """Initialize AWS Bedrock client"""
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise EmbeddingProviderError(
                "boto3 library not installed. Install with: pip install boto3"
            )

        try:
            # Configure boto3 client
            config = Config(
                region_name=self.region,
                retries={"max_attempts": self.max_retries, "mode": "adaptive"},
                read_timeout=self.timeout,
            )

            # Create session with credentials
            if self.profile_name:
                session = boto3.Session(profile_name=self.profile_name)
                self.client = session.client("bedrock-runtime", config=config)
            elif self.aws_access_key_id and self.aws_secret_access_key:
                self.client = boto3.client(
                    "bedrock-runtime",
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    aws_session_token=self.aws_session_token,
                    config=config,
                )
            else:
                # Use default credentials (IAM role, env vars, etc.)
                self.client = boto3.client("bedrock-runtime", config=config)

            logger.info(f"Bedrock client initialized for region: {self.region}")

        except Exception as e:
            raise EmbeddingProviderError(f"Failed to initialize Bedrock client: {e}")

    def _set_model_capabilities(self):
        """Set capabilities based on the selected model"""
        model_specs = {
            "amazon.titan-embed-text-v1": {
                "embedding_dimension": 1536,
                "max_sequence_length": 8000,
                "input_format": "titan",
            },
            "amazon.titan-embed-text-v2:0": {
                "embedding_dimension": 1024,
                "max_sequence_length": 8000,
                "input_format": "titan_v2",
            },
            "cohere.embed-english-v3": {
                "embedding_dimension": 1024,
                "max_sequence_length": 512,
                "input_format": "cohere",
            },
            "cohere.embed-multilingual-v3": {
                "embedding_dimension": 1024,
                "max_sequence_length": 512,
                "input_format": "cohere",
            },
        }

        if self.model_id not in model_specs:
            logger.warning(f"Unknown model {self.model_id}, using default capabilities")
            specs = {
                "embedding_dimension": 1536,
                "max_sequence_length": 8000,
                "input_format": "titan",
            }
        else:
            specs = model_specs[self.model_id]

        self.CAPABILITIES["embedding_dimension"] = specs["embedding_dimension"]
        self.CAPABILITIES["max_sequence_length"] = specs["max_sequence_length"]
        self._input_format = specs["input_format"]

        logger.info(
            f"Model capabilities set: {specs['embedding_dimension']} dimensions, "
            f"{specs['max_sequence_length']} max tokens"
        )

    def embed_documents(
        self, texts: List[str], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generate embeddings for multiple documents"""
        if not texts:
            raise ValueError("No texts provided for embedding")

        logger.info(f"Generating embeddings for {len(texts)} documents")
        start_time = time.time()

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            if self._supports_batch_processing():
                # Use batch processing if supported
                batch_embeddings = self._embed_batch(batch)
            else:
                # Process individually
                batch_embeddings = []
                for text in batch:
                    embedding = self._embed_single(text)
                    batch_embeddings.append(embedding)

            all_embeddings.extend(batch_embeddings)

            # Small delay between batches
            if i + self.batch_size < len(texts):
                time.sleep(0.1)

        # Convert to tensor
        embeddings_tensor = torch.tensor(all_embeddings, dtype=torch.float32)

        # Move to specified device
        if device is not None:
            embeddings_tensor = embeddings_tensor.to(device)

        elapsed = time.time() - start_time
        logger.info(
            f"Generated {len(texts)} embeddings in {elapsed:.2f}s "
            f"({len(texts)/elapsed:.1f} docs/sec)"
        )

        return embeddings_tensor

    def embed_query(
        self, query: str, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generate embedding for a single query"""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        embedding = self._embed_single(query)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)

        if device is not None:
            embedding_tensor = embedding_tensor.to(device)

        return embedding_tensor

    def _supports_batch_processing(self) -> bool:
        """Check if the model supports batch processing"""
        # Currently, most Bedrock models process one text at a time
        return False

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts (for models that support it)"""
        # This would be implemented for models that support batch processing
        # For now, fall back to individual processing
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text with retry logic"""
        for attempt in range(self.max_retries + 1):
            try:
                # Prepare request body based on model format
                body = self._prepare_request_body(text)

                # Make Bedrock API call
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )

                # Parse response
                response_body = json.loads(response["body"].read())
                embedding = self._extract_embedding(response_body)

                logger.debug(f"Successfully embedded text of length {len(text)}")
                return embedding

            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Bedrock API error (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Bedrock API failed after {self.max_retries + 1} attempts: {e}"
                    )
                    raise EmbeddingProviderError(f"Bedrock API error: {e}")

    def _prepare_request_body(self, text: str) -> Dict[str, Any]:
        """Prepare request body based on model format"""
        if self._input_format == "titan":
            return {"inputText": text}
        elif self._input_format == "titan_v2":
            return {
                "inputText": text,
                "dimensions": self.CAPABILITIES["embedding_dimension"],
            }
        elif self._input_format == "cohere":
            return {"texts": [text], "input_type": "search_document"}
        else:
            # Default to Titan format
            return {"inputText": text}

    def _extract_embedding(self, response_body: Dict[str, Any]) -> List[float]:
        """Extract embedding from response based on model format"""
        if self._input_format in ["titan", "titan_v2"]:
            return response_body["embedding"]
        elif self._input_format == "cohere":
            return response_body["embeddings"][0]
        else:
            # Try common response formats
            if "embedding" in response_body:
                return response_body["embedding"]
            elif "embeddings" in response_body:
                return response_body["embeddings"][0]
            else:
                raise EmbeddingProviderError(
                    f"Unknown response format: {response_body}"
                )

    def get_capabilities(self) -> Dict[str, Any]:
        """Return provider capabilities"""
        capabilities = super().get_capabilities()
        capabilities.update(
            {
                "model_id": self.model_id,
                "aws_region": self.region,
                "api_based": True,
                "requires_internet": True,
                "supports_batch_processing": self._supports_batch_processing(),
            }
        )
        return capabilities

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        # Rough approximation: ~4 characters per token
        return len(text) // 4

    def validate_text_length(self, text: str) -> bool:
        """Validate that text is within model limits"""
        estimated_tokens = self.estimate_tokens(text)
        max_tokens = self.CAPABILITIES.get("max_sequence_length", 8000)
        return estimated_tokens <= max_tokens

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"BedrockEmbeddingProvider("
            f"model_id={self.model_id}, "
            f"region={self.region}, "
            f"dimensions={self.CAPABILITIES.get('embedding_dimension')})"
        )
