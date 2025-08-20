"""
Ollama Embeddings Provider

Provides access to local Ollama embedding models via HTTP API.
Automatically manages Ollama server lifecycle and model downloads.
"""

import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

import requests
import torch

from ..sem_module_reg import ConfigParameter
from .mod_embeddings_base import EmbeddingProviderBase, EmbeddingProviderError

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProviderBase):
    """Ollama local embeddings provider with automatic server management"""

    CONFIG_PARAMETERS = [
        ConfigParameter(
            key_name="model",
            value_type="str",
            config_description="Ollama model name for embeddings",
            required=False,
            value_opt_default="snowflake-arctic-embed2",
        ),
        ConfigParameter(
            key_name="base_url",
            value_type="str",
            config_description="Ollama server base URL",
            required=False,
            value_opt_default="http://localhost:11434",
        ),
        ConfigParameter(
            key_name="batch_size",
            value_type="numeric",
            config_description="Number of texts to process in one batch",
            required=False,
            value_opt_default=10,  # Conservative for local processing
            value_opt_regex=r"^[1-9][0-9]*$",
        ),
        ConfigParameter(
            key_name="timeout",
            value_type="numeric",
            config_description="Request timeout in seconds",
            required=False,
            value_opt_default=60,
            value_opt_regex=r"^[1-9][0-9]*$",
        ),
        ConfigParameter(
            key_name="auto_start_server",
            value_type="bool",
            config_description="Automatically start Ollama server if not running",
            required=False,
            value_opt_default=True,
        ),
        ConfigParameter(
            key_name="auto_pull_model",
            value_type="bool",
            config_description="Automatically pull model if not available",
            required=False,
            value_opt_default=True,
        ),
        ConfigParameter(
            key_name="server_startup_timeout",
            value_type="numeric",
            config_description="Timeout for server startup in seconds",
            required=False,
            value_opt_default=30,
            value_opt_regex=r"^[1-9][0-9]*$",
        ),
    ]

    CAPABILITIES = {
        "embedding_dimension": None,  # Detected from model
        "max_sequence_length": None,  # Detected from model
        "supports_batching": False,  # Ollama processes one at a time
        "requires_local_server": True,
        "supports_custom_models": True,
        "offline_capable": True,
        "tokenizer_type": "ollama",
    }

    def __init__(self, **config):
        """Initialize Ollama embedding provider"""
        super().__init__(**config)

        self.model = config.get("model", "snowflake-arctic-embed2")
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.batch_size = config.get("batch_size", 10)
        self.timeout = config.get("timeout", 60)
        self.auto_start_server = config.get("auto_start_server", True)
        self.auto_pull_model = config.get("auto_pull_model", True)
        self.server_startup_timeout = config.get("server_startup_timeout", 30)

        # Check Ollama installation
        self._check_ollama_installation()

        # Ensure server is running
        self._ensure_server_running()

        # Ensure model is available
        self._ensure_model_available()

        # Detect model capabilities
        self._detect_model_capabilities()

        logger.info(f"Ollama embedding provider initialized: {self.model}")

    def _check_ollama_installation(self):
        """Check if Ollama is installed and accessible"""
        try:
            result = subprocess.run(
                ["which", "ollama"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise EmbeddingProviderError(
                    "Ollama not found in PATH. Please install Ollama from https://ollama.ai"
                )

            self.ollama_path = result.stdout.strip()
            logger.info(f"Found Ollama at: {self.ollama_path}")

        except subprocess.TimeoutExpired:
            raise EmbeddingProviderError("Timeout checking for Ollama installation")
        except Exception as e:
            raise EmbeddingProviderError(f"Error checking Ollama installation: {e}")

    def _is_server_running(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _start_server(self):
        """Start Ollama server in background"""
        if self._is_server_running():
            logger.info("Ollama server already running")
            return

        if not self.auto_start_server:
            raise EmbeddingProviderError(
                "Ollama server not running and auto_start_server is disabled. "
                "Please start Ollama server manually: ollama serve"
            )

        logger.info("Starting Ollama server...")

        try:
            # Start server in background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            # Wait for server to start
            start_time = time.time()
            while time.time() - start_time < self.server_startup_timeout:
                if self._is_server_running():
                    logger.info("Ollama server started successfully")
                    return
                time.sleep(1)

            raise EmbeddingProviderError(
                f"Ollama server failed to start within {self.server_startup_timeout}s"
            )

        except Exception as e:
            raise EmbeddingProviderError(f"Failed to start Ollama server: {e}")

    def _ensure_server_running(self):
        """Ensure Ollama server is running"""
        if not self._is_server_running():
            self._start_server()

    def _list_models(self) -> List[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            return [model["name"] for model in data.get("models", [])]

        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []

    def _pull_model(self, model_name: str):
        """Pull a model using Ollama CLI"""
        logger.info(f"Pulling model: {model_name}")

        try:
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes for model download
            )

            if result.returncode != 0:
                raise EmbeddingProviderError(
                    f"Failed to pull model {model_name}: {result.stderr}"
                )

            logger.info(f"Successfully pulled model: {model_name}")

        except subprocess.TimeoutExpired:
            raise EmbeddingProviderError(
                f"Timeout pulling model {model_name} (5 minutes)"
            )
        except Exception as e:
            raise EmbeddingProviderError(f"Error pulling model {model_name}: {e}")

    def _ensure_model_available(self):
        """Ensure the specified model is available"""
        available_models = self._list_models()

        # Check if exact model name exists
        if self.model in available_models:
            logger.info(f"Model {self.model} is available")
            return

        # Check if model name without tag exists (Ollama adds :latest automatically)
        model_base = self.model.split(":")[0]
        for available in available_models:
            if available.startswith(model_base):
                logger.info(f"Found model variant: {available}")
                return

        # Model not found, try to pull it
        if not self.auto_pull_model:
            raise EmbeddingProviderError(
                f"Model {self.model} not available and auto_pull_model is disabled. "
                f"Available models: {available_models}"
            )

        self._pull_model(self.model)

    def _detect_model_capabilities(self):
        """Detect model capabilities by making a test embedding call"""
        try:
            # Make a test embedding call
            test_response = self._embed_single("test")

            if test_response:
                self.CAPABILITIES["embedding_dimension"] = len(test_response)
                logger.info(f"Detected embedding dimension: {len(test_response)}")

            # Set reasonable defaults for sequence length
            # Most embedding models handle 512-2048 tokens well
            self.CAPABILITIES["max_sequence_length"] = 2048

        except Exception as e:
            logger.warning(f"Failed to detect model capabilities: {e}")
            # Set conservative defaults
            self.CAPABILITIES["embedding_dimension"] = 384
            self.CAPABILITIES["max_sequence_length"] = 512

    def embed_documents(
        self, texts: List[str], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generate embeddings for multiple documents"""
        if not texts:
            raise ValueError("No texts provided for embedding")

        logger.info(f"Generating embeddings for {len(texts)} documents")
        start_time = time.time()

        all_embeddings = []

        # Process individually (Ollama doesn't support batch processing)
        for i, text in enumerate(texts):
            if i > 0 and i % 10 == 0:
                logger.info(f"Processed {i}/{len(texts)} documents")

            embedding = self._embed_single(text)
            all_embeddings.append(embedding)

            # Small delay to avoid overwhelming local server
            if i < len(texts) - 1:
                time.sleep(0.01)

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

    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text using Ollama API"""
        try:
            payload = {"model": self.model, "prompt": text}

            response = requests.post(
                f"{self.base_url}/api/embeddings", json=payload, timeout=self.timeout
            )

            response.raise_for_status()
            data = response.json()

            if "embedding" not in data:
                raise EmbeddingProviderError(f"No embedding in response: {data}")

            return data["embedding"]

        except requests.exceptions.RequestException as e:
            raise EmbeddingProviderError(f"Ollama API request failed: {e}")
        except Exception as e:
            raise EmbeddingProviderError(f"Error generating embedding: {e}")

    def get_capabilities(self) -> Dict[str, Any]:
        """Return provider capabilities"""
        capabilities = super().get_capabilities()
        capabilities.update(
            {
                "model_name": self.model,
                "server_url": self.base_url,
                "local_processing": True,
                "requires_internet": False,  # After model is downloaded
                "supports_custom_models": True,
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
        max_tokens = self.CAPABILITIES.get("max_sequence_length", 512)
        return estimated_tokens <= max_tokens

    def list_available_models(self) -> List[str]:
        """List all available models on the Ollama server"""
        return self._list_models()

    def pull_model(self, model_name: str):
        """Pull a new model"""
        self._pull_model(model_name)

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"OllamaEmbeddingProvider("
            f"model={self.model}, "
            f"url={self.base_url}, "
            f"dimensions={self.CAPABILITIES.get('embedding_dimension')})"
        )
