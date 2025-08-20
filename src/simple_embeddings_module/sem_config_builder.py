"""
Configuration Builder for Dependency Chain Management

Handles the complex dependency chain:
Embedding Provider → Chunking Strategy → Index Structure → Database Configuration

Ensures compatibility and provides migration tools.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .chunking.mod_chunking_base import ChunkingProviderBase
from .embeddings.mod_embeddings_base import EmbeddingProviderBase
from .sem_module_reg import create_module_registry, discover_all_modules

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityResult:
    """Result of compatibility checking between providers"""

    is_compatible: bool
    reasons: List[str]
    migration_required: bool
    migration_strategy: Optional[str] = None

    def __post_init__(self):
        """Add best effort warning to all compatibility results"""
        if self.is_compatible:
            self.reasons.insert(
                0,
                "⚠️  BEST EFFORT: Even 'compatible' switches may produce different results due to:",
            )
            self.reasons.insert(
                1, "   • Different hardware (NVIDIA vs AMD vs Apple Silicon)"
            )
            self.reasons.insert(2, "   • Different RNG seeds and numerical precision")
            self.reasons.insert(3, "   • Different CUDA/ROCm/MPS implementations")
            self.reasons.insert(4, "   • Different quantization methods and rounding")
            self.reasons.insert(
                5, "   • Always validate results after switching providers!"
            )
            self.reasons.insert(6, "")  # Blank line for readability


class SEMConfigBuilder:
    """Builder for creating compatible SEM configurations"""

    def __init__(self):
        self.config = {
            "embedding": {},
            "chunking": {},
            "storage": {},
            "serialization": {},
            "index": {},
        }
        self._embedding_provider: Optional[EmbeddingProviderBase] = None
        self._chunking_provider: Optional[ChunkingProviderBase] = None
        self._registry = None

    def set_embedding_provider(
        self, provider_name: str, **provider_config
    ) -> "SEMConfigBuilder":
        """Set embedding provider and configure chunking constraints

        Args:
            provider_name: Name of embedding provider
            **provider_config: Provider configuration parameters

        Returns:
            Self for method chaining
        """
        # Ensure modules are discovered
        discover_all_modules()

        # Create temporary registry to instantiate provider
        if self._registry is None:
            self._registry = create_module_registry("config_builder")

        # Instantiate embedding provider to get capabilities
        self._embedding_provider = self._registry.instantiate_module(
            "embeddings", provider_name, provider_config
        )

        # Store configuration
        self.config["embedding"] = {"provider": provider_name, **provider_config}

        logger.info(f"Set embedding provider: {provider_name}")
        return self

    def set_chunking_strategy(
        self, strategy_name: str, **chunking_config
    ) -> "SEMConfigBuilder":
        """Set chunking strategy with embedding provider constraints

        Args:
            strategy_name: Name of chunking strategy
            **chunking_config: Chunking configuration parameters

        Returns:
            Self for method chaining
        """
        if self._embedding_provider is None:
            raise ValueError("Must set embedding provider before chunking strategy")

        # Instantiate chunking provider with embedding constraints
        chunking_config_with_embedding = {
            **chunking_config,
            "embedding_provider": self._embedding_provider,
        }

        logger.info(
            f"Instantiating chunking provider '{strategy_name}' with config keys: {list(chunking_config_with_embedding.keys())}"
        )

        self._chunking_provider = self._registry.instantiate_module(
            "chunking", strategy_name, chunking_config_with_embedding
        )

        # Store configuration
        self.config["chunking"] = {"strategy": strategy_name, **chunking_config}

        logger.info(f"Set chunking strategy: {strategy_name}")
        return self

    def auto_configure_chunking(
        self, strategy_preference: Optional[str] = None
    ) -> "SEMConfigBuilder":
        """Auto-configure chunking based on embedding provider capabilities

        Args:
            strategy_preference: Preferred chunking strategy (auto-select if None)

        Returns:
            Self for method chaining
        """
        if self._embedding_provider is None:
            raise ValueError(
                "Must set embedding provider before auto-configuring chunking"
            )

        embedding_caps = self._embedding_provider.get_capabilities()

        # Determine optimal chunking strategy
        if strategy_preference:
            strategy_name = strategy_preference
        else:
            strategy_name = self._select_optimal_chunking_strategy(embedding_caps)

        # Configure chunking parameters based on embedding constraints
        chunking_config = self._generate_chunking_config(embedding_caps)

        return self.set_chunking_strategy(strategy_name, **chunking_config)

    def _select_optimal_chunking_strategy(self, embedding_caps: Dict[str, Any]) -> str:
        """Select optimal chunking strategy based on embedding capabilities"""
        max_length = embedding_caps.get("max_sequence_length", 512)
        model_name = embedding_caps.get("model_name", "").lower()

        # Strategy selection logic - use simple text chunking for MVP
        if "code" in model_name and max_length > 2048:
            return (
                "text"  # Use text chunking for now (code chunking not implemented yet)
            )
        elif max_length < 256:
            return "text"  # Simple text chunking for short context models
        else:
            return "text"  # Use text chunking as default for MVP

    def _generate_chunking_config(
        self, embedding_caps: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate chunking configuration based on embedding capabilities"""
        max_length = embedding_caps.get("max_sequence_length", 512)

        # Conservative chunk size (80% of max to allow for tokenization overhead)
        chunk_size = int(max_length * 0.8)

        # Overlap size (10% of chunk size for context preservation)
        overlap_size = int(chunk_size * 0.1)

        return {
            "max_chunk_size": chunk_size,
            "overlap_size": overlap_size,
            "min_chunk_size": max(50, chunk_size // 10),
        }

    def set_storage_backend(
        self, backend_name: str, **storage_config
    ) -> "SEMConfigBuilder":
        """Set storage backend configuration

        Args:
            backend_name: Name of storage backend
            **storage_config: Storage configuration parameters

        Returns:
            Self for method chaining
        """
        self.config["storage"] = {"backend": backend_name, **storage_config}

        logger.info(f"Set storage backend: {backend_name}")
        return self

    def set_serialization_provider(
        self, provider_name: str, **serialization_config
    ) -> "SEMConfigBuilder":
        """Set serialization provider configuration

        Args:
            provider_name: Name of serialization provider
            **serialization_config: Serialization configuration parameters

        Returns:
            Self for method chaining
        """
        self.config["serialization"] = {
            "provider": provider_name,
            **serialization_config,
        }

        logger.info(f"Set serialization provider: {provider_name}")
        return self

    def set_index_config(self, index_name: str, **index_config) -> "SEMConfigBuilder":
        """Set index configuration

        Args:
            index_name: Name of the index
            **index_config: Index configuration parameters

        Returns:
            Self for method chaining
        """
        self.config["index"] = {"name": index_name, **index_config}

        logger.info(f"Set index config: {index_name}")
        return self

    def build(self) -> Dict[str, Any]:
        """Build final configuration with validation

        Returns:
            Complete validated configuration
        """
        # Set defaults for unspecified components
        if not self.config["storage"]:
            self.set_storage_backend("local_disk", path="./indexes")

        if not self.config["serialization"]:
            self.set_serialization_provider("orjson")

        if not self.config["index"]:
            self.set_index_config(
                "default", max_documents=100000, similarity_threshold=0.7
            )

        # Validate configuration consistency
        self._validate_configuration()

        return self.config.copy()

    def _validate_configuration(self) -> None:
        """Validate that configuration is internally consistent"""
        if not self.config["embedding"]:
            raise ValueError("Embedding provider must be configured")

        # Skip compatibility check during initial setup - providers are compatible with themselves
        # Compatibility checks are for switching between different providers

    def check_provider_compatibility(
        self, new_provider_name: str, **new_provider_config
    ) -> CompatibilityResult:
        """Check if switching to a new embedding provider is compatible

        STRICT COMPATIBILITY RULES:
        ✅ Compatible: Same embedding dimensions, similar max_length (±20%), IDENTICAL MODEL (except quantization)
        ❌ Incompatible: Different dimensions, different limits, different model family, different model size

        Args:
            new_provider_name: Name of new embedding provider
            **new_provider_config: New provider configuration

        Returns:
            CompatibilityResult with detailed compatibility information
        """
        if self._embedding_provider is None:
            return CompatibilityResult(
                is_compatible=True, reasons=[], migration_required=False
            )

        # Instantiate new provider
        try:
            new_provider = self._registry.instantiate_module(
                "embedding", new_provider_name, new_provider_config
            )
        except Exception as e:
            return CompatibilityResult(
                is_compatible=False,
                reasons=[f"Failed to instantiate new provider: {e}"],
                migration_required=True,
                migration_strategy="fix_configuration",
            )

        # Get capabilities for detailed comparison
        current_caps = self._embedding_provider.get_capabilities()
        new_caps = new_provider.get_capabilities()

        reasons = []

        # 1. STRICT: Embedding dimensions must be identical
        if current_caps["embedding_dim"] != new_caps["embedding_dim"]:
            reasons.append(
                f"❌ Embedding dimensions differ: {current_caps['embedding_dim']} vs {new_caps['embedding_dim']} (INCOMPATIBLE)"
            )

        # 2. STRICT: Models must be identical (except quantization)
        current_model = current_caps.get("model_name", "")
        new_model = new_caps.get("model_name", "")

        if not self._are_models_compatible(current_model, new_model):
            reasons.append(
                f"❌ Models are not compatible: '{current_model}' vs '{new_model}' (INCOMPATIBLE)"
            )

        # 3. Sequence length compatibility (±20% allowed)
        current_max_length = current_caps["max_sequence_length"]
        new_max_length = new_caps["max_sequence_length"]
        length_ratio = abs(new_max_length - current_max_length) / current_max_length

        if length_ratio > 0.2:
            reasons.append(
                f"❌ Max sequence length differs significantly: {current_max_length} vs {new_max_length} ({length_ratio:.1%} difference)"
            )
        elif length_ratio > 0.05:
            reasons.append(
                f"⚠️  Max sequence length differs moderately: {current_max_length} vs {new_max_length} ({length_ratio:.1%} difference)"
            )

        # 4. Check chunking compatibility if chunking provider exists
        if self._chunking_provider:
            chunk_compatible, chunk_reasons = (
                self._chunking_provider.is_compatible_with_embedding_provider(
                    new_provider
                )
            )
            if not chunk_compatible:
                reasons.extend([f"❌ Chunking: {reason}" for reason in chunk_reasons])

        # 5. Check tokenization compatibility
        current_tokenizer = current_caps.get("tokenizer_type")
        new_tokenizer = new_caps.get("tokenizer_type")
        if current_tokenizer and new_tokenizer and current_tokenizer != new_tokenizer:
            reasons.append(
                f"⚠️  Tokenizer types differ: {current_tokenizer} vs {new_tokenizer}"
            )

        # 6. Check normalization compatibility
        current_norm = current_caps.get("normalization", "l2")
        new_norm = new_caps.get("normalization", "l2")
        if current_norm != new_norm:
            reasons.append(
                f"❌ Normalization differs: {current_norm} vs {new_norm} (INCOMPATIBLE)"
            )

        # Determine compatibility
        critical_issues = [r for r in reasons if r.startswith("❌")]
        is_compatible = len(critical_issues) == 0

        # Determine migration strategy
        migration_strategy = None
        if not is_compatible:
            if any("dimensions differ" in r for r in critical_issues):
                migration_strategy = "full_reindex_different_dimensions"
            elif any("Models are not compatible" in r for r in critical_issues):
                migration_strategy = "full_reindex_different_model"
            elif any("Normalization differs" in r for r in critical_issues):
                migration_strategy = "full_reindex_different_normalization"
            else:
                migration_strategy = "full_reindex_chunking_incompatible"

        return CompatibilityResult(
            is_compatible=is_compatible,
            reasons=reasons,
            migration_required=not is_compatible,
            migration_strategy=migration_strategy,
        )

    def _are_models_compatible(self, model1: str, model2: str) -> bool:
        """Check if two models are compatible using strict rules

        Compatible models:
        - Identical model names
        - Same model with different quantization (q4, q8, fp16, etc.)
        - Same model with different file formats (.bin, .safetensors, etc.)

        Args:
            model1: First model name
            model2: Second model name

        Returns:
            True if models are compatible
        """
        if model1 == model2:
            return True

        # Normalize for comparison
        norm1 = self._normalize_model_name(model1)
        norm2 = self._normalize_model_name(model2)

        return norm1 == norm2

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name for strict compatibility checking"""
        if not model_name:
            return ""

        normalized = model_name.lower().strip()

        # Remove organization prefix (e.g., "sentence-transformers/", "microsoft/")
        if "/" in normalized:
            parts = normalized.split("/")
            if len(parts) == 2:
                normalized = parts[1]  # Keep only the model name part

        # Remove file extensions
        extensions = [".bin", ".safetensors", ".ggu", ".ggml"]
        for ext in extensions:
            if normalized.endswith(ext):
                normalized = normalized[: -len(ext)]

        # Remove quantization suffixes (these are compatible variants)
        quant_patterns = [
            r"-q\d+(_\d+)?$",  # -q4, -q8, -q4_0, etc.
            r"-int\d+$",  # -int4, -int8
            r"-fp\d+$",  # -fp16, -fp32
            r"-bf\d+$",  # -bf16
            r"-(awq|gptq|bnb|bitsandbytes)$",  # quantization methods
            r"-ggml$",  # GGML format
            r"-gguf$",  # GGUF format
        ]

        import re

        for pattern in quant_patterns:
            normalized = re.sub(pattern, "", normalized)

        # Remove version suffixes that don't affect compatibility
        version_patterns = [
            r"-v\d+(\.\d+)*$",  # -v1, -v2.1, etc.
            r"-r\d+$",  # -r1, -r2, etc.
        ]

        for pattern in version_patterns:
            normalized = re.sub(pattern, "", normalized)

        # Clean up any trailing separators
        normalized = normalized.rstrip("-_.")

        return normalized

    def create_migration_config(
        self, new_provider_name: str, **new_provider_config
    ) -> Dict[str, Any]:
        """Create configuration for migrating to a new embedding provider

        Args:
            new_provider_name: Name of new embedding provider
            **new_provider_config: New provider configuration

        Returns:
            New configuration with updated embedding provider and compatible chunking
        """
        # Create new builder with new embedding provider
        new_builder = SEMConfigBuilder()
        new_builder.set_embedding_provider(new_provider_name, **new_provider_config)

        # Auto-configure chunking for new provider
        new_builder.auto_configure_chunking()

        # Copy other configurations
        if self.config["storage"]:
            new_builder.config["storage"] = self.config["storage"].copy()

        if self.config["serialization"]:
            new_builder.config["serialization"] = self.config["serialization"].copy()

        if self.config["index"]:
            new_builder.config["index"] = self.config["index"].copy()

        return new_builder.build()


# Convenience functions
def create_default_config() -> Dict[str, Any]:
    """Create a default configuration for quick setup"""
    builder = SEMConfigBuilder()
    builder.set_embedding_provider("sentence_transformers", model="all-MiniLM-L6-v2")
    builder.auto_configure_chunking()
    return builder.build()


def create_code_search_config() -> Dict[str, Any]:
    """Create configuration optimized for code search"""
    builder = SEMConfigBuilder()
    builder.set_embedding_provider(
        "sentence_transformers", model="Salesforce/SFR-Embedding-Code-400M_R"
    )
    builder.auto_configure_chunking("code")
    builder.set_index_config("code_search", max_documents=500000)
    return builder.build()


def create_research_config() -> Dict[str, Any]:
    """Create configuration optimized for research papers"""
    builder = SEMConfigBuilder()
    builder.set_embedding_provider("sentence_transformers", model="all-MiniLM-L6-v2")
    builder.auto_configure_chunking("text")
    builder.set_index_config(
        "research_papers", max_documents=100000, similarity_threshold=0.75
    )
    return builder.build()
