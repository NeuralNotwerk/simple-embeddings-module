"""
Chunk Multiplexer - Auto-selects chunking strategies based on content type
Capable of auto-selecting other modules for chunking based on configured regex patterns.
Routes different document types to appropriate chunking strategies.
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..embeddings.mod_embeddings_base import EmbeddingProviderBase
from ..sem_module_reg import ConfigParameter
from .mod_chunking_base import (
    ChunkedDocument,
    ChunkingProviderBase,
    ChunkingProviderError,
)


class ChunkMultiplexer(ChunkingProviderBase):
    """Multiplexer that routes documents to appropriate chunking strategies"""
    CONFIG_PARAMETERS = [
        ConfigParameter(
            key_name="routing_rules",
            value_type="list",
            config_description="List of routing rules with patterns and chunking strategies",
            required=True,
        ),
        ConfigParameter(
            key_name="default_strategy",
            value_type="str",
            config_description="Default chunking strategy when no rules match",
            required=True,
            value_opt_default="text",
        ),
        ConfigParameter(
            key_name="content_detection",
            value_type="bool",
            config_description="Enable automatic content type detection",
            required=False,
            value_opt_default=True,
        ),
    ]
    CAPABILITIES = {
        "supports_multiple_strategies": True,
        "auto_content_detection": True,
        "configurable_routing": True,
    }

    def __init__(self, embedding_provider: EmbeddingProviderBase, **config):
        """Initialize chunk multiplexer with routing configuration"""
        super().__init__(embedding_provider, **config)
        self.routing_rules = config.get("routing_rules", [])
        self.default_strategy = config.get("default_strategy", "text")
        self.content_detection = config.get("content_detection", True)
        # Registry of available chunking strategies
        self._chunking_strategies: Dict[str, ChunkingProviderBase] = {}
        # Compiled regex patterns for performance
        self._compiled_rules = []
        self._compile_routing_rules()

    def _configure_for_embedding_provider(self) -> None:
        """Configure multiplexer based on embedding provider capabilities"""
        # Multiplexer inherits constraints from embedding provider
        self.max_chunk_size = self.max_sequence_length
        self.min_chunk_size = 50  # Reasonable minimum
        self.overlap_size = 0  # Let individual strategies handle overlap

    def _compile_routing_rules(self) -> None:
        """Compile regex patterns for routing rules"""
        for rule in self.routing_rules:
            if not isinstance(rule, dict):
                raise ChunkingProviderError("Invalid routing rule format: %s" % rule)
            pattern = rule.get("pattern")
            strategy = rule.get("strategy")
            rule_type = rule.get("type", "content")  # 'content', 'filename', 'metadata'
            if not pattern or not strategy:
                raise ChunkingProviderError("Routing rule missing pattern or strategy: %s" % rule)
            try:
                compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                self._compiled_rules.append(
                    {
                        "pattern": compiled_pattern,
                        "strategy": strategy,
                        "type": rule_type,
                        "original_pattern": pattern,
                    }
                )
            except re.error as e:
                raise ChunkingProviderError("Invalid regex pattern '%s': %s" % (pattern, e))

    def register_chunking_strategy(self, name: str, strategy: ChunkingProviderBase) -> None:
        """Register a chunking strategy with the multiplexer"""
        self._chunking_strategies[name] = strategy

    def _select_chunking_strategy(self, document_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Select appropriate chunking strategy for a document"""
        # Check routing rules in order
        for rule in self._compiled_rules:
            if self._rule_matches(rule, document_id, text, metadata):
                strategy_name = rule["strategy"]
                if strategy_name in self._chunking_strategies:
                    return strategy_name
                else:
                    # Strategy not registered, fall through to default
                    continue
        # Auto-detect content type if enabled
        if self.content_detection:
            detected_strategy = self._auto_detect_content_type(text)
            if detected_strategy and detected_strategy in self._chunking_strategies:
                return detected_strategy
        # Use default strategy
        return self.default_strategy

    def _rule_matches(
        self,
        rule: Dict[str, Any],
        document_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]],
    ) -> bool:
        """Check if a routing rule matches the document"""
        rule_type = rule["type"]
        pattern = rule["pattern"]
        if rule_type == "filename":
            # Extract filename from document_id or metadata
            filename = self._extract_filename(document_id, metadata)
            return bool(pattern.search(filename))
        elif rule_type == "content":
            # Match against document content
            return bool(pattern.search(text))
        elif rule_type == "metadata":
            # Match against metadata fields
            if metadata:
                metadata_str = str(metadata)
                return bool(pattern.search(metadata_str))
            return False
        return False

    def _extract_filename(self, document_id: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Extract filename from document_id or metadata"""
        if metadata and "filename" in metadata:
            return metadata["filename"]
        # Try to extract from document_id
        if "/" in document_id or "\\" in document_id:
            return Path(document_id).name
        return document_id

    def _auto_detect_content_type(self, text: str) -> Optional[str]:
        """Auto-detect content type based on text patterns"""
        text_sample = text[:2000]  # Check first 2000 characters
        # Code detection patterns
        code_patterns = [
            r"def\s+\w+\s*\(",  # Python functions
            r"function\s+\w+\s*\(",  # JavaScript functions
            r"class\s+\w+\s*[{:]",  # Class definitions
            r"#include\s*<",  # C/C++ includes
            r"import\s+\w+",  # Import statements
            r"^\s*//.*$",  # Single-line comments
            r"/\*.*?\*/",  # Multi-line comments
        ]
        code_score = sum(1 for pattern in code_patterns if re.search(pattern, text_sample, re.MULTILINE))
        if code_score >= 2:
            return "code"
        # CSV detection
        if self._looks_like_csv(text_sample):
            return "csv"
        # Default to text
        return "text"

    def _looks_like_csv(self, text: str) -> bool:
        """Detect if text looks like CSV format"""
        lines = text.split("\n")[:10]  # Check first 10 lines
        if len(lines) < 2:
            return False
        # Check for consistent comma/tab separation
        separators = [",", "\t", ";"]
        for sep in separators:
            if all(sep in line for line in lines if line.strip()):
                # Check if separator count is consistent
                counts = [line.count(sep) for line in lines if line.strip()]
                if len(set(counts)) <= 2:  # Allow some variation
                    return True
        return False

    def chunk_document(self, document_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkedDocument:
        """Route document to appropriate chunking strategy"""
        strategy_name = self._select_chunking_strategy(document_id, text, metadata)
        if strategy_name not in self._chunking_strategies:
            raise ChunkingProviderError("Chunking strategy '%s' not registered" % strategy_name)
        strategy = self._chunking_strategies[strategy_name]
        # Delegate to selected strategy
        chunked_doc = strategy.chunk_document(document_id, text, metadata)
        # Add multiplexer metadata
        chunked_doc.chunking_strategy = f"mux:{strategy_name}"
        return chunked_doc

    def chunk_query(self, query: str) -> List[str]:
        """Chunk query using default text strategy"""
        if self.default_strategy in self._chunking_strategies:
            strategy = self._chunking_strategies[self.default_strategy]
            return strategy.chunk_query(query)
        # Fallback: simple length-based chunking
        if len(query) <= self.max_sequence_length:
            return [query]
        # Split into chunks at word boundaries
        words = query.split()
        chunks = []
        current_chunk = []
        current_length = 0
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > self.max_sequence_length:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    # Single word too long, truncate
                    chunks.append(word[: self.max_sequence_length])
            else:
                current_chunk.append(word)
                current_length += word_length
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def get_routing_info(self) -> Dict[str, Any]:
        """Get information about routing configuration"""
        return {
            "routing_rules": [
                {
                    "pattern": rule["original_pattern"],
                    "strategy": rule["strategy"],
                    "type": rule["type"],
                }
                for rule in self._compiled_rules
            ],
            "default_strategy": self.default_strategy,
            "registered_strategies": list(self._chunking_strategies.keys()),
            "content_detection": self.content_detection,
        }

    def __repr__(self) -> str:
        return (
            f"ChunkMultiplexer("
            f"strategies={len(self._chunking_strategies)}, "
            f"rules={len(self._compiled_rules)}, "
            f"default={self.default_strategy})"
        )
