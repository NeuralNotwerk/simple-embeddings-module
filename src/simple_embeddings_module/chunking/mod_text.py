"""
Text Chunking Provider
Implements intelligent text chunking with sentence and paragraph boundary detection.
Optimized for embedding provider constraints and semantic coherence.
"""
import logging
import re
from typing import Any, Dict, List, Optional

from ..sem_module_reg import ConfigParameter
from .mod_chunking_base import (
    ChunkBoundary,
    ChunkedDocument,
    ChunkingProviderBase,
    ChunkingProviderError,
    ChunkMetadata,
)

logger = logging.getLogger(__name__)
class TextChunkingProvider(ChunkingProviderBase):
    """Intelligent text chunking with sentence and paragraph awareness"""
    CONFIG_PARAMETERS = [
        ConfigParameter(
            key_name="chunk_size",
            value_type="numeric",
            config_description="Target chunk size in characters (auto-configured from embedding provider if not set)",
            required=False,
            value_opt_regex=r"^[1-9]\d*$",
        ),
        ConfigParameter(
            key_name="overlap_size",
            value_type="numeric",
            config_description="Overlap size in characters between chunks",
            required=False,
            value_opt_default=50,
            value_opt_regex=r"^[0-9]\d*$",
        ),
        ConfigParameter(
            key_name="boundary_type",
            value_type="str",
            config_description="Preferred chunk boundary type",
            required=False,
            value_opt_default="sentence",
            value_opt_regex=r"^(sentence|paragraph|word|character)$",
        ),
        ConfigParameter(
            key_name="min_chunk_size",
            value_type="numeric",
            config_description="Minimum chunk size in characters",
            required=False,
            value_opt_default=50,
            value_opt_regex=r"^[1-9]\d*$",
        ),
        ConfigParameter(
            key_name="preserve_paragraphs",
            value_type="bool",
            config_description="Try to keep paragraphs together when possible",
            required=False,
            value_opt_default=True,
        ),
    ]
    CAPABILITIES = {
        "boundary_type": "sentence",
        "supports_metadata": True,
        "preserves_structure": True,
        "handles_overlap": True,
    }
    def __init__(self, **config):
        """Initialize text chunking provider"""
        # Extract embedding provider from config
        embedding_provider = config.pop("embedding_provider", None)
        if embedding_provider is None:
            raise ChunkingProviderError("embedding_provider is required for chunking initialization")
        # Set attributes that will be needed by base class
        self.overlap_size = config.get("overlap_size", 50)
        self.boundary_type = ChunkBoundary(config.get("boundary_type", "sentence"))
        self.min_chunk_size = config.get("min_chunk_size", 50)
        self.preserve_paragraphs = config.get("preserve_paragraphs", True)
        # Chunk size will be set in _configure_for_embedding_provider
        self.chunk_size = config.get("chunk_size")  # May be None initially
        # Initialize base class
        super().__init__(embedding_provider, **config)
        # Compile regex patterns for sentence detection
        self._sentence_endings = re.compile(r"[.!?]+\s+")
        self._paragraph_breaks = re.compile(r"\n\s*\n")
        logger.info("TextChunkingProvider initialized with embedding constraints")
    def _configure_for_embedding_provider(self) -> None:
        """Configure chunking parameters based on embedding provider"""
        # Use 80% of max sequence length as target chunk size (conservative)
        if self.chunk_size is None:
            self.chunk_size = int(self.max_sequence_length * 0.8)
        # Ensure chunk size doesn't exceed embedding limits
        self.chunk_size = min(self.chunk_size, self.max_sequence_length)
        # Adjust overlap if it's too large relative to chunk size
        if self.overlap_size >= self.chunk_size * 0.5:
            self.overlap_size = int(self.chunk_size * 0.1)  # 10% overlap
        # Ensure minimum chunk size is reasonable
        self.min_chunk_size = max(self.min_chunk_size, 20)
        self.min_chunk_size = min(self.min_chunk_size, self.chunk_size // 4)
        logger.info(
            "Configured for embedding provider: chunk_size=%s, "
            "overlap=%s, min_size=%s", self.chunk_size, self.overlap_size, self.min_chunk_size
        )
    def chunk_document(self, document_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkedDocument:
        """Chunk a document into embedding-ready pieces"""
        if not text.strip():
            raise ChunkingProviderError("Document text cannot be empty")
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        # Generate chunks based on boundary type
        if self.boundary_type == ChunkBoundary.PARAGRAPH:
            chunks = self._chunk_by_paragraphs(cleaned_text)
        elif self.boundary_type == ChunkBoundary.SENTENCE:
            chunks = self._chunk_by_sentences(cleaned_text)
        elif self.boundary_type == ChunkBoundary.WORD:
            chunks = self._chunk_by_words(cleaned_text)
        else:  # CHARACTER
            chunks = self._chunk_by_characters(cleaned_text)
        # Create chunk metadata
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            # Find position in original text
            start_pos = text.find(chunk)
            if start_pos == -1:
                # Fallback for cleaned text differences
                start_pos = i * (len(text) // len(chunks)) if chunks else 0
            chunk_meta = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{i}",
                document_id=document_id,
                chunk_index=i,
                start_position=start_pos,
                end_position=start_pos + len(chunk),
                chunk_type=self.boundary_type.value,
            )
            chunk_metadata.append(chunk_meta)
        # Create chunked document
        chunked_doc = ChunkedDocument(
            document_id=document_id,
            original_text=text,
            chunks=chunks,
            chunk_metadata=chunk_metadata,
            chunking_strategy="text",
            embedding_constraints=self.embedding_capabilities,
        )
        # Validate result
        if not chunked_doc.validate():
            raise ChunkingProviderError("Chunk validation failed")
        logger.debug("Chunked document '%s' into %s chunks", document_id, len(chunks))
        return chunked_doc
    def chunk_query(self, query: str) -> List[str]:
        """Chunk a query if it exceeds embedding limits"""
        if len(query) <= self.max_sequence_length:
            return [query]
        # For queries, use simple word-based chunking without overlap
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
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for chunking"""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Normalize line breaks
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\r", "\n", text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """Chunk text by paragraph boundaries"""
        paragraphs = self._paragraph_breaks.split(text)
        chunks = []
        current_chunk = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            # If paragraph fits in current chunk, add it
            if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:  # +2 for \n\n
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)
                # If paragraph is too long, split it by sentences
                if len(paragraph) > self.chunk_size:
                    sentence_chunks = self._chunk_by_sentences(paragraph)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        return self._apply_overlap(chunks)
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text by sentence boundaries"""
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Check if sentence fits in current chunk
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:  # +1 for space
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)
                # If sentence is too long, split by words
                if len(sentence) > self.chunk_size:
                    word_chunks = self._chunk_by_words(sentence)
                    chunks.extend(word_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        return self._apply_overlap(chunks)
    def _chunk_by_words(self, text: str) -> List[str]:
        """Chunk text by word boundaries"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        for word in words:
            word_length = len(word) + (1 if current_chunk else 0)  # +1 for space
            if current_length + word_length > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    # Single word too long, truncate
                    chunks.append(word[: self.chunk_size])
            else:
                current_chunk.append(word)
                current_length += word_length
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return self._apply_overlap(chunks)
    def _chunk_by_characters(self, text: str) -> List[str]:
        """Chunk text by character count (fallback method)"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap_size):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        return chunks
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Simple sentence splitting - can be enhanced with more sophisticated methods
        sentences = self._sentence_endings.split(text)
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)
        return cleaned_sentences
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between chunks"""
        if len(chunks) <= 1 or self.overlap_size == 0:
            return chunks
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - no prefix overlap
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]
                # Get overlap text from end of previous chunk
                overlap_text = prev_chunk[-self.overlap_size :] if len(prev_chunk) > self.overlap_size else prev_chunk
                # Find good break point in overlap (prefer word boundaries)
                overlap_words = overlap_text.split()
                if len(overlap_words) > 1:
                    # Use last few words as overlap
                    overlap_text = " ".join(overlap_words[-3:])  # Last 3 words
                # Combine overlap with current chunk
                overlapped_chunk = overlap_text + " " + chunk
                # Ensure chunk doesn't exceed size limits
                if len(overlapped_chunk) > self.chunk_size:
                    overlapped_chunk = overlapped_chunk[: self.chunk_size]
                overlapped_chunks.append(overlapped_chunk)
        return overlapped_chunks
    def get_capabilities(self) -> Dict[str, Any]:
        """Get chunking capabilities"""
        capabilities = super().get_capabilities()
        capabilities.update(
            {
                "chunk_size": self.chunk_size,
                "overlap_size": self.overlap_size,
                "min_chunk_size": self.min_chunk_size,
                "boundary_type": self.boundary_type.value,
                "preserve_paragraphs": self.preserve_paragraphs,
            }
        )
        return capabilities
    def __repr__(self) -> str:
        return (
            f"TextChunkingProvider("
            f"chunk_size={self.chunk_size}, "
            f"overlap={self.overlap_size}, "
            f"boundary={self.boundary_type.value})"
        )
