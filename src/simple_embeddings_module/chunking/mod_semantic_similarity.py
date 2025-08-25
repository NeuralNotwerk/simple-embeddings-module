"""
Semantic similarity-based chunking provider.
Groups content by semantic similarity rather than arbitrary boundaries.
"""
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from .mod_chunking_base import ChunkingProviderBase

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
class SemanticSimilarityChunkingProvider(ChunkingProviderBase):
    """
    Chunks content by semantic similarity rather than arbitrary boundaries.
    Process:
    1. Split content into logical units (sentences, paragraphs, functions)
    2. Embed each unit individually
    3. Cluster units by semantic similarity
    4. Group similar units into coherent chunks
    5. Re-embed final chunks for storage
    """
    PROVIDER_NAME = "semantic_similarity"
    CONFIG_PARAMETERS = {
        "similarity_threshold": {
            "type": float,
            "default": 0.75,
            "description": "Minimum cosine similarity for grouping units (0.0-1.0)",
        },
        "min_cluster_size": {"type": int, "default": 2, "description": "Minimum number of units per cluster"},
        "max_cluster_size": {"type": int, "default": 10, "description": "Maximum number of units per cluster"},
        "unit_type": {
            "type": str,
            "default": "sentence",
            "description": "Type of logical unit: sentence, paragraph, or auto",
        },
        "clustering_method": {
            "type": str,
            "default": "threshold",
            "description": "Clustering method: threshold or hierarchical",
        },
    }
    def __init__(self):
        super().__init__()
        self.similarity_threshold = 0.75
        self.min_cluster_size = 2
        self.max_cluster_size = 10
        self.unit_type = "sentence"
        self.clustering_method = "threshold"
        self.embedding_provider = None
    def initialize(self, config: Dict[str, Any], embedding_provider) -> None:
        """Initialize the semantic similarity chunking provider."""
        self.embedding_provider = embedding_provider
        # Update configuration
        self.similarity_threshold = config.get("similarity_threshold", 0.75)
        self.min_cluster_size = config.get("min_cluster_size", 2)
        self.max_cluster_size = config.get("max_cluster_size", 10)
        self.unit_type = config.get("unit_type", "sentence")
        self.clustering_method = config.get("clustering_method", "threshold")
        # Validate configuration
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.unit_type not in ["sentence", "paragraph", "auto"]:
            raise ValueError("unit_type must be 'sentence', 'paragraph', or 'auto'")
        if self.clustering_method not in ["threshold", "hierarchical"]:
            raise ValueError("clustering_method must be 'threshold' or 'hierarchical'")
        logger.info("Initialized semantic similarity chunking with threshold=%s", self.similarity_threshold)
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text using semantic similarity clustering.
        Args:
            text: Input text to chunk
            metadata: Optional metadata (file_path, etc.)
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        try:
            # Step 1: Split into logical units
            units = self._split_into_units(text, metadata)
            if len(units) <= 1:
                # Not enough units to cluster
                return [{"text": text, "metadata": metadata or {}}]
            # Step 2: Embed each unit
            unit_embeddings = self._embed_units(units)
            # Step 3: Cluster by semantic similarity
            clusters = self._cluster_units(units, unit_embeddings)
            # Step 4: Create chunks from clusters
            chunks = self._create_chunks_from_clusters(clusters, metadata)
            logger.info("Created %s semantic chunks from %s units", len(chunks), len(units))
            return chunks
        except Exception as e:
            logger.error("Error in semantic similarity chunking: %s", e)
            # Fallback to single chunk
            return [{"text": text, "metadata": metadata or {}}]
    def _split_into_units(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Split text into logical units based on unit_type."""
        if self.unit_type == "sentence":
            return self._split_into_sentences(text)
        elif self.unit_type == "paragraph":
            return self._split_into_paragraphs(text)
        else:  # auto
            return self._auto_split_units(text, metadata)
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Enhanced sentence splitting pattern
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_pattern, text.strip())
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r"\n\s*\n", text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    def _auto_split_units(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Automatically determine best unit type based on content."""
        # Check if it's code (has file extension or code patterns)
        if metadata and metadata.get("file_path"):
            file_path = metadata["file_path"]
            code_extensions = {".py", ".js", ".java", ".cpp", ".c", ".rs", ".go", ".rb", ".php"}
            if any(file_path.endswith(ext) for ext in code_extensions):
                # For code, use line-based splitting for now
                # TODO: Integrate with tree-sitter for function-level units
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                return lines
        # For regular text, prefer paragraphs if they exist, otherwise sentences
        paragraphs = self._split_into_paragraphs(text)
        if len(paragraphs) > 1:
            return paragraphs
        else:
            return self._split_into_sentences(text)
    def _embed_units(self, units: List[str]) -> np.ndarray:
        """Embed each unit individually."""
        embeddings = []
        for unit in units:
            try:
                # Use the embedding provider to get embeddings
                embedding = self.embedding_provider.embed_text(unit)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning("Failed to embed unit: %s", e)
                # Create zero embedding as fallback
                dim = getattr(self.embedding_provider, "embedding_dimension", 384)
                embeddings.append(np.zeros(dim))
        return np.array(embeddings)
    def _cluster_units(self, units: List[str], embeddings: np.ndarray) -> List[List[int]]:
        """Cluster units by semantic similarity."""
        if self.clustering_method == "threshold":
            return self._threshold_clustering(units, embeddings)
        else:
            return self._hierarchical_clustering(units, embeddings)
    def _threshold_clustering(self, units: List[str], embeddings: np.ndarray) -> List[List[int]]:
        """Simple threshold-based clustering."""
        clusters = []
        used = set()
        for i in range(len(units)):
            if i in used:
                continue
            cluster = [i]
            used.add(i)
            # Find similar units
            for j in range(i + 1, len(units)):
                if j in used:
                    continue
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if similarity >= self.similarity_threshold and len(cluster) < self.max_cluster_size:
                    cluster.append(j)
                    used.add(j)
            clusters.append(cluster)
        return clusters
    def _hierarchical_clustering(self, units: List[str], embeddings: np.ndarray) -> List[List[int]]:
        """Hierarchical clustering using sklearn."""
        try:
            # Use agglomerative clustering
            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=1.0 - self.similarity_threshold, linkage="average", metric="cosine"
            )
            cluster_labels = clustering.fit_predict(embeddings)
            # Group indices by cluster label
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)
            # Filter clusters by size constraints
            filtered_clusters = []
            for cluster in clusters.values():
                if self.min_cluster_size <= len(cluster) <= self.max_cluster_size:
                    filtered_clusters.append(cluster)
                else:
                    # Split large clusters or merge small ones
                    if len(cluster) > self.max_cluster_size:
                        # Split into smaller clusters
                        for i in range(0, len(cluster), self.max_cluster_size):
                            sub_cluster = cluster[i : i + self.max_cluster_size]
                            if len(sub_cluster) >= self.min_cluster_size:
                                filtered_clusters.append(sub_cluster)
                    else:
                        # Add small clusters as individual items
                        for idx in cluster:
                            filtered_clusters.append([idx])
            return filtered_clusters
        except Exception as e:
            logger.warning("Hierarchical clustering failed: %s, falling back to threshold", e)
            return self._threshold_clustering(units, embeddings)
    def _create_chunks_from_clusters(
        self, clusters: List[List[int]], metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create final chunks from unit clusters."""
        chunks = []
        for i, cluster in enumerate(clusters):
            # Combine units in cluster
            cluster_texts = []
            for unit_idx in cluster:
                if unit_idx < len(self._current_units):  # Safety check
                    cluster_texts.append(self._current_units[unit_idx])
            if cluster_texts:
                chunk_text = "\n\n".join(cluster_texts)
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata["cluster_id"] = i
                chunk_metadata["cluster_size"] = len(cluster)
                chunk_metadata["semantic_chunking"] = True
                chunks.append({"text": chunk_text, "metadata": chunk_metadata})
        return chunks
    def get_chunk_size_estimate(self, text: str) -> int:
        """Estimate chunk size for semantic similarity chunking."""
        # This is harder to estimate since it depends on semantic similarity
        # Return a rough estimate based on unit count and clustering parameters
        units = self._split_into_units(text)
        estimated_clusters = max(1, len(units) // self.max_cluster_size)
        avg_chunk_size = len(text) // estimated_clusters
        return avg_chunk_size
    def validate_configuration(self, config: Dict[str, Any], embedding_provider) -> List[str]:
        """Validate semantic similarity chunking configuration."""
        errors = []
        similarity_threshold = config.get("similarity_threshold", 0.75)
        if not isinstance(similarity_threshold, (int, float)) or not (0.0 <= similarity_threshold <= 1.0):
            errors.append("similarity_threshold must be a number between 0.0 and 1.0")
        min_cluster_size = config.get("min_cluster_size", 2)
        if not isinstance(min_cluster_size, int) or min_cluster_size < 1:
            errors.append("min_cluster_size must be a positive integer")
        max_cluster_size = config.get("max_cluster_size", 10)
        if not isinstance(max_cluster_size, int) or max_cluster_size < min_cluster_size:
            errors.append("max_cluster_size must be an integer >= min_cluster_size")
        unit_type = config.get("unit_type", "sentence")
        if unit_type not in ["sentence", "paragraph", "auto"]:
            errors.append("unit_type must be 'sentence', 'paragraph', or 'auto'")
        clustering_method = config.get("clustering_method", "threshold")
        if clustering_method not in ["threshold", "hierarchical"]:
            errors.append("clustering_method must be 'threshold' or 'hierarchical'")
        return errors
# Store current units for cluster creation (temporary solution)
SemanticSimilarityChunkingProvider._current_units = []
def _store_units_for_clustering(self, units):
    """Temporary method to store units for cluster creation."""
    self._current_units = units
# Monkey patch the method
SemanticSimilarityChunkingProvider._store_units_for_clustering = _store_units_for_clustering
# Update the chunk_text method to store units
original_chunk_text = SemanticSimilarityChunkingProvider.chunk_text
def enhanced_chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Enhanced chunk_text that stores units for clustering."""
    if not text.strip():
        return []
    try:
        # Step 1: Split into logical units
        units = self._split_into_units(text, metadata)
        self._store_units_for_clustering(units)  # Store for later use
        if len(units) <= 1:
            return [{"text": text, "metadata": metadata or {}}]
        # Continue with original logic
        unit_embeddings = self._embed_units(units)
        clusters = self._cluster_units(units, unit_embeddings)
        chunks = self._create_chunks_from_clusters(clusters, metadata)
        logger.info("Created %s semantic chunks from %s units", len(chunks), len(units))
        return chunks
    except Exception as e:
        logger.error("Error in semantic similarity chunking: %s", e)
        return [{"text": text, "metadata": metadata or {}}]
SemanticSimilarityChunkingProvider.chunk_text = enhanced_chunk_text
