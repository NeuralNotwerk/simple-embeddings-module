"""Integration module for hierarchy-constrained grouping with SEM architecture."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..embeddings.mod_embeddings_base import EmbeddingProviderBase
from .mod_chunking_base import ChunkingProviderBase
from .mod_hierarchy_grouping import (
    ChunkMetadata,
    HierarchyConstrainedGrouping,
    SemanticGroup,
    process_file_with_hierarchy_grouping,
    process_multiple_files_with_grouping,
)

logger = logging.getLogger(__name__)
class HierarchyGroupingProvider(ChunkingProviderBase):
    """Chunking provider that implements hierarchy-constrained semantic grouping."""
    def __init__(self, embedding_provider: EmbeddingProviderBase, config: Optional[Dict[str, Any]] = None):
        """Initialize the hierarchy grouping provider."""
        super().__init__(config or {})
        self.embedding_provider = embedding_provider
        self.grouper = HierarchyConstrainedGrouping(embedding_provider)
        # Configure from config
        if config:
            self.grouper.similarity_threshold = config.get("similarity_threshold", 0.7)
            self.grouper.min_group_size = config.get("min_group_size", 2)
            self.grouper.max_group_size = config.get("max_group_size", 10)

    def chunk_text(self, text: str, file_path: Optional[str] = None) -> List[str]:
        """Chunk text and return simple text chunks (for compatibility)."""
        if not file_path:
            # Fallback to basic chunking if no file path provided
            logger.warning("No file path provided for hierarchy grouping, using basic chunking")
            return [text]
        try:
            chunks, _ = process_file_with_hierarchy_grouping(
                file_path, self.embedding_provider, self.grouper.similarity_threshold
            )
            return [chunk.text for chunk in chunks]
        except Exception as e:
            logger.error("Failed to chunk text with hierarchy grouping: %s", e)
            return [text]

    def chunk_file(self, file_path: str) -> List[str]:
        """Chunk file and return simple text chunks (for compatibility)."""
        try:
            chunks, _ = process_file_with_hierarchy_grouping(
                file_path, self.embedding_provider, self.grouper.similarity_threshold
            )
            return [chunk.text for chunk in chunks]
        except Exception as e:
            logger.error("Failed to chunk file with hierarchy grouping: %s", e)
            return []

    def chunk_file_with_metadata(self, file_path: str) -> Tuple[List[ChunkMetadata], List[SemanticGroup]]:
        """Chunk file and return chunks with metadata and semantic groups."""
        return process_file_with_hierarchy_grouping(
            file_path, self.embedding_provider, self.grouper.similarity_threshold
        )

    def chunk_multiple_files_with_metadata(
        self, file_paths: List[str]
    ) -> Tuple[List[ChunkMetadata], List[SemanticGroup]]:
        """Chunk multiple files and return chunks with metadata and semantic groups."""
        return process_multiple_files_with_grouping(
            file_paths, self.embedding_provider, self.grouper.similarity_threshold
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities."""
        return {
            "chunk_types": ["semantic", "code", "hierarchy_grouped"],
            "supports_files": True,
            "supports_text": True,
            "supports_metadata": True,
            "supports_grouping": True,
            "hierarchy_constrained": True,
            "embedding_provider": self.embedding_provider.__class__.__name__,
            "similarity_threshold": self.grouper.similarity_threshold,
            "min_group_size": self.grouper.min_group_size,
            "max_group_size": self.grouper.max_group_size,
        }

    def get_config_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration parameters."""
        return {
            "similarity_threshold": {
                "type": float,
                "default": 0.7,
                "description": "Minimum similarity threshold for grouping chunks within same hierarchy",
            },
            "min_group_size": {
                "type": int,
                "default": 2,
                "description": "Minimum number of chunks required to form a semantic group",
            },
            "max_group_size": {
                "type": int,
                "default": 10,
                "description": "Maximum number of chunks allowed in a semantic group",
            },
            "fallback_to_basic": {
                "type": bool,
                "default": True,
                "description": "Fall back to basic chunking if hierarchy grouping fails",
            },
        }

class HierarchyGroupingStorage:
    """Storage utilities for hierarchy-constrained grouping data."""
    @staticmethod
    def serialize_chunks_and_groups(chunks: List[ChunkMetadata], groups: List[SemanticGroup]) -> Dict[str, Any]:
        """Serialize chunks and groups to JSON-compatible format."""
        serialized_chunks = []
        for chunk in chunks:
            chunk_data = {
                "id": chunk.id,
                "text": chunk.text,
                "embedding": chunk.embedding.tolist() if chunk.embedding is not None else None,
                "metadata": {
                    "file_path": chunk.file_path,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "parent_hierarchy": chunk.parent_hierarchy,
                    "chunk_type": chunk.chunk_type,
                    "language": chunk.language,
                },
            }
            serialized_chunks.append(chunk_data)
        serialized_groups = []
        for group in groups:
            group_data = {
                "group_id": group.group_id,
                "parent_scope": group.parent_scope,
                "chunk_ids": group.chunk_ids,
                "group_embedding": group.group_embedding.tolist() if group.group_embedding is not None else None,
                "similarity_threshold": group.similarity_threshold,
                "group_theme": group.group_theme,
                "group_type": group.group_type,
                "creation_timestamp": group.creation_timestamp.isoformat(),
            }
            serialized_groups.append(group_data)
        return {
            "metadata": {
                "version": "2.0",
                "created": datetime.now().isoformat(),
                "total_chunks": len(chunks),
                "total_groups": len(groups),
                "hierarchy_constrained": True,
            },
            "chunks": serialized_chunks,
            "semantic_groups": serialized_groups,
        }

    @staticmethod
    def deserialize_chunks_and_groups(data: Dict[str, Any]) -> Tuple[List[ChunkMetadata], List[SemanticGroup]]:
        """Deserialize chunks and groups from JSON-compatible format."""
        import torch
        chunks = []
        for chunk_data in data.get("chunks", []):
            embedding = None
            if chunk_data.get("embedding"):
                embedding = torch.tensor(chunk_data["embedding"])
            metadata = chunk_data.get("metadata", {})
            chunk = ChunkMetadata(
                id=chunk_data["id"],
                text=chunk_data["text"],
                embedding=embedding,
                file_path=metadata.get("file_path", ""),
                line_start=metadata.get("line_start", 1),
                line_end=metadata.get("line_end", 1),
                parent_hierarchy=metadata.get("parent_hierarchy", []),
                chunk_type=metadata.get("chunk_type", "code"),
                language=metadata.get("language", "unknown"),
            )
            chunks.append(chunk)
        groups = []
        for group_data in data.get("semantic_groups", []):
            group_embedding = None
            if group_data.get("group_embedding"):
                group_embedding = torch.tensor(group_data["group_embedding"])
            creation_timestamp = datetime.fromisoformat(
                group_data.get("creation_timestamp", datetime.now().isoformat())
            )
            group = SemanticGroup(
                group_id=group_data["group_id"],
                parent_scope=group_data["parent_scope"],
                chunk_ids=group_data["chunk_ids"],
                group_embedding=group_embedding,
                similarity_threshold=group_data.get("similarity_threshold", 0.7),
                group_theme=group_data.get("group_theme"),
                group_type=group_data.get("group_type", "unknown"),
                creation_timestamp=creation_timestamp,
            )
            groups.append(group)
        return chunks, groups

    @staticmethod
    def save_to_file(chunks: List[ChunkMetadata], groups: List[SemanticGroup], file_path: str) -> None:
        """Save chunks and groups to a JSON file."""
        data = HierarchyGroupingStorage.serialize_chunks_and_groups(chunks, groups)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved %s chunks and %s groups to %s", len(chunks), len(groups), file_path)

    @staticmethod
    def load_from_file(file_path: str) -> Tuple[List[ChunkMetadata], List[SemanticGroup]]:
        """Load chunks and groups from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks, groups = HierarchyGroupingStorage.deserialize_chunks_and_groups(data)
        logger.info("Loaded %s chunks and %s groups from %s", len(chunks), len(groups), file_path)
        return chunks, groups

class HierarchyGroupingSearch:
    """Search utilities for hierarchy-constrained grouping."""
    def __init__(self, embedding_provider: EmbeddingProviderBase):
        """Initialize with embedding provider."""
        self.embedding_provider = embedding_provider

    def search_chunks_and_groups(
        self,
        query: str,
        chunks: List[ChunkMetadata],
        groups: List[SemanticGroup],
        top_k: int = 5,
        include_groups: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search both individual chunks and semantic groups."""
        from sklearn.metrics.pairwise import cosine_similarity
        # Embed the query
        query_embedding = self.embedding_provider.embed_query(query)
        query_np = query_embedding.cpu().numpy().reshape(1, -1)  # Convert to CPU first
        results = []
        # Search individual chunks
        for chunk in chunks:
            if chunk.embedding is not None:
                chunk_np = chunk.embedding.cpu().numpy().reshape(1, -1)  # Convert to CPU first
                similarity = cosine_similarity(query_np, chunk_np)[0][0]
                results.append(
                    {
                        "type": "chunk",
                        "id": chunk.id,
                        "text": chunk.text,
                        "score": float(similarity),
                        "metadata": {
                            "file_path": chunk.file_path,
                            "line_start": chunk.line_start,
                            "line_end": chunk.line_end,
                            "parent_hierarchy": chunk.parent_hierarchy,
                            "chunk_type": chunk.chunk_type,
                            "language": chunk.language,
                        },
                    }
                )
        # Search semantic groups if requested
        if include_groups:
            for group in groups:
                if group.group_embedding is not None:
                    group_np = group.group_embedding.cpu().numpy().reshape(1, -1)  # Convert to CPU first
                    similarity = cosine_similarity(query_np, group_np)[0][0]
                    # Get combined text from all chunks in the group
                    group_chunks = [c for c in chunks if c.id in group.chunk_ids]
                    combined_text = "\n\n".join(c.text for c in group_chunks)
                    results.append(
                        {
                            "type": "group",
                            "id": group.group_id,
                            "text": combined_text,
                            "score": float(similarity),
                            "metadata": {
                                "parent_scope": group.parent_scope,
                                "chunk_ids": group.chunk_ids,
                                "group_theme": group.group_theme,
                                "group_type": group.group_type,
                                "chunk_count": len(group.chunk_ids),
                            },
                        }
                    )
        # Sort by similarity score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search_within_scope(
        self, query: str, scope: str, chunks: List[ChunkMetadata], groups: List[SemanticGroup], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search within a specific hierarchy scope."""
        # Filter chunks and groups by scope
        scope_chunks = [c for c in chunks if scope in "::".join(c.parent_hierarchy)]
        scope_groups = [g for g in groups if g.parent_scope == scope]
        return self.search_chunks_and_groups(query, scope_chunks, scope_groups, top_k)

def create_hierarchy_grouping_demo(
    file_paths: List[str], embedding_provider: EmbeddingProviderBase, output_file: Optional[str] = None
) -> Tuple[List[ChunkMetadata], List[SemanticGroup]]:
    """Create a demo of hierarchy-constrained grouping with multiple files."""
    logger.info("Creating hierarchy grouping demo with %s files", len(file_paths))
    # Process all files
    chunks, groups = process_multiple_files_with_grouping(file_paths, embedding_provider, similarity_threshold=0.6)
    # Save to file if requested
    if output_file:
        HierarchyGroupingStorage.save_to_file(chunks, groups, output_file)
    # Print summary
    print("🎯 Hierarchy-Constrained Grouping Demo Results:")
    print(f"   📦 Total chunks: {len(chunks)}")
    print(f"   🔗 Total groups: {len(groups)}")
    # Group by file
    file_stats = {}
    for chunk in chunks:
        file_path = chunk.file_path
        if file_path not in file_stats:
            file_stats[file_path] = {"chunks": 0, "groups": 0}
        file_stats[file_path]["chunks"] += 1
    for group in groups:
        # Find which file this group belongs to by checking first chunk
        if group.chunk_ids:
            first_chunk = next((c for c in chunks if c.id == group.chunk_ids[0]), None)
            if first_chunk:
                file_path = first_chunk.file_path
                if file_path in file_stats:
                    file_stats[file_path]["groups"] += 1
    print("\n📊 Per-file statistics:")
    for file_path, stats in file_stats.items():
        file_name = Path(file_path).name
        logger.info("File %s: %s chunks, %s groups", file_name, stats["chunks"], stats["groups"])
    return chunks, groups
