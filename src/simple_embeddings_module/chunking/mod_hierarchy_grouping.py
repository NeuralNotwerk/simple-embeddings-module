"""Hierarchy-constrained semantic grouping for code chunks."""
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from ..embeddings.mod_embeddings_base import EmbeddingProviderBase
from .mod_chunking_ts import ts_get_code_chunks

logger = logging.getLogger(__name__)
@dataclass
class ChunkMetadata:
    """Rich metadata for code chunks with hierarchy information."""
    id: str
    text: str
    embedding: Optional[torch.Tensor]
    file_path: str
    line_start: int
    line_end: int
    parent_hierarchy: List[str]  # ["file.py", "class_name", "function_name"]
    chunk_type: str  # "function", "class", "module", "import", etc.
    language: str
@dataclass
class SemanticGroup:
    """Semantic group of related chunks within same hierarchy scope."""
    group_id: str
    parent_scope: str  # "file.py::class_name"
    chunk_ids: List[str]
    group_embedding: Optional[torch.Tensor]
    similarity_threshold: float
    group_theme: Optional[str]
    group_type: str  # "related_functions", "related_methods", etc.
    creation_timestamp: datetime
class HierarchyConstrainedGrouping:
    """Implements hierarchy-constrained semantic grouping for code chunks."""
    def __init__(self, embedding_provider: EmbeddingProviderBase):
        """Initialize with an embedding provider."""
        self.embedding_provider = embedding_provider
        self.similarity_threshold = 0.7
        self.min_group_size = 2
        self.max_group_size = 10

    def extract_chunks_with_metadata(self, file_path: str) -> List[ChunkMetadata]:
        """Extract semantic chunks with rich metadata from a code file."""
        logger.info("Extracting chunks with metadata from %s", file_path)
        # Get semantic chunks using existing tree-sitter functionality
        chunks = ts_get_code_chunks(file_path)
        if not chunks:
            logger.warning("No chunks extracted from %s", file_path)
            return []
        # Read the original file to get line information
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception as e:
            logger.error("Failed to read file %s: %s", file_path, e)
            return []
        file_lines = file_content.split("\n")
        chunk_metadata_list = []
        # Process each chunk to extract metadata
        for i, chunk_text in enumerate(chunks):
            try:
                metadata = self._extract_chunk_metadata(chunk_text, file_path, file_lines, i)
                if metadata:
                    chunk_metadata_list.append(metadata)
            except Exception as e:
                logger.error("Failed to extract metadata for chunk %s in %s: %s", i, file_path, e)
                continue
        # Generate embeddings for all chunks
        if chunk_metadata_list:
            self._generate_embeddings(chunk_metadata_list)
        logger.info("Extracted %s chunks with metadata from %s", len(chunk_metadata_list), file_path)
        return chunk_metadata_list

    def _extract_chunk_metadata(
        self, chunk_text: str, file_path: str, file_lines: List[str], chunk_index: int
    ) -> Optional[ChunkMetadata]:
        """Extract metadata for a single chunk."""
        # Find line numbers by searching for the chunk in the file
        line_start, line_end = self._find_chunk_lines(chunk_text, file_lines)
        # Extract hierarchy information from the chunk
        parent_hierarchy, chunk_type = self._analyze_chunk_hierarchy(chunk_text, file_path)
        # Generate unique chunk ID
        chunk_id = self._generate_chunk_id(file_path, parent_hierarchy, chunk_index)
        # Detect language from file extension
        language = self._detect_language(file_path)
        return ChunkMetadata(
            id=chunk_id,
            text=chunk_text,
            embedding=None,  # Will be filled later
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            parent_hierarchy=parent_hierarchy,
            chunk_type=chunk_type,
            language=language,
        )

    def _find_chunk_lines(self, chunk_text: str, file_lines: List[str]) -> Tuple[int, int]:
        """Find the line numbers where a chunk appears in the file."""
        chunk_lines = chunk_text.strip().split("\n")
        if not chunk_lines:
            return 1, 1
        first_line = chunk_lines[0].strip()
        last_line = chunk_lines[-1].strip()
        # Search for the first line
        line_start = 1
        for i, file_line in enumerate(file_lines, 1):
            if first_line in file_line.strip():
                line_start = i
                break
        # Calculate end line
        line_end = line_start + len(chunk_lines) - 1
        return line_start, line_end

    def _analyze_chunk_hierarchy(self, chunk_text: str, file_path: str) -> Tuple[List[str], str]:
        """Analyze chunk to extract hierarchy and type information."""
        file_name = Path(file_path).name
        hierarchy = [file_name]
        chunk_type = "code"
        lines = chunk_text.strip().split("\n")
        if not lines:
            return hierarchy, chunk_type
        # Analyze first few lines to determine hierarchy and type
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            # Python patterns
            if line.startswith("class "):
                class_name = self._extract_name_from_definition(line, "class")
                hierarchy.append(class_name)
                chunk_type = "class"
                break
            elif line.startswith("def ") or line.startswith("async def "):
                func_name = self._extract_name_from_definition(line, "de")
                hierarchy.append(func_name)
                chunk_type = "function"
                break
            elif line.startswith("@"):
                chunk_type = "decorated"
                continue
            # JavaScript/TypeScript patterns
            elif "function " in line:
                func_name = self._extract_js_function_name(line)
                if func_name:
                    hierarchy.append(func_name)
                    chunk_type = "function"
                    break
            elif line.startswith("class "):
                class_name = self._extract_name_from_definition(line, "class")
                hierarchy.append(class_name)
                chunk_type = "class"
                break
            # Java patterns
            elif "public class " in line or "private class " in line:
                class_name = self._extract_java_class_name(line)
                if class_name:
                    hierarchy.append(class_name)
                    chunk_type = "class"
                    break
            elif ("public " in line or "private " in line) and "(" in line and ")" in line:
                method_name = self._extract_java_method_name(line)
                if method_name:
                    hierarchy.append(method_name)
                    chunk_type = "method"
                    break
            # Import statements
            elif line.startswith("import ") or line.startswith("from "):
                chunk_type = "import"
                break
        return hierarchy, chunk_type

    def _extract_name_from_definition(self, line: str, keyword: str) -> str:
        """Extract name from a definition line (class, def, etc.)."""
        try:
            # Remove the keyword and everything after '(' or ':'
            after_keyword = line.split(keyword, 1)[1].strip()
            name = after_keyword.split("(")[0].split(":")[0].strip()
            return name if name else "unknown"
        except Exception:
            return "unknown"

    def _extract_js_function_name(self, line: str) -> Optional[str]:
        """Extract function name from JavaScript/TypeScript function definition."""
        try:
            if "function " in line:
                parts = line.split("function ")
                if len(parts) > 1:
                    name_part = parts[1].split("(")[0].strip()
                    return name_part if name_part else None
            return None
        except Exception:
            return None

    def _extract_java_class_name(self, line: str) -> Optional[str]:
        """Extract class name from Java class definition."""
        try:
            if "class " in line:
                parts = line.split("class ")
                if len(parts) > 1:
                    name_part = parts[1].split()[0].strip()
                    return name_part if name_part else None
            return None
        except Exception:
            return None

    def _extract_java_method_name(self, line: str) -> Optional[str]:
        """Extract method name from Java method definition."""
        try:
            # Look for pattern: [modifiers] returnType methodName(
            parts = line.split("(")[0].strip().split()
            if len(parts) >= 2:
                return parts[-1]  # Last part before '(' should be method name
            return None
        except Exception:
            return None

    def _generate_chunk_id(self, file_path: str, hierarchy: List[str], chunk_index: int) -> str:
        """Generate a unique ID for a chunk."""
        file_name = Path(file_path).stem
        hierarchy_str = "_".join(h.replace(" ", "_") for h in hierarchy[1:])  # Skip file name
        if hierarchy_str:
            return f"{file_name}_{hierarchy_str}_{chunk_index}"
        else:
            return f"{file_name}_chunk_{chunk_index}"

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        extension = Path(file_path).suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".lua": "lua",
            ".sh": "bash",
            ".bash": "bash",
        }
        return language_map.get(extension, "unknown")

    def _generate_embeddings(self, chunks: List[ChunkMetadata]) -> None:
        """Generate embeddings for all chunks using the embedding provider."""
        logger.info("Generating embeddings for %s chunks", len(chunks))
        try:
            # Extract and truncate text from all chunks
            texts = []
            for chunk in chunks:
                text = chunk.text
                # Very aggressive truncation for all-MiniLM-L6-v2 (256 tokens max)
                # Use only first 200 characters to be safe
                max_chars = 200
                if len(text) > max_chars:
                    text = text[:max_chars] + "..."
                    logger.debug("Truncated chunk %s from %s to %s chars", chunk.id, len(chunk.text), len(text))
                texts.append(text)
            # Generate embeddings in batch
            embeddings = self.embedding_provider.embed_documents(texts)
            # Assign embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
        except Exception as e:
            logger.error("Failed to generate embeddings: %s", e)
            # Set empty embeddings as fallback
            embedding_dim = self.embedding_provider.get_embedding_dimension()
            for chunk in chunks:
                chunk.embedding = torch.zeros(embedding_dim)

    def group_chunks_within_hierarchy(self, chunks: List[ChunkMetadata]) -> List[SemanticGroup]:
        """Group chunks within same hierarchy scope based on semantic similarity."""
        logger.info("Grouping %s chunks within hierarchy constraints", len(chunks))
        # Group chunks by parent scope
        scope_groups = self._group_by_parent_scope(chunks)
        semantic_groups = []
        # Process each scope independently
        for parent_scope, scope_chunks in scope_groups.items():
            if len(scope_chunks) < self.min_group_size:
                logger.debug("Skipping scope %s with only %s chunks", parent_scope, len(scope_chunks))
                continue
            # Find semantic groups within this scope
            scope_semantic_groups = self._find_semantic_groups_in_scope(parent_scope, scope_chunks)
            semantic_groups.extend(scope_semantic_groups)
        logger.info("Created %s semantic groups", len(semantic_groups))
        return semantic_groups

    def _group_by_parent_scope(self, chunks: List[ChunkMetadata]) -> Dict[str, List[ChunkMetadata]]:
        """Group chunks by their parent scope."""
        scope_groups = {}
        for chunk in chunks:
            # Create parent scope string (exclude the last element which is the chunk itself)
            if len(chunk.parent_hierarchy) > 1:
                parent_scope = "::".join(chunk.parent_hierarchy[:-1])
            else:
                parent_scope = chunk.parent_hierarchy[0] if chunk.parent_hierarchy else "root"
            if parent_scope not in scope_groups:
                scope_groups[parent_scope] = []
            scope_groups[parent_scope].append(chunk)
        return scope_groups

    def _find_semantic_groups_in_scope(self, parent_scope: str, chunks: List[ChunkMetadata]) -> List[SemanticGroup]:
        """Find semantic groups within a single parent scope."""
        if len(chunks) < self.min_group_size:
            return []
        # Extract embeddings
        embeddings = []
        valid_chunks = []
        for chunk in chunks:
            if chunk.embedding is not None:
                # Convert to CPU first if on MPS/CUDA device
                embedding_cpu = chunk.embedding.cpu().numpy()
                embeddings.append(embedding_cpu)
                valid_chunks.append(chunk)
        if len(embeddings) < self.min_group_size:
            logger.debug("Not enough valid embeddings in scope %s", parent_scope)
            return []
        # Compute similarity matrix
        embeddings_array = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        # Find groups using similarity clustering
        groups = self._cluster_by_similarity(similarity_matrix, valid_chunks)
        # Create SemanticGroup objects
        semantic_groups = []
        for i, group_chunk_indices in enumerate(groups):
            if len(group_chunk_indices) >= self.min_group_size:
                group_chunks = [valid_chunks[idx] for idx in group_chunk_indices]
                semantic_group = self._create_semantic_group(parent_scope, group_chunks, i)
                semantic_groups.append(semantic_group)
        return semantic_groups

    def _cluster_by_similarity(self, similarity_matrix: np.ndarray, chunks: List[ChunkMetadata]) -> List[List[int]]:
        """Cluster chunks by similarity using a simple threshold-based approach."""
        n_chunks = len(chunks)
        visited = set()
        groups = []
        for i in range(n_chunks):
            if i in visited:
                continue
            # Start a new group
            current_group = [i]
            visited.add(i)
            # Find similar chunks
            for j in range(i + 1, n_chunks):
                if j in visited:
                    continue
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    current_group.append(j)
                    visited.add(j)
            # Only keep groups with minimum size
            if len(current_group) >= self.min_group_size:
                groups.append(current_group)
        return groups

    def _create_semantic_group(self, parent_scope: str, chunks: List[ChunkMetadata], group_index: int) -> SemanticGroup:
        """Create a SemanticGroup from a list of chunks."""
        # Generate group ID
        scope_clean = parent_scope.replace("::", "_").replace(".", "_")
        group_id = f"{scope_clean}_group_{group_index}"
        # Extract chunk IDs
        chunk_ids = [chunk.id for chunk in chunks]
        # Generate group embedding by averaging chunk embeddings
        group_embedding = self._create_group_embedding(chunks)
        # Determine group theme and type
        group_theme = self._infer_group_theme(chunks)
        group_type = self._infer_group_type(chunks)
        return SemanticGroup(
            group_id=group_id,
            parent_scope=parent_scope,
            chunk_ids=chunk_ids,
            group_embedding=group_embedding,
            similarity_threshold=self.similarity_threshold,
            group_theme=group_theme,
            group_type=group_type,
            creation_timestamp=datetime.now(),
        )

    def _create_group_embedding(self, chunks: List[ChunkMetadata]) -> Optional[torch.Tensor]:
        """Create a group embedding by combining chunk texts and re-embedding."""
        try:
            # Combine all chunk texts
            combined_text = "\n\n".join(chunk.text for chunk in chunks)
            # Very aggressive truncation for short sequence models
            max_chars = 200
            if len(combined_text) > max_chars:
                combined_text = combined_text[:max_chars] + "..."
                logger.debug("Truncated group text to %s chars", len(combined_text))
            # Generate new embedding for the combined text
            group_embedding = self.embedding_provider.embed_query(combined_text)
            return group_embedding
        except Exception as e:
            logger.error("Failed to create group embedding: %s", e)
            return None

    def _infer_group_theme(self, chunks: List[ChunkMetadata]) -> Optional[str]:
        """Infer a theme for the group based on chunk content."""
        # Simple heuristic: look for common keywords in chunk texts
        all_text = " ".join(chunk.text.lower() for chunk in chunks)
        # Common themes in code
        themes = {
            "database": ["db", "database", "sql", "query", "connection", "table"],
            "authentication": ["auth", "login", "password", "token", "user", "session"],
            "data_processing": ["process", "transform", "parse", "convert", "format"],
            "validation": ["validate", "check", "verify", "ensure", "assert"],
            "utility": ["util", "helper", "common", "shared", "tool"],
            "api": ["api", "endpoint", "route", "request", "response", "http"],
            "testing": ["test", "mock", "assert", "expect", "should"],
            "configuration": ["config", "setting", "option", "parameter", "env"],
        }
        for theme, keywords in themes.items():
            if any(keyword in all_text for keyword in keywords):
                return theme
        return None

    def _infer_group_type(self, chunks: List[ChunkMetadata]) -> str:
        """Infer the type of group based on chunk types."""
        chunk_types = [chunk.chunk_type for chunk in chunks]
        if all(ct == "function" for ct in chunk_types):
            return "related_functions"
        elif all(ct == "method" for ct in chunk_types):
            return "related_methods"
        elif all(ct == "class" for ct in chunk_types):
            return "related_classes"
        elif "function" in chunk_types and "method" in chunk_types:
            return "mixed_functions_methods"
        else:
            return "mixed_code_units"

def process_file_with_hierarchy_grouping(
    file_path: str, embedding_provider: EmbeddingProviderBase, similarity_threshold: float = 0.7
) -> Tuple[List[ChunkMetadata], List[SemanticGroup]]:
    """Process a single file with hierarchy-constrained semantic grouping."""
    grouper = HierarchyConstrainedGrouping(embedding_provider)
    grouper.similarity_threshold = similarity_threshold
    # Extract chunks with metadata
    chunks = grouper.extract_chunks_with_metadata(file_path)
    # Group chunks within hierarchy constraints
    groups = grouper.group_chunks_within_hierarchy(chunks)
    return chunks, groups

def process_multiple_files_with_grouping(
    file_paths: List[str], embedding_provider: EmbeddingProviderBase, similarity_threshold: float = 0.7
) -> Tuple[List[ChunkMetadata], List[SemanticGroup]]:
    """Process multiple files with hierarchy-constrained semantic grouping."""
    all_chunks = []
    all_groups = []
    for file_path in file_paths:
        try:
            chunks, groups = process_file_with_hierarchy_grouping(file_path, embedding_provider, similarity_threshold)
            all_chunks.extend(chunks)
            all_groups.extend(groups)
        except Exception as e:
            logger.error("Failed to process file %s: %s", file_path, e)
            continue
    return all_chunks, all_groups
