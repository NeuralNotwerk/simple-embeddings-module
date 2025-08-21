"""
Semantic code clustering provider.
Groups code functions/methods by semantic similarity within parent scopes.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .mod_chunking_base import ChunkingProviderBase
from .mod_chunking_ts import ts_get_code_chunks
from .mod_chunking_ts_lang_lazy import TreeSitterLanguageLoader
import tree_sitter

logger = logging.getLogger(__name__)

class SemanticCodeClusteringProvider(ChunkingProviderBase):
    """
    Advanced code chunking that clusters functions by semantic similarity
    within parent scopes (classes, modules, namespaces).
    
    Instead of grouping adjacent functions, this clusters semantically
    similar functions together regardless of their physical position.
    """
    
    PROVIDER_NAME = "semantic_code"
    
    CONFIG_PARAMETERS = {
        'similarity_threshold': {
            'type': float,
            'default': 0.7,
            'description': 'Minimum cosine similarity for grouping functions (0.0-1.0)'
        },
        'min_cluster_size': {
            'type': int,
            'default': 1,
            'description': 'Minimum number of functions per cluster'
        },
        'max_cluster_size': {
            'type': int,
            'default': 8,
            'description': 'Maximum number of functions per cluster'
        },
        'include_context': {
            'type': bool,
            'default': True,
            'description': 'Include parent class/module context in chunks'
        },
        'cluster_scope': {
            'type': str,
            'default': 'class',
            'description': 'Clustering scope: function, class, or module'
        }
    }
    
    def __init__(self):
        super().__init__()
        self.similarity_threshold = 0.7
        self.min_cluster_size = 1
        self.max_cluster_size = 8
        self.include_context = True
        self.cluster_scope = 'class'
        self.embedding_provider = None
        self.ts_loader = TreeSitterLanguageLoader()
    
    def initialize(self, config: Dict[str, Any], embedding_provider) -> None:
        """Initialize the semantic code clustering provider."""
        self.embedding_provider = embedding_provider
        
        # Update configuration
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.min_cluster_size = config.get('min_cluster_size', 1)
        self.max_cluster_size = config.get('max_cluster_size', 8)
        self.include_context = config.get('include_context', True)
        self.cluster_scope = config.get('cluster_scope', 'class')
        
        # Validate configuration
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        
        if self.cluster_scope not in ['function', 'class', 'module']:
            raise ValueError("cluster_scope must be 'function', 'class', or 'module'")
        
        logger.info(f"Initialized semantic code clustering with threshold={self.similarity_threshold}")
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk code using semantic similarity clustering of functions.
        
        Args:
            text: Input code to chunk
            metadata: Optional metadata (file_path, etc.)
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        try:
            # Determine if this is code and what language
            file_path = metadata.get('file_path', '') if metadata else ''
            language = self._detect_language(text, file_path)
            
            if not language:
                # Not code or unsupported language, fall back to regular chunking
                logger.info("Not code or unsupported language, using fallback chunking")
                return self._fallback_chunking(text, metadata)
            
            # Parse code into AST
            parser = tree_sitter.Parser(language)
            tree = parser.parse(text.encode('utf-8'))
            
            # Extract semantic units (functions, classes, etc.)
            semantic_units = self._extract_semantic_units(tree.root_node, text, language)
            
            if len(semantic_units) <= 1:
                # Not enough units to cluster
                return [{'text': text, 'metadata': metadata or {}}]
            
            # Group units by parent scope
            scoped_units = self._group_by_scope(semantic_units)
            
            # Cluster within each scope
            all_chunks = []
            for scope_name, units in scoped_units.items():
                scope_chunks = self._cluster_units_in_scope(units, scope_name, metadata)
                all_chunks.extend(scope_chunks)
            
            logger.info(f"Created {len(all_chunks)} semantic code chunks from {len(semantic_units)} units")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error in semantic code clustering: {e}")
            # Fallback to regular chunking
            return self._fallback_chunking(text, metadata)
    
    def _detect_language(self, text: str, file_path: str) -> Optional[tree_sitter.Language]:
        """Detect programming language from file path or content."""
        try:
            if file_path:
                return self.ts_loader.get_language_for_file(file_path)
            
            # Try to detect from content patterns
            if 'def ' in text and 'class ' in text:
                return self.ts_loader.get_language_for_file('test.py')
            elif 'function ' in text or 'class ' in text and '{' in text:
                return self.ts_loader.get_language_for_file('test.js')
            elif 'fn ' in text and 'struct ' in text:
                return self.ts_loader.get_language_for_file('test.rs')
            elif 'public class ' in text or 'private ' in text:
                return self.ts_loader.get_language_for_file('test.java')
            
            return None
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return None
    
    def _extract_semantic_units(self, root_node: tree_sitter.Node, text: str, language: tree_sitter.Language) -> List[Dict[str, Any]]:
        """Extract semantic units (functions, classes, methods) from AST."""
        units = []
        
        # Language-specific node types for semantic units
        semantic_node_types = {
            'python': ['function_definition', 'class_definition', 'async_function_definition'],
            'javascript': ['function_declaration', 'function_expression', 'class_declaration', 'method_definition'],
            'rust': ['function_item', 'struct_item', 'impl_item', 'trait_item'],
            'java': ['method_declaration', 'class_declaration', 'interface_declaration'],
            'cpp': ['function_definition', 'class_specifier', 'struct_specifier'],
            'c': ['function_definition', 'struct_specifier']
        }
        
        # Detect language name from tree-sitter language object
        lang_name = self._get_language_name(language)
        node_types = semantic_node_types.get(lang_name, ['function_definition', 'class_definition'])
        
        def extract_units_recursive(node: tree_sitter.Node, parent_scope: str = ''):
            """Recursively extract semantic units from AST nodes."""
            if node.type in node_types:
                # Extract the text for this unit
                start_byte = node.start_byte
                end_byte = node.end_byte
                unit_text = text[start_byte:end_byte]
                
                # Get unit name if possible
                unit_name = self._extract_unit_name(node, text)
                
                # Determine scope
                scope = parent_scope
                if node.type in ['class_definition', 'class_declaration', 'class_specifier', 'struct_item']:
                    scope = unit_name or f"class_{len(units)}"
                elif not parent_scope:
                    scope = 'module'
                
                units.append({
                    'text': unit_text,
                    'name': unit_name,
                    'type': node.type,
                    'scope': scope,
                    'start_line': node.start_point[0],
                    'end_line': node.end_point[0]
                })
                
                # If this is a class/struct, extract its methods with this as parent scope
                if node.type in ['class_definition', 'class_declaration', 'class_specifier', 'struct_item', 'impl_item']:
                    new_scope = unit_name or scope
                    for child in node.children:
                        extract_units_recursive(child, new_scope)
            else:
                # Continue searching in children
                for child in node.children:
                    extract_units_recursive(child, parent_scope)
        
        extract_units_recursive(root_node)
        return units
    
    def _get_language_name(self, language: tree_sitter.Language) -> str:
        """Get language name from tree-sitter language object."""
        # This is a bit hacky, but tree-sitter doesn't expose language names directly
        lang_str = str(language)
        if 'python' in lang_str.lower():
            return 'python'
        elif 'javascript' in lang_str.lower():
            return 'javascript'
        elif 'rust' in lang_str.lower():
            return 'rust'
        elif 'java' in lang_str.lower():
            return 'java'
        elif 'cpp' in lang_str.lower() or 'c++' in lang_str.lower():
            return 'cpp'
        elif 'c' in lang_str.lower():
            return 'c'
        else:
            return 'unknown'
    
    def _extract_unit_name(self, node: tree_sitter.Node, text: str) -> Optional[str]:
        """Extract the name of a function/class from its AST node."""
        try:
            # Look for identifier nodes in children
            for child in node.children:
                if child.type == 'identifier':
                    return text[child.start_byte:child.end_byte]
            
            # Fallback: try to extract from text patterns
            unit_text = text[node.start_byte:node.end_byte]
            lines = unit_text.split('\n')
            if lines:
                first_line = lines[0].strip()
                # Simple regex patterns for common cases
                import re
                
                # Python: def function_name( or class ClassName:
                python_match = re.search(r'(?:def|class)\s+(\w+)', first_line)
                if python_match:
                    return python_match.group(1)
                
                # JavaScript: function functionName( or class ClassName
                js_match = re.search(r'(?:function|class)\s+(\w+)', first_line)
                if js_match:
                    return js_match.group(1)
                
                # Rust: fn function_name( or struct StructName
                rust_match = re.search(r'(?:fn|struct|impl|trait)\s+(\w+)', first_line)
                if rust_match:
                    return rust_match.group(1)
                
                # Java: public/private type methodName(
                java_match = re.search(r'(?:public|private|protected)?\s*\w+\s+(\w+)\s*\(', first_line)
                if java_match:
                    return java_match.group(1)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract unit name: {e}")
            return None
    
    def _group_by_scope(self, units: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group semantic units by their parent scope."""
        scoped_units = {}
        
        for unit in units:
            scope = unit['scope']
            if scope not in scoped_units:
                scoped_units[scope] = []
            scoped_units[scope].append(unit)
        
        return scoped_units
    
    def _cluster_units_in_scope(self, units: List[Dict[str, Any]], scope_name: str, metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster semantic units within a scope by similarity."""
        if len(units) <= 1:
            # Single unit, return as-is
            chunks = []
            for unit in units:
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    'scope': scope_name,
                    'unit_name': unit['name'],
                    'unit_type': unit['type'],
                    'semantic_clustering': True
                })
                chunks.append({
                    'text': unit['text'],
                    'metadata': chunk_metadata
                })
            return chunks
        
        try:
            # Embed each unit
            embeddings = []
            for unit in units:
                embedding = self.embedding_provider.embed_text(unit['text'])
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings)
            
            # Cluster by similarity
            clusters = self._cluster_by_similarity(units, embeddings)
            
            # Create chunks from clusters
            chunks = []
            for i, cluster_indices in enumerate(clusters):
                cluster_units = [units[idx] for idx in cluster_indices]
                chunk_text = self._combine_cluster_units(cluster_units, scope_name)
                
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    'scope': scope_name,
                    'cluster_id': i,
                    'cluster_size': len(cluster_indices),
                    'unit_names': [unit['name'] for unit in cluster_units],
                    'unit_types': [unit['type'] for unit in cluster_units],
                    'semantic_clustering': True
                })
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error clustering units in scope {scope_name}: {e}")
            # Fallback: return units as individual chunks
            return self._units_as_individual_chunks(units, scope_name, metadata)
    
    def _cluster_by_similarity(self, units: List[Dict[str, Any]], embeddings: np.ndarray) -> List[List[int]]:
        """Cluster units by semantic similarity."""
        clusters = []
        used = set()
        
        for i in range(len(units)):
            if i in used:
                continue
            
            cluster = [i]
            used.add(i)
            
            # Find similar units
            for j in range(len(units)):
                if j in used or j == i:
                    continue
                
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                
                if similarity >= self.similarity_threshold and len(cluster) < self.max_cluster_size:
                    cluster.append(j)
                    used.add(j)
            
            # Only keep clusters that meet size requirements
            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)
            else:
                # Add as individual units if cluster too small
                for idx in cluster:
                    if idx not in [item for sublist in clusters for item in sublist]:
                        clusters.append([idx])
        
        return clusters
    
    def _combine_cluster_units(self, cluster_units: List[Dict[str, Any]], scope_name: str) -> str:
        """Combine units in a cluster into a single chunk text."""
        if self.include_context and scope_name != 'module':
            # Include scope context
            context_header = f"# Scope: {scope_name}\n# Related functions:\n\n"
        else:
            context_header = ""
        
        # Sort units by line number to maintain some logical order
        sorted_units = sorted(cluster_units, key=lambda u: u['start_line'])
        
        unit_texts = []
        for unit in sorted_units:
            unit_header = f"# {unit['type']}: {unit['name']}\n" if unit['name'] else ""
            unit_texts.append(unit_header + unit['text'])
        
        return context_header + '\n\n'.join(unit_texts)
    
    def _units_as_individual_chunks(self, units: List[Dict[str, Any]], scope_name: str, metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert units to individual chunks as fallback."""
        chunks = []
        for unit in units:
            chunk_metadata = (metadata or {}).copy()
            chunk_metadata.update({
                'scope': scope_name,
                'unit_name': unit['name'],
                'unit_type': unit['type'],
                'semantic_clustering': False  # Fallback mode
            })
            chunks.append({
                'text': unit['text'],
                'metadata': chunk_metadata
            })
        return chunks
    
    def _fallback_chunking(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback to regular chunking when semantic clustering fails."""
        try:
            # Try tree-sitter chunking first
            chunks = ts_get_code_chunks(text)
            result = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata['fallback_chunking'] = True
                chunk_metadata['chunk_id'] = i
                result.append({
                    'text': chunk,
                    'metadata': chunk_metadata
                })
            return result
        except Exception:
            # Ultimate fallback: single chunk
            return [{'text': text, 'metadata': metadata or {}}]
    
    def get_chunk_size_estimate(self, text: str) -> int:
        """Estimate chunk size for semantic code clustering."""
        # Rough estimate based on function count and clustering parameters
        function_count = text.count('def ') + text.count('function ') + text.count('fn ')
        if function_count == 0:
            return len(text)
        
        estimated_clusters = max(1, function_count // self.max_cluster_size)
        avg_chunk_size = len(text) // estimated_clusters
        return avg_chunk_size
    
    def validate_configuration(self, config: Dict[str, Any], embedding_provider) -> List[str]:
        """Validate semantic code clustering configuration."""
        errors = []
        
        similarity_threshold = config.get('similarity_threshold', 0.7)
        if not isinstance(similarity_threshold, (int, float)) or not (0.0 <= similarity_threshold <= 1.0):
            errors.append("similarity_threshold must be a number between 0.0 and 1.0")
        
        min_cluster_size = config.get('min_cluster_size', 1)
        if not isinstance(min_cluster_size, int) or min_cluster_size < 1:
            errors.append("min_cluster_size must be a positive integer")
        
        max_cluster_size = config.get('max_cluster_size', 8)
        if not isinstance(max_cluster_size, int) or max_cluster_size < min_cluster_size:
            errors.append("max_cluster_size must be an integer >= min_cluster_size")
        
        cluster_scope = config.get('cluster_scope', 'class')
        if cluster_scope not in ['function', 'class', 'module']:
            errors.append("cluster_scope must be 'function', 'class', or 'module'")
        
        return errors
