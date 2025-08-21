"""Code chunking provider using tree-sitter for semantic boundaries."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .mod_chunking_base import ChunkingProviderBase
from .mod_chunking_ts import ts_get_code_chunks

logger = logging.getLogger(__name__)

# Configuration parameters for the code chunking provider
CONFIG_PARAMETERS = {
    "fallback_to_text": {
        "type": bool,
        "default": True,
        "description": "Fall back to text chunking if tree-sitter parsing fails"
    },
    "min_chunk_size": {
        "type": int,
        "default": 50,
        "description": "Minimum chunk size in characters (smaller chunks will be merged)"
    },
    "max_chunk_size": {
        "type": int,
        "default": 8000,
        "description": "Maximum chunk size in characters (larger chunks will be split)"
    },
    "preserve_imports": {
        "type": bool,
        "default": True,
        "description": "Include import statements with the first code chunk"
    },
    "supported_extensions": {
        "type": list,
        "default": [
            ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h", 
            ".cpp", ".cc", ".cxx", ".hpp", ".cs", ".php", ".rb", ".go", 
            ".rs", ".swift", ".kt", ".kts", ".scala", ".sc", ".lua", 
            ".sh", ".bash", ".zsh"
        ],
        "description": "File extensions that will use semantic code chunking"
    }
}

# Provider capabilities
CAPABILITIES = {
    "chunk_types": ["semantic", "code"],
    "supports_files": True,
    "supports_text": True,
    "language_aware": True,
    "preserves_structure": True,
    "optimal_for": ["code", "programming", "software"],
    "dependencies": ["tree-sitter", "tree-sitter language parsers"]
}


class CodeChunkingProvider(ChunkingProviderBase):
    """Code chunking provider using tree-sitter for semantic boundaries."""
    
    def __init__(self, embedding_provider, **config):
        """Initialize the code chunking provider."""
        # Validate and set defaults for config before calling parent
        validated_config = self._validate_config(config)
        super().__init__(embedding_provider, **validated_config)
        
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set default configuration values."""
        validated = {}
        
        for param_name, param_info in CONFIG_PARAMETERS.items():
            if param_name in config:
                validated[param_name] = config[param_name]
            else:
                validated[param_name] = param_info["default"]
                
        return validated
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """
        Chunk text using semantic code boundaries.
        
        Args:
            text: Source code text to chunk
            **kwargs: Additional parameters (file_path for language detection)
            
        Returns:
            List of code chunks
        """
        file_path = kwargs.get('file_path', '')
        
        # Check if this looks like code and we should use semantic chunking
        if self._should_use_semantic_chunking(text, file_path):
            try:
                return self._chunk_code_semantically(text, file_path)
            except Exception as e:
                logger.warning(f"Semantic chunking failed: {e}")
                if self.config["fallback_to_text"]:
                    logger.info("Falling back to text-based chunking")
                    return self._fallback_text_chunking(text)
                else:
                    raise
        else:
            # Not code or unsupported language - use text chunking
            return self._fallback_text_chunking(text)
    
    def chunk_file(self, file_path: str) -> List[str]:
        """
        Chunk a code file using semantic boundaries.
        
        Args:
            file_path: Path to the code file
            
        Returns:
            List of code chunks
        """
        try:
            # Use tree-sitter semantic chunking
            chunks = ts_get_code_chunks(file_path)
            
            if not chunks:
                logger.warning(f"No chunks generated for {file_path}")
                # Fallback: read file and use text chunking
                if self.config["fallback_to_text"]:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    return self._fallback_text_chunking(text)
                else:
                    return []
            
            # Post-process chunks
            processed_chunks = self._post_process_chunks(chunks, file_path)
            
            logger.debug(f"Generated {len(processed_chunks)} semantic chunks for {file_path}")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {e}")
            if self.config["fallback_to_text"]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    return self._fallback_text_chunking(text)
                except Exception as fallback_error:
                    logger.error(f"Fallback chunking also failed: {fallback_error}")
                    return []
            else:
                raise
    
    def _should_use_semantic_chunking(self, text: str, file_path: str) -> bool:
        """Determine if we should use semantic chunking for this text/file."""
        # Check file extension
        if file_path:
            path_obj = Path(file_path)
            if path_obj.suffix.lower() in self.config["supported_extensions"]:
                return True
            if path_obj.name in ["Dockerfile", "Makefile", "CMakeLists.txt"]:
                return True
        
        # Check if text looks like code (heuristic)
        if self._looks_like_code(text):
            return True
            
        return False
    
    def _looks_like_code(self, text: str) -> bool:
        """Heuristic to determine if text looks like code."""
        code_indicators = [
            'def ', 'function ', 'class ', 'import ', 'from ', 'include ',
            '#include', 'public class', 'private ', 'public ', 'protected ',
            'fn ', 'func ', 'let ', 'const ', 'var ', 'if (', 'for (', 'while (',
            '{', '}', ';', '//', '/*', '*/', '#', 'return ', 'print(', 'console.log'
        ]
        
        # Count code-like patterns
        matches = sum(1 for indicator in code_indicators if indicator in text)
        
        # If we have multiple code indicators, it's probably code
        return matches >= 3
    
    def _chunk_code_semantically(self, text: str, file_path: str) -> List[str]:
        """Chunk code text using semantic boundaries."""
        # Create a temporary file for tree-sitter processing
        import tempfile
        import os
        
        # Determine file extension for proper language detection
        if file_path:
            suffix = Path(file_path).suffix
        else:
            # Try to guess from content
            suffix = self._guess_file_extension(text)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(text)
            temp_path = f.name
        
        try:
            chunks = ts_get_code_chunks(temp_path)
            return self._post_process_chunks(chunks, file_path or temp_path)
        finally:
            os.unlink(temp_path)
    
    def _guess_file_extension(self, text: str) -> str:
        """Guess file extension from code content."""
        # Simple heuristics for common languages
        if 'def ' in text and 'import ' in text:
            return '.py'
        elif 'function ' in text and ('var ' in text or 'let ' in text or 'const ' in text):
            return '.js'
        elif 'public class' in text and 'import java' in text:
            return '.java'
        elif '#include' in text and ('int main' in text or 'void ' in text):
            return '.c'
        elif 'fn ' in text and ('let ' in text or 'use ' in text):
            return '.rs'
        elif 'func ' in text and 'package ' in text:
            return '.go'
        else:
            return '.txt'  # Default fallback
    
    def _post_process_chunks(self, chunks: List[str], file_path: str) -> List[str]:
        """Post-process chunks to handle size limits and imports."""
        if not chunks:
            return chunks
        
        processed = []
        imports_chunk = ""
        
        # Extract imports if configured
        if self.config["preserve_imports"]:
            imports_chunk = self._extract_imports(chunks[0])
        
        for i, chunk in enumerate(chunks):
            # Add imports to first code chunk
            if i == 0 and imports_chunk and imports_chunk not in chunk:
                chunk = imports_chunk + "\n\n" + chunk
            
            # Handle size limits
            if len(chunk) < self.config["min_chunk_size"]:
                # Try to merge with next chunk
                if i + 1 < len(chunks) and len(chunk + chunks[i + 1]) <= self.config["max_chunk_size"]:
                    chunks[i + 1] = chunk + "\n\n" + chunks[i + 1]
                    continue
            
            if len(chunk) > self.config["max_chunk_size"]:
                # Split large chunks
                split_chunks = self._split_large_chunk(chunk)
                processed.extend(split_chunks)
            else:
                processed.append(chunk)
        
        return [chunk for chunk in processed if chunk.strip()]
    
    def _extract_imports(self, first_chunk: str) -> str:
        """Extract import statements from the first chunk."""
        lines = first_chunk.split('\n')
        import_lines = []
        
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith(('import ', 'from ', '#include', 'using ', 'use ')) or
                stripped.startswith('package ') or
                stripped.startswith('namespace ')):
                import_lines.append(line)
            elif stripped and not stripped.startswith(('#', '//', '/*', '"""', "'''")):
                # Stop at first non-import, non-comment line
                break
        
        return '\n'.join(import_lines)
    
    def _split_large_chunk(self, chunk: str) -> List[str]:
        """Split a chunk that's too large."""
        # Simple line-based splitting
        lines = chunk.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > self.config["max_chunk_size"] and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _fallback_text_chunking(self, text: str) -> List[str]:
        """Fallback to simple text-based chunking."""
        # Simple paragraph-based chunking
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            para_size = len(paragraph)
            
            if current_size + para_size > self.config["max_chunk_size"] and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_size = para_size
            else:
                current_chunk.append(paragraph)
                current_size += para_size + 2  # +2 for \n\n
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _configure_for_embedding_provider(self) -> None:
        """Configure chunking parameters based on embedding provider capabilities."""
        if hasattr(self, 'embedding_provider') and self.embedding_provider:
            capabilities = self.embedding_provider.get_capabilities()
            max_sequence_length = capabilities.get('max_sequence_length', 512)
            
            # Adjust max chunk size based on embedding model limits
            # Use 80% of max sequence length to leave room for special tokens
            estimated_max_chars = int(max_sequence_length * 0.8 * 4)  # ~4 chars per token
            
            if estimated_max_chars < self.config["max_chunk_size"]:
                self.config["max_chunk_size"] = estimated_max_chars
                logger.info(f"Adjusted max_chunk_size to {estimated_max_chars} based on embedding model")
    
    def chunk_document(
        self, document_id: str, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document and return structured chunk information.
        
        Args:
            document_id: Unique identifier for the document
            text: Document text to chunk
            metadata: Optional document metadata
            
        Returns:
            List of chunk dictionaries with metadata
        """
        # Get file path from metadata if available
        file_path = ""
        if metadata:
            file_path = metadata.get('file_path', metadata.get('filename', ''))
        
        # Chunk the text
        chunks = self.chunk_text(text, file_path=file_path)
        
        # Create structured chunk information
        chunk_info = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                'document_id': document_id,
                'chunk_index': i,
                'chunk_count': len(chunks),
                'chunk_type': 'semantic_code' if self._should_use_semantic_chunking(text, file_path) else 'text',
                'char_count': len(chunk_text),
                'line_count': chunk_text.count('\n') + 1
            }
            
            # Add original metadata
            if metadata:
                chunk_metadata.update({k: v for k, v in metadata.items() 
                                     if k not in chunk_metadata})
            
            chunk_info.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return chunk_info
    
    def chunk_query(self, query: str) -> List[str]:
        """
        Chunk a query if it exceeds embedding provider limits.
        
        Args:
            query: Query text to chunk
            
        Returns:
            List of query chunks (usually just one for queries)
        """
        # For queries, we typically don't need semantic chunking
        # Just ensure it fits within size limits
        if len(query) <= self.config["max_chunk_size"]:
            return [query]
        
        # If query is too long, split it at sentence boundaries
        sentences = query.split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence) + 2  # +2 for '. '
            
            if current_size + sentence_size > self.config["max_chunk_size"] and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunk_text = '. '.join(current_chunk)
            if not chunk_text.endswith('.'):
                chunk_text += '.'
            chunks.append(chunk_text)
        
        return chunks if chunks else [query]
        """Get provider capabilities."""
        return CAPABILITIES.copy()
    
    def get_config_parameters(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return CONFIG_PARAMETERS.copy()


# Module exports for auto-discovery
__all__ = ["CodeChunkingProvider", "CONFIG_PARAMETERS", "CAPABILITIES"]
