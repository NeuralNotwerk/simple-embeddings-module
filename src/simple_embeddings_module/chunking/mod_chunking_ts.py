"""Tree-sitter based semantic code chunking."""

import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple
import tree_sitter

from .mod_chunking_ts_lang_lazy import get_language_for_file

logger = logging.getLogger(__name__)

# Node types that represent semantic boundaries for different languages
SEMANTIC_BOUNDARY_NODES = {
    'python': {
        'function_definition',
        'class_definition', 
        'async_function_definition',
        'decorated_definition'
    },
    'javascript': {
        'function_declaration',
        'function_expression',
        'arrow_function',
        'class_declaration',
        'method_definition'
    },
    'typescript': {
        'function_declaration',
        'function_expression', 
        'arrow_function',
        'class_declaration',
        'method_definition',
        'interface_declaration',
        'type_alias_declaration'
    },
    'java': {
        'method_declaration',
        'class_declaration',
        'interface_declaration',
        'constructor_declaration',
        'enum_declaration'
    },
    'c': {
        'function_definition',
        'struct_specifier',
        'enum_specifier',
        'union_specifier'
    },
    'cpp': {
        'function_definition',
        'class_specifier',
        'struct_specifier',
        'namespace_definition',
        'template_declaration'
    },
    'rust': {
        'function_item',
        'struct_item',
        'enum_item',
        'impl_item',
        'trait_item',
        'mod_item'
    },
    'go': {
        'function_declaration',
        'method_declaration',
        'type_declaration',
        'struct_type',
        'interface_type'
    },
    'ruby': {
        'method',
        'class',
        'module',
        'singleton_method'
    },
    'swift': {
        'function_declaration',
        'class_declaration',
        'struct_declaration',
        'protocol_declaration',
        'extension_declaration'
    },
    'kotlin': {
        'function_declaration',
        'class_declaration',
        'object_declaration',
        'interface_declaration'
    },
    'scala': {
        'function_definition',
        'class_definition',
        'object_definition',
        'trait_definition'
    },
    'php': {
        'function_definition',
        'class_declaration',
        'method_declaration',
        'interface_declaration',
        'trait_declaration'
    },
    'c#': {
        'method_declaration',
        'class_declaration',
        'interface_declaration',
        'struct_declaration',
        'namespace_declaration'
    }
}

# Default fallback for unknown languages
DEFAULT_BOUNDARY_NODES = {
    'function_definition',
    'function_declaration', 
    'class_definition',
    'class_declaration',
    'method_definition',
    'method_declaration'
}


def ts_get_code_chunks(file_name: str) -> List[str]:
    """
    Extract semantic code chunks from a file using tree-sitter parsing.
    
    Args:
        file_name: Path to the code file to chunk
        
    Returns:
        List of code chunks as strings
    """
    try:
        # Read the file
        file_path = Path(file_name)
        if not file_path.exists():
            logger.error(f"File not found: {file_name}")
            return []
            
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            
        if not code.strip():
            logger.warning(f"Empty file: {file_name}")
            return []
            
        # Get line break positions for semantic boundaries
        line_breaks = _get_code_line_breaks(code, file_name)
        
        if not line_breaks:
            # No semantic boundaries found, return whole file as single chunk
            logger.debug(f"No semantic boundaries found in {file_name}, returning as single chunk")
            return [code]
            
        # Split code into lines for processing
        lines = code.splitlines(keepends=True)
        chunks = []
        
        # Process each boundary pair
        for i in range(0, len(line_breaks), 2):
            if i + 1 < len(line_breaks):
                start_line = line_breaks[i]
                end_line = line_breaks[i + 1]
                
                # Extract chunk (convert from 0-based to 1-based indexing)
                chunk_lines = lines[start_line:end_line + 1]
                chunk = ''.join(chunk_lines).strip()
                
                if chunk:
                    chunks.append(chunk)
                    
        # If we have leftover content after the last boundary, add it
        if line_breaks and len(line_breaks) % 2 == 0:
            last_end = line_breaks[-1]
            if last_end + 1 < len(lines):
                remaining_lines = lines[last_end + 1:]
                remaining_chunk = ''.join(remaining_lines).strip()
                if remaining_chunk:
                    chunks.append(remaining_chunk)
                    
        return chunks if chunks else [code]
        
    except Exception as e:
        logger.error(f"Error chunking file {file_name}: {e}")
        # Fallback: return whole file as single chunk
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                return [f.read()]
        except:
            return []


def _get_code_line_breaks(code: str, file_name: str = "") -> List[int]:
    """
    Get line numbers where semantic boundaries occur in code.
    
    Args:
        code: Source code as string
        file_name: Optional file name for language detection
        
    Returns:
        List of line numbers (0-based) where semantic boundaries start and end.
        Format: [start1, end1, start2, end2, ...] sorted numerically.
    """
    try:
        # Get tree-sitter language parser
        language = get_language_for_file(file_name) if file_name else None
        
        if language is None:
            logger.debug(f"No tree-sitter parser available for {file_name}")
            return []
            
        # Create parser and parse code
        parser = tree_sitter.Parser(language)
        
        tree = parser.parse(bytes(code, 'utf-8'))
        root_node = tree.root_node
        
        # Detect language from file extension for boundary node selection
        lang_name = _detect_language_name(file_name)
        boundary_nodes = SEMANTIC_BOUNDARY_NODES.get(lang_name, DEFAULT_BOUNDARY_NODES)
        
        # Find all semantic boundary nodes
        boundaries = []
        _find_semantic_boundaries(root_node, boundary_nodes, boundaries)
        
        # Convert to line numbers and create start/end pairs
        line_breaks = []
        for node in boundaries:
            start_line = node.start_point[0]  # 0-based line number
            end_line = node.end_point[0]      # 0-based line number
            
            # Add start and end line for this semantic unit
            line_breaks.extend([start_line, end_line])
            
        # Sort and remove duplicates
        line_breaks = sorted(set(line_breaks))
        
        logger.debug(f"Found {len(boundaries)} semantic boundaries in {file_name}")
        return line_breaks
        
    except Exception as e:
        logger.error(f"Error parsing code with tree-sitter: {e}")
        return []


def _find_semantic_boundaries(node: tree_sitter.Node, boundary_types: Set[str], boundaries: List[tree_sitter.Node]) -> None:
    """
    Recursively find nodes that represent semantic boundaries.
    
    Args:
        node: Current tree-sitter node to examine
        boundary_types: Set of node types that represent boundaries
        boundaries: List to append found boundary nodes to
    """
    # Check if current node is a semantic boundary
    if node.type in boundary_types:
        boundaries.append(node)
        # Don't recurse into children of boundary nodes to avoid nested boundaries
        return
        
    # Recurse into children
    for child in node.children:
        _find_semantic_boundaries(child, boundary_types, boundaries)


def _detect_language_name(file_name: str) -> str:
    """
    Detect programming language name from file extension.
    
    Args:
        file_name: File name or path
        
    Returns:
        Language name (e.g., 'python', 'javascript') or 'unknown'
    """
    if not file_name:
        return 'unknown'
        
    # Handle special cases
    if file_name.endswith('Dockerfile') or 'Dockerfile' in file_name:
        return 'dockerfile'
        
    # Get extension
    if '.' not in file_name:
        return 'unknown'
        
    ext = '.' + file_name.split('.')[-1].lower()
    
    # Map extensions to language names
    ext_to_lang = {
        '.py': 'python',
        '.js': 'javascript',
        '.mjs': 'javascript', 
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.hpp': 'cpp',
        '.cs': 'c#',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.kts': 'kotlin',
        '.scala': 'scala',
        '.sc': 'scala',
        '.lua': 'lua',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'bash'
    }
    
    return ext_to_lang.get(ext, 'unknown')


# Convenience function for testing
def test_chunking(file_name: str) -> None:
    """Test the chunking functionality on a file."""
    print(f"🔍 Testing chunking on: {file_name}")
    print("=" * 50)
    
    chunks = ts_get_code_chunks(file_name)
    print(f"Found {len(chunks)} chunks:")
    
    for i, chunk in enumerate(chunks, 1):
        lines = chunk.count('\n') + 1
        chars = len(chunk)
        preview = chunk[:100].replace('\n', '\\n')
        if len(chunk) > 100:
            preview += "..."
        print(f"  Chunk {i}: {lines} lines, {chars} chars")
        print(f"    Preview: {preview}")
        print()
