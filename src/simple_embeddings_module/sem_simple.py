"""
SEMSimple - One-line semantic search

The simplest possible interface to SEM. Just import and go!
"""

import logging
from pathlib import Path
from typing import List, Optional

from .sem_config_builder import SEMConfigBuilder
from .sem_core import SEMDatabase

logger = logging.getLogger(__name__)


class SEMSimple:
    """
    Ultra-simple semantic search interface.
    
    Perfect for getting started quickly with sensible defaults:
    - Uses sentence-transformers/all-MiniLM-L6-v2 model
    - Stores indexes in ./sem_indexes/ 
    - Automatic chunking and GPU acceleration
    - No configuration needed!
    
    Example:
        >>> from simple_embeddings_module import SEMSimple
        >>> sem = SEMSimple()
        >>> sem.add_text("Machine learning is transforming software.")
        >>> results = sem.search("AI technology")
        >>> print(results[0]['text'])
    """
    
    def __init__(self, index_name: str = "default", storage_path: str = "./sem_indexes"):
        """
        Initialize SEMSimple with sensible defaults.
        
        Args:
            index_name: Name for the search index (default: "default")
            storage_path: Where to store the index files (default: "./sem_indexes")
        """
        self.index_name = index_name
        self.storage_path = Path(storage_path)
        self._db = None
        self._initialized = False
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SEMSimple initialized: index='{index_name}', path='{storage_path}'")
    
    def _ensure_initialized(self):
        """Lazy initialization of the database."""
        if not self._initialized:
            try:
                # Create default configuration
                builder = SEMConfigBuilder()
                builder.set_embedding_provider("sentence_transformers", model="all-MiniLM-L6-v2")
                builder.auto_configure_chunking()
                builder.set_storage_backend("local_disk", path=str(self.storage_path))
                builder.set_serialization_provider("orjson")
                builder.set_index_config(self.index_name)
                
                config = builder.build()
                
                # Create database
                self._db = SEMDatabase(config=config)
                self._initialized = True
                
                logger.info("SEMSimple database initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize SEMSimple: {e}")
                raise RuntimeError(f"SEMSimple initialization failed: {e}")
    
    def add_text(self, text: str, doc_id: Optional[str] = None) -> bool:
        """
        Add a single text document to the search index.
        
        Args:
            text: The text content to add
            doc_id: Optional document ID (auto-generated if not provided)
            
        Returns:
            True if successful
            
        Example:
            >>> sem.add_text("Machine learning is amazing!")
            True
        """
        return self.add_texts([text], [doc_id] if doc_id else None)
    
    def add_texts(self, texts: List[str], doc_ids: Optional[List[str]] = None) -> bool:
        """
        Add multiple text documents to the search index.
        
        Args:
            texts: List of text contents to add
            doc_ids: Optional list of document IDs (auto-generated if not provided)
            
        Returns:
            True if successful
            
        Example:
            >>> texts = ["First document", "Second document"]
            >>> sem.add_texts(texts)
            True
        """
        self._ensure_initialized()
        
        try:
            # Generate doc_ids if not provided
            if doc_ids is None:
                import uuid
                doc_ids = [f"doc_{uuid.uuid4().hex[:8]}" for _ in texts]
            elif len([d for d in doc_ids if d is not None]) != len(texts):
                # Handle case where some doc_ids are None
                import uuid
                doc_ids = [
                    doc_id if doc_id is not None else f"doc_{uuid.uuid4().hex[:8]}"
                    for doc_id in doc_ids
                ]
            
            if len(texts) != len(doc_ids):
                raise ValueError("Number of texts and doc_ids must match")
            
            # Add documents - this should append to existing index
            success = self._db.add_documents(texts, document_ids=doc_ids)
            
            if success:
                logger.info(f"Added {len(texts)} text documents")
                return True
            else:
                logger.warning(f"Failed to add {len(texts)} text documents")
                return False
                
        except Exception as e:
            logger.error(f"Error adding texts: {e}")
            return False
    
    def add_file(self, file_path: str) -> bool:
        """
        Add a text file to the search index.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            True if successful
            
        Example:
            >>> sem.add_file("document.txt")
            True
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use filename as doc_id
            doc_id = file_path.stem
            
            return self.add_text(content, doc_id=doc_id)
            
        except Exception as e:
            logger.error(f"Error adding file {file_path}: {e}")
            return False
    
    def add_files(self, file_paths: List[str]) -> bool:
        """
        Add multiple text files to the search index.
        
        Args:
            file_paths: List of paths to text files
            
        Returns:
            True if successful
            
        Example:
            >>> sem.add_files(["doc1.txt", "doc2.txt"])
            True
        """
        try:
            texts = []
            doc_ids = []
            
            for file_path in file_paths:
                file_path = Path(file_path)
                if not file_path.exists():
                    logger.warning(f"File not found, skipping: {file_path}")
                    continue
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                texts.append(content)
                doc_ids.append(file_path.stem)
            
            if not texts:
                logger.error("No valid files found")
                return False
            
            return self.add_texts(texts, doc_ids=doc_ids)
            
        except Exception as e:
            logger.error(f"Error adding files: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query text
            top_k: Number of results to return (default: 5)
            
        Returns:
            List of result dictionaries with 'id', 'text', and 'score' keys
            
        Example:
            >>> results = sem.search("machine learning", top_k=3)
            >>> for result in results:
            ...     print(f"Score: {result['score']:.3f} - {result['text']}")
        """
        self._ensure_initialized()
        
        try:
            results = self._db.search(query, top_k=top_k)
            
            # Convert to simple dictionary format
            simple_results = []
            for result in results:
                simple_results.append({
                    'id': result.get('document_id', 'unknown'),
                    'text': result.get('document', ''),
                    'score': result.get('similarity_score', 0.0)
                })
            
            logger.info(f"Search completed: found {len(simple_results)} results for '{query}'")
            return simple_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def count(self) -> int:
        """
        Get the number of documents in the index.
        
        Returns:
            Number of documents
            
        Example:
            >>> sem.count()
            42
        """
        self._ensure_initialized()
        
        try:
            # Get index info
            info = self._db.get_index_info()
            return info.get('document_count', 0) if info else 0
            
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def clear(self) -> bool:
        """
        Clear all documents from the index.
        
        Returns:
            True if successful
            
        Example:
            >>> sem.clear()
            True
        """
        try:
            if self._db and self._initialized:
                # Delete the index
                storage = self._db._storage_backend
                if storage.index_exists(self.index_name):
                    success = storage.delete_index(self.index_name)
                    if success:
                        # Reset initialization to force recreation
                        self._initialized = False
                        self._db = None
                        logger.info("Index cleared successfully")
                        return True
            
            return True  # Nothing to clear
            
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return False
    
    def info(self) -> dict:
        """
        Get information about the search index.
        
        Returns:
            Dictionary with index information
            
        Example:
            >>> info = sem.info()
            >>> print(f"Documents: {info['document_count']}")
        """
        self._ensure_initialized()
        
        try:
            info = self._db.get_index_info()
            return info if info else {}
            
        except Exception as e:
            logger.error(f"Error getting index info: {e}")
            return {}
    
    def __repr__(self) -> str:
        """String representation of SEMSimple."""
        doc_count = self.count() if self._initialized else "?"
        return f"SEMSimple(index='{self.index_name}', documents={doc_count}, path='{self.storage_path}')"
