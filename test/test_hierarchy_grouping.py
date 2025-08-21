#!/usr/bin/env python3
"""Test script for hierarchy-constrained semantic grouping."""

import tempfile
import os
import sys
import argparse
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simple_embeddings_module.chunking.mod_hierarchy_grouping import (
    HierarchyConstrainedGrouping,
    process_file_with_hierarchy_grouping,
    ChunkMetadata,
    SemanticGroup
)
from src.simple_embeddings_module.embeddings.mod_sentence_transformers import SentenceTransformersProvider

# Global verbose flag
VERBOSE = False

def print_verbose(message: str):
    """Print message only in verbose mode."""
    if VERBOSE:
        print(message)

def show_raw_input(title: str, content: str, max_lines: int = 20):
    """Show raw input content in verbose mode."""
    if not VERBOSE:
        return
    
    print(f"\n📥 RAW INPUT: {title}")
    print("=" * 60)
    lines = content.split('\n')
    for i, line in enumerate(lines[:max_lines], 1):
        print(f"{i:3d}: {line}")
    if len(lines) > max_lines:
        print(f"... ({len(lines) - max_lines} more lines)")
    print("=" * 60)

def show_raw_output(title: str, chunks: List[ChunkMetadata], groups: List[SemanticGroup]):
    """Show raw output data in verbose mode."""
    if not VERBOSE:
        return
    
    print(f"\n📤 RAW OUTPUT: {title}")
    print("=" * 60)
    
    print(f"CHUNKS ({len(chunks)} total):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  ID: {chunk.id}")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Language: {chunk.language}")
        print(f"  File: {chunk.file_path}")
        print(f"  Lines: {chunk.line_start}-{chunk.line_end}")
        print(f"  Hierarchy: {' → '.join(chunk.parent_hierarchy)}")
        print(f"  Embedding shape: {chunk.embedding.shape if chunk.embedding is not None else 'None'}")
        print(f"  Text length: {len(chunk.text)} chars")
        print(f"  Text preview: {repr(chunk.text[:100])}...")
    
    print(f"\nGROUPS ({len(groups)} total):")
    for i, group in enumerate(groups, 1):
        print(f"\nGroup {i}:")
        print(f"  ID: {group.group_id}")
        print(f"  Parent Scope: {group.parent_scope}")
        print(f"  Type: {group.group_type}")
        print(f"  Theme: {group.group_theme}")
        print(f"  Similarity Threshold: {group.similarity_threshold}")
        print(f"  Chunk IDs: {group.chunk_ids}")
        print(f"  Group Embedding shape: {group.group_embedding.shape if group.group_embedding is not None else 'None'}")
        print(f"  Created: {group.creation_timestamp}")
    
    print("=" * 60)

def show_embedding_details(chunks: List[ChunkMetadata]):
    """Show detailed embedding information in verbose mode."""
    if not VERBOSE:
        return
    
    print(f"\n🧠 EMBEDDING DETAILS")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks, 1):
        if chunk.embedding is not None:
            embedding = chunk.embedding.cpu().numpy()
            print(f"\nChunk {i} ({chunk.id}):")
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding dtype: {embedding.dtype}")
            print(f"  Embedding range: [{embedding.min():.6f}, {embedding.max():.6f}]")
            print(f"  Embedding mean: {embedding.mean():.6f}")
            print(f"  Embedding std: {embedding.std():.6f}")
            print(f"  First 10 values: {embedding[:10].tolist()}")
        else:
            print(f"\nChunk {i} ({chunk.id}): No embedding")
    
    print("=" * 60)

def show_similarity_matrix(chunks: List[ChunkMetadata]):
    """Show similarity matrix between chunks in verbose mode."""
    if not VERBOSE or len(chunks) < 2:
        return
    
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    print(f"\n📊 SIMILARITY MATRIX")
    print("=" * 60)
    
    # Extract embeddings
    embeddings = []
    valid_chunks = []
    
    for chunk in chunks:
        if chunk.embedding is not None:
            embeddings.append(chunk.embedding.cpu().numpy())
            valid_chunks.append(chunk)
    
    if len(embeddings) < 2:
        print("Not enough valid embeddings for similarity matrix")
        return
    
    # Compute similarity matrix
    embeddings_array = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings_array)
    
    # Print matrix with chunk IDs
    print("Similarity Matrix (cosine similarity):")
    print("Chunks:", [f"C{i+1}" for i in range(len(valid_chunks))])
    
    for i, row in enumerate(similarity_matrix):
        row_str = f"C{i+1:2d}: "
        for j, sim in enumerate(row):
            if i == j:
                row_str += "  1.000"  # Self-similarity
            else:
                row_str += f" {sim:6.3f}"
        print(row_str)
    
    # Show chunk ID mapping
    print("\nChunk ID Mapping:")
    for i, chunk in enumerate(valid_chunks):
        print(f"  C{i+1}: {chunk.id}")
    
    print("=" * 60)
SAMPLE_PYTHON_CODE = '''#!/usr/bin/env python3
"""Sample module for testing hierarchy-constrained grouping."""

import os
import sys
from typing import List, Dict

# Global constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
    
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            # Database connection logic here
            self.connection = self._create_connection()
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def execute_query(self, query: str) -> List[Dict]:
        """Execute a database query."""
        if not self.connection:
            raise RuntimeError("Not connected to database")
        
        cursor = self.connection.cursor()
        cursor.execute(query)
        return cursor.fetchall()
    
    def _create_connection(self):
        """Private method to create connection."""
        # Connection creation logic
        return None

class UserManager:
    """Manages user operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.users_cache = {}
    
    def create_user(self, username: str, email: str) -> bool:
        """Create a new user."""
        if self._validate_email(email):
            query = f"INSERT INTO users (username, email) VALUES ('{username}', '{email}')"
            self.db_manager.execute_query(query)
            return True
        return False
    
    def get_user(self, username: str) -> Dict:
        """Get user by username."""
        if username in self.users_cache:
            return self.users_cache[username]
        
        query = f"SELECT * FROM users WHERE username = '{username}'"
        result = self.db_manager.execute_query(query)
        
        if result:
            user_data = result[0]
            self.users_cache[username] = user_data
            return user_data
        return {}
    
    def update_user(self, username: str, **kwargs) -> bool:
        """Update user information."""
        if not kwargs:
            return False
        
        set_clause = ", ".join(f"{k} = '{v}'" for k, v in kwargs.items())
        query = f"UPDATE users SET {set_clause} WHERE username = '{username}'"
        
        try:
            self.db_manager.execute_query(query)
            # Clear cache for this user
            if username in self.users_cache:
                del self.users_cache[username]
            return True
        except Exception:
            return False
    
    def _validate_email(self, email: str) -> bool:
        """Private method to validate email format."""
        return "@" in email and "." in email

def hash_password(password: str) -> str:
    """Utility function to hash passwords."""
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

def generate_token(length: int = 32) -> str:
    """Utility function to generate random tokens."""
    import secrets
    import string
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def validate_input(data: Dict) -> bool:
    """Utility function to validate input data."""
    required_fields = ['username', 'email']
    return all(field in data for field in required_fields)

if __name__ == "__main__":
    # Main execution
    db = DatabaseManager("sqlite:///test.db")
    user_mgr = UserManager(db)
    
    if db.connect():
        print("Database connected successfully")
        user_mgr.create_user("testuser", "test@example.com")
    else:
        print("Failed to connect to database")
'''

# Sample JavaScript code for testing
SAMPLE_JAVASCRIPT_CODE = '''// Sample JavaScript module for testing
const express = require('express');
const bcrypt = require('bcrypt');

// Global configuration
const PORT = process.env.PORT || 3000;
const SALT_ROUNDS = 12;

class AuthService {
    constructor(database) {
        this.db = database;
        this.sessions = new Map();
    }
    
    async hashPassword(password) {
        return await bcrypt.hash(password, SALT_ROUNDS);
    }
    
    async verifyPassword(password, hash) {
        return await bcrypt.compare(password, hash);
    }
    
    generateSessionToken() {
        return Math.random().toString(36).substr(2, 9);
    }
    
    createSession(userId) {
        const token = this.generateSessionToken();
        this.sessions.set(token, {
            userId: userId,
            createdAt: new Date(),
            lastAccessed: new Date()
        });
        return token;
    }
    
    validateSession(token) {
        return this.sessions.has(token);
    }
}

class UserService {
    constructor(database, authService) {
        this.db = database;
        this.auth = authService;
        this.userCache = new Map();
    }
    
    async createUser(userData) {
        const hashedPassword = await this.auth.hashPassword(userData.password);
        const user = {
            ...userData,
            password: hashedPassword,
            createdAt: new Date()
        };
        
        const userId = await this.db.insert('users', user);
        return userId;
    }
    
    async getUserById(id) {
        if (this.userCache.has(id)) {
            return this.userCache.get(id);
        }
        
        const user = await this.db.findById('users', id);
        if (user) {
            this.userCache.set(id, user);
        }
        return user;
    }
    
    async updateUser(id, updates) {
        const result = await this.db.update('users', id, updates);
        if (result) {
            this.userCache.delete(id); // Invalidate cache
        }
        return result;
    }
}

// Utility functions
function validateEmail(email) {
    const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return emailRegex.test(email);
}

function sanitizeInput(input) {
    return input.trim().toLowerCase();
}

function formatResponse(data, success = true) {
    return {
        success: success,
        data: data,
        timestamp: new Date().toISOString()
    };
}

// Express app setup
const app = express();
app.use(express.json());

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
'''

def test_hierarchy_grouping():
    """Test the hierarchy-constrained semantic grouping functionality."""
    print("🧩 Testing Hierarchy-Constrained Semantic Grouping")
    print("=" * 60)
    
    # Initialize embedding provider
    print("🔧 Initializing embedding provider...")
    embedding_provider = SentenceTransformersProvider(model="all-MiniLM-L6-v2")
    
    # Create temporary files
    test_files = {
        "sample_python.py": SAMPLE_PYTHON_CODE,
        "sample_javascript.js": SAMPLE_JAVASCRIPT_CODE
    }
    
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Created temporary directory: {temp_dir}")
    
    try:
        file_paths = []
        for filename, content in test_files.items():
            file_path = Path(temp_dir) / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            file_paths.append(str(file_path))
            print(f"   📄 Created {filename}")
            
            # Show raw input in verbose mode
            show_raw_input(f"File: {filename}", content)
        
        # Test each file individually
        for file_path in file_paths:
            print(f"\n🔍 Processing {Path(file_path).name}")
            print("-" * 40)
            
            # Process file with hierarchy grouping
            chunks, groups = process_file_with_hierarchy_grouping(
                file_path, embedding_provider, similarity_threshold=0.6
            )
            
            print(f"📦 Extracted {len(chunks)} chunks with metadata")
            print(f"🔗 Created {len(groups)} semantic groups")
            
            # Show raw output in verbose mode
            show_raw_output(f"File: {Path(file_path).name}", chunks, groups)
            
            # Show embedding details in verbose mode
            show_embedding_details(chunks)
            
            # Show similarity matrix in verbose mode
            show_similarity_matrix(chunks)
            
            # Display chunk information
            print(f"\n📋 Chunk Details:")
            for i, chunk in enumerate(chunks, 1):
                hierarchy_str = " → ".join(chunk.parent_hierarchy)
                print(f"  {i:2d}. {chunk.id}")
                print(f"      Type: {chunk.chunk_type}")
                print(f"      Lines: {chunk.line_start}-{chunk.line_end}")
                print(f"      Hierarchy: {hierarchy_str}")
                print(f"      Preview: {chunk.text[:60].replace(chr(10), ' ')}...")
                
                if VERBOSE:
                    print(f"      Full text: {repr(chunk.text)}")
                print()
            
            # Display group information
            if groups:
                print(f"🔗 Semantic Groups:")
                for i, group in enumerate(groups, 1):
                    print(f"  {i}. {group.group_id}")
                    print(f"     Scope: {group.parent_scope}")
                    print(f"     Type: {group.group_type}")
                    print(f"     Theme: {group.group_theme}")
                    print(f"     Chunks: {len(group.chunk_ids)}")
                    print(f"     Chunk IDs: {', '.join(group.chunk_ids)}")
                    print(f"     Threshold: {group.similarity_threshold}")
                    
                    if VERBOSE:
                        print(f"     Creation time: {group.creation_timestamp}")
                        if group.group_embedding is not None:
                            emb = group.group_embedding.cpu().numpy()
                            print(f"     Group embedding shape: {emb.shape}")
                            print(f"     Group embedding sample: {emb[:5].tolist()}")
                    print()
            else:
                print("   No semantic groups created (chunks may be too dissimilar)")
                if VERBOSE:
                    print("   This could be due to:")
                    print("   - Similarity threshold too high (try lowering from 0.6)")
                    print("   - Chunks are semantically different")
                    print("   - Not enough chunks in same scope")
        
        # Test grouping constraints
        print(f"\n🔒 Testing Hierarchy Constraints:")
        print("-" * 40)
        
        # Process all files together to verify no cross-file grouping
        all_chunks = []
        all_groups = []
        
        for file_path in file_paths:
            chunks, groups = process_file_with_hierarchy_grouping(
                file_path, embedding_provider, similarity_threshold=0.6
            )
            all_chunks.extend(chunks)
            all_groups.extend(groups)
        
        # Show raw constraint validation data in verbose mode
        if VERBOSE:
            print(f"\n🔍 RAW CONSTRAINT VALIDATION DATA")
            print("=" * 60)
            print("All chunk scopes:")
            for chunk in all_chunks:
                scope = "::".join(chunk.parent_hierarchy[:-1]) if len(chunk.parent_hierarchy) > 1 else chunk.parent_hierarchy[0]
                print(f"  {chunk.id} -> {scope}")
            
            print("\nAll group scopes:")
            for group in all_groups:
                print(f"  {group.group_id} -> {group.parent_scope}")
                print(f"    Contains chunks: {group.chunk_ids}")
            print("=" * 60)
        
        # Verify hierarchy constraints
        print(f"📊 Total chunks across all files: {len(all_chunks)}")
        print(f"📊 Total groups across all files: {len(all_groups)}")
        
        # Check that no group contains chunks from different parent scopes
        constraint_violations = 0
        for group in all_groups:
            chunk_scopes = set()
            for chunk_id in group.chunk_ids:
                # Find the chunk
                chunk = next((c for c in all_chunks if c.id == chunk_id), None)
                if chunk:
                    scope = "::".join(chunk.parent_hierarchy[:-1]) if len(chunk.parent_hierarchy) > 1 else chunk.parent_hierarchy[0]
                    chunk_scopes.add(scope)
            
            if len(chunk_scopes) > 1:
                constraint_violations += 1
                print(f"   ❌ Constraint violation in group {group.group_id}: {chunk_scopes}")
                if VERBOSE:
                    print(f"      Group chunks: {group.chunk_ids}")
                    print(f"      Different scopes found: {list(chunk_scopes)}")
        
        if constraint_violations == 0:
            print("   ✅ All hierarchy constraints respected!")
        else:
            print(f"   ❌ Found {constraint_violations} constraint violations")
        
        # Display scope distribution
        scope_distribution = {}
        for chunk in all_chunks:
            scope = "::".join(chunk.parent_hierarchy[:-1]) if len(chunk.parent_hierarchy) > 1 else chunk.parent_hierarchy[0]
            if scope not in scope_distribution:
                scope_distribution[scope] = 0
            scope_distribution[scope] += 1
        
        print(f"\n📈 Scope Distribution:")
        for scope, count in sorted(scope_distribution.items()):
            print(f"   {scope}: {count} chunks")
        
        if VERBOSE:
            print(f"\n📋 RAW SCOPE DISTRIBUTION DATA:")
            print("=" * 60)
            for scope, count in scope_distribution.items():
                print(f"Scope: {scope}")
                scope_chunks = [c for c in all_chunks if ("::".join(c.parent_hierarchy[:-1]) if len(c.parent_hierarchy) > 1 else c.parent_hierarchy[0]) == scope]
                for chunk in scope_chunks:
                    print(f"  - {chunk.id} ({chunk.chunk_type})")
            print("=" * 60)
        
        print(f"\n✅ Hierarchy-constrained semantic grouping test completed!")
        
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temporary directory")


def test_grouping_with_different_thresholds():
    """Test grouping behavior with different similarity thresholds."""
    print(f"\n🎯 Testing Different Similarity Thresholds")
    print("=" * 50)
    
    embedding_provider = SentenceTransformersProvider(model="all-MiniLM-L6-v2")
    
    # Create a temporary Python file
    temp_dir = tempfile.mkdtemp()
    file_path = Path(temp_dir) / "threshold_test.py"
    
    with open(file_path, 'w') as f:
        f.write(SAMPLE_PYTHON_CODE)
    
    # Show raw input for threshold testing in verbose mode
    show_raw_input("Threshold Test File", SAMPLE_PYTHON_CODE)
    
    try:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        if VERBOSE:
            print(f"\n📊 RAW THRESHOLD TESTING DATA")
            print("=" * 60)
        
        for threshold in thresholds:
            print(f"\n🔍 Testing threshold: {threshold}")
            chunks, groups = process_file_with_hierarchy_grouping(
                str(file_path), embedding_provider, similarity_threshold=threshold
            )
            
            print(f"   Chunks: {len(chunks)}, Groups: {len(groups)}")
            
            if VERBOSE:
                print(f"   Raw threshold data for {threshold}:")
                print(f"     Total chunks processed: {len(chunks)}")
                print(f"     Chunks with embeddings: {sum(1 for c in chunks if c.embedding is not None)}")
                print(f"     Groups created: {len(groups)}")
                
                if groups:
                    for group in groups:
                        print(f"     Group {group.group_id}:")
                        print(f"       Threshold used: {group.similarity_threshold}")
                        print(f"       Chunks in group: {len(group.chunk_ids)}")
                        print(f"       Group type: {group.group_type}")
            
            if groups:
                avg_group_size = sum(len(g.chunk_ids) for g in groups) / len(groups)
                print(f"   Average group size: {avg_group_size:.1f}")
                
                for group in groups:
                    print(f"   - {group.group_id}: {len(group.chunk_ids)} chunks ({group.group_type})")
                    
                    if VERBOSE:
                        # Show which specific chunks were grouped
                        print(f"     Grouped chunk IDs: {group.chunk_ids}")
                        # Show the actual similarity scores that led to grouping
                        if len(chunks) >= 2:
                            show_similarity_matrix(chunks)
    
    finally:
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Parse command line arguments for verbose mode
    parser = argparse.ArgumentParser(description='Test hierarchy-constrained semantic grouping')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Enable verbose output showing raw inputs and outputs')
    args = parser.parse_args()
    
    # Set global verbose flag
    VERBOSE = args.verbose
    
    if VERBOSE:
        print("🔍 VERBOSE MODE ENABLED - Showing raw inputs and outputs")
        print("=" * 60)
    
    test_hierarchy_grouping()
    test_grouping_with_different_thresholds()
