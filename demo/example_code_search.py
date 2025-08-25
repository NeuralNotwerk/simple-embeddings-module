#!/usr/bin/env python3
"""Example: Semantic code search using tree-sitter chunking."""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our modules
from simple_embeddings_module import SEMSimple
from src.simple_embeddings_module.chunking.mod_chunking_ts import ts_get_code_chunks

# Sample Python codebase for demonstration
SAMPLE_FILES = {
    "utils.py": '''
"""Utility functions for data processing."""

import json
import hashlib
from typing import Dict, List, Optional

def hash_string(text: str) -> str:
    """Generate SHA-256 hash of a string."""
    return hashlib.sha256(text.encode()).hexdigest()

def load_json_file(filepath: str) -> Dict:
    """Load and parse a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json_file(data: Dict, filepath: str) -> None:
    """Save data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

class DataValidator:
    """Validates data structures."""

    def __init__(self, schema: Dict):
        self.schema = schema

    def validate_dict(self, data: Dict) -> bool:
        """Validate a dictionary against the schema."""
        for key, expected_type in self.schema.items():
            if key not in data:
                return False
            if not isinstance(data[key], expected_type):
                return False
        return True
''',

    "database.py": '''
"""Database connection and operations."""

import sqlite3
from typing import List, Dict, Optional
from contextlib import contextmanager

class DatabaseManager:
    """Manages SQLite database connections and operations."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self) -> None:
        """Initialize the database with required tables."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL
                )
            """)

    @contextmanager
    def get_connection(self):
        """Get a database connection with automatic cleanup."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_user(self, name: str, email: str) -> int:
        """Create a new user and return their ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)",
                (name, email)
            )
            return cursor.lastrowid

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email address."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, name, email FROM users WHERE email = ?",
                (email,)
            )
            row = cursor.fetchone()
            if row:
                return {"id": row[0], "name": row[1], "email": row[2]}
            return None
''',

    "api.py": '''
"""REST API endpoints."""

from flask import Flask, request, jsonify
from database import DatabaseManager
from utils import DataValidator

app = Flask(__name__)
db = DatabaseManager("app.db")

# User validation schema
USER_SCHEMA = {
    "name": str,
    "email": str
}
validator = DataValidator(USER_SCHEMA)

@app.route("/users", methods=["POST"])
def create_user():
    """Create a new user."""
    data = request.get_json()

    if not validator.validate_dict(data):
        return jsonify({"error": "Invalid user data"}), 400

    try:
        user_id = db.create_user(data["name"], data["email"])
        return jsonify({"id": user_id, "message": "User created"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/users/<email>", methods=["GET"])
def get_user(email: str):
    """Get user by email."""
    user = db.get_user_by_email(email)
    if user:
        return jsonify(user)
    return jsonify({"error": "User not found"}), 404

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "user-api"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
'''
}

def main():
    print("🔍 Code Search with Semantic Chunking Demo")
    print("=" * 60)

    # Create temporary directory for our sample codebase
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Created sample codebase in: {temp_dir}")

    try:
        # Write sample files
        file_paths = []
        for filename, content in SAMPLE_FILES.items():
            file_path = Path(temp_dir) / filename
            with open(file_path, 'w') as f:
                f.write(content)
            file_paths.append(str(file_path))
            print(f"   📄 {filename}")

        print("\n🌳 Chunking code files with tree-sitter...")

        # Initialize semantic search
        sem = SEMSimple()

        # Process each file with semantic chunking
        total_chunks = 0
        for file_path in file_paths:
            filename = Path(file_path).name
            print(f"\n📦 Processing {filename}:")

            # Get semantic chunks
            chunks = ts_get_code_chunks(file_path)
            total_chunks += len(chunks)

            # Add each chunk to the search index
            for i, chunk in enumerate(chunks, 1):
                # Create a meaningful document ID
                doc_id = f"{filename}:chunk_{i}"
                sem.add_text(chunk, doc_id=doc_id)

                # Show what we're indexing
                first_line = chunk.split('\n')[0].strip()
                if len(first_line) > 50:
                    first_line = first_line[:50] + "..."
                print(f"   ✅ Chunk {i}: {first_line}")

        print(f"\n📊 Indexed {total_chunks} semantic chunks from {len(file_paths)} files")

        # Now demonstrate semantic search
        print("\n🔍 Semantic Code Search Examples:")
        print("-" * 40)

        search_queries = [
            "database connection and SQL operations",
            "JSON file handling and parsing",
            "user validation and data checking",
            "Flask REST API endpoints",
            "hash function for strings"
        ]

        for query in search_queries:
            print(f"\n🔎 Query: '{query}'")
            results = sem.search(query, top_k=2)

            for i, result in enumerate(results, 1):
                score = result['score']
                doc_id = result.get('doc_id', 'unknown')
                text_preview = result['text'][:100].replace('\n', ' ')
                if len(result['text']) > 100:
                    text_preview += "..."

                print(f"   {i}. {doc_id} (score: {score:.3f})")
                print(f"      {text_preview}")

        print("\n✨ Benefits of Semantic Code Chunking:")
        print("   • Each chunk is a complete semantic unit (function, class, etc.)")
        print("   • Better search precision - finds exact functions/classes")
        print("   • Preserves code context and structure")
        print("   • Works with any programming language via tree-sitter")
        print("   • Perfect for code documentation and exploration")

    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        print("\n🧹 Cleaned up temporary files")

if __name__ == "__main__":
    main()
