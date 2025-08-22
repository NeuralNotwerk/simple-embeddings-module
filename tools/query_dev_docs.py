#!/usr/bin/env python3
"""
Interactive Dev Docs Query Tool

This script embeds all documentation files from ./dev_docs/ using AWS Bedrock + S3
and provides an interactive command-line interface for semantic search queries.

Usage:
    python query_dev_docs.py

Features:
- Embeds all .md and .py files from ./dev_docs/
- Interactive query interface with colored output
- Shows similarity scores and source files
- Supports various query commands
- Persistent storage in S3 with reuse capability
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from simple_embeddings_module import simple_aws
except ImportError:
    print("❌ Error: Could not import simple_embeddings_module")
    print("   Make sure you're in the project directory and the module is installed")
    sys.exit(1)

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def load_dev_docs() -> Dict[str, str]:
    """Load all documentation files from ./dev_docs directory."""
    dev_docs_path = Path("./dev_docs")
    docs = {}
    
    if not dev_docs_path.exists():
        print(f"{Colors.RED}❌ dev_docs directory not found at {dev_docs_path}{Colors.END}")
        return {}
    
    print(f"{Colors.CYAN}📁 Loading documentation files from {dev_docs_path}{Colors.END}")
    
    # Load markdown files
    for md_file in dev_docs_path.glob("*.md"):
        if md_file.name == ".DS_Store":
            continue
        
        try:
            content = md_file.read_text(encoding='utf-8')
            docs[md_file.name] = content
            print(f"   {Colors.GREEN}📄 Loaded {md_file.name} ({len(content):,} chars){Colors.END}")
        except Exception as e:
            print(f"   {Colors.RED}❌ Failed to load {md_file.name}: {e}{Colors.END}")
    
    # Load Python files (as code examples)
    for py_file in dev_docs_path.glob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            docs[py_file.name] = content
            print(f"   {Colors.GREEN}🐍 Loaded {py_file.name} ({len(content):,} chars){Colors.END}")
        except Exception as e:
            print(f"   {Colors.RED}❌ Failed to load {py_file.name}: {e}{Colors.END}")
    
    print(f"{Colors.GREEN}✅ Loaded {len(docs)} documentation files{Colors.END}")
    return docs

def setup_semantic_search(bucket_name: Optional[str] = None) -> 'simple_aws':
    """Set up AWS semantic search with dev docs."""
    print(f"{Colors.HEADER}🚀 Setting up AWS semantic search...{Colors.END}")
    
    # Create semantic search instance with consistent bucket naming
    start_time = time.time()
    if bucket_name:
        sem = simple_aws(bucket_name=bucket_name)
    else:
        # Use a consistent bucket name for dev docs
        sem = simple_aws(bucket_name="sem-simple-dev-docs")
    setup_time = time.time() - start_time
    
    print(f"{Colors.GREEN}✅ AWS setup completed in {setup_time:.2f}s{Colors.END}")
    
    # Show setup info
    info = sem.get_info()
    print(f"{Colors.BLUE}📊 Setup information:{Colors.END}")
    print(f"   S3 Bucket: {info.get('s3_bucket')}")
    print(f"   Bedrock Model: {info.get('bedrock_model')}")
    print(f"   AWS Region: {info.get('aws_region')}")
    print(f"   Document Count: {info.get('document_count', 0)}")
    
    return sem

def embed_documents(sem: 'simple_aws', docs: Dict[str, str], force_reindex: bool = False) -> None:
    """Embed all documents into the semantic search index."""
    info = sem.get_info()
    current_doc_count = info.get('document_count', 0)
    
    if current_doc_count > 0 and not force_reindex:
        print(f"{Colors.YELLOW}📚 Found existing index with {current_doc_count} documents{Colors.END}")
        response = input(f"{Colors.CYAN}Do you want to reindex all documents? (y/N): {Colors.END}").strip().lower()
        if response not in ['y', 'yes']:
            print(f"{Colors.GREEN}✅ Using existing index{Colors.END}")
            return
    
    print(f"{Colors.CYAN}📝 Embedding documentation content...{Colors.END}")
    
    doc_ids = []
    total_chars = 0
    
    start_time = time.time()
    for filename, content in docs.items():
        # For large documents, we'll let the chunking system handle them
        # The system will automatically chunk them appropriately
        doc_id = sem.add_text(content, document_id=filename)
        doc_ids.append(doc_id)
        total_chars += len(content)
        print(f"   {Colors.GREEN}✅ Embedded {filename} -> {doc_id}{Colors.END}")
    
    embed_time = time.time() - start_time
    
    print(f"{Colors.GREEN}✅ Embedded {len(doc_ids)} documents in {embed_time:.2f}s{Colors.END}")
    print(f"   Total content: {total_chars:,} characters")
    print(f"   Average processing: {embed_time / len(doc_ids):.2f}s per document")

def format_search_results(results: List[Dict], query: str) -> None:
    """Format and display search results with colors."""
    if not results:
        print(f"{Colors.YELLOW}   No results found above similarity threshold{Colors.END}")
        return
    
    print(f"{Colors.GREEN}   Found {len(results)} result(s):{Colors.END}")
    print()
    
    for i, result in enumerate(results, 1):
        score = result.get('similarity_score', 0)
        doc_id = result.get('document_id', 'unknown')
        text = result.get('document', '')
        
        # Determine file type for icon
        if doc_id.endswith('.py'):
            icon = "🐍"
            file_type = "Python"
        elif doc_id.endswith('.md'):
            icon = "📄"
            file_type = "Markdown"
        else:
            icon = "📝"
            file_type = "Text"
        
        # Show first 200 chars of result with highlighting
        preview = text[:200].replace('\n', ' ').strip()
        if len(text) > 200:
            preview += "..."
        
        print(f"{Colors.BOLD}{i}. {icon} {doc_id} ({file_type}){Colors.END}")
        print(f"   {Colors.CYAN}Similarity: {score:.3f}{Colors.END}")
        print(f"   {Colors.YELLOW}Preview: {preview}{Colors.END}")
        print()

def interactive_query_loop(sem: 'simple_aws') -> None:
    """Run interactive query loop."""
    print(f"{Colors.HEADER}🔍 Interactive Dev Docs Query Interface{Colors.END}")
    print(f"{Colors.CYAN}Enter your queries to search through the embedded documentation.{Colors.END}")
    print(f"{Colors.CYAN}Commands:{Colors.END}")
    print(f"  {Colors.YELLOW}help{Colors.END} - Show this help")
    print(f"  {Colors.YELLOW}info{Colors.END} - Show database information")
    print(f"  {Colors.YELLOW}examples{Colors.END} - Show example queries")
    print(f"  {Colors.YELLOW}quit{Colors.END} - Exit the program")
    print()
    
    while True:
        try:
            query = input(f"{Colors.BOLD}Query> {Colors.END}").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print(f"{Colors.GREEN}👋 Goodbye!{Colors.END}")
                break
            
            elif query.lower() == 'help':
                print(f"{Colors.CYAN}Available commands:{Colors.END}")
                print(f"  {Colors.YELLOW}help{Colors.END} - Show this help")
                print(f"  {Colors.YELLOW}info{Colors.END} - Show database information")
                print(f"  {Colors.YELLOW}examples{Colors.END} - Show example queries")
                print(f"  {Colors.YELLOW}quit{Colors.END} - Exit the program")
                print(f"{Colors.CYAN}Or enter any search query to find relevant documentation.{Colors.END}")
                print()
                continue
            
            elif query.lower() == 'info':
                info = sem.get_info()
                print(f"{Colors.BLUE}📊 Database Information:{Colors.END}")
                for key, value in info.items():
                    print(f"   {key}: {value}")
                print()
                continue
            
            elif query.lower() == 'examples':
                examples = [
                    "GPU acceleration and CUDA",
                    "JSON serialization performance",
                    "vector similarity search",
                    "PyTorch tensor operations",
                    "FAISS index types",
                    "installation and setup",
                    "Python code examples",
                    "performance benchmarks"
                ]
                print(f"{Colors.CYAN}Example queries you can try:{Colors.END}")
                for example in examples:
                    print(f"  • {example}")
                print()
                continue
            
            # Perform search
            print(f"{Colors.CYAN}🔍 Searching for: '{query}'{Colors.END}")
            start_time = time.time()
            
            results = sem.search(query, top_k=5)
            search_time = time.time() - start_time
            
            print(f"{Colors.BLUE}   Search completed in {search_time:.3f}s{Colors.END}")
            format_search_results(results, query)
            
        except KeyboardInterrupt:
            print(f"\n{Colors.GREEN}👋 Goodbye!{Colors.END}")
            break
        except Exception as e:
            print(f"{Colors.RED}❌ Error: {e}{Colors.END}")

def main():
    """Main function."""
    print(f"{Colors.HEADER}{Colors.BOLD}🌟 Dev Docs Semantic Search Tool 🌟{Colors.END}")
    print(f"{Colors.CYAN}Embedding and querying documentation with AWS Bedrock + S3{Colors.END}")
    print("=" * 60)
    
    # Check environment
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print(f"{Colors.RED}❌ AWS credentials not found in environment{Colors.END}")
        print(f"{Colors.YELLOW}   Load from .env: export $(cat .env | grep -v '^#' | xargs){Colors.END}")
        return False
    
    print(f"{Colors.GREEN}✅ AWS credentials found{Colors.END}")
    
    try:
        # Load documentation files
        docs = load_dev_docs()
        if not docs:
            print(f"{Colors.RED}❌ No documentation files found{Colors.END}")
            return False
        
        # Setup semantic search
        sem = setup_semantic_search()
        
        # Embed documents
        embed_documents(sem, docs)
        
        # Start interactive query loop
        interactive_query_loop(sem)
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ Error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
