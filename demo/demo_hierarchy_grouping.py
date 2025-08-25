#!/usr/bin/env python3
"""Comprehensive demo of hierarchy-constrained semantic grouping."""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simple_embeddings_module.chunking.mod_hierarchy_integration import (
    HierarchyGroupingProvider,
    HierarchyGroupingSearch,
    HierarchyGroupingStorage,
    create_hierarchy_grouping_demo,
)
from src.simple_embeddings_module.embeddings.mod_sentence_transformers import (
    SentenceTransformersProvider,
)


def print_banner():
    """Print demo banner."""
    print("🌟" * 70)
    print("🚀 HIERARCHY-CONSTRAINED SEMANTIC GROUPING DEMO 🚀")
    print("🌟" * 70)
    print("🧠 Intelligent code chunking with hierarchy boundaries + semantic grouping!")
    print("🔒 Groups chunks ONLY within same parent scope (classes, modules)")
    print("📊 Flat storage with rich metadata for efficient search")
    print("🌟" * 70)
    print()


def demo_with_existing_files():
    """Demo using existing files in the project."""
    print("📁 DEMO WITH EXISTING PROJECT FILES")
    print("=" * 50)

    # Initialize embedding provider
    print("🔧 Initializing sentence-transformers embedding provider...")
    embedding_provider = SentenceTransformersProvider(model="all-MiniLM-L6-v2")

    # Find existing Python files to demo with
    project_root = Path(__file__).parent
    demo_files = []

    # Look for demo files
    for pattern in ["demo_*.py", "example_*.py", "test_*.py"]:
        demo_files.extend(project_root.glob(pattern))

    # Also include some source files
    src_dir = project_root / "src" / "simple_embeddings_module"
    if src_dir.exists():
        demo_files.extend((src_dir / "chunking").glob("mod_*.py"))

    # Limit to first 5 files for demo
    demo_files = [str(f) for f in demo_files[:5] if f.stat().st_size > 1000]  # Skip empty files

    if not demo_files:
        print("❌ No suitable demo files found")
        return

    print(f"📄 Found {len(demo_files)} files for demo:")
    for file_path in demo_files:
        file_size = Path(file_path).stat().st_size
        print(f"   • {Path(file_path).name} ({file_size:,} bytes)")

    # Process files with hierarchy grouping
    print("\n🔄 Processing files with hierarchy-constrained grouping...")
    chunks, groups = create_hierarchy_grouping_demo(
        demo_files, embedding_provider, output_file="hierarchy_demo_results.json"
    )

    # Analyze results
    analyze_grouping_results(chunks, groups)

    # Demo search functionality
    demo_search_functionality(chunks, groups, embedding_provider)

    return chunks, groups


def analyze_grouping_results(chunks, groups):
    """Analyze and display grouping results."""
    print("\n🔍 DETAILED ANALYSIS")
    print("=" * 50)

    # Hierarchy distribution
    hierarchy_stats = {}
    for chunk in chunks:
        hierarchy_key = "::".join(chunk.parent_hierarchy[:-1]) if len(chunk.parent_hierarchy) > 1 else chunk.parent_hierarchy[0]
        if hierarchy_key not in hierarchy_stats:
            hierarchy_stats[hierarchy_key] = {"chunks": 0, "types": set()}
        hierarchy_stats[hierarchy_key]["chunks"] += 1
        hierarchy_stats[hierarchy_key]["types"].add(chunk.chunk_type)

    print("📊 Hierarchy Distribution:")
    for hierarchy, stats in sorted(hierarchy_stats.items()):
        types_str = ", ".join(sorted(stats["types"]))
        print(f"   {hierarchy}: {stats['chunks']} chunks ({types_str})")

    # Group analysis
    if groups:
        print("\n🔗 Semantic Groups Analysis:")
        group_types = {}
        for group in groups:
            if group.group_type not in group_types:
                group_types[group.group_type] = 0
            group_types[group.group_type] += 1

        for group_type, count in sorted(group_types.items()):
            print(f"   {group_type}: {count} groups")

        print("\n📋 Group Details:")
        for i, group in enumerate(groups, 1):
            print(f"   {i:2d}. {group.group_id}")
            print(f"       Scope: {group.parent_scope}")
            print(f"       Type: {group.group_type}")
            print(f"       Theme: {group.group_theme or 'None detected'}")
            print(f"       Chunks: {len(group.chunk_ids)}")
            print(f"       Threshold: {group.similarity_threshold}")
            print()
    else:
        print("\n🔗 No semantic groups created")
        print("   This could mean:")
        print("   • Chunks within scopes are too dissimilar")
        print("   • Similarity threshold is too high")
        print("   • Not enough chunks per scope for grouping")

    # Chunk type distribution
    chunk_types = {}
    for chunk in chunks:
        if chunk.chunk_type not in chunk_types:
            chunk_types[chunk.chunk_type] = 0
        chunk_types[chunk.chunk_type] += 1

    print("📈 Chunk Type Distribution:")
    for chunk_type, count in sorted(chunk_types.items()):
        print(f"   {chunk_type}: {count} chunks")


def demo_search_functionality(chunks, groups, embedding_provider):
    """Demo search functionality with hierarchy grouping."""
    print("\n🔍 SEARCH FUNCTIONALITY DEMO")
    print("=" * 50)

    # Initialize search
    search = HierarchyGroupingSearch(embedding_provider)

    # Demo queries
    demo_queries = [
        "function definition and implementation",
        "class methods and attributes",
        "import statements and dependencies",
        "error handling and exceptions",
        "data processing and transformation",
        "configuration and settings",
        "testing and validation",
        "database operations"
    ]

    print("🎯 Search Results:")
    for query in demo_queries:
        print(f"\n🔎 Query: '{query}'")
        results = search.search_chunks_and_groups(
            query, chunks, groups, top_k=3, include_groups=True
        )

        if results:
            for i, result in enumerate(results, 1):
                score = result['score']
                result_type = result['type']
                result_id = result['id']

                # Get preview text
                preview = result['text'][:100].replace('\n', ' ')
                if len(result['text']) > 100:
                    preview += "..."

                print(f"   {i}. [{result_type.upper()}] {result_id} (score: {score:.3f})")
                print(f"      {preview}")

                # Show metadata
                if result_type == "chunk":
                    metadata = result['metadata']
                    hierarchy = " → ".join(metadata['parent_hierarchy'])
                    print(f"      Hierarchy: {hierarchy}")
                    print(f"      Lines: {metadata['line_start']}-{metadata['line_end']}")
                elif result_type == "group":
                    metadata = result['metadata']
                    print(f"      Scope: {metadata['parent_scope']}")
                    print(f"      Theme: {metadata['group_theme'] or 'None'}")
                    print(f"      Chunks: {metadata['chunk_count']}")
                print()
        else:
            print("   No results found")


def demo_scope_specific_search(chunks, groups, embedding_provider):
    """Demo scope-specific search functionality."""
    print("\n🎯 SCOPE-SPECIFIC SEARCH DEMO")
    print("=" * 50)

    # Get available scopes
    scopes = set()
    for group in groups:
        scopes.add(group.parent_scope)

    if not scopes:
        print("❌ No scopes available for scope-specific search")
        return

    search = HierarchyGroupingSearch(embedding_provider)

    # Demo scope-specific searches
    for scope in list(scopes)[:3]:  # Limit to first 3 scopes
        print(f"\n🔍 Searching within scope: {scope}")
        results = search.search_within_scope(
            "function implementation", scope, chunks, groups, top_k=2
        )

        for i, result in enumerate(results, 1):
            result_type = result['type']
            result_id = result['id']
            score = result['score']
            print(f"   {i}. [{result_type.upper()}] {result_id} (score: {score:.3f})")


def demo_storage_and_loading():
    """Demo storage and loading functionality."""
    print("\n💾 STORAGE AND LOADING DEMO")
    print("=" * 50)

    # Check if demo results file exists
    results_file = "hierarchy_demo_results.json"
    if not Path(results_file).exists():
        print(f"❌ Results file {results_file} not found")
        return

    # Load from file
    print(f"📂 Loading results from {results_file}...")
    chunks, groups = HierarchyGroupingStorage.load_from_file(results_file)

    print(f"✅ Loaded {len(chunks)} chunks and {len(groups)} groups")

    # Show storage format
    print("\n📋 Storage Format Preview:")
    with open(results_file, 'r') as f:
        data = json.load(f)

    print(f"   Version: {data['metadata']['version']}")
    print(f"   Created: {data['metadata']['created']}")
    print(f"   Hierarchy Constrained: {data['metadata']['hierarchy_constrained']}")
    print(f"   Total Chunks: {data['metadata']['total_chunks']}")
    print(f"   Total Groups: {data['metadata']['total_groups']}")

    # Show sample chunk structure
    if data['chunks']:
        sample_chunk = data['chunks'][0]
        print("\n📦 Sample Chunk Structure:")
        print(f"   ID: {sample_chunk['id']}")
        print(f"   Type: {sample_chunk['metadata']['chunk_type']}")
        print(f"   Hierarchy: {' → '.join(sample_chunk['metadata']['parent_hierarchy'])}")
        print(f"   Lines: {sample_chunk['metadata']['line_start']}-{sample_chunk['metadata']['line_end']}")

    # Show sample group structure
    if data['semantic_groups']:
        sample_group = data['semantic_groups'][0]
        print("\n🔗 Sample Group Structure:")
        print(f"   ID: {sample_group['group_id']}")
        print(f"   Scope: {sample_group['parent_scope']}")
        print(f"   Type: {sample_group['group_type']}")
        print(f"   Chunks: {len(sample_group['chunk_ids'])}")


def demo_provider_integration():
    """Demo integration with existing SEM provider architecture."""
    print("\n🔧 PROVIDER INTEGRATION DEMO")
    print("=" * 50)

    # Initialize embedding provider
    embedding_provider = SentenceTransformersProvider(model="all-MiniLM-L6-v2")

    # Create hierarchy grouping provider
    config = {
        "similarity_threshold": 0.6,
        "min_group_size": 2,
        "max_group_size": 8
    }

    provider = HierarchyGroupingProvider(embedding_provider, config)

    # Show provider capabilities
    print("🔧 Provider Capabilities:")
    capabilities = provider.get_capabilities()
    for key, value in capabilities.items():
        print(f"   {key}: {value}")

    # Show configuration parameters
    print("\n⚙️  Configuration Parameters:")
    config_params = provider.get_config_parameters()
    for param, info in config_params.items():
        print(f"   {param}: {info['default']} - {info['description']}")

    # Test with a sample file
    sample_files = [f for f in Path(".").glob("demo_*.py") if f.stat().st_size > 1000]
    if sample_files:
        sample_file = str(sample_files[0])
        print(f"\n📄 Testing with file: {Path(sample_file).name}")

        # Test basic chunking (compatibility mode)
        basic_chunks = provider.chunk_file(sample_file)
        print(f"   Basic chunks: {len(basic_chunks)}")

        # Test with metadata
        chunks, groups = provider.chunk_file_with_metadata(sample_file)
        print(f"   Metadata chunks: {len(chunks)}")
        print(f"   Semantic groups: {len(groups)}")


def main():
    """Run the comprehensive hierarchy grouping demo."""
    print_banner()

    try:
        # Main demo with existing files
        chunks, groups = demo_with_existing_files()

        # Storage and loading demo
        demo_storage_and_loading()

        # Provider integration demo
        demo_provider_integration()

        # Scope-specific search demo
        if chunks and groups:
            embedding_provider = SentenceTransformersProvider(model="all-MiniLM-L6-v2")
            demo_scope_specific_search(chunks, groups, embedding_provider)

        # Final summary
        print("\n🎉 DEMO COMPLETE!")
        print("=" * 50)
        print("✨ Key Features Demonstrated:")
        print("  🧠 Hierarchy-constrained semantic grouping")
        print("  🔒 Respects code structure boundaries")
        print("  📊 Flat storage with rich metadata")
        print("  🔍 Advanced search with groups and individual chunks")
        print("  💾 JSON serialization and loading")
        print("  🔧 Integration with existing SEM architecture")
        print("  🚀 Real embedding providers (no mocks!)")
        print()
        print("🌟 Hierarchy-Constrained Semantic Grouping - Making code search smarter! 🌟")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
