#!/usr/bin/env python3
"""
SEM Auto-Resolution System
Automatically resolves database names to paths when there are no conflicts.
Provides smart path resolution for CLI commands.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
class DatabaseResolver:
    """Resolves database names to paths automatically."""
    def __init__(self):
        self.discovered_databases = {}
        self._scan_complete = False

    def discover_all_databases(self) -> Dict[str, List[Dict]]:
        """
        Discover all databases in standard locations.
        Returns:
            Dict mapping database names to list of location info
        """
        if self._scan_complete:
            return self.discovered_databases
        self.discovered_databases = {}
        # 1. Scan for simple databases
        self._scan_simple_databases()
        # 2. Scan for configured databases
        self._scan_configured_databases()
        # 3. Scan for AWS cached databases
        self._scan_aws_cached_databases()
        self._scan_complete = True
        return self.discovered_databases

    def _scan_simple_databases(self):
        """Scan for simple databases (metadata.json + index.json.gz)."""
        search_paths = [
            ("./sem_indexes", 1),  # Current directory - highest priority
            (Path.home() / "sem_indexes", 2),  # Home directory - lower priority
        ]
        for search_path_info, priority in search_paths:
            search_path = Path(search_path_info)
            if not search_path.exists() or not search_path.is_dir():
                continue
            for db_dir in search_path.iterdir():
                if not db_dir.is_dir() or db_dir.name.startswith("."):
                    continue
                metadata_file = db_dir / "metadata.json"
                index_file = db_dir / "index.json.gz"
                if metadata_file.exists() or index_file.exists():
                    db_name = db_dir.name
                    if db_name not in self.discovered_databases:
                        self.discovered_databases[db_name] = []
                    self.discovered_databases[db_name].append(
                        {
                            "type": "simple_local",
                            "path": str(db_dir),
                            "search_path": str(search_path),
                            "priority": priority,
                            "metadata_file": str(metadata_file) if metadata_file.exists() else None,
                            "index_file": str(index_file) if index_file.exists() else None,
                        }
                    )

    def _scan_configured_databases(self):
        """Scan for configured databases (config.json files)."""
        config_search_paths = [
            Path("."),
            Path("./indexes"),
            Path("./test_indexes"),
            Path.home() / "indexes",
        ]
        for search_path in config_search_paths:
            if not search_path.exists() or not search_path.is_dir():
                continue
            config_file = search_path / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        config_data = json.load(f)
                    db_name = config_data.get("index_name", search_path.name)
                    if db_name not in self.discovered_databases:
                        self.discovered_databases[db_name] = []
                    self.discovered_databases[db_name].append(
                        {
                            "type": "configured",
                            "path": str(search_path),
                            "config_file": str(config_file),
                            "priority": 1 if str(search_path).startswith("./") else 2,
                            "model": config_data.get("embedding_model", "unknown"),
                        }
                    )
                except Exception as e:
                    logger.debug("Could not read config %s: %s", config_file, e)

    def _scan_aws_cached_databases(self):
        """Scan for AWS cached databases."""
        cache_paths = [
            Path("./sem_indexes/.aws_cache.json"),
            Path.home() / "sem_indexes" / ".aws_cache.json",
        ]
        for cache_path in cache_paths:
            if cache_path.exists():
                try:
                    with open(cache_path, "r") as f:
                        cached_config = json.load(f)
                    bucket_name = cached_config.get("bucket_name", "unknown")
                    db_name = f"{bucket_name}"  # Remove (AWS) suffix for resolution
                    if db_name not in self.discovered_databases:
                        self.discovered_databases[db_name] = []
                    self.discovered_databases[db_name].append(
                        {
                            "type": "simple_aws_cached",
                            "path": str(cache_path),
                            "bucket_name": bucket_name,
                            "priority": 3,  # Lower priority than local
                            "cache_file": str(cache_path),
                        }
                    )
                except Exception as e:
                    logger.debug("Could not read AWS cache %s: %s", cache_path, e)

    def resolve_database(self, db_name: str, allow_conflicts: bool = True) -> Optional[Dict]:
        """
        Resolve a database name to a specific path.
        Args:
            db_name: Database name to resolve
            allow_conflicts: If True, use priority to resolve conflicts automatically
        Returns:
            Database info dict if resolution found, None otherwise
        """
        databases = self.discover_all_databases()
        if db_name not in databases:
            return None
        locations = databases[db_name]
        if len(locations) == 1:
            # Unique database - return it
            return locations[0]
        if allow_conflicts:
            # Multiple locations - return highest priority (lowest number)
            locations.sort(key=lambda x: x.get("priority", 999))
            return locations[0]
        else:
            # Multiple locations and conflicts not allowed - return None
            return None

    def list_conflicts(self) -> Dict[str, List[Dict]]:
        """
        List databases with naming conflicts.
        Returns:
            Dict of database names with multiple locations
        """
        databases = self.discover_all_databases()
        return {name: locations for name, locations in databases.items() if len(locations) > 1}

    def suggest_path_for_database(self, db_name: str) -> Optional[str]:
        """
        Suggest the best path for a database operation.
        Args:
            db_name: Database name
        Returns:
            Suggested path string or None if not found
        """
        resolved = self.resolve_database(db_name)
        if not resolved:
            return None
        if resolved["type"] == "simple_local":
            return resolved["search_path"]
        elif resolved["type"] == "configured":
            return resolved["path"]
        elif resolved["type"] == "simple_aws_cached":
            # For AWS, return the bucket name for --bucket parameter
            return resolved["bucket_name"]
        return None

    def get_database_type(self, db_name: str) -> Optional[str]:
        """Get the type of a resolved database."""
        resolved = self.resolve_database(db_name)
        return resolved["type"] if resolved else None


# Global resolver instance
_resolver = DatabaseResolver()


def resolve_database_path(
    db_name: str, allow_conflicts: bool = True
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Resolve database name to path, handling conflicts with priority.
    Args:
        db_name: Database name to resolve
        allow_conflicts: If True, use priority to resolve conflicts automatically
    Returns:
        Tuple of (path, database_type, error_message)
        - path: Resolved path or None
        - database_type: Type of database or None
        - error_message: Error description if resolution failed
    """
    # Try to resolve using priority system
    databases = _resolver.discover_all_databases()
    if db_name not in databases:
        return None, None, f"Database '{db_name}' not found"

    locations = databases[db_name]
    if len(locations) > 1 and not allow_conflicts:
        # Multiple locations found and conflicts not allowed - show error
        conflict_info = []
        for i, loc in enumerate(locations):
            conflict_info.append(f"{loc['path']} ({loc['type']})")
        error_msg = f"Multiple locations found for '{db_name}': {', '.join(conflict_info)}. Use --path to specify which one to use."
        return None, None, error_msg

    resolved = _resolver.resolve_database(db_name, allow_conflicts=allow_conflicts)
    if not resolved:
        return None, None, f"Could not resolve database '{db_name}'"

    path = _resolver.suggest_path_for_database(db_name)
    db_type = resolved["type"]
    return path, db_type, None

def auto_resolve_command_args(args, operation_type: str = "document", backend_type: str = "local"):
    """
    Auto-resolve database arguments in CLI args.
    Args:
        args: Parsed CLI arguments
        operation_type: Type of operation ("document", "database", etc.)
        backend_type: Backend type ("local", "aws", "gcp") for cloud precedence
    Returns:
        Modified args with resolved paths, or original args if no resolution needed
    """
    # If path/config already specified, don't auto-resolve
    if getattr(args, "path", None) and getattr(args, "path") != "./sem_indexes":
        return args
    if getattr(args, "config", None):
        return args

    # Get database name (always try to resolve, even default)
    db_name = getattr(args, "db", None)
    if not db_name:
        return args

    logger.debug("Attempting auto-resolution for database: %s (backend: %s)", db_name, backend_type)

    # Try to resolve the database with conflict checking
    path, db_type, error = resolve_database_path(db_name, allow_conflicts=False)
    if error:
        logger.warning("Auto-resolution failed: %s", error)
        # For conflicts, we should show the error to user
        if "Multiple locations found" in error:
            logger.error("Database name conflict: %s", error)
        return args

    if path and db_type:
        logger.info("Auto-resolved database '%s' to %s (%s)", db_name, path, db_type)
        # Set the appropriate argument based on database type
        if db_type == "simple_local":
            args.path = path
        elif db_type == "configured":
            args.path = path
        elif db_type == "simple_aws_cached" and backend_type == "aws":
            args.bucket = path  # For AWS, path is actually bucket name
        elif db_type == "simple_gcp_cached" and backend_type == "gcp":
            args.bucket = path  # For GCP, path is actually bucket name

    return args
    """
    Auto-resolve database arguments in CLI args.
    Args:
        args: Parsed CLI arguments
        operation_type: Type of operation ("document", "database", etc.)
    Returns:
        Modified args with resolved paths, or original args if no resolution needed
    """
    # If path/config already specified, don't auto-resolve
    if getattr(args, "path", None) or getattr(args, "config", None):
        return args
    # If no database name specified, can't auto-resolve
    db_name = getattr(args, "db", None)
    if not db_name:
        return args
    # Try to resolve the database
    path, db_type, error = resolve_database_path(db_name)
    if error:
        logger.debug("Auto-resolution failed: %s", error)
        return args
    if path and db_type:
        logger.info("Auto-resolved database '%s' to %s (%s)", db_name, path, db_type)
        # Set the appropriate argument based on database type
        if db_type == "simple_local":
            args.path = path
        elif db_type == "configured":
            args.path = path
        elif db_type == "simple_aws_cached":
            args.bucket = path  # For AWS, path is actually bucket name
    return args

def list_available_databases() -> Dict[str, List[Dict]]:
    """List all available databases for user reference."""
    return _resolver.discover_all_databases()

def get_conflict_details(db_name: str) -> Optional[Dict]:
    """
    Get detailed information about database name conflicts.
    Args:
        db_name: Database name to check
    Returns:
        Dict with conflict details or None if no conflicts
    """
    databases = _resolver.discover_all_databases()
    if db_name not in databases or len(databases[db_name]) <= 1:
        return None
    locations = databases[db_name]
    return {
        "database_name": db_name,
        "conflict_count": len(locations),
        "locations": [
            {"path": loc["path"], "type": loc["type"], "priority": loc.get("priority", 999), "is_primary": i == 0}
            for i, loc in enumerate(sorted(locations, key=lambda x: x.get("priority", 999)))
        ],
    }
def list_all_conflicts() -> Dict[str, Dict]:
    """List all database name conflicts with details."""
    databases = _resolver.discover_all_databases()
    conflicts = {}
    for db_name, locations in databases.items():
        if len(locations) > 1:
            conflicts[db_name] = get_conflict_details(db_name)
    return conflicts
def suggest_resolution_commands(db_name: str) -> List[str]:
    """
    Suggest specific commands to resolve database conflicts.
    Args:
        db_name: Conflicted database name
    Returns:
        List of suggested command strings
    """
    conflict_details = get_conflict_details(db_name)
    if not conflict_details:
        return []
    suggestions = []
    for loc in conflict_details["locations"]:
        if loc["type"] == "simple_local":
            suggestions.append(f"sem-cli list --path {loc['path']}")
            suggestions.append(f"sem-cli search 'query' --path {loc['path']}")
        elif loc["type"] == "configured":
            suggestions.append(f"sem-cli list --path {loc['path']}")
            suggestions.append(f"sem-cli search 'query' --path {loc['path']}")
        elif loc["type"] == "simple_aws_cached":
            bucket = loc["path"]  # For AWS, path is bucket name
            suggestions.append(f"sem-cli simple aws list --bucket {bucket}")
            suggestions.append(f"sem-cli simple aws search 'query' --bucket {bucket}")
    return suggestions
