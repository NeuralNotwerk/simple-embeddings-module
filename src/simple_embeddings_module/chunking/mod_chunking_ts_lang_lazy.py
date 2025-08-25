"""Lazy loading tree-sitter language parsers."""
import importlib
import logging
from functools import lru_cache
from typing import Any, Dict, Optional

import tree_sitter

logger = logging.getLogger(__name__)
# Language mapping: file extension -> tree-sitter package name
LANGUAGE_MAP = {
    # Core programming languages
    ".py": "tree_sitter_python",
    ".js": "tree_sitter_javascript",
    ".mjs": "tree_sitter_javascript",
    ".jsx": "tree_sitter_javascript",
    ".ts": "tree_sitter_typescript",
    ".tsx": "tree_sitter_typescript",
    ".java": "tree_sitter_java",
    ".c": "tree_sitter_c",
    ".h": "tree_sitter_c",
    ".cpp": "tree_sitter_cpp",
    ".cc": "tree_sitter_cpp",
    ".cxx": "tree_sitter_cpp",
    ".hpp": "tree_sitter_cpp",
    ".cs": "tree_sitter_c_sharp",
    ".php": "tree_sitter_php",
    ".rb": "tree_sitter_ruby",
    ".go": "tree_sitter_go",
    ".rs": "tree_sitter_rust",
    ".swift": "tree_sitter_swift",
    ".kt": "tree_sitter_kotlin",
    ".kts": "tree_sitter_kotlin",
    ".scala": "tree_sitter_scala",
    ".sc": "tree_sitter_scala",
    ".lua": "tree_sitter_lua",
    ".sh": "tree_sitter_bash",
    ".bash": "tree_sitter_bash",
    ".zsh": "tree_sitter_bash",
    # Web and markup
    ".html": "tree_sitter_html",
    ".htm": "tree_sitter_html",
    ".css": "tree_sitter_css",
    ".scss": "tree_sitter_css",
    ".sass": "tree_sitter_css",
    ".json": "tree_sitter_json",
    ".yaml": "tree_sitter_yaml",
    ".yml": "tree_sitter_yaml",
    ".toml": "tree_sitter_toml",
    ".xml": "tree_sitter_xml",
    ".sql": "tree_sitter_sql",
    ".md": "tree_sitter_markdown",
    ".markdown": "tree_sitter_markdown",
    # Special files
    "Dockerfile": "tree_sitter_dockerfile",
    ".dockerfile": "tree_sitter_dockerfile",
}
# Reverse mapping for language names
LANGUAGE_NAMES = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "java": "tree_sitter_java",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "c++": "tree_sitter_cpp",
    "csharp": "tree_sitter_c_sharp",
    "c#": "tree_sitter_c_sharp",
    "php": "tree_sitter_php",
    "ruby": "tree_sitter_ruby",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "swift": "tree_sitter_swift",
    "kotlin": "tree_sitter_kotlin",
    "scala": "tree_sitter_scala",
    "lua": "tree_sitter_lua",
    "bash": "tree_sitter_bash",
    "shell": "tree_sitter_bash",
    "html": "tree_sitter_html",
    "css": "tree_sitter_css",
    "json": "tree_sitter_json",
    "yaml": "tree_sitter_yaml",
    "toml": "tree_sitter_toml",
    "xml": "tree_sitter_xml",
    "sql": "tree_sitter_sql",
    "markdown": "tree_sitter_markdown",
    "dockerfile": "tree_sitter_dockerfile",
}
class TreeSitterLanguageLoader:
    """Lazy loader for tree-sitter language parsers."""
    def __init__(self):
        self._loaded_languages: Dict[str, Any] = {}
        self._failed_imports: set = set()

    @lru_cache(maxsize=32)
    def get_language(self, package_name: str) -> Optional[Any]:
        """Lazy load a tree-sitter language parser.
        Args:
            package_name: Name of the tree-sitter package (e.g., 'tree_sitter_python')
        Returns:
            Language parser object or None if import fails
        """
        if package_name in self._failed_imports:
            return None
        if package_name in self._loaded_languages:
            return self._loaded_languages[package_name]
        try:
            # Import the package
            module = importlib.import_module(package_name)
            # Get the language function - try multiple common patterns
            language_capsule = None
            for attr_name in ["language", "Language", "LANGUAGE"]:
                if hasattr(module, attr_name):
                    attr = getattr(module, attr_name)
                    if callable(attr):
                        language_capsule = attr()
                    else:
                        language_capsule = attr
                    break
            # Special cases for packages with different naming
            if language_capsule is None:
                # Some packages might have the language as a direct attribute
                lang_name = package_name.split("_")[-1]
                for attr_name in [lang_name, lang_name.upper(), f"{lang_name}_language"]:
                    if hasattr(module, attr_name):
                        attr = getattr(module, attr_name)
                        if callable(attr):
                            language_capsule = attr()
                        else:
                            language_capsule = attr
                        break
            if language_capsule is None:
                logger.warning("Could not find language function in %s", package_name)
                self._failed_imports.add(package_name)
                return None
            # Wrap the PyCapsule in a tree_sitter.Language object
            language = tree_sitter.Language(language_capsule)
            self._loaded_languages[package_name] = language
            logger.debug("Loaded tree-sitter language: %s", package_name)
            return language
        except ImportError as e:
            logger.debug("Failed to import %s: %s", package_name, e)
            self._failed_imports.add(package_name)
            return None
        except Exception as e:
            logger.warning("Error loading %s: %s", package_name, e)
            self._failed_imports.add(package_name)
            return None

    def get_language_for_file(self, filename: str) -> Optional[Any]:
        """Get language parser for a file based on its extension.
        Args:
            filename: File name or path
        Returns:
            Language parser object or None
        """
        # Handle special cases first
        if filename.endswith("Dockerfile") or "Dockerfile" in filename:
            return self.get_language("tree_sitter_dockerfile")
        # Get extension
        if "." not in filename:
            return None
        ext = "." + filename.split(".")[-1].lower()
        package_name = LANGUAGE_MAP.get(ext)
        if package_name:
            return self.get_language(package_name)
        return None

    def get_language_by_name(self, language_name: str) -> Optional[Any]:
        """Get language parser by language name.
        Args:
            language_name: Language name (e.g., 'python', 'javascript')
        Returns:
            Language parser object or None
        """
        package_name = LANGUAGE_NAMES.get(language_name.lower())
        if package_name:
            return self.get_language(package_name)
        return None

    def list_available_languages(self) -> list[str]:
        """List all available language parsers (that can be imported).
        Returns:
            List of available language names
        """
        available = []
        for lang_name, package_name in LANGUAGE_NAMES.items():
            if package_name not in self._failed_imports:
                # Try to load it
                if self.get_language(package_name) is not None:
                    available.append(lang_name)
        return sorted(available)

    def preload_languages(self, languages: list[str]) -> None:
        """Preload specific languages for better performance.
        Args:
            languages: List of language names to preload
        """
        for lang in languages:
            if lang in LANGUAGE_NAMES:
                self.get_language(LANGUAGE_NAMES[lang])
            elif lang in LANGUAGE_MAP.values():
                self.get_language(lang)

    def clear_cache(self) -> None:
        """Clear the language cache."""
        self._loaded_languages.clear()
        self._failed_imports.clear()
        self.get_language.cache_clear()


# Global instance for lazy loading
_language_loader = TreeSitterLanguageLoader()
# Convenience functions
def get_language_for_file(filename: str) -> Optional[Any]:
    """Get tree-sitter language parser for a file."""
    return _language_loader.get_language_for_file(filename)

def get_language_by_name(language_name: str) -> Optional[Any]:
    """Get tree-sitter language parser by name."""
    return _language_loader.get_language_by_name(language_name)

def list_available_languages() -> list[str]:
    """List available tree-sitter languages."""
    return _language_loader.list_available_languages()

def preload_common_languages() -> None:
    """Preload commonly used languages."""
    common = ["python", "javascript", "typescript", "java", "c", "cpp", "go", "rust"]
    _language_loader.preload_languages(common)
