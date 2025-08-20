"""
Utility functions for Simple Embeddings Module

Provides helper functions for configuration management, validation, and common operations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import orjson

from .sem_config_builder import SEMConfigBuilder, create_default_config

logger = logging.getLogger(__name__)


class SEMConfigObject:
    """Configuration object for Simple Embeddings Module"""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation support"""
        keys = key.split(".")
        config = self.config

        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set final value
        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.config.copy()

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)


def load_config(json_file: str) -> SEMConfigObject:
    """Load configuration from JSON file

    Args:
        json_file: Path to JSON configuration file

    Returns:
        SEMConfigObject with loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid JSON
    """
    config_path = Path(json_file)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {json_file}")

    try:
        with open(config_path, "rb") as f:
            config_data = orjson.loads(f.read())

        logger.info(f"Loaded configuration from: {json_file}")
        return SEMConfigObject(config_data)

    except orjson.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file '{json_file}': {e}")
    except Exception as e:
        raise ValueError(f"Failed to load configuration from '{json_file}': {e}")


def save_config(config: Union[SEMConfigObject, Dict[str, Any]], json_file: str) -> None:
    """Save configuration to JSON file

    Args:
        config: Configuration object or dictionary to save
        json_file: Path to output JSON file

    Raises:
        ValueError: If config cannot be serialized
    """
    config_path = Path(json_file)

    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary if needed
    if isinstance(config, SEMConfigObject):
        config_data = config.to_dict()
    else:
        config_data = config

    try:
        # Serialize with orjson (pretty printed)
        json_bytes = orjson.dumps(
            config_data, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
        )

        with open(config_path, "wb") as f:
            f.write(json_bytes)

        logger.info(f"Saved configuration to: {json_file}")

    except Exception as e:
        raise ValueError(f"Failed to save configuration to '{json_file}': {e}")


def generate_config_template(
    json_file: str = None, **kwargs
) -> Union[bool, SEMConfigObject]:
    """Generate configuration template

    Writes config template to json_file if provided -> bool
    Generates SEMConfigObject if no json_file -> SEMConfigObject
    Accepts kwargs for named settings outside of default

    * Source of truth for config options

    Args:
        json_file: Optional path to write template file
        **kwargs: Configuration overrides

    Returns:
        bool if json_file provided (True on success), SEMConfigObject otherwise
    """
    # Start with default configuration
    template_config = create_default_config()

    # Apply any overrides from kwargs
    if kwargs:
        # Use config builder to apply overrides safely
        builder = SEMConfigBuilder()

        # Set embedding provider if specified
        if "embedding_provider" in kwargs:
            provider = kwargs.pop("embedding_provider")
            model = kwargs.pop("embedding_model", "all-MiniLM-L6-v2")
            builder.set_embedding_provider(provider, model=model)

        # Set storage backend if specified
        if "storage_backend" in kwargs:
            backend = kwargs.pop("storage_backend")
            path = kwargs.pop("storage_path", "./indexes")
            builder.set_storage_backend(backend, path=path)

        # Set other overrides
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys
                parts = key.split(".")
                config_section = template_config
                for part in parts[:-1]:
                    if part not in config_section:
                        config_section[part] = {}
                    config_section = config_section[part]
                config_section[parts[-1]] = value
            else:
                template_config[key] = value

    # Add template metadata
    template_config["_template_info"] = {
        "generated_by": "sem_utils.generate_config_template",
        "description": "Simple Embeddings Module configuration template",
        "version": "1.0",
    }

    if json_file:
        # Write to file
        try:
            save_config(template_config, json_file)
            logger.info(f"Generated configuration template: {json_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate template '{json_file}': {e}")
            return False
    else:
        # Return as object
        return SEMConfigObject(template_config)


def validate_config(
    config: Union[SEMConfigObject, Dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Validate configuration for completeness and correctness

    Args:
        config: Configuration to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Convert to dict if needed
    if isinstance(config, SEMConfigObject):
        config_dict = config.to_dict()
    else:
        config_dict = config

    # Required sections
    required_sections = ["embedding", "storage", "serialization", "index"]

    for section in required_sections:
        if section not in config_dict:
            errors.append(f"Missing required section: {section}")
            continue

        section_config = config_dict[section]
        if not isinstance(section_config, dict):
            errors.append(f"Section '{section}' must be a dictionary")
            continue

        # Section-specific validation
        if section == "embedding":
            if "provider" not in section_config:
                errors.append("embedding.provider is required")
            if "model" not in section_config:
                errors.append("embedding.model is required")

        elif section == "storage":
            if "backend" not in section_config:
                errors.append("storage.backend is required")
            if (
                section_config.get("backend") == "local_disk"
                and "path" not in section_config
            ):
                errors.append("storage.path is required for local_disk backend")

        elif section == "serialization":
            if "provider" not in section_config:
                errors.append("serialization.provider is required")

        elif section == "index":
            if "name" not in section_config:
                errors.append("index.name is required")

    return len(errors) == 0, errors


def merge_configs(
    base_config: Dict[str, Any], override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two configurations, with override taking precedence

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def get_default_config_path() -> Path:
    """Get default configuration file path"""
    return Path.home() / ".sem" / "config.json"


def ensure_config_dir() -> Path:
    """Ensure configuration directory exists"""
    config_dir = Path.home() / ".sem"
    config_dir.mkdir(exist_ok=True)
    return config_dir


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging for SEM

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure logging
    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    logger.info(f"Logging configured: level={level}, file={log_file}")


# Convenience functions for common operations
def create_quick_config(
    embedding_model: str = "all-MiniLM-L6-v2",
    storage_path: str = "./indexes",
    index_name: str = "default",
) -> SEMConfigObject:
    """Create a quick configuration for common use cases"""
    builder = SEMConfigBuilder()
    builder.set_embedding_provider("sentence_transformers", model=embedding_model)
    builder.auto_configure_chunking()
    builder.set_storage_backend("local_disk", path=storage_path)
    builder.set_serialization_provider("orjson")
    builder.set_index_config(index_name)

    config = builder.build()
    return SEMConfigObject(config)


def get_config_info(config: Union[SEMConfigObject, Dict[str, Any]]) -> Dict[str, Any]:
    """Get summary information about a configuration"""
    if isinstance(config, SEMConfigObject):
        config_dict = config.to_dict()
    else:
        config_dict = config

    info = {
        "embedding_provider": config_dict.get("embedding", {}).get(
            "provider", "unknown"
        ),
        "embedding_model": config_dict.get("embedding", {}).get("model", "unknown"),
        "storage_backend": config_dict.get("storage", {}).get("backend", "unknown"),
        "storage_path": config_dict.get("storage", {}).get("path", "unknown"),
        "serialization_provider": config_dict.get("serialization", {}).get(
            "provider", "unknown"
        ),
        "index_name": config_dict.get("index", {}).get("name", "unknown"),
        "chunking_strategy": config_dict.get("chunking", {}).get("strategy", "auto"),
    }

    return info
