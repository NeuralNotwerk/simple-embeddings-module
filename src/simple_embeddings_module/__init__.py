"""Simple Embeddings Module - Cross-platform semantic search"""

from .sem_config_builder import SEMConfigBuilder, create_default_config
from .sem_core import SEMDatabase, create_database, discover_modules
from .sem_simple import SEMSimple
from .sem_utils import generate_config_template, load_config, save_config

__version__ = "0.1.0"
__all__ = [
    "SEMSimple",
    "SEMDatabase",
    "create_database",
    "discover_modules",
    "SEMConfigBuilder",
    "create_default_config",
    "load_config",
    "save_config",
    "generate_config_template",
]
