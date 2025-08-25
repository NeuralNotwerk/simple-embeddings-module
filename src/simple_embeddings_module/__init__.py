"""Simple Embeddings Module - Cross-platform semantic search"""
from .sem_config_builder import SEMConfigBuilder, create_default_config
from .sem_core import SEMDatabase, create_database, discover_modules
from .sem_simple import SEMSimple
from .sem_utils import generate_config_template, load_config, save_config

# AWS integration (optional - requires boto3)
try:
    from .sem_simple_aws import SEMSimpleAWS, simple_aws
    AWS_AVAILABLE = True
    __all_aws__ = ["SEMSimpleAWS", "simple_aws"]
except ImportError:
    AWS_AVAILABLE = False
    __all_aws__ = []

# Web API integration (optional - requires fastapi)
try:
    from .sem_web_client import SEMWebClient, SEMSimpleWebClient, simple_web, simple_web_aws
    WEB_AVAILABLE = True
    __all_web__ = ["SEMWebClient", "SEMSimpleWebClient", "simple_web", "simple_web_aws"]
except ImportError:
    WEB_AVAILABLE = False
    __all_web__ = []

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
] + __all_aws__ + __all_web__
