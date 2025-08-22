"""Storage backends for Simple Embeddings Module"""

from .mod_storage_base import StorageBackendBase

# Import storage backends with optional dependencies
try:
    from .mod_local_disk import LocalDiskStorage
    __all__ = ["StorageBackendBase", "LocalDiskStorage"]
except ImportError:
    __all__ = ["StorageBackendBase"]

try:
    from .mod_s3 import S3Storage
    __all__.append("S3Storage")
except ImportError:
    pass  # S3 storage requires boto3
