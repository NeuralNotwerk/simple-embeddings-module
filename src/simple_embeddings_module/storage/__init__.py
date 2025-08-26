"""Storage backends for Simple Embeddings Module"""
from .mod_storage_base import StorageBackendBase

# Import storage backends with optional dependencies
from .mod_local_disk import LocalDiskStorage

__all__ = ["StorageBackendBase", "LocalDiskStorage"]
try:
    from .mod_s3 import S3Storage
    __all__ += ["S3Storage"]
except ImportError:
    pass
try:
    from .mod_gcs import GCSStorage
    __all__ += ["GCSStorage"]
except ImportError:
    pass
