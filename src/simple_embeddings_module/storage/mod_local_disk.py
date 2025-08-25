"""
Local Disk Storage Backend
Implements storage backend for local file system using secure orjson serialization.
Provides atomic operations and compression support.
"""
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..sem_module_reg import ConfigParameter
from .mod_storage_base import StorageBackendBase, StorageBackendError

logger = logging.getLogger(__name__)
class LocalDiskStorage(StorageBackendBase):
    """Local file system storage backend with secure serialization"""
    CONFIG_PARAMETERS = [
        ConfigParameter(
            key_name="path",
            value_type="str",
            config_description="Base directory for storing indexes",
            required=True,
            value_opt_default="./indexes",
        ),
        ConfigParameter(
            key_name="compression",
            value_type="bool",
            config_description="Enable compression for stored files",
            required=False,
            value_opt_default=True,
        ),
        ConfigParameter(
            key_name="atomic_writes",
            value_type="bool",
            config_description="Use atomic writes for data safety",
            required=False,
            value_opt_default=True,
        ),
        ConfigParameter(
            key_name="backup_count",
            value_type="numeric",
            config_description="Number of backup copies to keep",
            required=False,
            value_opt_default=3,
            value_opt_regex=r"^[0-9]$",
        ),
    ]
    CAPABILITIES = {
        "max_index_size": -1,  # Unlimited (disk space dependent)
        "supports_streaming": False,
        "supports_partial_updates": True,
        "supports_concurrent_access": False,  # File locking not implemented
        "compression_supported": True,
        "encryption_supported": False,
    }
    def __init__(self, **config):
        """Initialize local disk storage backend"""
        super().__init__(**config)
        self.base_path = Path(config.get("path", "./indexes"))
        self.compression = config.get("compression", True)
        self.atomic_writes = config.get("atomic_writes", True)
        self.backup_count = config.get("backup_count", 3)
        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        # Initialize serializer
        self._init_serializer()
        logger.info("LocalDiskStorage initialized: %s", self.base_path)
    def _init_serializer(self):
        """Initialize the orjson serializer"""
        try:
            import orjson
            self._orjson = orjson
        except ImportError:
            raise StorageBackendError("orjson library not installed. Install with: pip install orjson")
    def save_index(self, vectors: torch.Tensor, metadata: Dict[str, Any], index_name: str) -> bool:
        """Save vector index and metadata to local disk"""
        self.validate_index_name(index_name)
        try:
            # Prepare metadata
            prepared_metadata = self.prepare_metadata(metadata, vectors)
            # Create index directory
            index_dir = self.base_path / index_name
            index_dir.mkdir(parents=True, exist_ok=True)
            # Prepare data for serialization
            index_data = {
                "vectors": vectors.cpu().numpy(),  # Convert to numpy for orjson
                "metadata": prepared_metadata,
                "format_version": "1.0",
                "compression": self.compression,
                "created_with": "LocalDiskStorage",
            }
            # Serialize with orjson
            serialized_data = self._orjson.dumps(
                index_data,
                option=self._orjson.OPT_SERIALIZE_NUMPY | self._orjson.OPT_INDENT_2,
            )
            # Compress if enabled
            if self.compression:
                serialized_data = self._compress_data(serialized_data)
            # Write to file (atomically if enabled)
            index_file = index_dir / "index.json"
            if self.compression:
                index_file = index_dir / "index.json.gz"
            if self.atomic_writes:
                self._atomic_write(index_file, serialized_data)
            else:
                with open(index_file, "wb") as f:
                    f.write(serialized_data)
            # Create backup if enabled
            if self.backup_count > 0:
                self._create_backup(index_file)
            # Write metadata summary for quick access
            self._write_metadata_summary(index_dir, prepared_metadata)
            logger.info("Index '%s' saved successfully", index_name)
            return True
        except Exception as e:
            logger.error("Failed to save index '%s': %s", index_name, e)
            raise StorageBackendError("Failed to save index: %s" % e)
    def load_index(self, index_name: str, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load vector index and metadata from local disk"""
        self.validate_index_name(index_name)
        if not self.index_exists(index_name):
            raise StorageBackendError("Index '%s' does not exist" % index_name)
        try:
            index_dir = self.base_path / index_name
            # Try compressed file first, then uncompressed
            index_file = index_dir / "index.json.gz"
            if not index_file.exists():
                index_file = index_dir / "index.json"
            if not index_file.exists():
                raise StorageBackendError("Index file not found for '%s'" % index_name)
            # Read file
            with open(index_file, "rb") as f:
                data = f.read()
            # Decompress if needed
            if index_file.suffix == ".gz":
                data = self._decompress_data(data)
            # Deserialize with orjson
            index_data = self._orjson.loads(data)
            # Validate format
            if index_data.get("format_version") != "1.0":
                raise StorageBackendError("Unsupported index format version")
            # Extract vectors and metadata
            vectors_array = index_data["vectors"]
            metadata = index_data["metadata"]
            # Convert back to numpy array (orjson converts numpy arrays to lists)
            import numpy as np
            if isinstance(vectors_array, list):
                vectors_array = np.array(vectors_array, dtype=np.float32)
            # Convert numpy array back to torch tensor
            vectors = torch.from_numpy(vectors_array)
            # Move to target device if specified
            if device is not None:
                vectors = vectors.to(device)
            logger.info("Index '%s' loaded successfully", index_name)
            return vectors, metadata
        except Exception as e:
            logger.error("Failed to load index '%s': %s", index_name, e)
            raise StorageBackendError("Failed to load index: %s" % e)
    def list_indexes(self) -> List[str]:
        """List all available indexes"""
        try:
            indexes = []
            for item in self.base_path.iterdir():
                if item.is_dir():
                    # Check if it contains an index file
                    if (item / "index.json").exists() or (item / "index.json.gz").exists():
                        indexes.append(item.name)
            return sorted(indexes)
        except Exception as e:
            logger.error("Failed to list indexes: %s", e)
            return []
    def delete_index(self, index_name: str) -> bool:
        """Delete an index"""
        self.validate_index_name(index_name)
        try:
            index_dir = self.base_path / index_name
            if not index_dir.exists():
                return True  # Already deleted
            # Remove directory and all contents
            shutil.rmtree(index_dir)
            logger.info("Index '%s' deleted successfully", index_name)
            return True
        except Exception as e:
            logger.error("Failed to delete index '%s': %s", index_name, e)
            raise StorageBackendError("Failed to delete index: %s" % e)
    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists"""
        try:
            index_dir = self.base_path / index_name
            return index_dir.is_dir() and (
                (index_dir / "index.json").exists() or (index_dir / "index.json.gz").exists()
            )
        except Exception:
            return False
    def _get_index_size(self, index_name: str) -> Optional[int]:
        """Get the size of an index in bytes"""
        try:
            index_dir = self.base_path / index_name
            total_size = 0
            for file_path in index_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return None
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip"""
        import gzip
        return gzip.compress(data)
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress gzip data"""
        import gzip
        return gzip.decompress(data)
    def _atomic_write(self, file_path: Path, data: bytes) -> None:
        """Write file atomically using temporary file"""
        temp_file = None
        try:
            # Create temporary file in same directory
            temp_fd, temp_path = tempfile.mkstemp(dir=file_path.parent, prefix=f".{file_path.name}.tmp")
            temp_file = Path(temp_path)
            # Write data to temporary file
            with os.fdopen(temp_fd, "wb") as f:
                f.write(data)
            # Atomic move (rename)
            temp_file.replace(file_path)
        except Exception as e:
            # Clean up temporary file on error
            if temp_file and temp_file.exists():
                temp_file.unlink()
            raise e
    def _create_backup(self, file_path: Path) -> None:
        """Create backup copies of the index file"""
        try:
            # Rotate existing backups
            for i in range(self.backup_count - 1, 0, -1):
                old_backup = file_path.with_suffix(f"{file_path.suffix}.bak{i}")
                new_backup = file_path.with_suffix(f"{file_path.suffix}.bak{i+1}")
                if old_backup.exists():
                    if new_backup.exists():
                        new_backup.unlink()
                    old_backup.rename(new_backup)
            # Create new backup
            if self.backup_count > 0:
                backup_path = file_path.with_suffix(f"{file_path.suffix}.bak1")
                if backup_path.exists():
                    backup_path.unlink()
                shutil.copy2(file_path, backup_path)
        except Exception as e:
            logger.warning("Failed to create backup: %s", e)
    def _write_metadata_summary(self, index_dir: Path, metadata: Dict[str, Any]) -> None:
        """Write metadata summary for quick access"""
        try:
            # Extract model name from embedding config if not directly available
            model_name = metadata.get("model_name")
            if not model_name:
                embedding_config = metadata.get("embedding_config", {})
                model_name = embedding_config.get("model", embedding_config.get("model_name"))
            summary = {
                "index_name": index_dir.name,
                "created_at": metadata.get("created_at"),
                "updated_at": metadata.get("updated_at"),
                "document_count": metadata.get("document_count", 0),
                "embedding_dim": metadata.get("embedding_dim"),
                "model_name": model_name,
                "storage_backend": metadata.get("storage_backend"),
            }
            summary_file = index_dir / "metadata.json"
            with open(summary_file, "wb") as f:
                f.write(self._orjson.dumps(summary, option=self._orjson.OPT_INDENT_2))
        except Exception as e:
            logger.warning("Failed to write metadata summary: %s", e)
    def get_index_info(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an index"""
        if not self.index_exists(index_name):
            return None
        try:
            index_dir = self.base_path / index_name
            # Try to read metadata summary first (faster)
            summary_file = index_dir / "metadata.json"
            if summary_file.exists():
                with open(summary_file, "rb") as f:
                    summary = self._orjson.loads(f.read())
                # Add size information
                summary["size_bytes"] = self._get_index_size(index_name)
                return summary
            # Fallback to loading full metadata
            return super().get_index_info(index_name)
        except Exception as e:
            logger.warning("Failed to get index info for '%s': %s", index_name, e)
            return None
    def __repr__(self) -> str:
        return (
            f"LocalDiskStorage("
            f"path={self.base_path}, "
            f"compression={self.compression}, "
            f"atomic={self.atomic_writes})"
        )
