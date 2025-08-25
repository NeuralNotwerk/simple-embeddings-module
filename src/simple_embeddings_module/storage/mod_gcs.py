#!/usr/bin/env python3
"""
Google Cloud Storage Backend for SEM
Provides Google Cloud Storage integration for the Simple Embeddings Module.
Supports Google Cloud Storage buckets with Vertex AI embeddings.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .mod_storage_base import StorageBackendBase

logger = logging.getLogger(__name__)
class GCSStorage(StorageBackendBase):
    """Google Cloud Storage backend implementation."""
    def __init__(
        self,
        bucket_name: str,
        project_id: str = None,
        credentials_path: str = None,
        prefix: str = "sem-gcs",
        region: str = "us-central1",
    ):
        """
        Initialize GCS storage backend.
        Args:
            bucket_name: GCS bucket name
            project_id: Google Cloud project ID (auto-detected if None)
            credentials_path: Path to service account JSON (uses default if None)
            prefix: Object prefix for organization
            region: GCS region
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.prefix = prefix
        self.region = region
        self._client = None
        self._bucket = None
        logger.info("Initialized GCS client for bucket '%s' in region '%s'", bucket_name, region)
    @property
    def client(self):
        """Lazy-loaded GCS client."""
        if self._client is None:
            try:
                from google.cloud import storage
                if self.credentials_path:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
                self._client = storage.Client(project=self.project_id)
                logger.debug("GCS client initialized successfully")
            except ImportError:
                raise ImportError(
                    "Google Cloud Storage dependencies not installed. " "Install with: pip install google-cloud-storage"
                )
            except Exception:
                raise RuntimeError("Failed to initialize GCS client: %s", e)
        return self._client
    @property
    def bucket(self):
        """Lazy-loaded GCS bucket."""
        if self._bucket is None:
            try:
                self._bucket = self.client.bucket(self.bucket_name)
                # Verify bucket exists and is accessible
                if not self._bucket.exists():
                    logger.info("Creating GCS bucket: %s", self.bucket_name)
                    self._bucket = self.client.create_bucket(self.bucket_name, location=self.region)
                    logger.info("Successfully created GCS bucket: %s", self.bucket_name)
                else:
                    logger.info("Successfully verified access to GCS bucket: %s", self.bucket_name)
            except Exception:
                raise RuntimeError("Failed to access GCS bucket '%s': %s", self.bucket_name, e)
        return self._bucket
    def _get_object_key(self, index_name: str, file_type: str) -> str:
        """Generate GCS object key."""
        return "%s/%s/%s" % (self.prefix, index_name, file_type)
    def save_index(self, index_name: str, data: Dict[str, Any]) -> bool:
        """Save index data to GCS."""
        try:
            # Save metadata
            metadata_key = self._get_object_key(index_name, "metadata.json")
            metadata = {
                "index_name": index_name,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "document_count": len(data.get("documents", [])),
                "embedding_dim": data.get("embedding_dim"),
                "model_name": data.get("model_name"),
                "version": "1.0",
            }
            metadata_blob = self.bucket.blob(metadata_key)
            metadata_blob.upload_from_string(json.dumps(metadata, indent=2), content_type="application/json")
            # Save index data (compressed)
            index_key = self._get_object_key(index_name, "index.json.gz")
            index_blob = self.bucket.blob(index_key)
            # Compress and upload
            import gzip

            import orjson
            compressed_data = gzip.compress(orjson.dumps(data))
            index_blob.upload_from_string(compressed_data, content_type="application/gzip")
            logger.info("Successfully saved index '%s' to GCS", index_name)
            return True
        except Exception as e:
            logger.error("Failed to save index '%s' to GCS: %s", index_name, e)
            return False
    def load_index(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Load index data from GCS."""
        try:
            index_key = self._get_object_key(index_name, "index.json.gz")
            index_blob = self.bucket.blob(index_key)
            if not index_blob.exists():
                logger.warning("Index '%s' not found in GCS bucket '%s'", index_name, self.bucket_name)
                return None
            # Download and decompress
            import gzip

            import orjson
            compressed_data = index_blob.download_as_bytes()
            decompressed_data = gzip.decompress(compressed_data)
            data = orjson.loads(decompressed_data)
            logger.info("Successfully loaded index '%s' from GCS", index_name)
            return data
        except Exception as e:
            logger.error("Failed to load index '%s' from GCS: %s", index_name, e)
            return None
    def index_exists(self, index_name: str) -> bool:
        """Check if index exists in GCS."""
        try:
            metadata_key = self._get_object_key(index_name, "metadata.json")
            metadata_blob = self.bucket.blob(metadata_key)
            return metadata_blob.exists()
        except Exception as e:
            logger.error("Error checking if index '%s' exists: %s", index_name, e)
            return False
    def delete_index(self, index_name: str) -> bool:
        """Delete index from GCS."""
        try:
            # Delete all objects with the index prefix
            prefix = "%s/%s/" % (self.prefix, index_name)
            blobs = self.bucket.list_blobs(prefix=prefix)
            deleted_count = 0
            for blob in blobs:
                blob.delete()
                deleted_count += 1
            logger.info("Successfully deleted index '%s' (%s objects) from GCS", index_name, deleted_count)
            return True
        except Exception as e:
            logger.error("Failed to delete index '%s' from GCS: %s", index_name, e)
            return False
    def list_indexes(self) -> List[str]:
        """List all indexes in GCS bucket."""
        try:
            prefix = "%s/" % self.prefix
            blobs = self.bucket.list_blobs(prefix=prefix, delimiter="/")
            # Extract index names from prefixes
            indexes = []
            for prefix_obj in blobs.prefixes:
                # Remove the base prefix and trailing slash
                index_name = prefix_obj[len(prefix) :].rstrip("/")
                if index_name:
                    indexes.append(index_name)
            logger.info("Found %s indexes in GCS bucket", len(indexes))
            return indexes
        except Exception as e:
            logger.error("Failed to list indexes from GCS: %s", e)
            return []
    def get_index_info(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get index metadata from GCS."""
        try:
            metadata_key = self._get_object_key(index_name, "metadata.json")
            metadata_blob = self.bucket.blob(metadata_key)
            if not metadata_blob.exists():
                return None
            metadata_json = metadata_blob.download_as_text()
            metadata = json.loads(metadata_json)
            # Add GCS-specific info
            metadata.update(
                {
                    "storage_backend": "gcs",
                    "bucket_name": self.bucket_name,
                    "project_id": self.project_id,
                    "region": self.region,
                    "size_bytes": metadata_blob.size,
                }
            )
            return metadata
        except Exception as e:
            logger.error("Failed to get info for index '%s': %s", index_name, e)
            return None
    def backup_index(self, index_name: str, backup_name: str = None) -> bool:
        """Create backup of index in GCS."""
        try:
            if not backup_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = "%s_backup_%s" % (index_name, timestamp)
            # Copy all objects to backup location
            source_prefix = "%s/%s/" % (self.prefix, index_name)
            backup_prefix = "%s_backups/%s/" % (self.prefix, backup_name)
            blobs = self.bucket.list_blobs(prefix=source_prefix)
            copied_count = 0
            for blob in blobs:
                # Create new blob name with backup prefix
                relative_path = blob.name[len(source_prefix) :]
                backup_blob_name = backup_prefix + relative_path
                # Copy blob
                backup_blob = self.bucket.copy_blob(blob, self.bucket, backup_blob_name)
                copied_count += 1
            logger.info("Successfully backed up index '%s' as '%s' (%s objects)", index_name, backup_name, copied_count)
            return True
        except Exception as e:
            logger.error("Failed to backup index '%s': %s", index_name, e)
            return False
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get GCS storage statistics."""
        try:
            # Count objects and calculate total size
            prefix = "%s/" % self.prefix
            blobs = self.bucket.list_blobs(prefix=prefix)
            total_objects = 0
            total_size = 0
            indexes = set()
            for blob in blobs:
                total_objects += 1
                total_size += blob.size or 0
                # Extract index name
                relative_path = blob.name[len(prefix) :]
                if "/" in relative_path:
                    index_name = relative_path.split("/")[0]
                    indexes.add(index_name)
            return {
                "storage_backend": "gcs",
                "bucket_name": self.bucket_name,
                "project_id": self.project_id,
                "region": self.region,
                "total_objects": total_objects,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "index_count": len(indexes),
                "indexes": list(indexes),
            }
        except Exception as e:
            logger.error("Failed to get GCS storage stats: %s", e)
            return {"error": str(e)}
# Factory function for easy instantiation
def create_gcs_storage(bucket_name: str, **kwargs) -> GCSStorage:
    """
    Create GCS storage backend with common defaults.
    Args:
        bucket_name: GCS bucket name
        **kwargs: Additional arguments for GCSStorage
    Returns:
        Configured GCSStorage instance
    """
    return GCSStorage(bucket_name=bucket_name, **kwargs)
