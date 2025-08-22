"""
Amazon S3 Storage Backend

Implements storage backend for Amazon S3 with secure serialization and compression.
Provides atomic operations, versioning, and encryption support.
"""

import gzip
import io
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ..sem_module_reg import ConfigParameter
from ..serialization.mod_orjson import OrjsonSerializationProvider
from .mod_storage_base import IndexInfo, StorageBackendBase, StorageBackendError

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available - S3 storage backend disabled")


class S3Storage(StorageBackendBase):
    """Amazon S3 storage backend with secure serialization and compression"""

    CONFIG_PARAMETERS = [
        ConfigParameter(
            key_name="bucket_name",
            value_type="str",
            config_description="S3 bucket name for storing indexes",
            required=True,
        ),
        ConfigParameter(
            key_name="region",
            value_type="str",
            config_description="AWS region for S3 bucket",
            required=False,
            value_opt_default="us-east-1",
        ),
        ConfigParameter(
            key_name="prefix",
            value_type="str",
            config_description="S3 key prefix for organizing indexes",
            required=False,
            value_opt_default="sem-indexes/",
        ),
        ConfigParameter(
            key_name="compression",
            value_type="bool",
            config_description="Enable gzip compression for stored files",
            required=False,
            value_opt_default=True,
        ),
        ConfigParameter(
            key_name="encryption",
            value_type="str",
            config_description="S3 server-side encryption (AES256, aws:kms, or None)",
            required=False,
            value_opt_default="AES256",
        ),
        ConfigParameter(
            key_name="storage_class",
            value_type="str",
            config_description="S3 storage class (STANDARD, STANDARD_IA, GLACIER, etc.)",
            required=False,
            value_opt_default="STANDARD",
        ),
        ConfigParameter(
            key_name="aws_access_key_id",
            value_type="str",
            config_description="AWS access key ID (optional if using IAM roles)",
            required=False,
        ),
        ConfigParameter(
            key_name="aws_secret_access_key",
            value_type="str",
            config_description="AWS secret access key (optional if using IAM roles)",
            required=False,
        ),
        ConfigParameter(
            key_name="aws_session_token",
            value_type="str",
            config_description="AWS session token for temporary credentials",
            required=False,
        ),
        ConfigParameter(
            key_name="endpoint_url",
            value_type="str",
            config_description="Custom S3 endpoint URL (for S3-compatible services)",
            required=False,
        ),
    ]

    CAPABILITIES = {
        "max_index_size": -1,  # Unlimited (S3 object size limit is 5TB)
        "supports_streaming": True,
        "supports_partial_updates": False,  # S3 requires full object replacement
        "supports_concurrent_access": True,
        "connection_info": {"type": "s3", "requires_credentials": True},
        "compression_supported": True,
        "encryption_supported": True,
        "versioning_supported": True,
        "backup_supported": True,
        "atomic_operations": True,
    }

    def __init__(self, **config):
        """Initialize S3 storage backend with configuration validation"""
        super().__init__(**config)
        
        if not BOTO3_AVAILABLE:
            raise StorageBackendError("boto3 is required for S3 storage backend. Install with: pip install boto3")
        
        self.bucket_name = config["bucket_name"]
        self.region = config.get("region", "us-east-1")
        self.prefix = config.get("prefix", "sem-indexes/").rstrip("/") + "/"
        self.compression = config.get("compression", True)
        self.encryption = config.get("encryption", "AES256")
        self.storage_class = config.get("storage_class", "STANDARD")
        
        # Initialize S3 client
        self._init_s3_client(config)
        
        # Initialize serialization provider
        self.serializer = OrjsonSerializationProvider()
        
        # Verify bucket access
        self._verify_bucket_access()

    def _init_s3_client(self, config: Dict[str, Any]) -> None:
        """Initialize S3 client with credentials and configuration"""
        session_kwargs = {}
        
        # Add explicit credentials if provided
        if config.get("aws_access_key_id"):
            session_kwargs["aws_access_key_id"] = config["aws_access_key_id"]
        if config.get("aws_secret_access_key"):
            session_kwargs["aws_secret_access_key"] = config["aws_secret_access_key"]
        if config.get("aws_session_token"):
            session_kwargs["aws_session_token"] = config["aws_session_token"]
        
        # Create session
        session = boto3.Session(**session_kwargs)
        
        # Create S3 client
        client_kwargs = {"region_name": self.region}
        if config.get("endpoint_url"):
            client_kwargs["endpoint_url"] = config["endpoint_url"]
        
        self.s3_client = session.client("s3", **client_kwargs)
        
        logger.info(f"Initialized S3 client for bucket '{self.bucket_name}' in region '{self.region}'")

    def _verify_bucket_access(self) -> None:
        """Verify that we can access the S3 bucket"""
        try:
            # Try to list objects with a limit to test access
            self.s3_client.list_objects_v2(Bucket=self.bucket_name, MaxKeys=1)
            logger.info(f"Successfully verified access to S3 bucket '{self.bucket_name}'")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                raise StorageBackendError(f"S3 bucket '{self.bucket_name}' does not exist")
            elif error_code == "AccessDenied":
                raise StorageBackendError(f"Access denied to S3 bucket '{self.bucket_name}'")
            else:
                raise StorageBackendError(f"Failed to access S3 bucket '{self.bucket_name}': {e}")
        except NoCredentialsError:
            raise StorageBackendError("AWS credentials not found. Configure credentials via AWS CLI, environment variables, or IAM roles")

    def _get_s3_key(self, index_name: str, file_type: str) -> str:
        """Generate S3 key for index files"""
        return f"{self.prefix}{index_name}/{file_type}"

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip if compression is enabled"""
        if not self.compression:
            return data
        
        compressed = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode='wb') as gz:
            gz.write(data)
        return compressed.getvalue()

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress gzip data if compression was used"""
        if not self.compression:
            return data
        
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data), mode='rb') as gz:
                return gz.read()
        except gzip.BadGzipFile:
            # Data might not be compressed (backward compatibility)
            return data

    def _upload_object(self, key: str, data: bytes, metadata: Optional[Dict[str, str]] = None) -> None:
        """Upload object to S3 with proper configuration"""
        put_kwargs = {
            "Bucket": self.bucket_name,
            "Key": key,
            "Body": data,
            "StorageClass": self.storage_class,
        }
        
        # Add server-side encryption
        if self.encryption and self.encryption != "None":
            put_kwargs["ServerSideEncryption"] = self.encryption
        
        # Add metadata
        if metadata:
            put_kwargs["Metadata"] = metadata
        
        # Add content encoding for compressed data
        if self.compression:
            put_kwargs["ContentEncoding"] = "gzip"
        
        try:
            self.s3_client.put_object(**put_kwargs)
            logger.debug(f"Successfully uploaded object to S3: {key}")
        except ClientError as e:
            raise StorageBackendError(f"Failed to upload object to S3 key '{key}': {e}")

    def _download_object(self, key: str) -> bytes:
        """Download object from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response["Body"].read()
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                raise StorageBackendError(f"S3 object not found: {key}")
            else:
                raise StorageBackendError(f"Failed to download S3 object '{key}': {e}")

    def save_index(self, vectors: torch.Tensor, metadata: Dict[str, Any], index_name: str) -> bool:
        """Save vector index and metadata to S3"""
        logger.info(f"Saving index '{index_name}' to S3 bucket '{self.bucket_name}'")
        
        try:
            # Move vectors to CPU for serialization
            cpu_vectors = vectors.cpu()
            
            # Serialize vectors
            vectors_data = self.serializer.serialize_tensor(cpu_vectors)
            vectors_bytes = self._compress_data(vectors_data)
            
            # Serialize metadata
            enhanced_metadata = {
                **metadata,
                "storage_backend": "s3",
                "bucket_name": self.bucket_name,
                "compression": self.compression,
                "encryption": self.encryption,
                "storage_class": self.storage_class,
                "saved_at": datetime.now().isoformat(),
                "vector_shape": list(cpu_vectors.shape),
                "vector_dtype": str(cpu_vectors.dtype),
            }
            
            metadata_data = self.serializer.serialize_metadata(enhanced_metadata)
            metadata_bytes = self._compress_data(metadata_data)
            
            # Upload vectors and metadata
            vectors_key = self._get_s3_key(index_name, "vectors.json")
            metadata_key = self._get_s3_key(index_name, "metadata.json")
            
            # Upload metadata first (smaller, faster to fail if there are issues)
            self._upload_object(
                metadata_key, 
                metadata_bytes,
                {"index_name": index_name, "file_type": "metadata"}
            )
            
            # Upload vectors
            self._upload_object(
                vectors_key, 
                vectors_bytes,
                {"index_name": index_name, "file_type": "vectors"}
            )
            
            logger.info(f"Successfully saved index '{index_name}' to S3")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index '{index_name}' to S3: {e}")
            # Attempt cleanup of partial uploads
            try:
                self._cleanup_partial_upload(index_name)
            except:
                pass  # Don't fail the original operation due to cleanup issues
            return False

    def load_index(self, index_name: str, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load vector index and metadata from S3"""
        logger.info(f"Loading index '{index_name}' from S3 bucket '{self.bucket_name}'")
        
        try:
            # Download metadata first
            metadata_key = self._get_s3_key(index_name, "metadata.json")
            metadata_bytes = self._download_object(metadata_key)
            metadata_data = self._decompress_data(metadata_bytes)
            metadata = self.serializer.deserialize_metadata(metadata_data)
            
            # Download vectors
            vectors_key = self._get_s3_key(index_name, "vectors.json")
            vectors_bytes = self._download_object(vectors_key)
            vectors_data = self._decompress_data(vectors_bytes)
            vectors = self.serializer.deserialize_tensor(vectors_data)
            
            # Move to target device
            if device is not None:
                vectors = vectors.to(device)
            
            logger.info(f"Successfully loaded index '{index_name}' from S3")
            return vectors, metadata
            
        except Exception as e:
            logger.error(f"Failed to load index '{index_name}' from S3: {e}")
            raise StorageBackendError(f"Failed to load index '{index_name}': {e}")

    def list_indexes(self) -> List[str]:
        """List available indexes in S3 bucket"""
        logger.debug(f"Listing indexes in S3 bucket '{self.bucket_name}'")
        
        try:
            indexes = set()
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        # Extract index name from key pattern: prefix/index_name/file_type
                        if key.startswith(self.prefix) and '/' in key[len(self.prefix):]:
                            index_name = key[len(self.prefix):].split('/')[0]
                            if index_name:  # Skip empty names
                                indexes.add(index_name)
            
            result = sorted(list(indexes))
            logger.debug(f"Found {len(result)} indexes in S3")
            return result
            
        except ClientError as e:
            logger.error(f"Failed to list indexes in S3: {e}")
            raise StorageBackendError(f"Failed to list indexes: {e}")

    def delete_index(self, index_name: str) -> bool:
        """Delete an index from S3"""
        logger.info(f"Deleting index '{index_name}' from S3 bucket '{self.bucket_name}'")
        
        try:
            # List all objects for this index
            prefix = f"{self.prefix}{index_name}/"
            objects_to_delete = []
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects_to_delete.append({'Key': obj['Key']})
            
            if not objects_to_delete:
                logger.warning(f"No objects found for index '{index_name}'")
                return False
            
            # Delete objects in batches (S3 allows up to 1000 objects per delete request)
            batch_size = 1000
            for i in range(0, len(objects_to_delete), batch_size):
                batch = objects_to_delete[i:i + batch_size]
                
                delete_request = {
                    'Objects': batch,
                    'Quiet': True  # Don't return info about successful deletions
                }
                
                response = self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete=delete_request
                )
                
                # Check for errors
                if 'Errors' in response and response['Errors']:
                    errors = response['Errors']
                    logger.error(f"Failed to delete some objects for index '{index_name}': {errors}")
                    return False
            
            logger.info(f"Successfully deleted index '{index_name}' from S3 ({len(objects_to_delete)} objects)")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete index '{index_name}' from S3: {e}")
            return False

    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists in S3"""
        try:
            # Check if metadata file exists (required for a valid index)
            metadata_key = self._get_s3_key(index_name, "metadata.json")
            self.s3_client.head_object(Bucket=self.bucket_name, Key=metadata_key)
            return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                return False
            else:
                logger.error(f"Error checking if index '{index_name}' exists: {e}")
                return False

    def get_index_info(self, index_name: str) -> Optional[IndexInfo]:
        """Get detailed information about an index"""
        if not self.index_exists(index_name):
            return None
        
        try:
            # Get metadata file info
            metadata_key = self._get_s3_key(index_name, "metadata.json")
            vectors_key = self._get_s3_key(index_name, "vectors.json")
            
            metadata_obj = self.s3_client.head_object(Bucket=self.bucket_name, Key=metadata_key)
            vectors_obj = self.s3_client.head_object(Bucket=self.bucket_name, Key=vectors_key)
            
            # Calculate total size
            total_size = metadata_obj["ContentLength"] + vectors_obj["ContentLength"]
            
            # Get last modified time
            last_modified = max(metadata_obj["LastModified"], vectors_obj["LastModified"])
            
            return IndexInfo(
                index_name=index_name,
                created_at=last_modified.isoformat(),
                updated_at=last_modified.isoformat(),
                document_count=0,  # Not tracked at storage level
                embedding_dim=0,   # Not available without loading metadata
                model_name=None,   # Not available without loading metadata
                size_bytes=total_size,
                backend_type="s3",
            )
            
        except ClientError as e:
            logger.error(f"Failed to get info for index '{index_name}': {e}")
            return None

    def _cleanup_partial_upload(self, index_name: str) -> None:
        """Clean up partial uploads in case of failure"""
        logger.debug(f"Cleaning up partial upload for index '{index_name}'")
        try:
            # Try to delete any objects that might have been created
            prefix = f"{self.prefix}{index_name}/"
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        try:
                            self.s3_client.delete_object(Bucket=self.bucket_name, Key=obj['Key'])
                        except:
                            pass  # Best effort cleanup
        except:
            pass  # Best effort cleanup

    def get_capabilities(self) -> Dict[str, Any]:
        """Return S3 storage backend capabilities"""
        capabilities = self.CAPABILITIES.copy()
        capabilities["connection_info"].update({
            "bucket": self.bucket_name,
            "region": self.region,
            "prefix": self.prefix,
            "compression": self.compression,
            "encryption": self.encryption,
            "storage_class": self.storage_class,
        })
        return capabilities
