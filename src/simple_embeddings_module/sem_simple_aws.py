"""
SEMSimpleAWS - One-liner AWS semantic search with Bedrock + S3
Provides the simplest possible interface for AWS-native semantic search:
- Automatic S3 bucket creation with random names
- Bedrock Titan embeddings with credential auto-detection
- Zero-configuration setup for immediate use
"""
import logging
import re
import sys
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
try:
    from .embeddings.mod_bedrock import BedrockEmbeddingProvider
    from .sem_config_builder import SEMConfigBuilder
    from .sem_core import SEMDatabase
    from .storage.mod_s3 import S3Storage
    SEM_AVAILABLE = True
except ImportError:
    SEM_AVAILABLE = False
class SEMSimpleAWS:
    """
    Super-simple AWS semantic search with automatic setup.
    Example usage:
        sem = SEMSimpleAWS()
        sem.add_text("Machine learning transforms software development")
        results = sem.search("AI technology")
    """
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize AWS semantic search with automatic setup.
        Args:
            bucket_name: S3 bucket name (auto-generated if None)
            embedding_model: Bedrock model ID or ARN (None = auto-detect from existing S3 config or use default)
            region: AWS region (default: us-east-1)
            **kwargs: Additional configuration options
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for AWS integration. Install with: pip install boto3")
        if not SEM_AVAILABLE:
            raise ImportError("SEM components not available. Check installation.")
        self.region = region or "us-east-1"
        self.bucket_name = bucket_name
        self.db = None
        self._setup_complete = False
        # Store additional config
        self.config_kwargs = kwargs
        # Determine which model to use: saved config > specified model > default
        self.embedding_model = self._determine_model(embedding_model)
        # Auto-setup on first use
        self._lazy_setup()

    def _determine_model(self, requested_model: Optional[str] = None) -> str:
        """
        Determine which model to use with hybrid local/S3 discovery:
        1. Explicit bucket + model from S3 (highest priority)
        2. Explicitly requested model
        3. Local cached config from previous S3 sync
        4. Default bucket 'sem_bucket' S3 config (with local caching)
        5. Default model (lowest priority)
        Args:
            requested_model: The model explicitly requested by user (can be None)
        Returns:
            The model to actually use
        """
        try:
            # Priority 1: Explicit bucket specified - check S3 directly
            if self.bucket_name and self._test_aws_credentials():
                model_from_s3 = self._get_model_from_s3(self.bucket_name)
                if model_from_s3:
                    if requested_model and model_from_s3 != self._parse_model_id(requested_model):
                        logger.warning(
                            "S3 bucket '%s' has saved model '%s' but '%s' was requested. Using saved model.",
                            self.bucket_name, model_from_s3, requested_model
                        )
                    else:
                        logger.info("Using saved model '%s' from S3 bucket '%s'", model_from_s3, self.bucket_name)
                    # Cache this config locally for future use
                    self._cache_aws_config_locally(self.bucket_name, model_from_s3)
                    return model_from_s3
            # Priority 2: Explicitly requested model
            if requested_model:
                parsed_model = self._parse_model_id(requested_model)
                logger.info("Using explicitly requested model: '%s'", parsed_model)
                return parsed_model
            # Priority 3: Check local cached config (from previous AWS usage)
            cached_config = self._get_cached_aws_config()
            if cached_config and cached_config.get("model"):
                logger.info("Using cached model '%s' from local AWS config cache", cached_config['model'])
                return cached_config["model"]
            # Priority 4: Check default bucket 'sem_bucket' if no bucket specified
            if not self.bucket_name and self._test_aws_credentials():
                default_bucket = "sem_bucket"
                model_from_default = self._get_model_from_s3_with_creation(default_bucket)
                if model_from_default:
                    logger.info("Using saved model '%s' from default S3 bucket '%s'", model_from_default, default_bucket)
                    # Set bucket name and cache config
                    self.bucket_name = default_bucket
                    self._cache_aws_config_locally(default_bucket, model_from_default)
                    return model_from_default
            # Priority 5: No existing config anywhere - use default
            default_model = "amazon.titan-embed-text-v2:0"
            logger.info("No existing AWS config found, using default: '%s'", default_model)
            return default_model
        except Exception as e:
            logger.warning("Could not check AWS configuration: %s", e)
            # Fallback logic
            if requested_model:
                return self._parse_model_id(requested_model)
            return "amazon.titan-embed-text-v2:0"

    def _get_model_from_s3_with_creation(self, bucket_name: str) -> Optional[str]:
        """
        Get model configuration from S3 bucket, creating bucket if it doesn't exist.
        Args:
            bucket_name: S3 bucket name to check/create
        Returns:
            Model name if found, None otherwise
        """
        try:
            # First try to get model from existing bucket
            model = self._get_model_from_s3(bucket_name)
            if model:
                return model
            # If bucket doesn't exist or has no config, try to create it
            logger.info("Default bucket '%s' not found or empty, attempting to create...", bucket_name)
            if self._create_s3_bucket(bucket_name):
                logger.info("Successfully created default S3 bucket: %s", bucket_name)
                # Bucket created but empty, so no model config yet
                return None
            else:
                logger.warning("Failed to create default S3 bucket: %s", bucket_name)
                return None
        except Exception as e:
            logger.debug("Could not check/create S3 bucket '%s': %s", bucket_name, e)
            return None

    def _get_model_from_s3(self, bucket_name: str) -> Optional[str]:
        """
        Get model configuration from S3 bucket.
        Args:
            bucket_name: S3 bucket name to check
        Returns:
            Model name if found, None otherwise
        """
        try:
            storage = S3Storage(bucket_name=bucket_name, region=self.region, prefix="sem-simple-aws/")
            # Check if index exists (using default index name)
            index_name = "sem_simple_index"
            if storage.index_exists(index_name):
                index_info = storage.get_index_info(index_name)
                if index_info and index_info.get("model_name"):
                    return self._parse_model_id(index_info["model_name"])
            return None
        except Exception as e:
            logger.debug("Could not check S3 bucket '%s': %s", bucket_name, e)
            return None

    def _get_cached_aws_config(self) -> Optional[Dict[str, Any]]:
        """
        Get cached AWS configuration from local storage.
        Returns:
            Cached config dict if found, None otherwise
        """
        try:
            import json
            from pathlib import Path
            # Check multiple cache locations
            cache_paths = [
                Path("./sem_indexes/.aws_cache.json"),  # Project-local cache
                Path.home() / "sem_indexes" / ".aws_cache.json",  # Home cache
            ]
            for cache_path in cache_paths:
                if cache_path.exists():
                    with open(cache_path, "r") as f:
                        cached_config = json.load(f)
                    # Check if cache is still valid (not too old)
                    from datetime import datetime, timedelta
                    cached_time = datetime.fromisoformat(cached_config.get("cached_at", "1970-01-01"))
                    if datetime.now() - cached_time < timedelta(hours=24):  # 24 hour cache
                        logger.debug("Found valid cached AWS config at: %s", cache_path)
                        return cached_config
                    else:
                        logger.debug("Cached AWS config expired: %s", cache_path)
            return None
        except Exception as e:
            logger.debug("Could not read cached AWS config: %s", e)
            return None

    def _cache_aws_config_locally(self, bucket_name: str, model: str) -> None:
        """
        Cache AWS configuration locally for faster future access.
        Args:
            bucket_name: S3 bucket name
            model: Model name to cache
        """
        try:
            import json
            from datetime import datetime
            from pathlib import Path
            # Prefer project-local cache, fallback to home cache
            cache_paths = [
                Path("./sem_indexes/.aws_cache.json"),  # Project-local cache
                Path.home() / "sem_indexes" / ".aws_cache.json",  # Home cache
            ]
            cache_data = {
                "bucket_name": bucket_name,
                "model": model,
                "region": self.region,
                "cached_at": datetime.now().isoformat(),
                "cache_version": "1.0",
            }
            # Try to write to the first available location
            for cache_path in cache_paths:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w") as f:
                        json.dump(cache_data, f, indent=2)
                    logger.debug("Cached AWS config to: %s", cache_path)
                    break
                except Exception as e:
                    logger.debug("Could not write to %s: %s", cache_path, e)
                    continue
        except Exception as e:
            logger.debug("Could not cache AWS config: %s", e)
            # Non-fatal error, continue without caching

    def _parse_model_id(self, model_input: str) -> str:
        """
        Parse model ID from ARN or direct model ID.
        Args:
            model_input: Model ARN or direct model ID
        Returns:
            Clean model ID
        """
        # Check if it's an ARN
        if model_input.startswith("arn:aws:bedrock:"):
            # Extract model ID from ARN: arn:aws:bedrock:region:account:foundation-model/model-id
            match = re.search(r"foundation-model/(.+)$", model_input)
            if match:
                return match.group(1)
            else:
                logger.warning("Could not parse model ID from ARN: %s", model_input)
                return "amazon.titan-embed-text-v2:0"  # Fallback
        # Direct model ID
        return model_input

    def _generate_bucket_name(self) -> str:
        """Generate a valid S3 bucket name with random GUID."""
        # S3 bucket naming rules:
        # - 3-63 characters
        # - lowercase letters, numbers, hyphens
        # - start and end with letter or number
        # - no consecutive periods or hyphens
        guid = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
        bucket_name = f"sem-{guid}"
        # Ensure it's valid (should be by construction)
        if not re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$", bucket_name):
            # Fallback to simple format
            bucket_name = f"sem{guid.replace('-', '')}"
        return bucket_name

    def _test_aws_credentials(self) -> bool:
        """Test if AWS credentials are available and working."""
        try:
            # Try to get caller identity
            sts = boto3.client("sts", region_name=self.region)
            sts.get_caller_identity()
            return True
        except (NoCredentialsError, ClientError) as e:
            logger.error("AWS credentials test failed: %s", e)
            return False

    def _create_s3_bucket(self, bucket_name: str) -> bool:
        """
        Create S3 bucket if it doesn't exist.
        Args:
            bucket_name: Name of bucket to create
        Returns:
            True if bucket exists or was created successfully
        """
        try:
            s3 = boto3.client("s3", region_name=self.region)
            # Check if bucket already exists
            try:
                s3.head_bucket(Bucket=bucket_name)
                logger.info("S3 bucket '%s' already exists", bucket_name)
                return True
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "404":
                    # Bucket doesn't exist, create it
                    pass
                elif error_code == "403":
                    # Bucket exists but we don't have access
                    logger.error("Access denied to bucket '%s'", bucket_name)
                    return False
                else:
                    logger.error("Error checking bucket '%s': %s", bucket_name, e)
                    return False
            # Create bucket
            create_kwargs = {"Bucket": bucket_name}
            # Add location constraint for regions other than us-east-1
            if self.region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
            s3.create_bucket(**create_kwargs)
            # Print bucket name to stderr as requested
            print(f"Created S3 bucket: {bucket_name}", file=sys.stderr)
            logger.info("Successfully created S3 bucket: %s", bucket_name)
            return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "BucketAlreadyOwnedByYou":
                logger.info("S3 bucket '%s' already owned by you", bucket_name)
                return True
            elif error_code == "BucketAlreadyExists":
                logger.error("S3 bucket name '%s' already exists globally", bucket_name)
                return False
            else:
                logger.error("Failed to create S3 bucket '%s': %s", bucket_name, e)
                return False
        except Exception as e:
            logger.error("Unexpected error creating S3 bucket '%s': %s", bucket_name, e)
            return False

    def _test_bedrock_access(self) -> bool:
        """
        Test Bedrock access with a simple embedding query.
        Returns:
            True if Bedrock access works
        """
        try:
            # Create Bedrock provider
            bedrock_config = {
                "model_id": self.embedding_model,
                "region": self.region,
            }
            provider = BedrockEmbeddingProvider(**bedrock_config)
            # Test with a simple query
            test_text = "Hello world"
            embedding = provider.embed_query(test_text)
            logger.info("Bedrock access test successful with model: %s", self.embedding_model)
            logger.debug("Test embedding shape: %s", embedding.shape)
            return True
        except Exception as e:
            logger.error("Bedrock access test failed: %s", e)
            return False

    def _lazy_setup(self):
        """Perform lazy setup on first use."""
        if self._setup_complete:
            return
        logger.info("Setting up AWS semantic search...")
        # Test AWS credentials
        if not self._test_aws_credentials():
            raise RuntimeError(
                "AWS credentials not available. Configure with:\n"
                "  - AWS CLI: aws configure\n"
                "  - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
                "  - IAM roles (for EC2/Lambda/ECS)\n"
                "  - AWS profiles: export AWS_PROFILE=myprofile"
            )
        # Generate bucket name if not provided
        if not self.bucket_name:
            max_attempts = 5
            for attempt in range(max_attempts):
                candidate_name = self._generate_bucket_name()
                if self._create_s3_bucket(candidate_name):
                    self.bucket_name = candidate_name
                    break
                logger.warning("Bucket name '%s' failed, trying another...", candidate_name)
            if not self.bucket_name:
                raise RuntimeError("Failed to create S3 bucket after %s attempts" % max_attempts)
        else:
            # Use provided bucket name
            if not self._create_s3_bucket(self.bucket_name):
                raise RuntimeError("Failed to create or access S3 bucket: %s" % self.bucket_name)
        # Test Bedrock access
        if not self._test_bedrock_access():
            raise RuntimeError(
                "Bedrock access failed for model: %s\n"
                "Check:\n"
                "  - Model is available in region: %s\n"
                "  - IAM permissions for bedrock:InvokeModel\n"
                "  - Model access is enabled in Bedrock console" % (self.embedding_model, self.region)
            )
        # Build SEM configuration
        builder = SEMConfigBuilder()
        # Configure Bedrock embeddings
        builder.set_embedding_provider("bedrock", model_id=self.embedding_model, region=self.region)
        # Configure S3 storage
        builder.set_storage_backend(
            "s3",
            bucket_name=self.bucket_name,
            region=self.region,
            prefix="sem-simple-aws/",
            compression=True,
            encryption="AES256",
            **self.config_kwargs,
        )
        # Auto-configure chunking
        builder.auto_configure_chunking()
        # Build and create database
        config = builder.build()
        self.db = SEMDatabase(config=config)
        # Check for existing index
        existing_info = self.db.get_index_info()
        if existing_info and existing_info.document_count > 0:
            logger.info("Found existing index with %s documents", existing_info.document_count)
            print(
                f"📚 Found existing semantic search index with {existing_info.document_count} documents",
                file=sys.stderr,
            )
            print("🔍 Ready to search! Use .search('your query') to find documents", file=sys.stderr)
        else:
            logger.info("No existing index found - ready to add documents")
            print("📝 Ready to add documents! Use .add_text('your content') to start", file=sys.stderr)
        self._setup_complete = True
        logger.info("AWS semantic search ready! Bucket: %s, Model: %s", self.bucket_name, self.embedding_model)
        # Show index info
        if existing_info:
            print(
                f"📊 Index: {existing_info.document_count} docs, {existing_info.embedding_dim}D embeddings",
                file=sys.stderr,
            )

    def add_text(self, text: str, document_id: Optional[str] = None) -> str:
        """
        Add a single text document.
        Args:
            text: Text content to add
            document_id: Optional document ID (auto-generated if None)
        Returns:
            Document ID that was used
        """
        self._lazy_setup()
        if document_id is None:
            # Get current document count from index info
            index_info = self.db.get_index_info()
            base_count = index_info.document_count if index_info else 0
            document_id = f"doc_{base_count + 1}"
        self.db.add_documents([text], document_ids=[document_id])
        return document_id

    def add_texts(self, texts: List[str], document_ids: Optional[List[str]] = None) -> List[str]:
        """
        Add multiple text documents.
        Args:
            texts: List of text contents
            document_ids: Optional list of document IDs (auto-generated if None)
        Returns:
            List of document IDs that were used
        """
        self._lazy_setup()
        if document_ids is None:
            # Get current document count from index info
            index_info = self.db.get_index_info()
            base_count = index_info.document_count if index_info else 0
            document_ids = [f"doc_{base_count + i + 1}" for i in range(len(texts))]
        if len(document_ids) != len(texts):
            raise ValueError("Number of document_ids must match number of texts")
        # documents = dict(zip(document_ids, texts))  # Appears unused, commenting out for scream test
        self.db.add_documents(texts, document_ids=document_ids)
        return document_ids

    def search(self, query: str, top_k: int = 5, similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (uses config default if None)
        Returns:
            List of search results with scores and content
        """
        self._lazy_setup()
        return self.db.search(query, top_k=top_k, similarity_threshold=similarity_threshold)

    def get_info(self) -> Dict[str, Any]:
        """Get information about the database."""
        self._lazy_setup()
        # Get basic info from storage backend
        index_info = self.db.get_index_info()
        # Build comprehensive info
        info = {
            "aws_region": self.region,
            "s3_bucket": self.bucket_name,
            "bedrock_model": self.embedding_model,
            "setup_complete": self._setup_complete,
            "index_name": "sem_simple_index",  # Always show the standardized name
            "document_count": 0,  # Will be updated if index exists
            "embedding_dim": 0,  # Will be updated if index exists
        }
        if index_info:
            info.update(
                {
                    "document_count": index_info.document_count,
                    "embedding_dim": index_info.embedding_dim,
                    "index_name": index_info.index_name,
                    "created_at": index_info.created_at,
                    "updated_at": index_info.updated_at,
                    "size_bytes": index_info.size_bytes,
                    "backend_type": index_info.backend_type,
                }
            )
        return info

    def list_documents(
        self, limit: Optional[int] = None, show_content: bool = True, max_content_length: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List documents in the AWS semantic search index.
        Args:
            limit: Maximum number of documents to return (None for all)
            show_content: Whether to include document content snippets
            max_content_length: Maximum length of content to show
        Returns:
            List of document dictionaries with 'id', 'text' (if show_content), and metadata
        Example:
            >>> aws_sem = simple_aws(bucket_name="my-bucket")
            >>> docs = aws_sem.list_documents(limit=5)
            >>> for doc in docs:
            ...     print(f"ID: {doc['id']}, Text: {doc['text'][:50]}...")
        """
        self._lazy_setup()
        try:
            return self.db.list_documents(limit=limit, show_content=show_content, max_content_length=max_content_length)
        except Exception as e:
            logger.error("Error listing documents: %s", e)
            return []

    def __repr__(self) -> str:
        """String representation."""
        if self._setup_complete:
            return (
                f"SEMSimpleAWS("
                f"bucket={self.bucket_name}, "
                f"model={self.embedding_model}, "
                f"region={self.region})"
            )
        else:
            return "SEMSimpleAWS(pending_setup)"
def simple_aws(
    bucket_name: Optional[str] = None, embedding_model: Optional[str] = None, region: Optional[str] = None, **kwargs
) -> SEMSimpleAWS:
    """
    Create a simple AWS semantic search instance.
    This is the one-liner function for AWS semantic search:
    Examples:
        # Simplest usage - auto-creates bucket, auto-detects model from existing S3 config or uses default
        sem = simple_aws()
        sem.add_text("Machine learning transforms software")
        results = sem.search("AI technology")
        # With custom bucket (will auto-detect model from existing S3 index if it exists)
        sem = simple_aws(bucket_name="my-semantic-search-bucket")
        # With explicit model (overrides any existing S3 config)
        sem = simple_aws(embedding_model="amazon.titan-embed-text-v1")
        # With ARN
        sem = simple_aws(embedding_model="arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0")
    Args:
        bucket_name: S3 bucket name (auto-generated if None)
        embedding_model: Bedrock model ID or ARN (None = auto-detect from existing S3 config or use default)
        region: AWS region (default: us-east-1)
        **kwargs: Additional S3 configuration options
    Returns:
        SEMSimpleAWS instance ready for use
    """
    return SEMSimpleAWS(bucket_name=bucket_name, embedding_model=embedding_model, region=region, **kwargs)
