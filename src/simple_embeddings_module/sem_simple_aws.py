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
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from .sem_config_builder import SEMConfigBuilder
    from .sem_core import SEMDatabase
    from .embeddings.mod_bedrock import BedrockEmbeddingProvider
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
        **kwargs
    ):
        """
        Initialize AWS semantic search with automatic setup.
        
        Args:
            bucket_name: S3 bucket name (auto-generated if None)
            embedding_model: Bedrock model ID or ARN (default: amazon.titan-embed-text-v2:0)
            region: AWS region (default: us-east-1)
            **kwargs: Additional configuration options
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for AWS integration. Install with: pip install boto3")
        
        if not SEM_AVAILABLE:
            raise ImportError("SEM components not available. Check installation.")
        
        self.region = region or "us-east-1"
        self.embedding_model = self._parse_model_id(embedding_model or "amazon.titan-embed-text-v2:0")
        self.bucket_name = bucket_name
        self.db = None
        self._setup_complete = False
        
        # Store additional config
        self.config_kwargs = kwargs
        
        # Auto-setup on first use
        self._lazy_setup()
    
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
                logger.warning(f"Could not parse model ID from ARN: {model_input}")
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
            sts = boto3.client('sts', region_name=self.region)
            sts.get_caller_identity()
            return True
        except (NoCredentialsError, ClientError) as e:
            logger.error(f"AWS credentials test failed: {e}")
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
            s3 = boto3.client('s3', region_name=self.region)
            
            # Check if bucket already exists
            try:
                s3.head_bucket(Bucket=bucket_name)
                logger.info(f"S3 bucket '{bucket_name}' already exists")
                return True
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    # Bucket doesn't exist, create it
                    pass
                elif error_code == '403':
                    # Bucket exists but we don't have access
                    logger.error(f"Access denied to bucket '{bucket_name}'")
                    return False
                else:
                    logger.error(f"Error checking bucket '{bucket_name}': {e}")
                    return False
            
            # Create bucket
            create_kwargs = {'Bucket': bucket_name}
            
            # Add location constraint for regions other than us-east-1
            if self.region != 'us-east-1':
                create_kwargs['CreateBucketConfiguration'] = {
                    'LocationConstraint': self.region
                }
            
            s3.create_bucket(**create_kwargs)
            
            # Print bucket name to stderr as requested
            print(f"Created S3 bucket: {bucket_name}", file=sys.stderr)
            logger.info(f"Successfully created S3 bucket: {bucket_name}")
            
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'BucketAlreadyOwnedByYou':
                logger.info(f"S3 bucket '{bucket_name}' already owned by you")
                return True
            elif error_code == 'BucketAlreadyExists':
                logger.error(f"S3 bucket name '{bucket_name}' already exists globally")
                return False
            else:
                logger.error(f"Failed to create S3 bucket '{bucket_name}': {e}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error creating S3 bucket '{bucket_name}': {e}")
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
            
            logger.info(f"Bedrock access test successful with model: {self.embedding_model}")
            logger.debug(f"Test embedding shape: {embedding.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Bedrock access test failed: {e}")
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
                logger.warning(f"Bucket name '{candidate_name}' failed, trying another...")
            
            if not self.bucket_name:
                raise RuntimeError(f"Failed to create S3 bucket after {max_attempts} attempts")
        else:
            # Use provided bucket name
            if not self._create_s3_bucket(self.bucket_name):
                raise RuntimeError(f"Failed to create or access S3 bucket: {self.bucket_name}")
        
        # Test Bedrock access
        if not self._test_bedrock_access():
            raise RuntimeError(
                f"Bedrock access failed for model: {self.embedding_model}\n"
                "Check:\n"
                "  - Model is available in region: {self.region}\n"
                "  - IAM permissions for bedrock:InvokeModel\n"
                "  - Model access is enabled in Bedrock console"
            )
        
        # Build SEM configuration
        builder = SEMConfigBuilder()
        
        # Configure Bedrock embeddings
        builder.set_embedding_provider("bedrock", 
            model_id=self.embedding_model,
            region=self.region
        )
        
        # Configure S3 storage
        builder.set_storage_backend("s3",
            bucket_name=self.bucket_name,
            region=self.region,
            prefix="sem-simple-aws/",
            compression=True,
            encryption="AES256",
            **self.config_kwargs
        )
        
        # Auto-configure chunking
        builder.auto_configure_chunking()
        
        # Build and create database
        config = builder.build()
        self.db = SEMDatabase(config=config)
        
        # Check for existing index
        existing_info = self.db.get_index_info()
        if existing_info and existing_info.document_count > 0:
            logger.info(f"Found existing index with {existing_info.document_count} documents")
            print(f"📚 Found existing semantic search index with {existing_info.document_count} documents", file=sys.stderr)
            print(f"🔍 Ready to search! Use .search('your query') to find documents", file=sys.stderr)
        else:
            logger.info("No existing index found - ready to add documents")
            print(f"📝 Ready to add documents! Use .add_text('your content') to start", file=sys.stderr)
        
        self._setup_complete = True
        logger.info(f"AWS semantic search ready! Bucket: {self.bucket_name}, Model: {self.embedding_model}")
        
        # Show index info
        if existing_info:
            print(f"📊 Index: {existing_info.document_count} docs, {existing_info.embedding_dim}D embeddings", file=sys.stderr)
    
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
        
        documents = dict(zip(document_ids, texts))
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
            "embedding_dim": 0,   # Will be updated if index exists
        }
        
        if index_info:
            info.update({
                "document_count": index_info.document_count,
                "embedding_dim": index_info.embedding_dim,
                "index_name": index_info.index_name,
                "created_at": index_info.created_at,
                "updated_at": index_info.updated_at,
                "size_bytes": index_info.size_bytes,
                "backend_type": index_info.backend_type,
            })
        
        return info
    
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
            return f"SEMSimpleAWS(pending_setup)"


def simple_aws(
    bucket_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    region: Optional[str] = None,
    **kwargs
) -> SEMSimpleAWS:
    """
    Create a simple AWS semantic search instance.
    
    This is the one-liner function for AWS semantic search:
    
    Examples:
        # Simplest usage - auto-creates bucket, uses Titan v2
        sem = simple_aws()
        sem.add_text("Machine learning transforms software")
        results = sem.search("AI technology")
        
        # With custom bucket
        sem = simple_aws(bucket_name="my-semantic-search-bucket")
        
        # With different model
        sem = simple_aws(embedding_model="amazon.titan-embed-text-v1")
        
        # With ARN
        sem = simple_aws(embedding_model="arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0")
    
    Args:
        bucket_name: S3 bucket name (auto-generated if None)
        embedding_model: Bedrock model ID or ARN (default: amazon.titan-embed-text-v2:0)
        region: AWS region (default: us-east-1)
        **kwargs: Additional S3 configuration options
        
    Returns:
        SEMSimpleAWS instance ready for use
    """
    return SEMSimpleAWS(
        bucket_name=bucket_name,
        embedding_model=embedding_model,
        region=region,
        **kwargs
    )
