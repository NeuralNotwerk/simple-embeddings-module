#!/usr/bin/env python3
"""
SEM Web Client - Python HTTP Client for SEM Web API
Provides a Python interface to interact with the SEM Web API server.

This client mirrors the library interface but communicates with a remote SEM API server,
enabling distributed semantic search deployments.

Features:
- Same interface as SEMSimple but over HTTP
- Automatic request/response handling
- Error handling and retries
- Authentication support
- Connection pooling
"""
import logging
from typing import Dict, Any, List, Optional, Union
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class SEMWebClient:
    """
    HTTP client for SEM Web API.

    Provides the same interface as SEMSimple but communicates with a remote API server.

    Example:
        >>> client = SEMWebClient("http://localhost:8000")
        >>> client.add_text("Machine learning is transforming software.")
        True
        >>> results = client.search("AI technology")
        >>> print(results[0]['text'])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        retries: int = 3
    ):
        """
        Initialize SEM Web Client.

        Args:
            base_url: Base URL of the SEM API server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            retries: Number of retry attempts for failed requests
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

        # Setup session with retries
        self.session = requests.Session()

        # Configure retries
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Setup headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'SEM-Web-Client/1.0.0'
        })

        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}'
            })

        logger.info("SEM Web Client initialized: %s", self.base_url)

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path
            data: Request data dictionary

        Returns:
            Response data dictionary

        Raises:
            Exception: If request fails or API returns error
        """
        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=self.timeout)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, timeout=self.timeout)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            result = response.json()

            # Check API-level success
            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown API error')
                logger.error("API error: %s", error_msg)
                raise Exception(error_msg)

            return result

        except requests.exceptions.RequestException as e:
            logger.error("HTTP request failed: %s", e)
            raise Exception(f"HTTP request failed: {e}")
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON response: %s", e)
            raise Exception(f"Invalid JSON response: {e}")

    def health_check(self) -> bool:
        """
        Check if the API server is healthy.

        Returns:
            True if server is healthy
        """
        try:
            result = self._make_request('GET', '/health')
            return result.get('success', False)
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return False

    def add_text(self, text: str, db: Optional[str] = None, path: Optional[str] = None) -> bool:
        """
        Add a single text document to the search index.

        Args:
            text: The text content to add
            db: Optional database name
            path: Optional storage path

        Returns:
            True if successful
        """
        try:
            data = {
                'text': text,
                'db': db,
                'path': path
            }
            result = self._make_request('POST', '/add', data)
            return result.get('success', False)
        except Exception as e:
            logger.error("Failed to add text: %s", e)
            return False

    def add_files(self, file_paths: List[str], db: Optional[str] = None, path: Optional[str] = None) -> bool:
        """
        Add multiple text files to the search index.

        Args:
            file_paths: List of paths to text files
            db: Optional database name
            path: Optional storage path

        Returns:
            True if successful
        """
        try:
            data = {
                'files': file_paths,
                'db': db,
                'path': path
            }
            result = self._make_request('POST', '/add', data)
            return result.get('success', False)
        except Exception as e:
            logger.error("Failed to add files: %s", e)
            return False

    def search(
        self,
        query: str,
        top_k: int = 5,
        db: Optional[str] = None,
        path: Optional[str] = None,
        output_format: str = "dict",
        delimiter: str = ";"
    ) -> Union[List[dict], str]:
        """
        Search for documents similar to the query.

        Args:
            query: Search query text
            top_k: Number of results to return
            db: Optional database name
            path: Optional storage path
            output_format: Output format ("dict", "cli", "json", "csv")
            delimiter: Delimiter for CLI format

        Returns:
            List of result dictionaries or formatted string
        """
        try:
            data = {
                'query': query,
                'top_k': top_k,
                'db': db,
                'path': path,
                'cli_format': output_format == "cli",
                'delimiter': delimiter
            }
            result = self._make_request('POST', '/search', data)
            return result.get('data', {}).get('results', [])
        except Exception as e:
            logger.error("Search failed: %s", e)
            return [] if output_format == "dict" else ""

    def list_documents(
        self,
        limit: Optional[int] = None,
        show_content: bool = True,
        db: Optional[str] = None,
        path: Optional[str] = None,
        output_format: str = "dict",
        delimiter: str = ";"
    ) -> Union[List[dict], str]:
        """
        List documents in the search index.

        Args:
            limit: Maximum number of documents to return
            show_content: Whether to include document content
            db: Optional database name
            path: Optional storage path
            output_format: Output format ("dict", "cli", "json", "csv")
            delimiter: Delimiter for CLI format

        Returns:
            List of document dictionaries or formatted string
        """
        try:
            data = {
                'limit': limit,
                'no_content': not show_content,
                'db': db,
                'path': path,
                'cli_format': output_format == "cli",
                'delimiter': delimiter
            }
            result = self._make_request('POST', '/list', data)
            return result.get('data', {}).get('documents', [])
        except Exception as e:
            logger.error("List documents failed: %s", e)
            return [] if output_format == "dict" else ""

    def info(self, db: Optional[str] = None, path: Optional[str] = None) -> dict:
        """
        Get information about the search index.

        Args:
            db: Optional database name
            path: Optional storage path

        Returns:
            Dictionary with index information
        """
        try:
            data = {
                'db': db,
                'path': path
            }
            result = self._make_request('POST', '/info', data)
            return result.get('data', {})
        except Exception as e:
            logger.error("Get info failed: %s", e)
            return {}

    def generate_config_template(self, output_path: Optional[str] = None) -> Union[Dict[str, Any], bool]:
        """
        Generate a configuration template.

        Args:
            output_path: Optional path to save config file

        Returns:
            Configuration dictionary if no output_path, True if file saved successfully
        """
        try:
            data = {
                'output': output_path
            }
            result = self._make_request('POST', '/config', data)

            if output_path:
                return result.get('success', False)
            else:
                return result.get('data', {})
        except Exception as e:
            logger.error("Generate config failed: %s", e)
            if output_path:
                return False
            else:
                return {}

    def list_databases(self) -> List[str]:
        """
        Get list of available databases.

        Returns:
            List of database names
        """
        try:
            result = self._make_request('GET', '/list-databases')
            databases = result.get('data', {}).get('databases', {})
            return list(databases.keys())
        except Exception as e:
            logger.error("List databases failed: %s", e)
            return []


class SEMSimpleWebClient:
    """
    Simple web client that mirrors the SEMSimple interface.

    This provides a drop-in replacement for SEMSimple that works over HTTP.

    Example:
        >>> sem = SEMSimpleWebClient("http://localhost:8000")
        >>> sem.add_text("Machine learning content")
        True
        >>> results = sem.search("AI technology")
        >>> print(results[0]['text'])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        index_name: str = "sem_simple_database",
        storage_path: str = "./sem_indexes",
        api_key: Optional[str] = None
    ):
        """
        Initialize simple web client.

        Args:
            base_url: Base URL of the SEM API server
            index_name: Database name to use
            storage_path: Storage path to use
            api_key: Optional API key for authentication
        """
        self.client = SEMWebClient(base_url=base_url, api_key=api_key)
        self.index_name = index_name
        self.storage_path = storage_path

        logger.info("SEMSimpleWebClient initialized: %s (db: %s)", base_url, index_name)

    def add_text(self, text: str, doc_id: Optional[str] = None) -> bool:
        """Add a single text document."""
        return self.client.add_text(text, db=self.index_name, path=self.storage_path)

    def add_files(self, file_paths: List[str]) -> bool:
        """Add multiple text files."""
        return self.client.add_files(file_paths, db=self.index_name, path=self.storage_path)

    def search(self, query: str, top_k: int = 5, output_format: str = "dict", delimiter: str = ";") -> Union[List[dict], str]:
        """Search for documents."""
        return self.client.search(
            query=query,
            top_k=top_k,
            db=self.index_name,
            path=self.storage_path,
            output_format=output_format,
            delimiter=delimiter
        )

    def list_documents(
        self,
        limit: Optional[int] = None,
        show_content: bool = True,
        output_format: str = "dict",
        delimiter: str = ";"
    ) -> Union[List[dict], str]:
        """List documents."""
        return self.client.list_documents(
            limit=limit,
            show_content=show_content,
            db=self.index_name,
            path=self.storage_path,
            output_format=output_format,
            delimiter=delimiter
        )

    def info(self) -> dict:
        """Get database information."""
        return self.client.info(db=self.index_name, path=self.storage_path)

    def generate_config_template(self, output_path: Optional[str] = None) -> Union[Dict[str, Any], bool]:
        """Generate configuration template."""
        return self.client.generate_config_template(output_path)

    def health_check(self) -> bool:
        """Check if the API server is healthy."""
        return self.client.health_check()

    def __repr__(self) -> str:
        """String representation."""
        return f"SEMSimpleWebClient(url='{self.client.base_url}', db='{self.index_name}')"


# Convenience functions
def simple_web(base_url: str = "http://localhost:8000", **kwargs) -> SEMSimpleWebClient:
    """
    Create a simple web client with default settings.

    Args:
        base_url: API server URL
        **kwargs: Additional arguments for SEMSimpleWebClient

    Returns:
        Configured SEMSimpleWebClient instance
    """
    return SEMSimpleWebClient(base_url=base_url, **kwargs)


def simple_web_aws(base_url: str, bucket_name: str, **kwargs) -> SEMWebClient:
    """
    Create a web client configured for AWS operations.

    Args:
        base_url: API server URL
        bucket_name: S3 bucket name
        **kwargs: Additional arguments

    Returns:
        Configured SEMWebClient instance
    """
    client = SEMWebClient(base_url=base_url, **kwargs)
    # Store AWS-specific settings
    client._aws_bucket = bucket_name
    return client
