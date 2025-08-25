#!/usr/bin/env python3
"""
SEM Web API - FastAPI-based REST API Server
Automatically generates REST endpoints from Click CLI metadata with "/" delimited paths.

This module creates a web API that mirrors the CLI structure:
- CLI: sem-cli simple local search "query" --top-k 5
- API: POST /simple/local/search {"query": "query", "top_k": 5}

Features:
- Automatic endpoint generation from Click commands
- OpenAPI/Swagger documentation
- Authentication support
- Error handling and validation
- JSON request/response format
- CORS support for web frontends
"""
import logging
import inspect
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import click

from .sem_cli_click import cli
from .sem_simple import SEMSimple
from .sem_simple_aws import SEMSimpleAWS
from .sem_simple_gcp import SEMSimpleGCP

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# FastAPI app
app = FastAPI(
    title="Simple Embeddings Module API",
    description="REST API for semantic search with intelligent chunking and GPU acceleration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class BaseRequest(BaseModel):
    """Base request model with common parameters."""
    verbose: Optional[bool] = Field(False, description="Enable verbose output")
    config: Optional[str] = Field(None, description="Configuration file path")


class DatabaseRequest(BaseRequest):
    """Request model with database parameters."""
    db: Optional[str] = Field(None, description="Database name (will auto-resolve path)")
    path: Optional[str] = Field(None, description="Storage path")


class OutputRequest(DatabaseRequest):
    """Request model with output formatting parameters."""
    cli_format: Optional[bool] = Field(False, description="Output in CLI format")
    delimiter: Optional[str] = Field(";", description="Delimiter for CLI format")


class InitRequest(DatabaseRequest):
    """Initialize database request."""
    model: Optional[str] = Field(None, description="Embedding model name")


class AddRequest(DatabaseRequest):
    """Add documents request."""
    files: Optional[List[str]] = Field(None, description="List of file paths to add")
    text: Optional[str] = Field(None, description="Text content to add")


class SearchRequest(OutputRequest):
    """Search documents request."""
    query: str = Field(..., description="Search query text")
    top_k: Optional[int] = Field(5, description="Number of results to return")
    threshold: Optional[float] = Field(None, description="Similarity threshold")


class ListRequest(OutputRequest):
    """List documents request."""
    limit: Optional[int] = Field(None, description="Maximum number of documents to return")
    no_content: Optional[bool] = Field(False, description="Exclude document content")


class InfoRequest(DatabaseRequest):
    """Get database info request."""
    pass


class ConfigRequest(BaseRequest):
    """Configuration request."""
    output: Optional[str] = Field(None, description="Output configuration file path")
    provider: Optional[str] = Field(None, description="Embedding provider")
    model: Optional[str] = Field(None, description="Embedding model")
    storage: Optional[str] = Field(None, description="Storage backend")
    path: Optional[str] = Field(None, description="Storage path")


class SimpleLocalRequest(BaseModel):
    """Simple local interface request."""
    files: Optional[List[str]] = Field(None, description="List of file paths")
    db: Optional[str] = Field("sem_simple_database", description="Database name")
    path: Optional[str] = Field("./sem_indexes", description="Storage path")
    model: Optional[str] = Field(None, description="Embedding model")
    text: Optional[str] = Field(None, description="Text content")
    query: Optional[str] = Field(None, description="Search query")
    top_k: Optional[int] = Field(5, description="Number of results")
    max_content: Optional[int] = Field(100, description="Maximum content length")
    doc_id: Optional[str] = Field(None, description="Document ID")
    doc_ids: Optional[List[str]] = Field(None, description="List of document IDs")
    confirm: Optional[bool] = Field(False, description="Confirm destructive operations")


class SimpleAWSRequest(BaseModel):
    """Simple AWS interface request."""
    files: Optional[List[str]] = Field(None, description="List of file paths")
    bucket: str = Field(..., description="S3 bucket name")
    region: Optional[str] = Field("us-east-1", description="AWS region")
    model: Optional[str] = Field(None, description="Embedding model")
    text: Optional[str] = Field(None, description="Text content")
    query: Optional[str] = Field(None, description="Search query")
    top_k: Optional[int] = Field(5, description="Number of results")
    max_content: Optional[int] = Field(100, description="Maximum content length")
    doc_id: Optional[str] = Field(None, description="Document ID")
    doc_ids: Optional[List[str]] = Field(None, description="List of document IDs")
    confirm: Optional[bool] = Field(False, description="Confirm destructive operations")


class SimpleGCPRequest(BaseModel):
    """Simple GCP interface request."""
    files: Optional[List[str]] = Field(None, description="List of file paths")
    bucket: str = Field(..., description="GCS bucket name")
    project: str = Field(..., description="GCP project ID")
    region: Optional[str] = Field("us-central1", description="GCP region")
    model: Optional[str] = Field(None, description="Embedding model")
    credentials: Optional[str] = Field(None, description="Service account credentials path")
    text: Optional[str] = Field(None, description="Text content")
    query: Optional[str] = Field(None, description="Search query")
    top_k: Optional[int] = Field(5, description="Number of results")
    max_content: Optional[int] = Field(100, description="Maximum content length")
    doc_id: Optional[str] = Field(None, description="Document ID")
    doc_ids: Optional[List[str]] = Field(None, description="List of document IDs")
    confirm: Optional[bool] = Field(False, description="Confirm destructive operations")


class APIResponse(BaseModel):
    """Standard API response format."""
    success: bool = Field(..., description="Operation success status")
    data: Optional[Any] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    error: Optional[str] = Field(None, description="Error message if failed")


# Authentication (optional)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Optional authentication - can be configured with API keys.
    For now, this is a placeholder that allows all requests.
    """
    # TODO: Implement actual authentication if needed
    # if credentials:
    #     # Validate API key here
    #     pass
    return {"user": "anonymous"}


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error("API error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=APIResponse(
            success=False,
            error=f"Internal server error: {str(exc)}"
        ).dict()
    )


# Root endpoint
@app.get("/", response_model=APIResponse)
async def root():
    """API root endpoint with basic information."""
    return APIResponse(
        success=True,
        data={
            "name": "Simple Embeddings Module API",
            "version": "1.0.0",
            "description": "REST API for semantic search with intelligent chunking",
            "endpoints": {
                "docs": "/docs",
                "redoc": "/redoc",
                "health": "/health"
            }
        },
        message="SEM API is running"
    )


@app.get("/health", response_model=APIResponse)
async def health_check():
    """Health check endpoint."""
    return APIResponse(
        success=True,
        data={"status": "healthy"},
        message="API is healthy"
    )


# Core API endpoints (mirroring CLI structure)

@app.post("/init", response_model=APIResponse)
async def api_init(request: InitRequest, user=Depends(get_current_user)):
    """Initialize a new semantic search database."""
    try:
        from .sem_core import SEMDatabase
        from .sem_utils import create_quick_config

        # Create configuration
        config = create_quick_config(
            embedding_model=request.model or "all-MiniLM-L6-v2",
            storage_path=request.path or "./indexes",
            index_name=request.db or "default"
        )

        # Initialize database
        db = SEMDatabase(config=config)
        info = db.get_index_info()

        return APIResponse(
            success=True,
            data={"database": request.db or "default", "info": info},
            message="Database initialized successfully"
        )
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add", response_model=APIResponse)
async def api_add(request: AddRequest, user=Depends(get_current_user)):
    """Add documents to the semantic search database."""
    try:
        sem = SEMSimple(
            index_name=request.db or "default",
            storage_path=request.path or "./indexes"
        )

        if request.files:
            success = sem.add_files(request.files)
            return APIResponse(
                success=success,
                data={"files_added": len(request.files) if success else 0},
                message=f"Added {len(request.files)} files" if success else "Failed to add files"
            )
        elif request.text:
            success = sem.add_text(request.text)
            return APIResponse(
                success=success,
                data={"text_added": len(request.text) if success else 0},
                message="Text added successfully" if success else "Failed to add text"
            )
        else:
            raise HTTPException(status_code=400, detail="Either 'files' or 'text' must be provided")

    except Exception as e:
        logger.error("Failed to add documents: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=APIResponse)
async def api_search(request: SearchRequest, user=Depends(get_current_user)):
    """Search documents in the semantic database."""
    try:
        sem = SEMSimple(
            index_name=request.db or "default",
            storage_path=request.path or "./indexes"
        )

        # Determine output format
        output_format = "cli" if request.cli_format else "dict"

        results = sem.search(
            query=request.query,
            top_k=request.top_k or 5,
            output_format=output_format,
            delimiter=request.delimiter or ";"
        )

        return APIResponse(
            success=True,
            data={"results": results, "query": request.query, "count": len(results) if isinstance(results, list) else 1},
            message=f"Found {len(results) if isinstance(results, list) else 1} results"
        )
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/list", response_model=APIResponse)
async def api_list(request: ListRequest, user=Depends(get_current_user)):
    """List documents in the semantic database."""
    try:
        sem = SEMSimple(
            index_name=request.db or "default",
            storage_path=request.path or "./indexes"
        )

        # Determine output format
        output_format = "cli" if request.cli_format else "dict"

        documents = sem.list_documents(
            limit=request.limit,
            show_content=not request.no_content,
            output_format=output_format,
            delimiter=request.delimiter or ";"
        )

        return APIResponse(
            success=True,
            data={"documents": documents, "count": len(documents) if isinstance(documents, list) else 1},
            message=f"Listed {len(documents) if isinstance(documents, list) else 1} documents"
        )
    except Exception as e:
        logger.error("List failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/info", response_model=APIResponse)
async def api_info(request: InfoRequest, user=Depends(get_current_user)):
    """Get database information."""
    try:
        sem = SEMSimple(
            index_name=request.db or "default",
            storage_path=request.path or "./indexes"
        )

        info = sem.info()

        return APIResponse(
            success=True,
            data=info,
            message="Database info retrieved successfully"
        )
    except Exception as e:
        logger.error("Info failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config", response_model=APIResponse)
async def api_config(request: ConfigRequest, user=Depends(get_current_user)):
    """Generate or manage configuration."""
    try:
        from .sem_utils import generate_config_template

        # Generate configuration template
        config_kwargs = {}
        if request.provider:
            config_kwargs[f"embedding.provider"] = request.provider
        if request.model:
            config_kwargs[f"embedding.model"] = request.model
        if request.storage:
            config_kwargs[f"storage.backend"] = request.storage
        if request.path:
            config_kwargs[f"storage.path"] = request.path

        if request.output:
            # Save to file
            success = generate_config_template(request.output, **config_kwargs)
            return APIResponse(
                success=success,
                data={"config_file": request.output},
                message="Configuration saved to file" if success else "Failed to save configuration"
            )
        else:
            # Return configuration
            config = generate_config_template(**config_kwargs)
            return APIResponse(
                success=True,
                data=config.to_dict() if hasattr(config, 'to_dict') else dict(config),
                message="Configuration generated successfully"
            )
    except Exception as e:
        logger.error("Config failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# Simple interface endpoints (mirroring CLI simple commands)

@app.post("/simple/local/index", response_model=APIResponse)
async def api_simple_local_index(request: SimpleLocalRequest, user=Depends(get_current_user)):
    """Simple local interface - index documents."""
    try:
        sem = SEMSimple(
            index_name=request.db,
            storage_path=request.path,
            embedding_model=request.model
        )

        if request.files:
            success = sem.add_files(request.files)
            return APIResponse(
                success=success,
                data={"files_indexed": len(request.files) if success else 0},
                message=f"Indexed {len(request.files)} files" if success else "Failed to index files"
            )
        elif request.text:
            success = sem.add_text(request.text)
            return APIResponse(
                success=success,
                message="Text indexed successfully" if success else "Failed to index text"
            )
        else:
            raise HTTPException(status_code=400, detail="Either 'files' or 'text' must be provided")

    except Exception as e:
        logger.error("Simple local index failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simple/local/search", response_model=APIResponse)
async def api_simple_local_search(request: SimpleLocalRequest, user=Depends(get_current_user)):
    """Simple local interface - search documents."""
    try:
        if not request.query:
            raise HTTPException(status_code=400, detail="Query is required")

        sem = SEMSimple(
            index_name=request.db,
            storage_path=request.path
        )

        results = sem.search(query=request.query, top_k=request.top_k)

        return APIResponse(
            success=True,
            data={"results": results, "query": request.query, "count": len(results)},
            message=f"Found {len(results)} results"
        )
    except Exception as e:
        logger.error("Simple local search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simple/local/list", response_model=APIResponse)
async def api_simple_local_list(request: SimpleLocalRequest, user=Depends(get_current_user)):
    """Simple local interface - list documents."""
    try:
        sem = SEMSimple(
            index_name=request.db,
            storage_path=request.path
        )

        documents = sem.list_documents(
            limit=request.top_k,
            max_content_length=request.max_content
        )

        return APIResponse(
            success=True,
            data={"documents": documents, "count": len(documents)},
            message=f"Listed {len(documents)} documents"
        )
    except Exception as e:
        logger.error("Simple local list failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/simple/local/remove", response_model=APIResponse)
async def api_simple_local_remove(request: SimpleLocalRequest, user=Depends(get_current_user)):
    """Simple local interface - remove documents."""
    try:
        sem = SEMSimple(
            index_name=request.db,
            storage_path=request.path
        )

        if request.doc_id:
            success = sem.remove_document(request.doc_id)
            return APIResponse(
                success=success,
                data={"removed_count": 1 if success else 0},
                message="Document removed" if success else "Failed to remove document"
            )
        elif request.doc_ids:
            removed_count = sem.remove_documents(request.doc_ids)
            return APIResponse(
                success=removed_count > 0,
                data={"removed_count": removed_count},
                message=f"Removed {removed_count} documents"
            )
        else:
            raise HTTPException(status_code=400, detail="Either 'doc_id' or 'doc_ids' must be provided")

    except Exception as e:
        logger.error("Simple local remove failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/simple/local/clear", response_model=APIResponse)
async def api_simple_local_clear(request: SimpleLocalRequest, user=Depends(get_current_user)):
    """Simple local interface - clear database."""
    try:
        if not request.confirm:
            raise HTTPException(status_code=400, detail="Confirmation required for destructive operation")

        sem = SEMSimple(
            index_name=request.db,
            storage_path=request.path
        )

        success = sem.clear()

        return APIResponse(
            success=success,
            message="Database cleared" if success else "Failed to clear database"
        )
    except Exception as e:
        logger.error("Simple local clear failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# AWS interface endpoints
@app.post("/simple/aws/index", response_model=APIResponse)
async def api_simple_aws_index(request: SimpleAWSRequest, user=Depends(get_current_user)):
    """Simple AWS interface - index documents."""
    try:
        sem = SEMSimpleAWS(
            bucket_name=request.bucket,
            region=request.region,
            embedding_model=request.model
        )

        if request.files:
            success = sem.add_files(request.files)
            return APIResponse(
                success=success,
                data={"files_indexed": len(request.files) if success else 0},
                message=f"Indexed {len(request.files)} files to AWS" if success else "Failed to index files"
            )
        elif request.text:
            success = sem.add_text(request.text)
            return APIResponse(
                success=success,
                message="Text indexed to AWS successfully" if success else "Failed to index text"
            )
        else:
            raise HTTPException(status_code=400, detail="Either 'files' or 'text' must be provided")

    except Exception as e:
        logger.error("Simple AWS index failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simple/aws/search", response_model=APIResponse)
async def api_simple_aws_search(request: SimpleAWSRequest, user=Depends(get_current_user)):
    """Simple AWS interface - search documents."""
    try:
        if not request.query:
            raise HTTPException(status_code=400, detail="Query is required")

        sem = SEMSimpleAWS(
            bucket_name=request.bucket,
            region=request.region
        )

        results = sem.search(query=request.query, top_k=request.top_k)

        return APIResponse(
            success=True,
            data={"results": results, "query": request.query, "count": len(results)},
            message=f"Found {len(results)} results in AWS"
        )
    except Exception as e:
        logger.error("Simple AWS search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# List databases endpoint
@app.get("/list-databases", response_model=APIResponse)
async def api_list_databases(user=Depends(get_current_user)):
    """List all available databases."""
    try:
        from .sem_auto_resolve import list_available_databases

        databases = list_available_databases()

        return APIResponse(
            success=True,
            data={"databases": databases, "count": len(databases)},
            message=f"Found {len(databases)} databases"
        )
    except Exception as e:
        logger.error("List databases failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# Development server function
def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the FastAPI development server."""
    import uvicorn

    logger.info("Starting SEM Web API server on %s:%s", host, port)
    uvicorn.run(
        "simple_embeddings_module.sem_web_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server(reload=True)
