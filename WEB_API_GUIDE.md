# SEM Web API Guide

🌐 **Complete REST API for Simple Embeddings Module**

The SEM Web API provides HTTP access to all SEM functionality, automatically generating REST endpoints from the existing Click CLI metadata with "/" delimited paths.

## 🚀 Quick Start

### Start the Server

```bash
# Start server with default settings
sem-cli serve

# Custom host and port
sem-cli serve --host 0.0.0.0 --port 8080

# Development mode with auto-reload
sem-cli serve --reload

# Production with multiple workers
sem-cli serve --workers 4
```

### Use the Python Client

```python
from simple_embeddings_module import SEMSimpleWebClient

# Create client (drop-in replacement for SEMSimple)
sem = SEMSimpleWebClient("http://localhost:8000")

# Use exactly like SEMSimple
sem.add_text("Machine learning is transforming software development.")
results = sem.search("AI technology")
print(results[0]['text'])
```

### Direct HTTP Calls

```python
import requests

# Add document
response = requests.post("http://localhost:8000/simple/local/index", json={
    "text": "Semantic search enables meaning-based document retrieval."
})

# Search documents
response = requests.post("http://localhost:8000/simple/local/search", json={
    "query": "semantic search",
    "top_k": 5
})
results = response.json()['data']['results']
```

## 🏗️ Architecture

### CLI-to-REST Mapping

The API automatically maps CLI commands to REST endpoints using "/" delimited paths:

| CLI Command | REST Endpoint | Method |
|-------------|---------------|---------|
| `sem-cli simple local search "query"` | `POST /simple/local/search` | POST |
| `sem-cli simple local index --files file.txt` | `POST /simple/local/index` | POST |
| `sem-cli simple aws search "query"` | `POST /simple/aws/search` | POST |
| `sem-cli list-databases` | `GET /list-databases` | GET |
| `sem-cli info --db mydb` | `POST /info` | POST |

### Request/Response Format

All endpoints use JSON format:

```json
// Request
{
  "query": "machine learning",
  "top_k": 5,
  "db": "my_database"
}

// Response
{
  "success": true,
  "data": {
    "results": [...],
    "count": 3
  },
  "message": "Found 3 results"
}
```

## 📋 API Reference

### Core Endpoints

#### POST /add
Add documents to the database.

```json
{
  "text": "Document content",          // Single text
  "files": ["file1.txt", "file2.txt"], // Or multiple files
  "db": "database_name",             // Optional
  "path": "./custom_path"            // Optional
}
```

#### POST /search
Search for similar documents.

```json
{
  "query": "search query",
  "top_k": 5,                       // Number of results
  "threshold": 0.7,                 // Optional similarity threshold
  "db": "database_name",            // Optional
  "cli_format": false,              // Output format
  "delimiter": ";"                  // CLI format delimiter
}
```

#### POST /list
List documents in the database.

```json
{
  "limit": 10,                      // Optional limit
  "no_content": false,              // Exclude content
  "db": "database_name",            // Optional
  "cli_format": false               // Output format
}
```

#### POST /info
Get database information.

```json
{
  "db": "database_name",            // Optional
  "path": "./custom_path"           // Optional
}
```

### Simple Interface Endpoints

#### POST /simple/local/index
Index documents locally (zero-config).

```json
{
  "text": "Document content",
  "files": ["file1.txt"],
  "db": "sem_simple_database",      // Default database
  "model": "all-MiniLM-L6-v2"       // Optional model
}
```

#### POST /simple/local/search
Search local documents.

```json
{
  "query": "search query",
  "top_k": 5,
  "db": "sem_simple_database"
}
```

#### POST /simple/aws/index
Index documents to AWS S3.

```json
{
  "text": "Document content",
  "bucket": "my-s3-bucket",         // Required
  "region": "us-east-1",            // Optional
  "model": "all-MiniLM-L6-v2"       // Optional
}
```

#### POST /simple/aws/search
Search AWS documents.

```json
{
  "query": "search query",
  "bucket": "my-s3-bucket",
  "region": "us-east-1",
  "top_k": 5
}
```

### Utility Endpoints

#### GET /health
Health check endpoint.

```json
// Response
{
  "success": true,
  "data": {"status": "healthy"},
  "message": "API is healthy"
}
```

#### GET /list-databases
List all available databases.

```json
// Response
{
  "success": true,
  "data": {
    "databases": {
      "my_db": [{"path": "...", "type": "local"}]
    },
    "count": 1
  }
}
```

#### GET /docs
Interactive API documentation (Swagger UI).

#### GET /redoc
Alternative API documentation (ReDoc).

## 🐍 Python Client

### SEMSimpleWebClient

Drop-in replacement for SEMSimple that works over HTTP:

```python
from simple_embeddings_module import SEMSimpleWebClient

# Same interface as SEMSimple
sem = SEMSimpleWebClient(
    base_url="http://localhost:8000",
    index_name="my_database",
    api_key="optional_api_key"
)

# All methods work identically
sem.add_text("Content")
sem.add_files(["file1.txt", "file2.txt"])
results = sem.search("query", top_k=10)
docs = sem.list_documents(limit=5)
info = sem.info()

# New output formats (Phase 1 parity)
cli_results = sem.search("query", output_format="cli", delimiter="|")
json_results = sem.search("query", output_format="json")
```

### SEMWebClient

Lower-level client for custom usage:

```python
from simple_embeddings_module import SEMWebClient

client = SEMWebClient("http://localhost:8000")

# Health check
healthy = client.health_check()

# Custom database operations
client.add_text("Content", db="custom_db", path="./custom_path")
results = client.search("query", db="custom_db", top_k=5)
```

### Convenience Functions

```python
from simple_embeddings_module import simple_web, simple_web_aws

# Local web client
sem = simple_web("http://localhost:8000")

# AWS web client
aws_client = simple_web_aws("http://localhost:8000", bucket_name="my-bucket")
```

## 🔒 Security

### Authentication (Optional)

The API supports Bearer token authentication:

```python
# Client with API key
client = SEMWebClient(
    base_url="http://localhost:8000",
    api_key="your_api_key_here"
)

# HTTP headers
headers = {"Authorization": "Bearer your_api_key_here"}
response = requests.post(url, json=data, headers=headers)
```

### CORS Support

CORS is enabled for web frontend integration:

```javascript
// JavaScript frontend
fetch('http://localhost:8000/simple/local/search', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({query: 'search term', top_k: 5})
})
.then(response => response.json())
.then(data => console.log(data.data.results));
```

## 🚀 Deployment

### Development

```bash
# Auto-reload for development
sem-cli serve --reload --host 127.0.0.1 --port 8000
```

### Production

```bash
# Multiple workers for production
sem-cli serve --host 0.0.0.0 --port 8000 --workers 4

# Or use gunicorn directly
pip install gunicorn
gunicorn simple_embeddings_module.sem_web_api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["sem-cli", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Environment Variables

```bash
# Optional configuration
export SEM_API_KEY="your_api_key"
export SEM_DEFAULT_MODEL="all-MiniLM-L6-v2"
export SEM_DEFAULT_PATH="./indexes"
```

## 📊 Monitoring

### Health Checks

```bash
# Simple health check
curl http://localhost:8000/health

# Detailed server info
curl http://localhost:8000/
```

### Logging

The server provides comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Server logs include:
# - Request/response details
# - Performance metrics
# - Error tracking
# - Database operations
```

## 🔧 Configuration

### Server Configuration

```python
# Custom server configuration
from simple_embeddings_module.sem_web_api import app
import uvicorn

# Configure server
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    workers=4,
    log_level="info",
    access_log=True
)
```

### Client Configuration

```python
# Client with custom settings
client = SEMWebClient(
    base_url="http://api.example.com",
    timeout=60,          # Request timeout
    retries=5,           # Retry attempts
    api_key="key"        # Authentication
)
```

## 🎯 Use Cases

### Distributed Semantic Search

```python
# Multiple clients connecting to central server
clients = [
    SEMSimpleWebClient("http://server1:8000"),
    SEMSimpleWebClient("http://server2:8000"),
    SEMSimpleWebClient("http://server3:8000")
]

# Distribute search across servers
for client in clients:
    results = client.search("query")
    process_results(results)
```

### Web Frontend Integration

```javascript
// React/Vue/Angular frontend
const searchDocuments = async (query) => {
  const response = await fetch('/api/simple/local/search', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query, top_k: 10})
  });
  return response.json();
};
```

### Microservices Architecture

```python
# Service A: Document ingestion
ingestion_client = SEMWebClient("http://ingestion-service:8000")
ingestion_client.add_files(document_files)

# Service B: Search interface
search_client = SEMWebClient("http://search-service:8000")
results = search_client.search(user_query)
```

## 📈 Performance

### Benchmarks

- **Startup Time**: ~2-3 seconds (model loading)
- **Request Latency**: 50-200ms (depending on operation)
- **Throughput**: 100+ requests/second (single worker)
- **Memory Usage**: ~2GB (model + index)

### Optimization

```python
# Connection pooling
client = SEMWebClient(
    base_url="http://localhost:8000",
    timeout=30,
    retries=3
)

# Batch operations
client.add_files(large_file_list)  # More efficient than individual adds
```

## 🔍 Troubleshooting

### Common Issues

**Server won't start:**
```bash
# Check dependencies
pip install fastapi uvicorn pydantic

# Check port availability
lsof -i :8000
```

**Client connection errors:**
```python
# Test server health
client = SEMWebClient("http://localhost:8000")
if not client.health_check():
    print("Server not responding")
```

**Performance issues:**
```bash
# Use multiple workers
sem-cli serve --workers 4

# Monitor resource usage
htop  # or Activity Monitor on macOS
```

## 🎉 Summary

The SEM Web API provides:

✅ **Complete CLI Functionality**: Every CLI command available as REST endpoint  
✅ **Automatic Mapping**: CLI structure directly maps to "/" delimited API paths  
✅ **Python Client**: Drop-in replacement for SEMSimple that works over HTTP  
✅ **Production Ready**: FastAPI server with full documentation and monitoring  
✅ **Easy Deployment**: Single command server startup with scaling options  
✅ **Web Integration**: CORS support and JSON API for frontend development  

The web API enables distributed semantic search deployments while maintaining the same simple interface users already know and love.

---

**Get Started**: `sem-cli serve` → Visit http://localhost:8000/docs  
**Documentation**: Interactive API docs with live testing  
**Client**: `SEMSimpleWebClient("http://localhost:8000")`
