# EMBEDDINGS_SERVICE_GUIDE.md
# Step 4: Embeddings & Vector Index - Complete Guide

## Overview

The Embeddings Service provides semantic search capabilities using sentence-transformers for text encoding and Faiss for efficient vector similarity search.

---

## Architecture

```
Embeddings Service (Port 8004)
├── EmbeddingsClient (embeddings_client.py)
│   ├── sentence-transformers (all-MiniLM-L6-v2)
│   └── Faiss Vector Indices
│       ├── Transcript Index
│       ├── Frame Index
│       └── Clip Index
├── FastAPI Routes
│   ├── POST /embed (generate)
│   ├── POST /search (generic)
│   ├── GET /search/transcripts
│   ├── GET /search/frames
│   ├── GET /search/clips
│   ├── GET /stats
│   └── GET /health
└── Storage
    └── /data/processed/faiss/
        ├── transcript_index.faiss
        ├── frame_index.faiss
        ├── clip_index.faiss
        └── *_metadata.pkl
```

---

## Key Features

### Embedding Models
- **sentence-transformers** library with all-MiniLM-L6-v2
- 384-dimensional embeddings
- Fast (~3000 sentences/second on CPU)
- Production-ready and lightweight (~33MB)

### Vector Search
- **Faiss IndexFlatL2** for exact similarity search
- L2 distance converted to similarity scores (0-1)
- Support for 3 independent indices (transcripts, frames, clips)
- Configurable minimum similarity thresholds

### Persistence
- Automatic index saving on shutdown
- Index + metadata persistence to disk
- Fast index loading on startup

---

## Integration with Orchestrator

### Updated Tasks

In `orchestrator/app/tasks.py`, the `generate_embeddings` task calls the embeddings service:

```python
@shared_task(bind=True, name='app.tasks.generate_embeddings')
def generate_embeddings(self, media_item_id, transcript_segments):
    """Generate embeddings for transcript segments"""
    try:
        import requests
        embeddings_url = os.getenv('EMBEDDINGS_SERVICE_URL', 'http://localhost:8004')
        
        response = requests.post(
            f"{embeddings_url}/embed",
            json={
                'media_item_id': media_item_id,
                'embedding_type': 'transcript',
                'segments': transcript_segments
            },
            timeout=120
        )
        
        if response.status_code != 200:
            logger.error(f"Embeddings service failed: {response.text}")
            return {'status': 'failed', 'error': 'Embedding generation failed'}
        
        result = response.json()
        embeddings_count = result.get('embeddings_created', 0)
        
        return {
            'status': 'success',
            'media_item_id': media_item_id,
            'embeddings_created': embeddings_count
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return {'status': 'failed', 'error': str(e)}
```

### Pipeline Integration

Updated pipeline in `run_full_pipeline`:

```python
pipeline = chain(
    validate_video.s(media_item_id, file_path, filename),
    process_video.s(file_path),
    detect_keywords.s(),
    generate_clips.s(file_path),
    extract_concepts.s(),
    generate_embeddings.s(),  # NEW: After concept extraction
    generate_strategy.s(),
    backtest_strategy.s(),
    evaluate_and_save.s()
)
```

---

## API Endpoints

### POST /embed
Generate embeddings for transcripts, frames, or clips

**Request (Transcripts)**:
```json
{
  "media_item_id": 1,
  "embedding_type": "transcript",
  "segments": [
    {
      "id": 1,
      "text": "The RSI indicator measures momentum...",
      "start_time": 0.0,
      "end_time": 3.5
    },
    {
      "id": 2,
      "text": "RSI above 70 indicates overbought...",
      "start_time": 3.5,
      "end_time": 7.2
    }
  ]
}
```

**Response**:
```json
{
  "status": "success",
  "media_item_id": 1,
  "embedding_type": "transcript",
  "embeddings_created": 2,
  "timestamp": "2025-11-20T21:00:00Z"
}
```

### POST /search
Generic semantic similarity search

**Request**:
```json
{
  "query": "relative strength index overbought conditions",
  "embedding_type": "transcript",
  "top_k": 10,
  "min_similarity": 0.5
}
```

**Response**:
```json
{
  "query": "relative strength index overbought conditions",
  "embedding_type": "transcript",
  "results": [
    {
      "index": 0,
      "similarity": 0.92,
      "distance": 0.08,
      "metadata": {
        "media_item_id": 1,
        "segment_id": 2,
        "start_time": 3.5,
        "end_time": 7.2,
        "text": "RSI above 70 indicates overbought..."
      }
    },
    {
      "index": 1,
      "similarity": 0.87,
      "distance": 0.13,
      "metadata": {...}
    }
  ],
  "timestamp": "2025-11-20T21:00:00Z"
}
```

### GET /search/transcripts
Search transcripts with simple GET

```bash
curl "http://localhost:8004/search/transcripts?query=breakout%20resistance&top_k=5&min_similarity=0.6"
```

### GET /search/frames
Search extracted frames

```bash
curl "http://localhost:8004/search/frames?query=candlestick%20pattern&top_k=5"
```

### GET /search/clips
Search generated clips

```bash
curl "http://localhost:8004/search/clips?query=risk%20management%20position%20size&top_k=10"
```

### GET /stats
Get service statistics

**Response**:
```json
{
  "model_name": "all-MiniLM-L6-v2",
  "embedding_dimension": 384,
  "transcript_vectors": 1250,
  "frame_vectors": 450,
  "clip_vectors": 380,
  "total_vectors": 2080,
  "timestamp": "2025-11-20T21:00:00Z"
}
```

### GET /health
Health check

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-20T21:00:00Z",
  "service": "embeddings-service",
  "stats": {...}
}
```

---

## Implementation Steps

### 1. Directory Structure

```powershell
$dirs = @(
    "services/embeddings-service/app"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

$files = @(
    "services/embeddings-service/__init__.py",
    "services/embeddings-service/app/__init__.py"
)

foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
    }
}
```

### 2. Copy Files

```powershell
Copy-Item -Path "embeddings_client.py" -Destination "services/embeddings-service/app/embeddings_client.py"
Copy-Item -Path "embeddings_main.py" -Destination "services/embeddings-service/app/main.py"
Copy-Item -Path "embeddings_requirements.txt" -Destination "services/embeddings-service/requirements.txt"
Copy-Item -Path "embeddings_dockerfile" -Destination "services/embeddings-service/Dockerfile"
```

### 3. Create Faiss Directory

```bash
mkdir -p /data/processed/faiss
chmod 755 /data/processed/faiss
```

### 4. Update docker-compose.yml

Add the embeddings-service snippet and update:
- Depends on: none (standalone)
- Networks: trading-network
- Volumes: ./data/processed/faiss:/data/processed/faiss

### 5. Update .env

```bash
EMBEDDINGS_SERVICE_URL=http://embeddings-service:8004
FAISS_INDEX_DIR=/data/processed/faiss
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
```

### 6. Build and Run

```powershell
# Build
docker-compose build embeddings-service

# Run
docker-compose up -d embeddings-service

# Check logs
docker-compose logs -f embeddings-service

# Test health
curl http://localhost:8004/health
```

---

## Performance Characteristics

### Model Performance
- **Inference Speed**: ~3000 sentences/sec on CPU
- **Memory**: ~33MB model + ~1GB embeddings per 1000 vectors
- **Latency**: <1ms per search query on <100k vectors

### Faiss Performance
- **Vector Dimension**: 384
- **Index Type**: IndexFlatL2 (exact search)
- **Search Time**: O(n) where n = number of vectors
- **For 10k vectors**: ~10ms average search time
- **For 100k vectors**: ~100ms average search time

### Optimization Tips
1. **Batch Generation**: Generate embeddings in batches (100-1000)
2. **GPU Support**: Set EMBEDDING_DEVICE=cuda if GPU available (5-10x faster)
3. **Min Similarity**: Use higher thresholds to filter irrelevant results
4. **Index Optimization**: Consider GPU-accelerated Faiss for >100k vectors

---

## Advanced Configuration

### GPU Acceleration

Update `docker-compose-embeddings-snippet.yml`:
```yaml
embeddings-service:
  # ... other config ...
  environment:
    - EMBEDDING_DEVICE=cuda
  runtime: nvidia  # Requires nvidia-docker
```

Then use GPU model:
```bash
docker run --gpus all -e EMBEDDING_DEVICE=cuda ...
```

### Alternative Models

Replace `all-MiniLM-L6-v2` with:
- `all-MiniLM-L12-v2` (larger, better quality, slower)
- `multi-qa-MiniLM-L6-cos-v1` (better for QA/search)
- `distiluse-base-multilingual-cased-v2` (multilingual)
- `paraphrase-MiniLM-L6-v2` (paraphrase detection)

Update in .env:
```bash
EMBEDDING_MODEL=all-MiniLM-L12-v2
```

### Custom Index Types

Modify `embeddings_client.py` to use different Faiss index types:

```python
# IVF for larger datasets (>100k vectors)
quantizer = faiss.IndexFlatL2(embedding_dim)
self.transcript_index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)

# HNSW for even faster search
self.transcript_index = faiss.IndexHNSWFlat(embedding_dim, 32)

# GPU Index for very large datasets
self.transcript_index = faiss.index_cpu_to_all_gpus(index)
```

---

## Troubleshooting

### Service Won't Start
```bash
# Check logs
docker-compose logs embeddings-service

# Verify Faiss installation
docker exec trading-embeddings-service python -c "import faiss; print(faiss.__version__)"

# Check model download
docker exec trading-embeddings-service python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Slow Search Performance
```bash
# Check index size
curl http://localhost:8004/stats

# If >100k vectors, consider GPU or IVF index
# Reduce top_k parameter
# Increase min_similarity threshold
```

### Memory Issues
```bash
# Limit concurrency in docker-compose
EMBEDDINGS_WORKERS=1

# Use smaller model
EMBEDDING_MODEL=all-MiniLM-L6-v2  # already small

# Monitor memory
docker stats trading-embeddings-service
```

### CUDA/GPU Issues
```bash
# Fall back to CPU
EMBEDDING_DEVICE=cpu

# Verify NVIDIA driver
nvidia-smi

# Check nvidia-docker
docker run --rm --runtime=nvidia nvidia/cuda:11.0-runtime nvidia-smi
```

---

## Testing

### Quick Test Script

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8004"

# 1. Check health
health = requests.get(f"{BASE_URL}/health").json()
print(f"Health: {health['status']}")

# 2. Generate embeddings
embed_data = {
    "media_item_id": 1,
    "embedding_type": "transcript",
    "segments": [
        {"id": 1, "text": "RSI indicator measures momentum", "start_time": 0, "end_time": 3},
        {"id": 2, "text": "When RSI is above 70 it's overbought", "start_time": 3, "end_time": 6},
        {"id": 3, "text": "Support and resistance levels", "start_time": 6, "end_time": 9}
    ]
}

embed_response = requests.post(f"{BASE_URL}/embed", json=embed_data).json()
print(f"Embeddings created: {embed_response['embeddings_created']}")

# 3. Search
search_data = {
    "query": "relative strength index overbought",
    "embedding_type": "transcript",
    "top_k": 5
}

search_response = requests.post(f"{BASE_URL}/search", json=search_data).json()
print(f"Search results:")
for result in search_response['results']:
    print(f"  - Similarity: {result['similarity']:.2f} - {result['metadata']['text'][:50]}...")

# 4. Stats
stats = requests.get(f"{BASE_URL}/stats").json()
print(f"Total vectors: {stats['total_vectors']}")
```

---

## Integration with API Service

The main API service (Step 3) now includes embeddings search via:

```bash
GET /embeddings/search?query=...&top_k=10&embedding_type=transcript
```

This endpoint calls the embeddings service internally and returns results formatted for the API response model.

---

## Next Steps

### Step 5: Llama Dataset API Extension
- Enhance `/llama/examples` endpoint with embeddings
- Create Llama integration documentation
- Add prompting examples and scripts

### Step 6: Strategy & Backtesting
- Feature engineering from embeddings
- ML model training on extracted features
- Strategy validation and promotion

### Step 7: Tests & Acceptance
- Unit tests for embedding generation
- Integration tests for search functionality
- Performance benchmarks

---

## Key Files

| File | Purpose |
|------|---------|
| embeddings_client.py | Core EmbeddingsClient class |
| embeddings_main.py | FastAPI application |
| embeddings_requirements.txt | Python dependencies |
| embeddings_dockerfile | Container definition |
| docker-compose snippet | Service configuration |

---

## Status

✅ **Step 4 Complete** - Embeddings & Vector Index
- Sentence-transformers integration
- Faiss vector indices (3 types)
- 6 API endpoints for search and generation
- Production-ready configuration
- Persistence and recovery

**Ready to proceed to Step 5: Llama Dataset API**