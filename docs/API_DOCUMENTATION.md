# API_DOCUMENTATION.md
# Trading Media Extraction API Documentation

## Overview

The Trading Media Extraction API provides RESTful endpoints for accessing processed trading videos, extracted clips, transcripts, keywords, embeddings, and Llama training datasets.

**Base URL**: `http://localhost:8003`

---

## Table of Contents

1. [Health & Info Endpoints](#health--info-endpoints)
2. [Media Items](#media-items)
3. [Ingest & Pipeline](#ingest--pipeline)
4. [Transcripts](#transcripts)
5. [Keywords](#keywords)
6. [Clips](#clips)
7. [Embeddings & Search](#embeddings--search)
8. [Llama Dataset API](#llama-dataset-api)
9. [Error Handling](#error-handling)
10. [Examples](#examples)

---

## Health & Info Endpoints

### GET /health
Health check endpoint

**Response**: 
```json
{
  "status": "healthy",
  "timestamp": "2025-11-20T20:50:00.000Z",
  "service": "trading-media-extraction-api"
}
```

### GET /
API information and documentation

**Response**:
```json
{
  "name": "Trading Media Extraction API",
  "version": "1.0.0",
  "description": "RESTful API for accessing processed trading videos...",
  "docs_url": "/docs",
  "redoc_url": "/redoc",
  "endpoints": {...}
}
```

---

## Media Items

### GET /media_items
List all processed videos with pagination

**Query Parameters**:
- `skip` (int, default: 0): Number of items to skip
- `limit` (int, default: 20, max: 100): Number of items to return
- `status` (string, optional): Filter by status (pending, processing, completed, failed)

**Example**:
```bash
curl "http://localhost:8003/media_items?skip=0&limit=10&status=completed"
```

**Response**:
```json
{
  "total": 42,
  "skip": 0,
  "limit": 10,
  "items": [
    {
      "id": 1,
      "video_id": "video_001",
      "filename": "trading_tutorial.mp4",
      "source_url": null,
      "duration_seconds": 1200.5,
      "file_size_bytes": 524288000,
      "status": "completed",
      "created_at": "2025-11-20T15:30:00Z",
      "updated_at": "2025-11-20T15:45:00Z",
      "clips_count": 24,
      "keywords_count": 18,
      "transcript_segments_count": 156
    }
  ]
}
```

### GET /media_items/{media_id}
Get details for a specific media item

**Path Parameters**:
- `media_id` (int): Media item ID

**Example**:
```bash
curl http://localhost:8003/media_items/1
```

**Response**:
```json
{
  "id": 1,
  "video_id": "video_001",
  "filename": "trading_tutorial.mp4",
  "duration_seconds": 1200.5,
  "status": "completed",
  "created_at": "2025-11-20T15:30:00Z",
  "clips_count": 24,
  "keywords_count": 18,
  "transcript_segments": 156
}
```

---

## Ingest & Pipeline

### POST /ingest
Trigger video processing pipeline

**Request**:
```json
{
  "video_path": "/data/videos/trading_tutorial.mp4",
  "source_url": "https://youtube.com/watch?v=abc123",
  "filename": "trading_tutorial.mp4"
}
```

**Response** (202 Accepted):
```json
{
  "status": "accepted",
  "task_id": "celery-task-uuid-1234",
  "media_item_id": 1,
  "message": "Video queued for processing",
  "estimated_duration_minutes": 5
}
```

**Example with curl**:
```bash
curl -X POST http://localhost:8003/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/data/videos/trading_tutorial.mp4",
    "source_url": "https://youtube.com/watch?v=abc123",
    "filename": "trading_tutorial.mp4"
  }'
```

---

## Transcripts

### GET /transcript/{media_id}
Get full timestamped transcript for a media item

**Path Parameters**:
- `media_id` (int): Media item ID

**Query Parameters**:
- `format` (string, default: "json"): Response format (json or text)

**Example**:
```bash
curl "http://localhost:8003/transcript/1?format=json"
```

**Response** (JSON format):
```json
{
  "media_id": 1,
  "segments": [
    {
      "index": 0,
      "start_time": 0.0,
      "end_time": 3.5,
      "text": "Today we're going to discuss the RSI indicator..."
    },
    {
      "index": 1,
      "start_time": 3.5,
      "end_time": 7.2,
      "text": "RSI stands for Relative Strength Index..."
    }
  ]
}
```

**Response** (Text format):
```json
{
  "text": "Today we're going to discuss the RSI indicator... RSI stands for Relative Strength Index..."
}
```

---

## Keywords

### GET /keywords
Get detected keywords for a media item

**Query Parameters**:
- `media_id` (int, optional): Filter by media item ID
- `min_confidence` (float, default: 0.7): Minimum confidence threshold (0.0-1.0)
- `limit` (int, default: 50, max: 500): Maximum keywords to return

**Example**:
```bash
curl "http://localhost:8003/keywords?media_id=1&min_confidence=0.8&limit=20"
```

**Response**:
```json
{
  "keywords": [
    {
      "id": 1,
      "keyword": "RSI",
      "category": "technical_indicator",
      "start_time": 5.3,
      "end_time": 5.8,
      "confidence": 0.95,
      "context": "...the RSI indicator is a momentum oscillator..."
    },
    {
      "id": 2,
      "keyword": "overbought",
      "category": "price_action",
      "start_time": 12.4,
      "end_time": 12.8,
      "confidence": 0.92,
      "context": "...when RSI goes above 70, the asset is considered overbought..."
    }
  ]
}
```

---

## Clips

### GET /clips
Search for video clips by keyword

**Query Parameters**:
- `keyword` (string, optional): Filter by keyword name
- `media_id` (int, optional): Filter by media item ID
- `limit` (int, default: 10, max: 100): Number of results
- `skip` (int, default: 0): Number of results to skip

**Example**:
```bash
curl "http://localhost:8003/clips?keyword=RSI&limit=5&skip=0"
```

**Response**:
```json
{
  "total": 8,
  "skip": 0,
  "limit": 5,
  "clips": [
    {
      "id": 1,
      "keyword": "RSI",
      "start_time": 3.0,
      "end_time": 13.0,
      "duration_seconds": 10.0,
      "download_url": "/clip/1/download",
      "media_item_id": 1,
      "created_at": "2025-11-20T15:35:00Z"
    },
    {
      "id": 2,
      "keyword": "RSI",
      "start_time": 45.2,
      "end_time": 55.2,
      "duration_seconds": 10.0,
      "download_url": "/clip/2/download",
      "media_item_id": 1,
      "created_at": "2025-11-20T15:35:00Z"
    }
  ]
}
```

### GET /clip/{clip_id}/download
Download video clip by ID

**Path Parameters**:
- `clip_id` (int): Clip ID

**Response**: Binary MP4 video file

**Example**:
```bash
curl -O http://localhost:8003/clip/1/download
```

Or from browser: `http://localhost:8003/clip/1/download`

---

## Embeddings & Search

### GET /embeddings/search
Semantic similarity search using embeddings

**Query Parameters**:
- `query` (string, required): Search text or concept
- `top_k` (int, default: 10, max: 100): Number of top results
- `embedding_type` (string, default: "transcript"): Type to search (transcript, frame, clip)

**Example**:
```bash
curl "http://localhost:8003/embeddings/search?query=relative%20strength%20index%20overbought&top_k=5&embedding_type=transcript"
```

**Response**:
```json
{
  "query": "relative strength index overbought",
  "top_k": 5,
  "results": [
    {
      "result_id": 1,
      "reference_type": "transcript",
      "reference_id": 45,
      "similarity_score": 0.92,
      "metadata": {
        "media_id": 1,
        "segment_index": 12,
        "text": "The RSI indicates overbought conditions when above 70..."
      }
    },
    {
      "result_id": 2,
      "reference_type": "transcript",
      "reference_id": 67,
      "similarity_score": 0.88,
      "metadata": {
        "media_id": 2,
        "segment_index": 8,
        "text": "Overbought RSI readings suggest potential reversals..."
      }
    }
  ]
}
```

---

## Llama Dataset API

### GET /llama/examples
Get structured training examples for Llama fine-tuning or prompting

**Query Parameters**:
- `keyword` (string, optional): Filter by keyword (e.g., "RSI", "breakout")
- `top_k` (int, default: 5, max: 50): Number of examples
- `include_embeddings` (boolean, default: false): Include embedding vectors
- `category` (string, optional): Filter by category (technical_indicator, price_action, etc.)

**Example**:
```bash
curl "http://localhost:8003/llama/examples?keyword=RSI&top_k=3&include_embeddings=true&category=technical_indicator"
```

**Response**:
```json
{
  "examples": [
    {
      "clip_id": "video_001_RSI_0",
      "transcript": "The Relative Strength Index (RSI) is a momentum oscillator that measures the velocity and magnitude of price changes. When RSI exceeds 70, it typically indicates overbought conditions...",
      "keyword": "RSI",
      "category": "technical_indicator",
      "timestamp": 45.2,
      "clip_url": "/clip/1/download",
      "frame_path": "/data/processed/clips/video_001/frames/video_001_RSI_0.jpg",
      "context_text": "...the RSI indicator is a momentum oscillator that measures the speed and magnitude of price changes...",
      "detected_concepts": ["momentum", "overbought", "oscillator"],
      "embeddings": [0.123, 0.456, 0.789, ...]
    },
    {
      "clip_id": "video_001_RSI_1",
      "transcript": "An RSI reading above 70 suggests the asset is overbought and may be due for a pullback. Below 30, it's oversold and may bounce...",
      "keyword": "RSI",
      "category": "technical_indicator",
      "timestamp": 120.5,
      "clip_url": "/clip/2/download",
      "frame_path": "/data/processed/clips/video_001/frames/video_001_RSI_1.jpg",
      "context_text": "...RSI above 70 is overbought, below 30 is oversold...",
      "detected_concepts": ["overbought", "oversold", "pullback"],
      "embeddings": [0.234, 0.567, 0.890, ...]
    }
  ],
  "total": 2,
  "keyword_filter": "RSI",
  "category_filter": "technical_indicator"
}
```

### Using Llama Examples with Ollama

**Python Example**:
```python
import requests
import json

# Get examples from API
response = requests.get(
    "http://localhost:8003/llama/examples",
    params={
        "keyword": "RSI",
        "top_k": 5,
        "include_embeddings": False
    }
)

examples = response.json()["examples"]

# Format for Llama prompt
context = "\n".join([
    f"- {ex['transcript'][:200]}... (mentioned at {ex['timestamp']}s)"
    for ex in examples
])

# Create prompt
prompt = f"""Based on these trading video excerpts about RSI:

{context}

Summarize the key concepts and strategies related to RSI. Be specific about overbought/oversold levels."""

# Send to Ollama
ollama_response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama2",
        "prompt": prompt,
        "stream": False
    }
)

print(ollama_response.json()["response"])
```

---

## Error Handling

### Error Response Format

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2025-11-20T20:50:00.000Z",
  "request_id": "req-uuid-1234"
}
```

### Common HTTP Status Codes

- `200 OK`: Successful GET request
- `201 Created`: Successful POST request
- `202 Accepted`: Request accepted for async processing
- `400 Bad Request`: Invalid parameters or malformed request
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

---

## Examples

### Complete Workflow Example

```bash
#!/bin/bash

# 1. Ingest a video
INGEST_RESPONSE=$(curl -X POST http://localhost:8003/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/data/videos/trading_tutorial.mp4",
    "filename": "trading_tutorial.mp4"
  }')

MEDIA_ID=$(echo $INGEST_RESPONSE | jq -r '.media_item_id')
echo "Media ID: $MEDIA_ID"

# 2. Wait for processing (poll status)
sleep 10

# 3. Get transcript
curl http://localhost:8003/transcript/$MEDIA_ID

# 4. Get keywords
curl "http://localhost:8003/keywords?media_id=$MEDIA_ID&limit=10"

# 5. Get clips for specific keyword
curl "http://localhost:8003/clips?keyword=RSI&media_id=$MEDIA_ID"

# 6. Get Llama examples
curl "http://localhost:8003/llama/examples?keyword=RSI&top_k=5"

# 7. Download first clip (if available)
curl -O http://localhost:8003/clip/1/download
```

---

## API Specification

Full OpenAPI/Swagger documentation available at: **http://localhost:8003/docs**

ReDoc documentation available at: **http://localhost:8003/redoc**