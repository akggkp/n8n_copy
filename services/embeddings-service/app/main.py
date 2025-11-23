# services/embeddings-service/app/main.py
# FastAPI service for generating and searching embeddings

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import os
from datetime import datetime
from app.embeddings_client import EmbeddingsClient

# Initialize FastAPI app
app = FastAPI(
    title="Embeddings Service",
    description="Service for generating and searching text/image embeddings using sentence-transformers and Faiss",
    version="1.0.0")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global embeddings client (initialized on startup)
embeddings_client: Optional[EmbeddingsClient] = None


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class EmbedRequest(BaseModel):
    """Request to generate embeddings"""
    media_item_id: int
    embedding_type: str  # 'transcript', 'frame', 'clip'
    segments: Optional[List[Dict]] = None  # for transcripts
    frames: Optional[List[Dict]] = None    # for frames
    clips: Optional[List[Dict]] = None     # for clips


class EmbedResponse(BaseModel):
    """Response from embedding generation"""
    status: str
    media_item_id: int
    embedding_type: str
    embeddings_created: int
    timestamp: datetime


class SearchRequest(BaseModel):
    """Request for similarity search"""
    query: str
    embedding_type: str  # 'transcript', 'frame', 'clip'
    top_k: int = 10
    min_similarity: float = 0.5


class SearchResult(BaseModel):
    """Individual search result"""
    index: int
    similarity: float
    distance: float
    metadata: Dict


class SearchResponse(BaseModel):
    """Response from similarity search"""
    query: str
    embedding_type: str
    results: List[SearchResult]
    timestamp: datetime


class StatsResponse(BaseModel):
    """Statistics about embeddings"""
    model_name: str
    embedding_dimension: int
    transcript_vectors: int
    frame_vectors: int
    clip_vectors: int
    total_vectors: int
    timestamp: datetime


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize embeddings client on startup"""
    try:
        faiss_dir = os.getenv("FAISS_INDEX_DIR", "/data/processed/faiss")
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        device = os.getenv("EMBEDDING_DEVICE", "cpu")

        logger.info(f"Initializing embeddings client with model: {model_name}")
            model_name = model_name,
            faiss_index_dir = faiss_dir,
            device = device
        )
        logger.info("Embeddings client initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing embeddings client: {str(e)}")
        raise


@ app.on_event("shutdown")
async def shutdown_event():
    """Save indices on shutdown"""
    try:
        if embeddings_client:
            logger.info("Saving embeddings indices...")
            embeddings_client.save_indices()
            logger.info("Embeddings indices saved successfully")
    except Exception as e:
        logger.error(f"Error saving embeddings: {str(e)}")


# ============================================================================
# HEALTH CHECK
# ============================================================================

@ app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if embeddings_client is None:
            return {"status": "initializing",
                    "timestamp": datetime.utcnow().isoformat()}

        stats = embeddings_client.get_stats()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "embeddings-service",
            "stats": stats
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 503


# ============================================================================
# EMBEDDING GENERATION ENDPOINTS
# ============================================================================

@app.post("/embed", response_model=EmbedResponse)
async def generate_embeddings(
        request: EmbedRequest,
        background_tasks: BackgroundTasks):
    """
    Generate embeddings for transcripts, frames, or clips

    Request:
    ```json
    {
      "media_item_id": 1,
      "embedding_type": "transcript",
      "segments": [
        {"id": 1, "text": "The RSI indicator...", "start_time": 0.0, "end_time": 3.5},
        {"id": 2, "text": "RSI stands for...", "start_time": 3.5, "end_time": 7.2}
      ]
    }
    ```
    """
    try:
        if embeddings_client is None:
            raise HTTPException(status_code=503,
                                detail="Embeddings client not initialized")

        embeddings_count = 0

        if request.embedding_type == "transcript" and request.segments:
            embeddings_count = embeddings_client.add_transcript_embeddings(
                request.media_item_id,
                request.segments
            )
        elif request.embedding_type == "frame" and request.frames:
            embeddings_count = embeddings_client.add_frame_embeddings(
                request.media_item_id,
                request.frames
            )
        elif request.embedding_type == "clip" and request.clips:
            embeddings_count = embeddings_client.add_clip_embeddings(
                request.media_item_id,
                request.clips
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid embedding_type or missing data")

        # Save indices in background
        background_tasks.add_task(embeddings_client.save_indices)

        return EmbedResponse(
            status="success",
            media_item_id=request.media_item_id,
            embedding_type=request.embedding_type,
            embeddings_created=embeddings_count,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SEARCH ENDPOINTS
# ============================================================================

@app.post("/search", response_model=SearchResponse)
async def search_embeddings(request: SearchRequest):
    """
    Perform semantic similarity search

    Request:
    ```json
    {
      "query": "relative strength index overbought",
      "embedding_type": "transcript",
      "top_k": 10,
      "min_similarity": 0.5
    }
    ```
    """
    try:
        if embeddings_client is None:
            raise HTTPException(status_code=503,
                                detail="Embeddings client not initialized")

        results = []

        if request.embedding_type == "transcript":
            results = embeddings_client.search_transcripts(
                query=request.query,
                top_k=request.top_k,
                min_similarity=request.min_similarity
            )
        elif request.embedding_type == "frame":
            results = embeddings_client.search_frames(
                query=request.query,
                top_k=request.top_k,
                min_similarity=request.min_similarity
            )
        elif request.embedding_type == "clip":
            results = embeddings_client.search_clips(
                query=request.query,
                top_k=request.top_k,
                min_similarity=request.min_similarity
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid embedding_type")

        # Format results
        formatted_results = [
            SearchResult(
                index=r['index'],
                similarity=r['similarity'],
                distance=r['distance'],
                metadata={
                    k: v for k,
                    v in r.items() if k not in [
                        'index',
                        'similarity',
                        'distance']}) for r in results]

        return SearchResponse(
            query=request.query,
            embedding_type=request.embedding_type,
            results=formatted_results,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/transcripts", response_model=SearchResponse)
async def search_transcripts_simple(
    query: str,
    top_k: int = 10,
    min_similarity: float = 0.5
):
    """Simple GET endpoint for transcript search"""
    try:
        if embeddings_client is None:
            raise HTTPException(status_code=503,
                                detail="Embeddings client not initialized")

        results = embeddings_client.search_transcripts(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity
        )

        formatted_results = [
            SearchResult(
                index=r['index'],
                similarity=r['similarity'],
                distance=r['distance'],
                metadata={
                    k: v for k,
                    v in r.items() if k not in [
                        'index',
                        'similarity',
                        'distance']}) for r in results]

        return SearchResponse(
            query=query,
            embedding_type="transcript",
            results=formatted_results,
            timestamp=datetime.utcnow()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching transcripts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/frames", response_model=SearchResponse)
async def search_frames_simple(
    query: str,
    top_k: int = 10,
    min_similarity: float = 0.5
):
    """Simple GET endpoint for frame search"""
    try:
        if embeddings_client is None:
            raise HTTPException(status_code=503,
                                detail="Embeddings client not initialized")

        results = embeddings_client.search_frames(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity
        )

        formatted_results = [
            SearchResult(
                index=r['index'],
                similarity=r['similarity'],
                distance=r['distance'],
                metadata={
                    k: v for k,
                    v in r.items() if k not in [
                        'index',
                        'similarity',
                        'distance']}) for r in results]

        return SearchResponse(
            query=query,
            embedding_type="frame",
            results=formatted_results,
            timestamp=datetime.utcnow()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching frames: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/clips", response_model=SearchResponse)
async def search_clips_simple(
    query: str,
    top_k: int = 10,
    min_similarity: float = 0.5
):
    """Simple GET endpoint for clip search"""
    try:
        if embeddings_client is None:
            raise HTTPException(status_code=503,
                                detail="Embeddings client not initialized")

        results = embeddings_client.search_clips(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity
        )

        formatted_results = [
            SearchResult(
                index=r['index'],
                similarity=r['similarity'],
                distance=r['distance'],
                metadata={
                    k: v for k,
                    v in r.items() if k not in [
                        'index',
                        'similarity',
                        'distance']}) for r in results]

        return SearchResponse(
            query=query,
            embedding_type="clip",
            results=formatted_results,
            timestamp=datetime.utcnow()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching clips: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STATISTICS ENDPOINT
# ============================================================================

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get embeddings service statistics"""
    try:
        if embeddings_client is None:
            raise HTTPException(status_code=503,
                                detail="Embeddings client not initialized")

        stats = embeddings_client.get_stats()
        stats['timestamp'] = datetime.utcnow()

        return StatsResponse(**stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# INFO ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Service information"""
    return {
        "name": "Embeddings Service",
        "version": "1.0.0",
        "description": "Semantic embedding generation and search service using sentence-transformers and Faiss",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "health": "/health",
            "embed": "/embed (POST)",
            "search": "/search (POST)",
            "search_transcripts": "/search/transcripts (GET)",
            "search_frames": "/search/frames (GET)",
            "search_clips": "/search/clips (GET)",
            "stats": "/stats"}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("EMBEDDINGS_PORT", 8004)),
        workers=int(os.getenv("EMBEDDINGS_WORKERS", 1))
    )
