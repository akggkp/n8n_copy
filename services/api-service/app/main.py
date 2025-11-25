# services/api-service/app/main.py
# FastAPI application for Trading Media Extraction Pipeline data access layer

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

# These models will be needed for the database-connected endpoints
from app.database import get_db
from app.llama_client import LlamaClient

# Initialize FastAPI app
app = FastAPI(
    title="Trading Media Extraction API",
    description="API for accessing processed videos, clips, transcripts, and embeddings",
    version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Llama client
llama_client: Optional[LlamaClient] = None


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize clients on startup"""
    global llama_client
    logger.info("API Server starting up...")
    try:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        llama_client = LlamaClient(ollama_url=ollama_url)
        if llama_client.health_check():
            logger.info("Llama client initialized and connected to Ollama.")
        else:
            logger.warning(
                "Llama client initialized, but could not connect to Ollama.")
    except Exception as e:
        logger.error(f"Error initializing Llama client: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("API Server shutting down...")
    # Close database connections


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "trading-media-extraction-api"
    }


# ============================================================================
# MEDIA ITEM ENDPOINTS
# ============================================================================

@app.get("/media_items", tags=["Media Items"])
async def list_media_items(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None)
):
    """
    List all processed videos with pagination

    Query Parameters:
    - skip: Number of items to skip (default: 0)
    - limit: Number of items to return (default: 20, max: 100)
    - status: Filter by status (pending, processing, completed, failed)

    Returns:
    - List of media items with metadata and statistics
    """
    try:
        # Implementation placeholder
        # from app.database import db
        # from app.models import MediaItem

        # query = db.session.query(MediaItem)
        # if status:
        #     query = query.filter_by(status=status)

        # total = query.count()
        # items = query.offset(skip).limit(limit).all()

        # return {
        #     "total": total,
        #     "skip": skip,
        #     "limit": limit,
        #     "items": [item.to_dict() for item in items]
        # }

        return {
            "message": "List media items endpoint - database integration needed",
            "skip": skip,
            "limit": limit,
            "status_filter": status}
    except Exception as e:
        logger.error(f"Error listing media items: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/media_items/{media_id}", tags=["Media Items"])
async def get_media_item(media_id: int):
    """
    Get details for a specific media item

    Returns:
    - Media item metadata with associated clips, transcripts, keywords count
    """
    try:
        # from app.models import MediaItem
        # from app.database import db

        # media_item = db.session.query(MediaItem).filter_by(id=media_id).first()
        # if not media_item:
        #     raise HTTPException(status_code=404, detail="Media item not found")

        # return {
        #     "id": media_item.id,
        #     "video_id": media_item.video_id,
        #     "filename": media_item.filename,
        #     "duration_seconds": media_item.duration_seconds,
        #     "status": media_item.status,
        #     "created_at": media_item.created_at.isoformat(),
        #     "clips_count": len(media_item.clips),
        #     "keywords_count": len(media_item.keyword_hits),
        #     "transcript_segments": len(media_item.transcripts)
        # }

        return {
            "message": f"Get media item {media_id} - database integration needed"}
    except Exception as e:
        logger.error(f"Error getting media item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# INGEST ENDPOINT - TRIGGER PIPELINE
# ============================================================================

@app.post("/ingest", tags=["Pipeline"], status_code=202)
async def trigger_ingest(
    video_path: str,
    source_url: Optional[str] = None,
    filename: Optional[str] = None
):
    """
    Trigger video processing pipeline

    Request Body:
    - video_path: Path to video file (local or S3)
    - source_url: Optional URL where video came from
    - filename: Optional custom filename

    Returns:
    - task_id: Celery task ID for tracking progress
    - status: 'accepted' or 'queued'
    - estimated_duration_minutes: Estimated processing time
    """
    try:
        pass

        # Validate file exists
        if not os.path.exists(video_path):
            raise HTTPException(status_code=400,
                                detail=f"Video file not found: {video_path}")

        # Get file info
        file_size = os.path.getsize(video_path)
        if not filename:
            filename = os.path.basename(video_path)

        logger.info(f"Ingesting video: {filename} ({file_size} bytes)")

        # Create media_item record in database
        # from app.models import MediaItem
        # from app.database import db

        # media_item = MediaItem(
        #     filename=filename,
        #     source_url=source_url,
        #     file_size_bytes=file_size,
        #     status='pending'
        # )
        # db.session.add(media_item)
        # db.session.commit()

        # Queue Celery task
        # from app.celery_app import celery_app
        # result = celery_app.send_task(
        #     'app.tasks.run_full_pipeline',
        #     args=[media_item.id, video_path, filename]
        # )

        return {
            "status": "accepted",
            "task_id": "placeholder-task-id",
            "media_item_id": "placeholder-media-id",
            "message": "Video queued for processing",
            "estimated_duration_minutes": 5
        }
    except Exception as e:
        logger.error(f"Error triggering ingest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TRANSCRIPT ENDPOINTS
# ============================================================================

@app.get("/transcript/{media_id}", tags=["Transcripts"])
async def get_transcript(
    media_id: int,
    format: str = Query("json", regex="^(json|text)$")
):
    """
    Get full timestamped transcript for a media item

    Path Parameters:
    - media_id: Media item ID

    Query Parameters:
    - format: Response format (json or text)

    Returns:
    - Full transcript with timestamps and segment indices
    """
    try:
        # from app.models import Transcript
        # from app.database import db

        # transcripts = db.session.query(Transcript).filter_by(media_item_id=media_id).order_by(Transcript.segment_index).all()
        # if not transcripts:
        #     raise HTTPException(status_code=404, detail="No transcript found")

        # if format == "json":
        #     return {
        #         "media_id": media_id,
        #         "segments": [
        #             {
        #                 "index": t.segment_index,
        #                 "start_time": t.start_time,
        #                 "end_time": t.end_time,
        #                 "text": t.text
        #             }
        #             for t in transcripts
        #         ]
        #     }
        # else:
        #     full_text = " ".join([t.text for t in transcripts])
        #     return {"text": full_text}

        return {
            "message": f"Get transcript for media {media_id} - database integration needed"}
    except Exception as e:
        logger.error(f"Error getting transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# KEYWORD & CLIPS ENDPOINTS
# ============================================================================

@app.get("/clips", tags=["Clips"])
async def search_clips(
    keyword: Optional[str] = Query(None),
    media_id: Optional[int] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    skip: int = Query(0, ge=0)
):
    """
    Search for video clips by keyword

    Query Parameters:
    - keyword: Filter by keyword name
    - media_id: Filter by media item ID
    - limit: Number of results (default: 10, max: 100)
    - skip: Number of results to skip (default: 0)

    Returns:
    - List of clip metadata with download URLs
    """
    try:
        # from app.models import Clip
        # from app.database import db

        # query = db.session.query(Clip)
        # if keyword:
        #     query = query.filter(Clip.keyword.ilike(f"%{keyword}%"))
        # if media_id:
        #     query = query.filter_by(media_item_id=media_id)

        # total = query.count()
        # clips = query.offset(skip).limit(limit).all()

        # return {
        #     "total": total,
        #     "skip": skip,
        #     "limit": limit,
        #     "clips": [
        #         {
        #             "id": c.id,
        #             "keyword": c.keyword,
        #             "start_time": c.start_time,
        #             "end_time": c.end_time,
        #             "duration_seconds": c.duration_seconds,
        #             "download_url": f"/clip/{c.id}/download"
        #         }
        #         for c in clips
        #     ]
        # }

        return {
            "message": "Search clips endpoint - database integration needed",
            "keyword": keyword,
            "media_id": media_id,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error searching clips: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clip/{clip_id}/download", tags=["Clips"])
async def download_clip(clip_id: int):
    """
    Download video clip by ID

    Returns:
    - Binary video file stream (mp4)
    """
    try:
        # from app.models import Clip
        # from app.database import db

        # clip = db.session.query(Clip).filter_by(id=clip_id).first()
        # if not clip:
        #     raise HTTPException(status_code=404, detail="Clip not found")

        # if not os.path.exists(clip.file_path):
        #     raise HTTPException(status_code=404, detail="Clip file not found on disk")

        # return FileResponse(
        #     path=clip.file_path,
        #     media_type="video/mp4",
        #     filename=f"clip_{clip_id}.mp4"
        # )

        raise HTTPException(
            status_code=404,
            detail="Clip not found - database integration needed")
    except Exception as e:
        logger.error(f"Error downloading clip: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/keywords", tags=["Keywords"])
async def get_keywords(
    media_id: Optional[int] = Query(None),
    min_confidence: float = Query(0.7, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500)
):
    """
    Get detected keywords for a media item

    Query Parameters:
    - media_id: Filter by media item ID
    - min_confidence: Minimum confidence threshold
    - limit: Maximum keywords to return

    Returns:
    - List of keywords with timestamps and confidence scores
    """
    try:
        # from app.models import KeywordHit
        # from app.database import db

        # query = db.session.query(KeywordHit).filter(KeywordHit.confidence >= min_confidence)
        # if media_id:
        #     query = query.filter_by(media_item_id=media_id)

        # keywords = query.limit(limit).all()

        # return {
        #     "keywords": [
        #         {
        #             "id": k.id,
        #             "keyword": k.keyword,
        #             "category": k.category,
        #             "start_time": k.start_time,
        #             "end_time": k.end_time,
        #             "confidence": k.confidence,
        #             "context": k.context_text
        #         }
        #         for k in keywords
        #     ]
        # }

        return {
            "message": "Get keywords endpoint - database integration needed",
            "media_id": media_id,
            "min_confidence": min_confidence
        }
    except Exception as e:
        logger.error(f"Error getting keywords: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EMBEDDINGS & SEMANTIC SEARCH ENDPOINTS
# ============================================================================

@app.get("/embeddings/search", tags=["Embeddings"])
async def search_embeddings(
    query: str, top_k: int = Query(
        10, ge=1, le=100), embedding_type: str = Query(
            "transcript", regex="^(transcript|frame|clip)$")):
    """
    Semantic similarity search using embeddings

    Query Parameters:
    - query: Search text or concept
    - top_k: Number of top results to return
    - embedding_type: Type of embeddings to search (transcript, frame, clip)

    Returns:
    - List of most similar transcripts/frames/clips with similarity scores
    """
    try:
        # from app.embeddings_service import EmbeddingsClient
        # embeddings_client = EmbeddingsClient()

        # query_embedding = embeddings_client.encode(query)
        # results = embeddings_client.search_faiss(
        #     query_embedding=query_embedding,
        #     embedding_type=embedding_type,
        #     top_k=top_k
        # )

        return {
            "message": "Semantic search endpoint - embeddings service integration needed",
            "query": query,
            "top_k": top_k,
            "embedding_type": embedding_type}
    except Exception as e:
        logger.error(f"Error in embeddings search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LLAMA DATASET ENDPOINT
# ============================================================================

def _query_keyword_hits(db: Session, keyword: Optional[str], category: Optional[str], top_k: int):
    """Queries the database for keyword hits."""
    from app.models import KeywordHit

    query = db.query(KeywordHit).filter(KeywordHit.confidence >= 0.7)

    if keyword:
        query = query.filter(KeywordHit.keyword.ilike(f"%{keyword}%"))
    if category:
        query = query.filter(KeywordHit.category == category)

    return query.limit(top_k).all()


def _get_media_item_for_hit(db, hit):
    from app.models import MediaItem
    return db.query(MediaItem).filter_by(id=hit.media_item_id).first()


def _get_clips_for_hit(db, hit):
    from app.models import Clip
    return db.query(Clip).filter_by(keyword_hit_id=hit.id).all()


def _get_transcripts_for_hit(db, hit):
    from app.models import Transcript
    return db.query(Transcript).filter_by(media_item_id=hit.media_item_id).all()


def _build_context_text(transcripts, hit):
    context_segments = [
        t for t in transcripts
        if hit.start_time - 10 <= t.start_time <= hit.end_time + 10
    ]
    return " ".join([t.text for t in context_segments])


def _get_embedding_for_hit(db, hit):
    from app.models import Embedding
    from sqlalchemy import and_
    embedding = db.query(Embedding).filter(
        and_(
            Embedding.media_item_id == hit.media_item_id,
            Embedding.embedding_type == "keyword",
            Embedding.reference_id == hit.id
        )
    ).first()
    return embedding.embedding_vector if embedding else None


def _detect_concepts(text):
    trading_terms = ["support", "resistance", "breakout", "momentum", "trend"]
    return [term for term in trading_terms if term.lower() in text.lower()]


def _build_example_dict(hit, media_item, clips, context_text, full_transcript, detected_concepts, embeddings_vector):
    example = {
        "clip_id": f"{media_item.video_id}_{hit.keyword}_{hit.id}",
        "transcript": context_text.strip() if context_text else full_transcript[:500],
        "keyword": hit.keyword,
        "category": hit.category,
        "timestamp": hit.start_time,
        "confidence": hit.confidence,
        "clip_url": f"/clip/{clips[0].id}/download" if clips else None,
        "detected_concepts": detected_concepts,
        "context_text": hit.context_text or context_text[:200]
    }
    if embeddings_vector:
        example["embeddings"] = embeddings_vector
    return example


def _create_example_from_hit(db: Session, hit: 'KeywordHit', include_embeddings: bool):
    """Creates a Llama example dictionary from a KeywordHit."""
    try:
        media_item = _get_media_item_for_hit(db, hit)
        if not media_item:
            return None

        clips = _get_clips_for_hit(db, hit)
        transcripts = _get_transcripts_for_hit(db, hit)

        context_text = _build_context_text(transcripts, hit)
        full_transcript = " ".join([t.text for t in transcripts])

        embeddings_vector = _get_embedding_for_hit(db, hit) if include_embeddings else None

        detected_concepts = _detect_concepts(context_text)

        return _build_example_dict(hit, media_item, clips, context_text, full_transcript, detected_concepts, embeddings_vector)

    except Exception as e:
        logger.error(f"Error processing keyword hit {hit.id}: {str(e)}")
        return None


@app.get("/llama/examples", tags=["Llama Dataset"])
async def get_llama_examples(
    keyword: Optional[str] = Query(None),
    top_k: int = Query(5, ge=1, le=50),
    include_embeddings: bool = Query(False),
    category: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Get structured training examples for Llama fine-tuning or prompting

    Query Parameters:
    - keyword: Filter by keyword (e.g., "RSI", "breakout")
    - top_k: Number of examples to return
    - include_embeddings: Whether to include embedding vectors
    - category: Filter by keyword category

    Returns:
    - List of examples with transcripts, clips, concepts, and optional embeddings
    """
    try:
        keyword_hits = _query_keyword_hits(db, keyword, category, top_k)

        examples = []
        for hit in keyword_hits:
            example = _create_example_from_hit(db, hit, include_embeddings)
            if example:
                examples.append(example)

        return {
            "examples": examples,
            "total": len(examples),
            "keyword_filter": keyword,
            "category_filter": category,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting Llama examples: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llama/generate-strategy", tags=["Llama Dataset"])
async def generate_llama_strategy(
    keyword: str,
    top_k: int = Query(5, ge=1, le=20),
    temperature: float = Query(0.7, ge=0.0, le=1.0),
    db: Session = Depends(get_db)
):
    """
    Generate trading strategy using Llama based on video examples

    Query Parameters:
    - keyword: Trading concept to generate strategy for
    - top_k: Number of examples to use (default: 5)
    - temperature: Generation creativity (0.0=deterministic, 1.0=creative)

    Returns:
    - Generated strategy text
    """
    try:
        if not llama_client:
            raise HTTPException(
                status_code=503,
                detail="Llama client not initialized")

        if not llama_client.health_check():
            raise HTTPException(
                status_code=503,
                detail="Ollama service not available")

        # Get examples
        examples_response = await get_llama_examples(
            keyword=keyword,
            top_k=top_k,
            include_embeddings=False,
            category=None,
            db=db
        )

        examples = examples_response.get('examples', [])
        if not examples:
            raise HTTPException(
                status_code=404,
                detail=f"No examples found for keyword: {keyword}")

        # Generate strategy
        strategy = llama_client.generate_strategy(
            examples=examples,
            keyword=keyword,
            temperature=temperature
        )

        if not strategy:
            raise HTTPException(
                status_code=500,
                detail="Strategy generation failed")

        return {
            "keyword": keyword,
            "strategy": strategy.strip(),
            "examples_used": len(examples),
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llama/summarize/{keyword}", tags=["Llama Dataset"])
async def summarize_keyword(
    keyword: str,
    top_k: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    Get AI-generated summary of trading keyword from video examples

    Path Parameters:
    - keyword: Trading concept to summarize

    Query Parameters:
    - top_k: Number of examples to use for summarization

    Returns:
    - Summary text of the trading concept
    """
    try:
        if not llama_client:
            raise HTTPException(
                status_code=503,
                detail="Llama client not initialized")

        if not llama_client.health_check():
            raise HTTPException(
                status_code=503,
                detail="Ollama service not available")

        # Get examples
        examples_response = await get_llama_examples(
            keyword=keyword,
            top_k=top_k,
            include_embeddings=False,
            category=None,
            db=db
        )

        examples = examples_response.get('examples', [])
        if not examples:
            raise HTTPException(
                status_code=404,
                detail=f"No examples found for keyword: {keyword}")

        # Summarize
        summary = llama_client.summarize_keyword(
            examples=examples,
            keyword=keyword
        )

        if not summary:
            raise HTTPException(status_code=500, detail="Summarization failed")

        return {
            "keyword": keyword,
            "summary": summary.strip(),
            "examples_used": len(examples),
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing keyword: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API information and documentation"""
    return {
        "name": "Trading Media Extraction API",
        "version": "1.0.0",
        "description": "RESTful API for accessing processed trading videos, clips, transcripts, and embeddings",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "health": "/health",
            "media_items": "/media_items",
            "ingest": "/ingest",
            "transcripts": "/transcript/{media_id}",
            "clips": "/clips",
            "embeddings": "/embeddings/search",
            "llama": "/llama/examples"}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8003)),
        workers=int(os.getenv("API_WORKERS", 2))
    )
