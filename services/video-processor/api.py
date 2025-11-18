"\"\"\"
Video Processor API - FastAPI wrapper for orchestrator
\"\"\"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=\"Video Processor API\",
    description=\"Video processing API for orchestration\",
    version=\"1.0.0\"
)

class ProcessRequest(BaseModel):
    video_id: str
    file_path: str
    use_cascade: bool = True
    confidence_threshold: float = 0.65

class ProcessResponse(BaseModel):
    status: str
    video_id: str
    transcription: str
    detected_charts: List[Dict[str, Any]]
    processing_stats: Dict[str, Any]
    message: str

@app.get(\"/health\")
async def health_check():
    \"\"\"Health check\"\"\"
    return {
        \"status\": \"healthy\",
        \"service\": \"video-processor\",
        \"timestamp\": datetime.now().isoformat()
    }

@app.post(\"/process\", response_model=ProcessResponse)
async def process_video(request: ProcessRequest):
    \"\"\"
    Process video: extract frames, detect charts, transcribe audio
    \"\"\"
    try:
        logger.info(f\"Processing video: {request.video_id}\")
        
        # Import processor
        from worker import VideoProcessorCascade
        
        processor = VideoProcessorCascade()
        success = processor.process_video(request.video_id, request.file_path)
        
        if not success:
            raise HTTPException(status_code=500, detail=\"Processing failed\")
        
        # Fetch results from database
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker
        
        DATABASE_URL = os.getenv(\"DATABASE_URL\")
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Query processed video
        query = text(\"\"\"
            SELECT transcription, detected_charts, processing_stats 
            FROM processed_videos 
            WHERE video_id = :vid
        \"\"\")
        
        result = session.execute(query, {\"vid\": request.video_id}).fetchone()
        session.close()
        
        if not result:
            raise HTTPException(status_code=404, detail=\"Processed video not found\")
        
        return ProcessResponse(
            status=\"success\",
            video_id=request.video_id,
            transcription=result[0] or \"\",
            detected_charts=result[1] or [],
            processing_stats=result[2] or {},
            message=\"Video processed successfully\"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f\"Error processing video: {str(e)}\")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == \"__main__\":
    import uvicorn
    uvicorn.run(app, host=\"0.0.0.0\", port=8000)
"