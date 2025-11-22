from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import whisper
import os
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Video Processor Service", version="1.0.0")

# Load Whisper model (using base model for speed)
try:
    whisper_model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None

# Request/Response models
class ProcessRequest(BaseModel):
    file_path: str

class ProcessResponse(BaseModel):
    status: str
    transcript: list
    frames: list
    metadata: dict

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "video-processor",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    whisper_status = "loaded" if whisper_model is not None else "not_loaded"
    return {
        "status": "healthy",
        "service": "video-processor",
        "whisper_model": whisper_status,
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.post("/process", response_model=ProcessResponse)
async def process_video(request: ProcessRequest):
    """
    Process video: extract audio, transcribe, and extract frames
    
    Args:
        request: ProcessRequest with file_path
    
    Returns:
        ProcessResponse with transcript, frames, and metadata
    """
    try:
        file_path = request.file_path
        logger.info(f"Processing video: {file_path}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Extract audio
        logger.info("Extracting audio...")
        audio_path = extract_audio(file_path)
        
        # Transcribe with Whisper
        logger.info("Transcribing audio...")
        transcript_segments = transcribe_with_whisper(audio_path)
        
        # Extract keyframes
        logger.info("Extracting keyframes...")
        frames = extract_keyframes(file_path, fps=1)
        
        # Get video metadata
        logger.info("Getting video metadata...")
        metadata = get_video_metadata(file_path)
        
        # Cleanup temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        logger.info(f"Processing complete. Transcripts: {len(transcript_segments)}, Frames: {len(frames)}")
        
        return ProcessResponse(
            status="success",
            transcript=transcript_segments,
            frames=frames,
            metadata=metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_audio(video_path: str) -> str:
    """
    Extract audio from video using FFmpeg
    
    Args:
        video_path: Path to video file
    
    Returns:
        Path to extracted audio file
    """
    audio_path = video_path.replace('.mp4', '_audio.wav')
    
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # WAV format
        '-ar', '16000',  # 16kHz sample rate
        '-ac', '1',  # Mono
        audio_path,
        '-y'  # Overwrite
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, stderr=subprocess.PIPE)
        return audio_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr.decode()}")
        raise

def transcribe_with_whisper(audio_path: str) -> list:
    """
    Transcribe audio with Whisper
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        List of transcript segments with timestamps
    """
    if whisper_model is None:
        logger.warning("Whisper model not loaded, returning empty transcript")
        return []
    
    try:
        result = whisper_model.transcribe(
            audio_path,
            word_timestamps=False,
            language='en'
        )
        
        segments = []
        for segment in result['segments']:
            segments.append({
                'start_time': float(segment['start']),
                'end_time': float(segment['end']),
                'text': segment['text'].strip()
            })
        
        return segments
    
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        return []

def extract_keyframes(video_path: str, fps: int = 1) -> list:
    """
    Extract keyframes from video at specified FPS
    
    Args:
        video_path: Path to video file
        fps: Frames per second to extract
    
    Returns:
        List of frame information with paths and timestamps
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frames = []
    frame_count = 0
    saved_count = 0
    
    # Create output directory
    output_dir = Path(video_path).parent / f"{Path(video_path).stem}_frames"
    output_dir.mkdir(exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = output_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            frames.append({
                'frame_number': saved_count,
                'timestamp': float(frame_count / video_fps),
                'file_path': str(frame_path)
            })
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {saved_count} frames")
    return frames

def get_video_metadata(video_path: str) -> dict:
    """
    Get video metadata using OpenCV
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return {}
    
    metadata = {
        'duration': float(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)),
        'fps': float(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    
    cap.release()
    return metadata

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
