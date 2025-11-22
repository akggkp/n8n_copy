# GEMINI_CLI_COMPLETE_IMPLEMENTATION.md
# Complete Implementation Guide - All Missing Components

## 🤖 OVERVIEW

This guide contains **ALL implementations** from the critical pipeline audit, formatted for automated execution by Gemini CLI. Every missing component is provided in full, ready to copy-paste.

---

## 📋 TABLE OF CONTENTS

**TASK GROUP 1**: Video Processor Service (5 files)  
**TASK GROUP 2**: ML Service (5 files)  
**TASK GROUP 3**: Complete Orchestrator Tasks (1 large file)  
**TASK GROUP 4**: Complete Database Models (1 file)  
**TASK GROUP 5**: Orchestrator Support Files (4 files)  
**TASK GROUP 6**: Docker & Config (3 files)  
**TASK GROUP 7**: Validation Scripts (2 files)

**Total**: 21 files to create/update

---

## 🎯 TASK GROUP 1: VIDEO PROCESSOR SERVICE

### Task 1.1: Create Directory
```bash
mkdir -p services/video-processor/app
```

### Task 1.2: Create services/video-processor/app/main.py

**Path**: `services/video-processor/app/main.py`

```python
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
```

### Task 1.3: Create services/video-processor/Dockerfile

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Task 1.4: Create services/video-processor/requirements.txt

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
opencv-python==4.8.1.78
openai-whisper==20231117
ffmpeg-python==0.2.0
numpy==1.24.3
torch==2.1.0
```

### Task 1.5: Create services/video-processor/app/__init__.py

```python
# Video Processor Service
__version__ = "1.0.0"
```

---

## 🎯 TASK GROUP 2: ML SERVICE

### Task 2.1: Create Directory
```bash
mkdir -p services/ml-service/app
```

### Task 2.2: Create services/ml-service/app/main.py

**Path**: `services/ml-service/app/main.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="ML Service", version="1.0.0")

# Trading Keywords Database - Comprehensive list of trading terms
KEYWORDS_DB = {
    "technical_indicator": [
        "rsi", "relative strength index", 
        "macd", "moving average convergence divergence",
        "bollinger bands", "bollinger",
        "moving average", "ma", "ema", "sma",
        "exponential moving average", "simple moving average",
        "stochastic", "stochastic oscillator",
        "adx", "average directional index",
        "momentum", "momentum indicator",
        "volume", "volume indicator",
        "fibonacci", "fibonacci retracement",
        "atr", "average true range",
        "cci", "commodity channel index",
        "williams %r", "parabolic sar",
        "ichimoku", "ichimoku cloud"
    ],
    "price_action": [
        "support", "support level",
        "resistance", "resistance level",
        "breakout", "breakdown",
        "trend", "uptrend", "downtrend", "sideways",
        "reversal", "trend reversal",
        "consolidation", "consolidation phase",
        "pullback", "retracement",
        "swing high", "swing low",
        "higher high", "higher low",
        "lower high", "lower low",
        "double top", "double bottom",
        "head and shoulders", "inverse head and shoulders",
        "triangle", "ascending triangle", "descending triangle",
        "flag", "pennant", "wedge"
    ],
    "candlestick_pattern": [
        "doji", "hammer", "inverted hammer",
        "engulfing", "bullish engulfing", "bearish engulfing",
        "shooting star", "hanging man",
        "morning star", "evening star",
        "three white soldiers", "three black crows",
        "harami", "bullish harami", "bearish harami",
        "piercing", "dark cloud cover",
        "spinning top", "marubozu"
    ],
    "risk_management": [
        "stop loss", "stop-loss", "sl",
        "take profit", "take-profit", "tp",
        "risk reward", "risk-reward ratio", "r:r",
        "position size", "position sizing",
        "risk management", "money management",
        "drawdown", "maximum drawdown",
        "portfolio", "diversification",
        "leverage", "margin"
    ],
    "order_type": [
        "market order", "limit order",
        "stop order", "stop-limit order",
        "trailing stop", "trailing stop-loss",
        "oco", "one cancels other",
        "bracket order", "conditional order",
        "good till cancelled", "gtc",
        "day order", "fill or kill", "fok"
    ],
    "trading_strategy": [
        "scalping", "day trading", "swing trading",
        "position trading", "trend following",
        "mean reversion", "breakout trading",
        "momentum trading", "range trading",
        "arbitrage", "hedging"
    ],
    "market_structure": [
        "bull market", "bear market",
        "market cycle", "market phase",
        "accumulation", "distribution",
        "markup", "markdown",
        "volatility", "liquidity",
        "bid-ask spread", "order flow"
    ]
}

# Request/Response models
class ConceptRequest(BaseModel):
    transcript: str

class KeywordMatch(BaseModel):
    keyword: str
    category: str
    confidence: float
    context: str
    start_pos: int
    end_pos: int

class ConceptResponse(BaseModel):
    status: str
    keywords: List[KeywordMatch]
    total_found: int

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ml-service",
        "version": "1.0.0",
        "status": "running",
        "keywords_loaded": sum(len(v) for v in KEYWORDS_DB.values())
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ml-service",
        "keywords_loaded": sum(len(v) for v in KEYWORDS_DB.values()),
        "categories": list(KEYWORDS_DB.keys()),
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.post("/extract_concepts", response_model=ConceptResponse)
async def extract_concepts(request: ConceptRequest):
    """
    Extract trading concepts from transcript
    
    Uses rule-based keyword matching with category classification
    
    Args:
        request: ConceptRequest with transcript text
    
    Returns:
        ConceptResponse with detected keywords and metadata
    """
    try:
        transcript = request.transcript
        transcript_lower = transcript.lower()
        
        logger.info(f"Extracting concepts from transcript ({len(transcript)} chars)")
        
        detected_keywords = []
        
        # Iterate through each category and keywords
        for category, keywords in KEYWORDS_DB.items():
            for keyword in keywords:
                # Create regex pattern for whole word matching
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = list(re.finditer(pattern, transcript_lower, re.IGNORECASE))
                
                for match in matches:
                    # Extract context (50 characters before and after)
                    start = max(0, match.start() - 50)
                    end = min(len(transcript), match.end() + 50)
                    context = transcript[start:end].strip()
                    
                    detected_keywords.append({
                        "keyword": keyword,
                        "category": category,
                        "confidence": 1.0,  # Rule-based = 100% confidence
                        "context": context,
                        "start_pos": match.start(),
                        "end_pos": match.end()
                    })
        
        # Remove duplicates (same keyword at same position)
        unique_keywords = []
        seen = set()
        
        for kw in detected_keywords:
            key = (kw['keyword'], kw['start_pos'])
            if key not in seen:
                seen.add(key)
                unique_keywords.append(kw)
        
        # Sort by position in text
        unique_keywords.sort(key=lambda x: x['start_pos'])
        
        logger.info(f"Found {len(unique_keywords)} unique keywords")
        
        return ConceptResponse(
            status="success",
            keywords=unique_keywords,
            total_found=len(unique_keywords)
        )
    
    except Exception as e:
        logger.error(f"Concept extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_patterns")
async def detect_patterns(frame_data: Dict):
    """
    Detect candlestick patterns in frame (placeholder for future CV model)
    
    Args:
        frame_data: Dictionary with frame information
    
    Returns:
        Dictionary with detected patterns (currently empty)
    """
    logger.info("Pattern detection called (not yet implemented)")
    
    # TODO: Implement actual pattern detection with computer vision model
    return {
        "status": "success",
        "patterns": [],
        "note": "Pattern detection not yet implemented - requires CV model"
    }

@app.get("/categories")
async def get_categories():
    """
    Get available keyword categories
    
    Returns:
        Dictionary with categories and keyword counts
    """
    categories = {}
    for category, keywords in KEYWORDS_DB.items():
        categories[category] = {
            "count": len(keywords),
            "sample_keywords": keywords[:5]
        }
    
    return {
        "status": "success",
        "categories": categories,
        "total_keywords": sum(len(v) for v in KEYWORDS_DB.values())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
```

### Task 2.3: Create services/ml-service/Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8002

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
```

### Task 2.4: Create services/ml-service/requirements.txt

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
```

### Task 2.5: Create services/ml-service/app/__init__.py

```python
# ML Service
__version__ = "1.0.0"
```

---

## 🎯 TASK GROUP 3: COMPLETE ORCHESTRATOR TASKS

⚠️ **CRITICAL**: This is the COMPLETE tasks.py implementation with all 10 tasks fully implemented.

### Task 3.1: Create/Update orchestrator/app/tasks.py

**Path**: `orchestrator/app/tasks.py`

**Action**: Replace entire file with this content

**THIS FILE CONTINUES** - See next message for remaining task groups...
