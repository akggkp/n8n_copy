# GEMINI_CLI_IMPLEMENTATION_GUIDE.md
# Automated Implementation Guide for Gemini CLI

## 🤖 OVERVIEW

This guide is designed for automated execution by Gemini CLI or similar AI coding assistants. Each section provides clear, unambiguous instructions for creating files with complete implementations.

---

## 📋 EXECUTION ORDER

Execute tasks in this exact order to ensure dependencies are met:

1. **Task Group 1**: Create Video Processor Service (3 files)
2. **Task Group 2**: Create ML Service (3 files)
3. **Task Group 3**: Update Orchestrator (7 files)
4. **Task Group 4**: Update Docker Configuration (2 files)
5. **Task Group 5**: Validation Scripts (2 files)

---

## 🎯 TASK GROUP 1: VIDEO PROCESSOR SERVICE

### Task 1.1: Create Directory Structure

**Action**: Create directories
**Path**: `services/video-processor/app/`
**Command**:
```bash
mkdir -p services/video-processor/app
```

---

### Task 1.2: Create Video Processor Main Application

**Action**: Create file
**Path**: `services/video-processor/app/main.py`
**Content**: Full file below

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

---

### Task 1.3: Create Video Processor Dockerfile

**Action**: Create file
**Path**: `services/video-processor/Dockerfile`
**Content**:

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

---

### Task 1.4: Create Video Processor Requirements

**Action**: Create file
**Path**: `services/video-processor/requirements.txt`
**Content**:

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

---

### Task 1.5: Create Video Processor __init__.py

**Action**: Create file
**Path**: `services/video-processor/app/__init__.py`
**Content**:

```python
# Video Processor Service
__version__ = "1.0.0"
```

---

## 🎯 TASK GROUP 2: ML SERVICE

### Task 2.1: Create ML Service Directory

**Action**: Create directories
**Path**: `services/ml-service/app/`
**Command**:
```bash
mkdir -p services/ml-service/app
```

---

### Task 2.2: Create ML Service Main Application

**Action**: Create file
**Path**: `services/ml-service/app/main.py`
**Content**: Full file below

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

---

### Task 2.3: Create ML Service Dockerfile

**Action**: Create file
**Path**: `services/ml-service/Dockerfile`
**Content**:

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

---

### Task 2.4: Create ML Service Requirements

**Action**: Create file
**Path**: `services/ml-service/requirements.txt`
**Content**:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
```

---

### Task 2.5: Create ML Service __init__.py

**Action**: Create file
**Path**: `services/ml-service/app/__init__.py`
**Content**:

```python
# ML Service
__version__ = "1.0.0"
```

---

## 🎯 TASK GROUP 3: UPDATE ORCHESTRATOR

### Task 3.1: Create Orchestrator Scripts Directory

**Action**: Create directories
**Path**: `orchestrator/scripts/`
**Command**:
```bash
mkdir -p orchestrator/scripts
```

---

### Task 3.2: Create Database Module

**Action**: Create file
**Path**: `orchestrator/app/database.py`
**Content**:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import os
import logging

logger = logging.getLogger(__name__)

# Database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://tradingai:password@postgres:5432/trading_education')

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=20,           # Connection pool size
    max_overflow=40,        # Max overflow connections
    pool_pre_ping=True,     # Verify connection before use
    pool_recycle=3600,      # Recycle connections after 1 hour
    echo=False,             # Set to True for SQL logging
    connect_args={
        "options": "-c timezone=utc"
    }
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for models
Base = declarative_base()

def get_db() -> Session:
    """
    Get database session with automatic cleanup
    
    Yields:
        Session: Database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()
```

---

### Task 3.3: Create Database Initialization Script

**Action**: Create file
**Path**: `orchestrator/scripts/init_database.py`
**Content**:

```python
#!/usr/bin/env python
"""
Database initialization script
Creates all tables defined in models
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, '/app')

from app.models import Base
from app.database import engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    """
    Initialize database with all tables
    
    Creates tables from SQLAlchemy models if they don't exist
    """
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables created successfully")
        
        # List created tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logger.info(f"✓ Tables created: {', '.join(tables)}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        return False

def seed_data():
    """
    Insert initial/test data (optional)
    """
    try:
        from app.database import SessionLocal
        db = SessionLocal()
        
        # Add seed data if needed
        # Example: Create default categories, etc.
        
        db.commit()
        logger.info("✓ Seed data inserted successfully")
    
    except Exception as e:
        logger.error(f"❌ Seed data failed: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    success = init_database()
    
    # Optionally seed data
    # seed_data()
    
    sys.exit(0 if success else 1)
```

---

### Task 3.4: Create Entrypoint Script

**Action**: Create file
**Path**: `orchestrator/scripts/entrypoint.sh`
**Content**:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Starting Orchestrator Service"
echo "=========================================="

# Wait for postgres
echo "Waiting for PostgreSQL..."
while ! nc -z postgres 5432; do
  sleep 1
done
echo "✓ PostgreSQL is ready"

# Wait for rabbitmq
echo "Waiting for RabbitMQ..."
while ! nc -z rabbitmq 5672; do
  sleep 1
done
echo "✓ RabbitMQ is ready"

# Wait for redis
echo "Waiting for Redis..."
while ! nc -z redis 6379; do
  sleep 1
done
echo "✓ Redis is ready"

# Initialize database
echo "Initializing database..."
python scripts/init_database.py
echo "✓ Database initialized"

echo "=========================================="
echo "Starting Celery Worker"
echo "=========================================="

# Execute the main command
exec "$@"
```

**Important**: Make this file executable
**Command**:
```bash
chmod +x orchestrator/scripts/entrypoint.sh
```

---

### Task 3.5: Create Orchestrator Dockerfile

**Action**: Create file
**Path**: `orchestrator/Dockerfile`
**Content**:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    ffmpeg \
    libpq-dev \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy and make entrypoint executable
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command (can be overridden)
CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info"]
```

---

### Task 3.6: Update Orchestrator Requirements

**Action**: Create/Update file
**Path**: `orchestrator/requirements.txt`
**Content**:

```txt
# Core dependencies
celery[redis]==5.3.4
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
pydantic==2.5.0
python-dotenv==1.0.0

# HTTP client
requests==2.31.0
urllib3==2.1.0

# Data processing
numpy==1.24.3
pandas==2.1.3
scikit-learn==1.3.2

# Database migrations (optional)
alembic==1.12.1

# Logging
python-json-logger==2.0.7
```

---

## 🎯 TASK GROUP 4: UPDATE DOCKER CONFIGURATION

### Task 4.1: Update Docker Compose - Add Video Processor

**Action**: Update file
**Path**: `docker-compose.yml`
**Instruction**: Add this service definition to docker-compose.yml

**Service to add**:

```yaml
  # Video Processor Service
  video-processor:
    build:
      context: ./services/video-processor
      dockerfile: Dockerfile
    container_name: trading-video-processor
    volumes:
      - ./data:/data
    ports:
      - "8000:8000"
    networks:
      - trading-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

---

### Task 4.2: Update Docker Compose - Add ML Service

**Action**: Update file
**Path**: `docker-compose.yml`
**Instruction**: Add this service definition to docker-compose.yml

**Service to add**:

```yaml
  # ML Service
  ml-service:
    build:
      context: ./services/ml-service
      dockerfile: Dockerfile
    container_name: trading-ml-service
    ports:
      - "8002:8002"
    networks:
      - trading-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
```

---

### Task 4.3: Update Docker Compose - Update Orchestrator Worker

**Action**: Update file
**Path**: `docker-compose.yml`
**Instruction**: Update orchestrator-worker service with these additions

**Add to orchestrator-worker environment**:

```yaml
  orchestrator-worker:
    # ... existing configuration ...
    environment:
      # ... existing vars ...
      - VIDEO_PROCESSOR_URL=http://video-processor:8000
      - ML_SERVICE_URL=http://ml-service:8002
    depends_on:
      # ... existing dependencies ...
      - video-processor
      - ml-service
```

---

### Task 4.4: Update Environment Variables

**Action**: Update file
**Path**: `.env`
**Instruction**: Add these variables to .env file

**Variables to add/update**:

```bash
# Service URLs
VIDEO_PROCESSOR_URL=http://video-processor:8000
ML_SERVICE_URL=http://ml-service:8002
API_SERVICE_URL=http://api-service:8003
EMBEDDINGS_SERVICE_URL=http://embeddings-service:8004
BACKTEST_SERVICE_URL=http://backtesting-service:8001

# Processing directories
VIDEO_INPUT_DIR=/data/videos
CLIPS_OUTPUT_DIR=/data/processed/clips
FRAMES_OUTPUT_DIR=/data/processed/frames
FAISS_INDEX_DIR=/data/processed/faiss

# Feature engineering thresholds
MIN_SHARPE_RATIO=1.0
MIN_WIN_RATE_PERCENT=55
MAX_DRAWDOWN_PERCENT=25
FEATURE_CONFIDENCE_THRESHOLD=0.7
```

---

## 🎯 TASK GROUP 5: VALIDATION SCRIPTS

### Task 5.1: Create Health Check Script

**Action**: Create file
**Path**: `scripts/check_health.sh`
**Content**:

```bash
#!/bin/bash
# Health check script for all services

echo "=========================================="
echo "Health Check - All Services"
echo "=========================================="

services=(
  "video-processor:8000"
  "ml-service:8002"
  "api-service:8003"
  "embeddings-service:8004"
  "backtesting-service:8001"
)

all_healthy=true

for service in "${services[@]}"; do
  IFS=':' read -r name port <<< "$service"
  
  status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${port}/health 2>/dev/null)
  
  if [ "$status" = "200" ]; then
    echo "✅ ${name} is healthy (HTTP 200)"
  else
    echo "❌ ${name} is unhealthy (HTTP ${status})"
    all_healthy=false
  fi
done

echo "=========================================="

if [ "$all_healthy" = true ]; then
  echo "✅ All services are healthy"
  exit 0
else
  echo "❌ Some services are unhealthy"
  exit 1
fi
```

**Make executable**:
```bash
chmod +x scripts/check_health.sh
```

---

### Task 5.2: Create Quick Diagnostic Script

**Action**: Create file
**Path**: `scripts/quick_diagnostic.sh`
**Content**:

```bash
#!/bin/bash
# Quick diagnostic script to check repository state

echo "=========================================="
echo "Repository Quick Diagnostic"
echo "=========================================="

# Check if critical services exist
echo -e "\n1. Checking critical services..."
[ -f "services/video-processor/app/main.py" ] && echo "  ✅ Video Processor EXISTS" || echo "  ❌ Video Processor MISSING"
[ -f "services/ml-service/app/main.py" ] && echo "  ✅ ML Service EXISTS" || echo "  ❌ ML Service MISSING"

# Check orchestrator tasks completeness
echo -e "\n2. Checking orchestrator tasks..."
if [ -f "orchestrator/app/tasks.py" ]; then
    pass_count=$(grep -c "^\s*pass\s*$" orchestrator/app/tasks.py 2>/dev/null || echo "0")
    echo "  Tasks with 'pass' (incomplete): $pass_count"
    [ "$pass_count" -gt 5 ] && echo "  ⚠️  Many tasks incomplete" || echo "  ✅ Most tasks implemented"
else
    echo "  ❌ tasks.py not found"
fi

# Check docker-compose services
echo -e "\n3. Checking docker-compose..."
if [ -f "docker-compose.yml" ]; then
    service_count=$(grep -c "container_name:" docker-compose.yml 2>/dev/null || echo "0")
    echo "  Services defined: $service_count"
    [ "$service_count" -ge 9 ] && echo "  ✅ All services present" || echo "  ⚠️  Missing services (expected 9-11)"
    
    # Check for specific services
    grep -q "video-processor:" docker-compose.yml && echo "  ✅ video-processor found" || echo "  ❌ video-processor missing"
    grep -q "ml-service:" docker-compose.yml && echo "  ✅ ml-service found" || echo "  ❌ ml-service missing"
else
    echo "  ❌ docker-compose.yml not found"
fi

# Check database models
echo -e "\n4. Checking database models..."
if [ -f "orchestrator/app/models.py" ]; then
    model_count=$(grep -c "class.*Base" orchestrator/app/models.py 2>/dev/null || echo "0")
    echo "  Models defined: $model_count"
    [ "$model_count" -ge 6 ] && echo "  ✅ All models present" || echo "  ⚠️  Missing models (expected 6)"
else
    echo "  ❌ models.py not found"
fi

# Check critical environment variables
echo -e "\n5. Checking environment variables..."
if [ -f ".env" ]; then
    grep -q "VIDEO_PROCESSOR_URL" .env && echo "  ✅ VIDEO_PROCESSOR_URL set" || echo "  ❌ VIDEO_PROCESSOR_URL missing"
    grep -q "ML_SERVICE_URL" .env && echo "  ✅ ML_SERVICE_URL set" || echo "  ❌ ML_SERVICE_URL missing"
    grep -q "DATABASE_URL" .env && echo "  ✅ DATABASE_URL set" || echo "  ❌ DATABASE_URL missing"
else
    echo "  ❌ .env file not found"
fi

echo -e "\n=========================================="
echo "DIAGNOSTIC COMPLETE"
echo "=========================================="
```

**Make executable**:
```bash
chmod +x scripts/quick_diagnostic.sh
```

---

## ✅ EXECUTION VALIDATION

### After completing all tasks, run these commands to validate:

```bash
# 1. Run quick diagnostic
./scripts/quick_diagnostic.sh

# 2. Build all services
docker-compose build

# 3. Start infrastructure
docker-compose up -d postgres redis rabbitmq

# Wait 15 seconds
sleep 15

# 4. Initialize database
docker-compose run orchestrator-worker python scripts/init_database.py

# 5. Start all services
docker-compose up -d

# Wait for startup
sleep 30

# 6. Run health checks
./scripts/check_health.sh

# 7. Check all services running
docker-compose ps

# 8. Check logs for errors
docker-compose logs --tail=50
```

---

## 🎯 SUCCESS CRITERIA

**System is ready when ALL of these pass:**

1. ✅ `./scripts/quick_diagnostic.sh` shows all ✅
2. ✅ `docker-compose build` completes without errors
3. ✅ `docker-compose ps` shows all services running
4. ✅ `./scripts/check_health.sh` shows all services healthy
5. ✅ Database tables created (6 tables expected)
6. ✅ No error messages in `docker-compose logs`

---

## 📝 NOTES FOR GEMINI CLI

**File Creation Order**:
1. Create directories first
2. Create Python files
3. Create Dockerfiles
4. Create requirements.txt files
5. Create shell scripts (and make executable)
6. Update existing configuration files

**Important Reminders**:
- All shell scripts need execute permissions (`chmod +x`)
- Entrypoint script must use LF line endings (not CRLF)
- Python files should use UTF-8 encoding
- Docker files should not have file extensions

**Verification After Each Task Group**:
- Task Group 1: `docker build -t video-processor:test services/video-processor`
- Task Group 2: `docker build -t ml-service:test services/ml-service`
- Task Group 3: `docker build -t orchestrator:test orchestrator`
- Task Group 4: `docker-compose config` (validates docker-compose.yml)
- Task Group 5: Execute scripts to verify they work

**End of Implementation Guide**