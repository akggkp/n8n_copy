# STEP_3_INTEGRATION_GUIDE.md
# Step 3: FastAPI Data Access Layer - Integration Guide

## Overview

This guide walks through integrating the FastAPI data access layer into your project.

---

## File Structure

```
services/
├── api-service/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── schemas.py           # Pydantic models
│   │   ├── models.py            # SQLAlchemy models (optional, shared with orchestrator)
│   │   ├── database.py          # Database session management
│   │   └── routes/              # Endpoint modules (optional)
│   │       ├── __init__.py
│   │       ├── media.py
│   │       ├── clips.py
│   │       ├── transcripts.py
│   │       ├── embeddings.py
│   │       └── llama.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
```

---

## Step-by-Step Integration

### 1. Create Directory Structure

PowerShell:
```powershell
$dirs = @(
    "services/api-service/app/routes",
    "services/api-service/tests"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

$files = @(
    "services/api-service/__init__.py",
    "services/api-service/app/__init__.py",
    "services/api-service/app/routes/__init__.py"
)

foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
    }
}
```

### 2. Copy Files from Generated Artifacts

```powershell
# Copy main API application
Copy-Item -Path "api_main.py" -Destination "services/api-service/app/main.py"

# Copy schemas
Copy-Item -Path "schemas.py" -Destination "services/api-service/app/schemas.py"

# Copy requirements
Copy-Item -Path "api_requirements.txt" -Destination "services/api-service/requirements.txt"

# Copy Dockerfile
Copy-Item -Path "api_dockerfile" -Destination "services/api-service/Dockerfile"
```

### 3. Create Database Models (Shared with Orchestrator)

Create `services/api-service/app/models.py`:

```python
# services/api-service/app/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class MediaItem(Base):
    __tablename__ = "media_items"
    
    id = Column(Integer, primary_key=True)
    video_id = Column(String(255), unique=True, nullable=False)
    source_url = Column(Text)
    filename = Column(String(255), nullable=False)
    duration_seconds = Column(Float)
    file_size_bytes = Column(Integer)
    status = Column(String(50), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    transcripts = relationship("Transcript", back_populates="media_item", cascade="all, delete-orphan")
    keyword_hits = relationship("KeywordHit", back_populates="media_item", cascade="all, delete-orphan")
    frames = relationship("Frame", back_populates="media_item", cascade="all, delete-orphan")
    clips = relationship("Clip", back_populates="media_item", cascade="all, delete-orphan")
    embeddings = relationship("Embedding", back_populates="media_item", cascade="all, delete-orphan")

class Transcript(Base):
    __tablename__ = "transcripts"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"))
    segment_index = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    language = Column(String(10), default="en")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    media_item = relationship("MediaItem", back_populates="transcripts")

class KeywordHit(Base):
    __tablename__ = "keyword_hits"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"))
    keyword = Column(String(100), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float)
    confidence = Column(Float, default=1.0)
    context_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    media_item = relationship("MediaItem", back_populates="keyword_hits")
    clips = relationship("Clip", back_populates="keyword_hit")

class Clip(Base):
    __tablename__ = "clips"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"))
    keyword_hit_id = Column(Integer, ForeignKey("keyword_hits.id", ondelete="SET NULL"))
    keyword = Column(String(100))
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration_seconds = Column(Float)
    file_path = Column(Text, nullable=False)
    file_size_bytes = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    media_item = relationship("MediaItem", back_populates="clips")
    keyword_hit = relationship("KeywordHit", back_populates="clips")

class Frame(Base):
    __tablename__ = "frames"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"))
    timestamp = Column(Float, nullable=False)
    file_path = Column(Text, nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    contains_chart = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    media_item = relationship("MediaItem", back_populates="frames")

class Embedding(Base):
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"))
    embedding_type = Column(String(50))
    reference_id = Column(Integer)
    embedding_model = Column(String(100))
    embedding_vector = Column(ARRAY(Float))
    vector_dimension = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    media_item = relationship("MediaItem", back_populates="embeddings")

class StrategiesFeature(Base):
    __tablename__ = "strategies_features"
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("proven_strategies.id", ondelete="CASCADE"))
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="SET NULL"))
    feature_name = Column(String(255), nullable=False)
    feature_value = Column(Float, nullable=False)
    feature_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 4. Create Database Session Management

Create `services/api-service/app/database.py`:

```python
# services/api-service/app/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import os
from typing import Generator

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://tradingai:password@localhost:5432/trading_education")

engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true",
    pool_size=20,
    max_overflow=10
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables"""
    from app.models import Base
    Base.metadata.create_all(bind=engine)
```

### 5. Create Environment Configuration

Create `.env`:

```bash
# Database
DATABASE_URL=postgresql://tradingai:your_password@postgres:5432/trading_education

# API Server
API_PORT=8003
API_WORKERS=2
SQL_ECHO=false

# Service URLs
VIDEO_PROCESSOR_URL=http://video-processor:8000
ML_SERVICE_URL=http://ml-service:8002
BACKTEST_SERVICE_URL=http://backtesting-service:8001
EMBEDDINGS_SERVICE_URL=http://embeddings-service:8004
OLLAMA_URL=http://ollama:11434

# Celery
CELERY_BROKER_URL=amqp://guest:guest@rabbitmq:5672//
CELERY_RESULT_BACKEND=redis://redis:6379/0

# Processing
CLIPS_OUTPUT_DIR=/data/processed/clips
MIN_SHARPE_RATIO=1.0
MIN_WIN_RATE_PERCENT=55
```

### 6. Update docker-compose.yml

Add to your `docker-compose.yml`:

```yaml
services:
  # ... existing services ...
  
  api-service:
    build:
      context: ./services/api-service
      dockerfile: Dockerfile
    container_name: trading-api-service
    ports:
      - "8003:8003"
    environment:
      - DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/${DB_NAME}
      - VIDEO_PROCESSOR_URL=http://video-processor:8000
      - ML_SERVICE_URL=http://ml-service:8002
      - BACKTEST_SERVICE_URL=http://backtesting-service:8001
      - EMBEDDINGS_SERVICE_URL=http://embeddings-service:8004
      - OLLAMA_URL=http://ollama:11434
      - API_PORT=8003
      - API_WORKERS=2
    depends_on:
      - postgres
      - redis
      - video-processor
    volumes:
      - ./services/api-service:/app
      - ./data:/data
    networks:
      - trading-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8003/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
```

### 7. Build and Run

PowerShell:
```powershell
# Build API service
docker-compose build api-service

# Start API service
docker-compose up -d api-service

# Check logs
docker-compose logs -f api-service

# Test health endpoint
curl http://localhost:8003/health

# Access API docs
# Browser: http://localhost:8003/docs (Swagger UI)
# Browser: http://localhost:8003/redoc (ReDoc)
```

---

## Testing the API

### 1. Health Check
```bash
curl http://localhost:8003/health
```

### 2. List Media Items
```bash
curl "http://localhost:8003/media_items?limit=10"
```

### 3. Ingest a Video
```bash
curl -X POST http://localhost:8003/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/data/videos/sample.mp4",
    "filename": "sample.mp4"
  }'
```

### 4. Get Transcript
```bash
curl http://localhost:8003/transcript/1
```

### 5. Search Keywords
```bash
curl "http://localhost:8003/keywords?media_id=1&limit=10"
```

### 6. Get Clips
```bash
curl "http://localhost:8003/clips?keyword=RSI&limit=5"
```

### 7. Semantic Search
```bash
curl "http://localhost:8003/embeddings/search?query=relative%20strength%20index&top_k=5"
```

### 8. Llama Examples
```bash
curl "http://localhost:8003/llama/examples?keyword=RSI&top_k=5&include_embeddings=false"
```

---

## Database Integration Notes

1. **Shared Models**: Copy `models.py` to both orchestrator and API service, or use a shared package
2. **Connection String**: Ensure DATABASE_URL points to your PostgreSQL instance
3. **Migrations**: Use Alembic if schema changes are frequent
4. **Connection Pooling**: FastAPI auto-manages connection pools via SQLAlchemy
5. **Async Support**: Consider using `async-sqlalchemy` for full async support (optional)

---

## API Documentation

Full documentation and interactive testing available at:
- **Swagger UI**: http://localhost:8003/docs
- **ReDoc**: http://localhost:8003/redoc

---

## Troubleshooting

### API Service Won't Start

```powershell
# Check logs
docker-compose logs api-service

# Verify database connection
docker exec trading-api-service python -c "from app.database import engine; engine.connect()"

# Check if port 8003 is in use
netstat -ano | findstr :8003
```

### Database Connection Errors

```bash
# Test PostgreSQL connection
docker exec trading-postgres psql -U tradingai -d trading_education -c "SELECT 1;"

# Verify connection string in .env
cat .env | grep DATABASE_URL
```

### Module Import Errors

```bash
# Reinstall dependencies
docker-compose exec api-service pip install -r requirements.txt

# Clear Python cache
docker-compose exec api-service find . -type d -name __pycache__ -exec rm -r {} +
```

---

## Next Steps

After Step 3 is complete:
1. Implement database integration in main.py endpoints
2. Create optional modular route files (routes/media.py, routes/clips.py, etc.)
3. Add database session dependency injection to endpoints
4. Test all endpoints with sample data
5. Proceed to **Step 4: Embeddings & Vector Index**