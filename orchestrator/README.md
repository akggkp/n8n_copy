# Trading AI Orchestrator

## Python-based Orchestration (Replaces n8n)

This service replaces n8n with a pure Python orchestration solution using **Celery + RabbitMQ**.

## Architecture

```
Video Upload → Celery Tasks → Processing Pipeline → Results
      ↓              ↓                  ↓              ↓
  File Watch    Task Queue        Microservices    Database
```

## Components

### 1. **Celery Worker** (`orchestrator-worker`)
- Executes pipeline tasks
- Processes videos through 7-step pipeline
- Handles retries and error recovery
- Memory-optimized for HP Victus

### 2. **Celery Beat** (`orchestrator-beat`)
- Scheduled batch processing (default: 2 AM daily)
- Cleanup old results
- Periodic health checks

### 3. **File Watcher** (`orchestrator-watcher`)
- Monitors `/data/videos` for new files
- Auto-triggers processing pipeline
- Supports: .mp4, .avi, .mov, .mkv

### 4. **Web UI** (`orchestrator-web`)
- Real-time dashboard at `http://localhost:8080`
- View statistics, videos, strategies
- Manually trigger processing
- Monitor Celery tasks

## Pipeline Flow

```
1. validate_video       → Check file, register in DB
2. process_video        → Extract frames, detect charts, transcribe
3. extract_concepts     → ML service extracts trading concepts  
4. generate_strategy    → LLaMA generates strategy from concepts
5. backtest_strategy    → Backtest on historical data
6. evaluate_and_save    → Save profitable, delete unprofitable
7. notify_completion    → Log results, update UI
```

## Usage

### Start All Services
```bash
docker-compose up -d
```

### View Logs
```bash
# Web UI
docker logs trading-orchestrator-web -f

# Celery Worker
docker logs trading-orchestrator-worker -f

# File Watcher
docker logs trading-orchestrator-watcher -f
```

### Access Dashboard
```
http://localhost:8080
```

### Manual Processing (API)
```bash
curl -X POST http://localhost:8080/api/trigger-processing \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "test_video_001",
    "file_path": "/data/videos/sample.mp4",
    "filename": "sample.mp4"
  }'
```

### Auto-Processing (File Watcher)
```bash
# Just copy video to watched directory
cp your_video.mp4 ./data/videos/

# Pipeline will auto-trigger
```

### Scheduled Batch Processing
- Runs daily at 2 AM (configurable via `BATCH_SCHEDULE_HOUR`)
- Processes up to 10 videos per batch (configurable via `BATCH_SIZE`)
- Automatically picks up pending videos from database

## Configuration

Edit `.env` file:

```bash
# Enable/disable file watcher
ENABLE_FILE_WATCHER=true

# Batch processing time (hour in 24h format)
BATCH_SCHEDULE_HOUR=2

# Number of videos per batch
BATCH_SIZE=10

# Cascade detection
USE_CASCADE=true
CONFIDENCE_THRESHOLD=0.65

# Backtesting thresholds
MIN_WIN_RATE_TO_SAVE=55
MIN_PROFIT_FACTOR=1.5
```

## Memory Optimization (HP Victus)

- **Celery concurrency**: 2 workers (configurable)
- **Worker max tasks**: 10 (restarts after 10 tasks)
- **Task timeout**: 1 hour per task
- **Memory limits** (docker-compose):
  - Web UI: 256 MB
  - Worker: 512 MB
  - Beat: 128 MB
  - Watcher: 128 MB

## Task Monitoring

### View Active Tasks
```bash
# Via Web UI
http://localhost:8080/api/celery-tasks

# Via CLI
docker exec trading-orchestrator-worker celery -A app.celery_app inspect active
```

### View Scheduled Tasks
```bash
docker exec trading-orchestrator-worker celery -A app.celery_app inspect scheduled
```

### Purge Queue (Clear all pending tasks)
```bash
docker exec trading-orchestrator-worker celery -A app.celery_app purge
```

## Comparison: n8n vs Python Orchestrator

| Feature | n8n | Python Orchestrator |
|---------|-----|--------------------|
| Setup Complexity | Medium | Low |
| Memory Usage | ~512 MB | ~900 MB total |
| Customization | Visual workflow | Full code control |
| Debugging | Limited | Full Python debugging |
| Performance | Good | Excellent |
| Scalability | Limited | Horizontal scaling |
| File Watching | Plugin required | Built-in |
| Scheduled Jobs | Built-in | Celery Beat |
| Web UI | Full-featured | Simple dashboard |

## Troubleshooting

### Tasks not executing
```bash
# Check RabbitMQ
docker logs trading-rabbitmq

# Check Celery worker
docker logs trading-orchestrator-worker -f

# Verify RabbitMQ queues
docker exec trading-rabbitmq rabbitmqctl list_queues
```

### File watcher not working
```bash
# Check watcher logs
docker logs trading-orchestrator-watcher -f

# Verify directory permissions
ls -la ./data/videos

# Test manual trigger
curl -X POST http://localhost:8080/api/trigger-processing ...
```

### Database connection errors
```bash
# Check PostgreSQL
docker exec -it trading-postgres psql -U tradingai -d trading_education

# Verify tables
\dt

# Check processed videos
SELECT video_id, status FROM processed_videos ORDER BY created_at DESC LIMIT 10;
```

## Development

### Run locally (without Docker)
```bash
cd orchestrator

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL=postgresql://...
export CELERY_BROKER_URL=amqp://...

# Run web UI
python -m app.web.app

# Run worker (separate terminal)
celery -A app.celery_app worker --loglevel=info

# Run beat (separate terminal)
celery -A app.celery_app beat --loglevel=info

# Run file watcher (separate terminal)
python -m app.file_watcher
```

### Add new tasks

Edit `app/tasks.py`:
```python
@celery_app.task(bind=True, name='app.tasks.my_custom_task')
def my_custom_task(self, data):
    # Your task logic
    pass
```

## Performance Tips

1. **Increase concurrency** (if you have more RAM):
   ```yaml
   command: celery -A app.celery_app worker --loglevel=info --concurrency=4
   ```

2. **Disable file watcher** (if only using API/scheduled):
   ```bash
   ENABLE_FILE_WATCHER=false
   ```

3. **Adjust batch size** (for faster processing):
   ```bash
   BATCH_SIZE=20
   ```

4. **Use Redis for results** (faster than RabbitMQ):
   ```bash
   CELERY_RESULT_BACKEND=redis://:password@redis:6379/0
   ```