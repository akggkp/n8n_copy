"# Trading AI Video Processing Pipeline
![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/akggkp/n8n_copy?utm_source=oss&utm_medium=github&utm_campaign=akggkp%2Fn8n_copy&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
## Overview

Automated pipeline for extracting trading strategies from educational videos using:
- **Video Processing**: Frame extraction, chart detection (YOLO cascade), audio transcription (Whisper)
- **ML Concept Extraction**: Trading concepts, indicators, patterns from transcriptions
- **Strategy Generation**: LLaMA-powered strategy synthesis
- **Backtesting**: Historical validation of strategies
- **Orchestration**: Python + Celery (replaces n8n)

## Architecture

```
Videos → [Python Orchestrator] → Processing Pipeline → Profitable Strategies
           ↓
    File Watcher / API / Scheduled
           ↓
    Celery Task Queue (RabbitMQ)
           ↓
    [Video Processor] → [ML Service] → [LLaMA] → [Backtester] → [Database]
```

## Key Features

✅ **Python Orchestration** - Replaced n8n with Celery + RabbitMQ  
✅ **Auto-processing** - File watcher for new videos  
✅ **REST API** - Trigger processing via HTTP  
✅ **Scheduled Batch** - Process pending videos daily (2 AM)  
✅ **Web Dashboard** - Monitor pipeline at `http://localhost:8080`  
✅ **Memory Optimized** - Designed for HP Victus (16GB RAM)  
✅ **Cascade Detection** - Intelligent YOLO model switching  
✅ **LLaMA Integration** - Local Ollama for strategy generation  

## Project Structure

```
/app/
├── orchestrator/          # Python orchestrator (replaces n8n)
│   ├── app/
│   │   ├── celery_app.py       # Celery configuration
│   │   ├── tasks.py            # Pipeline tasks (7 steps)
│   │   ├── file_watcher.py     # Auto-process new videos
│   │   └── web/                # Flask dashboard
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
├── services/
│   ├── video-processor/   # Frame extraction, chart detection
│   ├── ml-service/        # Concept extraction
│   ├── backtesting-service/  # Strategy validation
│   ├── database/          # PostgreSQL migrations
│   └── memory-cleaner/    # Memory optimization
├── data/
│   ├── videos/            # Input videos (watched directory)
│   ├── processed/         # Intermediate outputs
│   ├── logs/              # Application logs
│   └── models/            # ML models
├── docker-compose.yml     # Service orchestration
├── init-db.sql            # Database schema
└── README.md              # This file
```

## Quick Start

### 1. Prerequisites

- Docker & Docker Compose
- 16GB+ RAM (32GB recommended)
- NVIDIA GPU (optional, for faster processing)

### 2. Setup

```bash
# Clone or navigate to project
cd /app

# Copy environment template
cp .env.template .env

# Edit .env with your passwords
nano .env

# Create data directories
mkdir -p data/videos data/processed data/logs data/models
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f orchestrator-worker
```

### 4. Load LLaMA Model

```bash
# Enter Ollama container
docker exec -it trading-ollama bash

# Pull LLaMA 2
ollama pull llama2

# Test
ollama run llama2 \"What is RSI?\"

# Exit
exit
```

### 5. Access Dashboard

Open browser: **http://localhost:8080**

## Usage

### Method 1: File Watcher (Automatic)

```bash
# Copy video to watched directory
cp your_video.mp4 ./data/videos/

# Pipeline automatically triggers
# Check logs:
docker logs trading-orchestrator-watcher -f
```

### Method 2: REST API (Manual)

```bash
curl -X POST http://localhost:8080/api/trigger-processing \
  -H \"Content-Type: application/json\" \
  -d '{
    \"video_id\": \"test_001\",
    \"file_path\": \"/data/videos/test.mp4\",
    \"filename\": \"test.mp4\"
  }'
```

### Method 3: Scheduled Batch (Automatic)

- Runs daily at 2 AM (default)
- Processes pending videos from database
- Configure in `.env`:
  ```bash
  BATCH_SCHEDULE_HOUR=2
  BATCH_SIZE=10
  ```

## Pipeline Flow

```
1. validate_video       → Check file exists, register in DB
2. process_video        → Extract frames, detect charts, transcribe
3. extract_concepts     → ML extracts trading concepts
4. generate_strategy    → LLaMA creates strategy
5. backtest_strategy    → Validate on historical data
6. evaluate_and_save    → Save profitable, delete unprofitable
7. notify_completion    → Log results, update dashboard
```

**Estimated Time**: 3-5 minutes per video

## Services

| Service | Port | Purpose |
|---------|------|---------|
| **orchestrator-web** | 8080 | Web dashboard |
| **orchestrator-worker** | - | Celery task execution |
| **orchestrator-beat** | - | Scheduled jobs |
| **orchestrator-watcher** | - | File monitoring |
| **video-processor** | 8000 | Video processing API |
| **ml-service** | 8002 | Concept extraction |
| **backtesting-service** | 8001 | Strategy validation |
| **ollama** | 11434 | LLaMA inference |
| **postgres** | 5432 | Database |
| **redis** | 6379 | Celery backend |
| **rabbitmq** | 5672, 15672 | Message queue |

## Monitoring

### Dashboard

**http://localhost:8080**
- Real-time statistics
- Recent videos
- Profitable strategies
- Manual trigger form

### Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker logs trading-orchestrator-worker -f

# Database queries
docker exec -it trading-postgres psql -U tradingai -d trading_education

# View recent videos
SELECT video_id, status, processed_at FROM processed_videos ORDER BY created_at DESC LIMIT 10;

# View profitable strategies
SELECT * FROM proven_strategies;
```

### Celery Tasks

```bash
# Active tasks
docker exec trading-orchestrator-worker celery -A app.celery_app inspect active

# Scheduled tasks
docker exec trading-orchestrator-worker celery -A app.celery_app inspect scheduled
```

## Configuration

Edit `.env` file:

```bash
# Video Processing
USE_CASCADE=true              # Enable cascade detection
CONFIDENCE_THRESHOLD=0.65     # Cascade trigger threshold

# File Watcher
ENABLE_FILE_WATCHER=true      # Auto-process new videos

# Batch Processing
BATCH_SCHEDULE_HOUR=2         # Hour to run (24h format)
BATCH_SIZE=10                 # Videos per batch

# Backtesting Thresholds
MIN_WIN_RATE_TO_SAVE=55       # Minimum win rate (%)
MIN_PROFIT_FACTOR=1.5         # Minimum profit factor
```

## Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose logs

# Rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Tasks not processing

```bash
# Check RabbitMQ
docker logs trading-rabbitmq

# Check worker
docker logs trading-orchestrator-worker -f

# Purge queue (caution: deletes pending tasks)
docker exec trading-orchestrator-worker celery -A app.celery_app purge
```

### Database connection errors

```bash
# Verify PostgreSQL
docker logs trading-postgres

# Test connection
docker exec -it trading-postgres psql -U tradingai -d trading_education -c \"SELECT 1;\"

# Restart
docker-compose restart postgres
```

### Out of memory

```bash
# Reduce concurrency (docker-compose.yml)
command: celery -A app.celery_app worker --loglevel=info --concurrency=1

# Disable file watcher
ENABLE_FILE_WATCHER=false

# Restart
docker-compose restart orchestrator-worker
```

## Performance Tips

### For 16GB RAM
```yaml
orchestrator-worker:
  mem_limit: 512m
  command: celery -A app.celery_app worker --concurrency=2
```

### For 32GB RAM
```yaml
orchestrator-worker:
  mem_limit: 1g
  command: celery -A app.celery_app worker --concurrency=4
```

## Development

### Run Locally (without Docker)

```bash
cd orchestrator

# Install dependencies
pip install -r requirements.txt

# Set environment
export DATABASE_URL=postgresql://...
export CELERY_BROKER_URL=amqp://...

# Run components (separate terminals)
python -m app.web.app                                    # Web UI
celery -A app.celery_app worker --loglevel=info         # Worker
celery -A app.celery_app beat --loglevel=info           # Scheduler
python -m app.file_watcher                               # Watcher
```

## Migration from n8n

If you had n8n previously:

```bash
# 1. Backup n8n workflows
docker exec trading-n8n n8n export:workflow --all

# 2. Stop n8n
docker-compose stop n8n

# 3. Remove n8n volume
docker volume rm trading-ai_n8n_data

# 4. Start Python orchestrator (already configured in docker-compose.yml)
docker-compose up -d orchestrator-web orchestrator-worker orchestrator-beat

# 5. Verify
http://localhost:8080
```

## Documentation

- **[Python Orchestration Setup](./PYTHON_ORCHESTRATION_SETUP.md)** - Detailed setup guide
- **[Orchestrator README](./orchestrator/README.md)** - Component details
- **[Cascade Implementation](./cascade-implementation.md)** - YOLO cascade approach

## Support

For issues:
1. Check logs: `docker-compose logs`
2. Verify services: `docker-compose ps`
3. Test individually: `docker exec -it <container> bash`
4. Review configuration: `.env`

## License

MIT License

## Contributors

Trading AI Team

---

**Status**: ✅ Production Ready
**Last Updated**: 2025
**Version**: 1.0.0
"
