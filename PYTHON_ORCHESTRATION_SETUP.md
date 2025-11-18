# Python Orchestration Setup Guide

## Overview

This guide helps you set up the Python-based orchestration system that **replaces n8n** for your trading AI video processing pipeline.

## What's Changed?

### Before (n8n)
- Visual workflow editor
- Limited customization
- Single orchestrator service
- Webhook-based triggers

### After (Python + Celery)
- Full Python control
- Highly customizable
- Distributed task queue
- Multiple trigger methods (API, file watch, scheduled)
- Simple web dashboard

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VIDEO INPUT                             │
├─────────────────────────────────────────────────────────────┤
│  File Watcher  │  REST API  │  Scheduled Batch (2 AM)      │
└──────┬───────────────┬──────────────────┬──────────────────┘
       │               │                   │
       v               v                   v
┌─────────────────────────────────────────────────────────────┐
│              CELERY TASK QUEUE (RabbitMQ)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│                   CELERY WORKER                             │
├─────────────────────────────────────────────────────────────┤
│  Task 1: Validate   │  Task 5: Backtest                     │
│  Task 2: Process    │  Task 6: Evaluate                     │
│  Task 3: Extract    │  Task 7: Notify                       │
│  Task 4: Strategy   │                                       │
└──────┬──────────────┴───────────────┬────────────────────────┘
       │                              │
       v                              v
┌────────────────┐          ┌────────────────────┐
│  Microservices │          │  PostgreSQL + Redis│
├────────────────┤          ├────────────────────┤
│ Video Processor│          │ Results Storage    │
│ ML Service     │          │ Task State         │
│ Backtesting    │          │ Profitable Strats  │
│ Ollama (LLaMA) │          └────────────────────┘
└────────────────┘
       │
       v
┌─────────────────────────────────────────────────────────────┐
│              WEB DASHBOARD (Flask)                          │
│  http://localhost:8080                                      │
└─────────────────────────────────────────────────────────────┘
```

## Setup Steps

### 1. Prerequisites

- Docker & Docker Compose installed
- HP Victus with AMD Ryzen 5000 series
- At least 16GB RAM
- GPU with CUDA support (for Ollama/YOLO)

### 2. Configure Environment

```bash
# Copy template
cp .env.template .env

# Edit .env file with your passwords
nano .env
```

Key configurations:
```bash
# Database
POSTGRES_USER=tradingai
POSTGRES_PASSWORD=YourSecurePassword123!

# Redis
REDIS_PASSWORD=YourRedisPassword123!

# RabbitMQ
RABBITMQ_DEFAULT_USER=tradingai
RABBITMQ_DEFAULT_PASS=YourRabbitPassword123!

# Features
ENABLE_FILE_WATCHER=true
USE_CASCADE=true
CONFIDENCE_THRESHOLD=0.65
```

### 3. Create Data Directories

```bash
mkdir -p data/videos
mkdir -p data/processed
mkdir -p data/logs
mkdir -p data/models
```

### 4. Build and Start Services

```bash
# Build all services
docker-compose build

# Start infrastructure first
docker-compose up -d postgres redis rabbitmq

# Wait 10 seconds for databases to initialize
sleep 10

# Start orchestrator services
docker-compose up -d orchestrator-web orchestrator-worker orchestrator-beat orchestrator-watcher

# Start processing services
docker-compose up -d video-processor backtesting-service ollama

# Start ML service (if available)
docker-compose up -d ml-service
```

### 5. Verify Services

```bash
# Check all services are running
docker-compose ps

# Should see:
# - trading-postgres (healthy)
# - trading-redis (running)
# - trading-rabbitmq (running)
# - trading-orchestrator-web (running)
# - trading-orchestrator-worker (running)
# - trading-orchestrator-beat (running)
# - trading-orchestrator-watcher (running)
# - trading-video-processor (running)
# - trading-backtesting (running)
# - trading-ollama (running)
```

### 6. Load LLaMA Model

```bash
# Enter Ollama container
docker exec -it trading-ollama bash

# Pull LLaMA 2 model
ollama pull llama2

# Test
ollama run llama2 "What is RSI indicator?"

# Exit
exit
```

### 7. Access Web Dashboard

Open browser: **http://localhost:8080**

You should see:
- Total Videos: 0
- Completed: 0
- Processing: 0
- Profitable Strategies: 0

## Usage Examples

### Method 1: File Watcher (Automatic)

```bash
# Simply copy a video to the watched directory
cp your_trading_video.mp4 ./data/videos/

# Watch logs
docker logs trading-orchestrator-watcher -f

# You'll see:
# 🎹 New video detected: your_trading_video.mp4
# 🚀 Starting pipeline for: video_abc123
# ✓ Pipeline triggered
```

### Method 2: REST API (Manual)

```bash
curl -X POST http://localhost:8080/api/trigger-processing \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "manual_video_001",
    "file_path": "/data/videos/trading_tutorial.mp4",
    "filename": "trading_tutorial.mp4"
  }'

# Response:
# {
#   "status": "success",
#   "task_id": "abc-123-def-456",
#   "video_id": "manual_video_001"
# }
```

### Method 3: Scheduled Batch (Automatic)

- Runs daily at 2 AM (default)
- Processes pending videos from database
- Configure via `.env`:

```bash
BATCH_SCHEDULE_HOUR=2  # 2 AM
BATCH_SIZE=10          # Process 10 videos max
```

## Monitoring

### Real-time Dashboard

**http://localhost:8080**

Features:
- Live statistics
- Recent videos list
- Profitable strategies
- Manual trigger form
- Auto-refresh every 30 seconds

### Logs

```bash
# Orchestrator web UI
docker logs trading-orchestrator-web -f

# Celery worker (task execution)
docker logs trading-orchestrator-worker -f

# File watcher
docker logs trading-orchestrator-watcher -f

# Video processor
docker logs trading-video-processor -f

# Backtesting service
docker logs trading-backtesting -f
```

### Database Queries

```bash
# Connect to database
docker exec -it trading-postgres psql -U tradingai -d trading_education

# View recent videos
SELECT video_id, filename, status, processed_at 
FROM processed_videos 
ORDER BY created_at DESC 
LIMIT 10;

# View profitable strategies
SELECT video_id, strategy_name, backtest_results 
FROM proven_strategies 
ORDER BY created_at DESC;

# View cascade statistics
SELECT * FROM cascade_statistics LIMIT 5;

# Exit
\q
```

## Testing

### Test End-to-End Pipeline

```bash
# 1. Download a sample trading video
wget -O ./data/videos/test_video.mp4 "https://sample-videos.com/trading_sample.mp4"

# 2. Watch orchestrator worker logs
docker logs trading-orchestrator-worker -f

# 3. File watcher will auto-trigger processing

# 4. Watch pipeline progress (7 tasks):
# [Task 1/7] Validating video
# [Task 2/7] Processing video
# [Task 3/7] Extracting concepts
# [Task 4/7] Generating strategy
# [Task 5/7] Backtesting strategy
# [Task 6/7] Evaluating results
# [Task 7/7] Pipeline complete

# 5. Check dashboard for results
# http://localhost:8080
```

## Troubleshooting

### Issue: Services won't start

```bash
# Check Docker resources
docker system df

# Clean up unused resources
docker system prune -a

# Rebuild
docker-compose build --no-cache
docker-compose up -d
```

### Issue: Tasks stuck in queue

```bash
# Check RabbitMQ
docker logs trading-rabbitmq

# Check worker status
docker exec trading-orchestrator-worker celery -A app.celery_app inspect active

# Purge queue (CAUTION: deletes all pending tasks)
docker exec trading-orchestrator-worker celery -A app.celery_app purge
```

### Issue: Database connection errors

```bash
# Verify PostgreSQL is running
docker logs trading-postgres

# Test connection
docker exec -it trading-postgres psql -U tradingai -d trading_education -c "SELECT 1;"

# If connection fails, restart PostgreSQL
docker-compose restart postgres
```

### Issue: Out of memory errors

```bash
# Reduce Celery concurrency (in docker-compose.yml)
command: celery -A app.celery_app worker --loglevel=info --concurrency=1

# Disable file watcher if not needed
ENABLE_FILE_WATCHER=false

# Restart services
docker-compose restart orchestrator-worker
```

## Performance Optimization

### For 16GB RAM

```yaml
# docker-compose.yml
orchestrator-worker:
  mem_limit: 512m
  command: celery -A app.celery_app worker --loglevel=info --concurrency=2
```

### For 32GB RAM

```yaml
orchestrator-worker:
  mem_limit: 1g
  command: celery -A app.celery_app worker --loglevel=info --concurrency=4
```

## Migration from n8n

If you had n8n running:

```bash
# 1. Export n8n workflows (backup)
docker exec trading-n8n n8n export:workflow --all

# 2. Stop n8n
docker-compose stop n8n

# 3. Remove n8n service from docker-compose.yml (already done)

# 4. Start Python orchestrator
docker-compose up -d orchestrator-web orchestrator-worker orchestrator-beat

# 5. Verify migration
http://localhost:8080
```

## Maintenance

### Daily Tasks
- Monitor dashboard for failed tasks
- Check disk space in `/data/videos`
- Review profitable strategies

### Weekly Tasks
- Review logs for errors
- Clean up old processed videos
- Optimize database (VACUUM)

### Monthly Tasks
- Update Docker images
- Review and adjust thresholds
- Backup database

## Next Steps

1. ✅ Set up orchestrator
2. ✅ Configure environment
3. ✅ Test with sample video
4. 🔄 Add your video library
5. 🔄 Monitor profitable strategies
6. 🔄 Deploy strategies to live trading

## Support

For issues:
1. Check logs: `docker-compose logs`
2. Verify services: `docker-compose ps`
3. Test components individually
4. Review configuration in `.env`

---

**Congratulations!** You've successfully replaced n8n with a pure Python orchestration system. 🎉