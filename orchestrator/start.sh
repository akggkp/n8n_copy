#!/bin/bash

# Start script for orchestrator service

echo "Starting Trading AI Orchestrator..."

# Determine which component to start based on environment variable
COMPONENT=${COMPONENT:-"all"}

case $COMPONENT in
  "web")
    echo "Starting Flask Web UI..."
    python -m app.web.app
    ;;
  
  "celery-worker")
    echo "Starting Celery Worker..."
    celery -A app.celery_app worker --loglevel=info --concurrency=2
    ;;
  
  "celery-beat")
    echo "Starting Celery Beat (Scheduler)..."
    celery -A app.celery_app beat --loglevel=info
    ;;
  
  "file-watcher")
    echo "Starting File Watcher..."
    python -m app.file_watcher
    ;;
  
  "all")
    echo "Starting all components..."
    # This is for development only
    python -m app.web.app &
    celery -A app.celery_app worker --loglevel=info --concurrency=2 &
    celery -A app.celery_app beat --loglevel=info &
    python -m app.file_watcher &
    wait
    ;;
  
  *)
    echo "Unknown component: $COMPONENT"
    echo "Valid options: web, celery-worker, celery-beat, file-watcher, all"
    exit 1
    ;;
esac