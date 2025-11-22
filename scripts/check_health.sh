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
