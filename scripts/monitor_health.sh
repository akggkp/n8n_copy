#!/bin/bash
# Health monitoring script

services=("api-service:8003" "embeddings-service:8004" "backtesting-service:8001")

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${port}/health)
    
    if [ $response -eq 200 ]; then
        echo "✓ ${name} is healthy"
    else
        echo "✗ ${name} is unhealthy (HTTP ${response})"
        # Send alert (email, Slack, PagerDuty, etc.)
    fi
done
