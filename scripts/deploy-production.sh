#!/bin/bash
# Production deployment script

set -e

echo "========================================"
echo "Trading Education Platform Deployment"
echo "========================================"

# Load environment variables
if [ -f .env.production ]; then
    export $(cat .env.production | xargs)
else
    echo "Error: .env.production not found"
    exit 1
fi

# Backup database
echo "1. Backing up database..."
docker exec trading-postgres-prod pg_dump -U ${DB_USER} ${DB_NAME} > backup_$(date +%Y%m%d_%H%M%S).sql

# Pull latest images
echo "2. Pulling latest Docker images..."
docker-compose -f docker-compose.prod.yml pull

# Stop services
echo "3. Stopping services..."
docker-compose -f docker-compose.prod.yml down

# Start services
echo "4. Starting services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo "5. Waiting for services to be healthy..."
sleep 30

# Health checks
echo "6. Running health checks..."
curl -f http://localhost:8003/health || echo "API service health check failed"
curl -f http://localhost:8004/health || echo "Embeddings service health check failed"
curl -f http://localhost:8001/health || echo "Backtest service health check failed"

echo "========================================"
echo "Deployment Complete!"
echo "========================================"
