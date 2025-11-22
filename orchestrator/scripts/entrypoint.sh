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
