#!/bin/bash

# Production startup script for AI Music Generation Backend

set -e

# Load environment variables
if [ -f .env.production ]; then
    export $(grep -v '^#' .env.production | xargs)
fi

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Create upload directories
mkdir -p /tmp/uploads /tmp/models

# Start the application with gunicorn
echo "Starting AI Music Generation Backend..."
exec gunicorn \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers ${WORKERS:-4} \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 300 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    main:app
