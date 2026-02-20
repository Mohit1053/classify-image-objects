#!/bin/bash

# Start script for Image Classification API
# This script can be used to start the application manually or through systemd

set -e

# Configuration
APP_DIR="/opt/image-classification-api"
VENV_DIR="$APP_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
GUNICORN="$VENV_DIR/bin/gunicorn"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    exit 1
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Change to application directory
cd "$APP_DIR"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Using default configuration."
fi

# Start the application
echo "Starting Image Classification API..."

# Option 1: Direct uvicorn (for development/testing)
if [ "$1" = "dev" ]; then
    echo "Starting in development mode..."
    exec $PYTHON main.py
fi

# Option 2: Gunicorn (for production)
echo "Starting with Gunicorn..."
exec $GUNICORN main:app \
    --bind 0.0.0.0:8001 \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 300 \
    --keep-alive 2 \
    --access-logfile /var/log/image-classification-api-access.log \
    --error-logfile /var/log/image-classification-api-error.log \
    --log-level info
