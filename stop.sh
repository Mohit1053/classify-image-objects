#!/bin/bash

# Stop script for Image Classification API

echo "Stopping Image Classification API..."

# Stop supervisor processes
if command -v supervisorctl &> /dev/null; then
    echo "Stopping supervisor processes..."
    sudo supervisorctl stop image-classification-api
    sudo supervisorctl stop image-classification-api-gunicorn
fi

# Stop any remaining Python processes
echo "Stopping any remaining processes..."
pkill -f "main.py"
pkill -f "gunicorn.*main:app"

# Stop nginx if needed
if command -v nginx &> /dev/null; then
    echo "Reloading nginx..."
    sudo nginx -s reload
fi

echo "Image Classification API stopped."
