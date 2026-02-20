#!/bin/bash

# EC2 Deployment Script for Image Classification API
# Run this script on your EC2 instance

set -e

echo "Starting deployment of Image Classification API..."

# Update system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-venv git curl nginx supervisor

# Install NVIDIA drivers and CUDA (if GPU instance)
echo "Installing NVIDIA drivers and CUDA..."
# Uncomment the following lines if you're using a GPU instance
# sudo apt install -y nvidia-driver-470
# wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
# sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit

# Create application directory
echo "Setting up application directory..."
sudo mkdir -p /opt/image-classification-api
sudo chown $USER:$USER /opt/image-classification-api
cd /opt/image-classification-api

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Copy application files (assuming they're in current directory)
echo "Copying application files..."
cp -r ~/classify_image_objects/* .

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Add python-dotenv to requirements if not present
echo "Installing additional dependencies..."
pip install python-dotenv gunicorn

# Create models cache directory
sudo mkdir -p /opt/image-classification-api/models
sudo chown $USER:$USER /opt/image-classification-api/models

echo "Deployment setup completed!"
echo "Next steps:"
echo "1. Copy your application files to /opt/image-classification-api/"
echo "2. Configure nginx (see nginx.conf)"
echo "3. Set up supervisor for process management"
echo "4. Start the application"
