#!/bin/bash

# Quick deployment script for EC2 instance
# Run this after uploading your files to EC2

set -e

echo "🚀 Starting Quick EC2 Deployment for Image Classification API..."

# Variables
APP_DIR="/opt/image-classification-api"
PROJECT_DIR="$HOME/classify_image_objects"

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Error: Project directory $PROJECT_DIR not found!"
    echo "Please upload your project files first using:"
    echo "scp -i your-key.pem -r . ubuntu@your-ec2-ip:/home/ubuntu/classify_image_objects/"
    exit 1
fi

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-venv git curl nginx supervisor htop

# Create application directory
echo "📁 Setting up application directory..."
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Copy files
echo "📋 Copying application files..."
cp -r $PROJECT_DIR/* $APP_DIR/

# Set up Python environment
echo "🐍 Setting up Python environment..."
cd $APP_DIR
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set permissions for scripts
echo "🔐 Setting file permissions..."
chmod +x start.sh stop.sh deploy.sh
sudo chmod +x start.sh stop.sh

# Configure nginx
echo "🌐 Configuring Nginx..."
sudo cp nginx.conf /etc/nginx/sites-available/image-classification-api

# Get EC2 public IP and update nginx config
EC2_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
if [ ! -z "$EC2_IP" ]; then
    echo "🌍 Detected EC2 public IP: $EC2_IP"
    sudo sed -i "s/your-domain.com/$EC2_IP/g" /etc/nginx/sites-available/image-classification-api
fi

sudo ln -sf /etc/nginx/sites-available/image-classification-api /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
sudo nginx -t

# Configure supervisor
echo "👨‍💼 Setting up Supervisor..."
sudo cp supervisor.conf /etc/supervisor/conf.d/image-classification-api.conf

# Start services
echo "🚀 Starting services..."

# Start nginx
sudo systemctl enable nginx
sudo systemctl restart nginx

# Start supervisor and application
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start image-classification-api-gunicorn

# Wait a moment for startup
echo "⏳ Waiting for application to start..."
sleep 10

# Test the deployment
echo "🧪 Testing deployment..."
if curl -s http://localhost:8001/healthz > /dev/null; then
    echo "✅ Local health check passed!"
else
    echo "❌ Local health check failed!"
fi

if [ ! -z "$EC2_IP" ]; then
    if curl -s http://$EC2_IP/healthz > /dev/null; then
        echo "✅ External health check passed!"
    else
        echo "❌ External health check failed! Check security group settings."
    fi
fi

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "📋 Next steps:"
echo "1. Configure your security group to allow HTTP traffic (port 80)"
echo "2. Optionally set up HTTPS with Let's Encrypt"
echo ""
echo "🔗 Your API is available at:"
echo "   Health Check: http://$EC2_IP/healthz"
echo "   API Docs: http://$EC2_IP/docs"
echo "   Classify URL: POST http://$EC2_IP/classify-url/"
echo "   Classify File: POST http://$EC2_IP/classify-file/"
echo ""
echo "📊 Monitor your application:"
echo "   Logs: sudo supervisorctl tail -f image-classification-api-gunicorn"
echo "   Status: sudo supervisorctl status"
echo "   Restart: sudo supervisorctl restart image-classification-api-gunicorn"
echo ""

# Show current status
echo "📈 Current Status:"
sudo supervisorctl status | grep image-classification
sudo systemctl status nginx --no-pager -l
