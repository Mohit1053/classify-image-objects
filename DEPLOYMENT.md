# Image Classification API - EC2 Deployment Guide

## Prerequisites

1. **EC2 Instance Requirements:**
   - Instance Type: `g4dn.xlarge` or `p3.2xlarge` (for GPU support) or `m5.2xlarge` (CPU only)
   - AMI: Ubuntu 20.04 LTS or Ubuntu 22.04 LTS
   - Storage: At least 50GB EBS volume
   - Security Group: Allow inbound traffic on ports 22 (SSH), 80 (HTTP), 443 (HTTPS), and 8001 (API)

2. **SSH Access:**
   - Ensure you have your `.pem` key file
   - Configure your security group for SSH access from your IP

## Deployment Steps

### 1. Upload Files to EC2

```bash
# From your local machine, upload all files to EC2
scp -i your-key.pem -r . ubuntu@your-ec2-ip:/home/ubuntu/classify_image_objects/
```

### 2. Connect to EC2 Instance

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### 3. Run Deployment Script

```bash
cd ~/classify_image_objects
chmod +x deploy.sh start.sh stop.sh
./deploy.sh
```

### 4. Configure Nginx

```bash
# Copy nginx configuration
sudo cp nginx.conf /etc/nginx/sites-available/image-classification-api
sudo ln -s /etc/nginx/sites-available/image-classification-api /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default  # Remove default site

# Update server_name in nginx.conf with your domain or EC2 public IP
sudo nano /etc/nginx/sites-available/image-classification-api

# Test and reload nginx
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl enable nginx
```

### 5. Set Up Process Management (Choose one option)

#### Option A: Using Supervisor (Recommended)

```bash
# Install supervisor
sudo apt install supervisor

# Copy supervisor configuration
sudo cp supervisor.conf /etc/supervisor/conf.d/image-classification-api.conf

# Update and start
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start image-classification-api-gunicorn
sudo supervisorctl status
```

#### Option B: Using Systemd

```bash
# Copy systemd service file
sudo cp image-classification-api.service /etc/systemd/system/

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable image-classification-api
sudo systemctl start image-classification-api
sudo systemctl status image-classification-api
```

### 6. Verify Deployment

```bash
# Check if the API is running
curl http://localhost:8001/healthz

# Check from external access (replace with your EC2 public IP)
curl http://your-ec2-ip/healthz
```

## Configuration

### Environment Variables

Edit `/opt/image-classification-api/.env` to customize:

```bash
sudo nano /opt/image-classification-api/.env
```

### Security Group Configuration

Make sure your EC2 security group allows:
- Port 22: SSH access
- Port 80: HTTP traffic
- Port 443: HTTPS traffic (if using SSL)
- Port 8001: Direct API access (optional, can be restricted)

## API Endpoints

Once deployed, your API will be available at:

- **Health Check:** `http://your-ec2-ip/healthz`
- **API Documentation:** `http://your-ec2-ip/docs`
- **Classify Single URL:** `POST http://your-ec2-ip/classify-url/`
- **Classify CSV File:** `POST http://your-ec2-ip/classify-file/`

## Monitoring and Logs

### View Application Logs

```bash
# If using supervisor
sudo tail -f /var/log/image-classification-api-gunicorn.log

# If using systemd
sudo journalctl -u image-classification-api -f

# Nginx logs
sudo tail -f /var/log/nginx/image-classification-api.access.log
sudo tail -f /var/log/nginx/image-classification-api.error.log
```

### Monitor System Resources

```bash
# Check GPU usage (if using GPU instance)
nvidia-smi

# Check memory and CPU usage
htop

# Check disk usage
df -h
```

## SSL/HTTPS Setup (Optional but Recommended)

### Using Let's Encrypt (Free SSL)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate (replace your-domain.com)
sudo certbot --nginx -d your-domain.com

# Auto-renewal setup
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
```

## Troubleshooting

### Common Issues

1. **Out of Memory:**
   ```bash
   # Check memory usage
   free -h
   
   # Reduce workers in gunicorn or use swap
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **CUDA/GPU Issues:**
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Install CUDA toolkit if needed
   # Follow NVIDIA's installation guide
   ```

3. **Port Issues:**
   ```bash
   # Check what's running on port 8001
   sudo lsof -i :8001
   
   # Kill process if needed
   sudo pkill -f gunicorn
   ```

### Service Management Commands

```bash
# Using supervisor
sudo supervisorctl status
sudo supervisorctl restart image-classification-api-gunicorn
sudo supervisorctl stop image-classification-api-gunicorn

# Using systemd
sudo systemctl status image-classification-api
sudo systemctl restart image-classification-api
sudo systemctl stop image-classification-api

# Nginx
sudo systemctl status nginx
sudo systemctl reload nginx
sudo nginx -t  # Test configuration
```

## Performance Optimization

1. **For GPU Instances:**
   - Use CUDA-enabled PyTorch
   - Set `DEVICE=cuda` in `.env`
   - Monitor GPU memory usage

2. **For CPU Instances:**
   - Increase worker count in gunicorn
   - Set `DEVICE=cpu` in `.env`
   - Consider using CPU-optimized instances

3. **General:**
   - Enable nginx caching for static content
   - Use load balancer for multiple instances
   - Monitor and optimize memory usage

## Backup and Updates

### Backup

```bash
# Backup application and models
sudo tar -czf /tmp/image-classification-backup-$(date +%Y%m%d).tar.gz \
  /opt/image-classification-api/
```

### Updates

```bash
# Pull latest code
cd /opt/image-classification-api
git pull  # if using git

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Restart service
sudo supervisorctl restart image-classification-api-gunicorn
# or
sudo systemctl restart image-classification-api
```

## Cost Optimization

1. **Use Spot Instances:** For development/testing
2. **Auto Scaling:** Set up based on CPU/memory usage
3. **Reserved Instances:** For production workloads
4. **Monitor Costs:** Use AWS Cost Explorer

## Support

For issues or questions:
1. Check application logs
2. Verify all dependencies are installed
3. Ensure proper file permissions
4. Check AWS security group settings
