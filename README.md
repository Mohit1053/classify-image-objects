# 🖼️ Image Classification API

> High-performance REST API for scrap/material image classification using ensemble deep learning models

[![CI](https://github.com/Mohit1053/classify-image-objects/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohit1053/classify-image-objects/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The **Image Classification API** is a production-ready FastAPI service that classifies images of scrap materials and pickup items using an ensemble of state-of-the-art deep learning models. It combines multiple CLIP variants, BLIP for captioning, and DETR for object detection to achieve robust classification results.

### Key Benefits

- **High Accuracy**: Ensemble of 5 models for robust predictions
- **Fast Processing**: Optimized for GPU acceleration
- **Flexible Input**: Supports URL, file upload, and CSV batch processing
- **Detailed Output**: Class prediction, confidence scores, and image descriptions
- **Production Ready**: Includes deployment scripts for AWS EC2

---

## ✨ Features

### Classification Models

| Model | Purpose | Capability |
|-------|---------|------------|
| **OpenAI CLIP ViT-B/32** | Zero-shot classification | Fast, general-purpose |
| **OpenAI CLIP ViT-L/14** | Zero-shot classification | Higher accuracy |
| **OpenCLIP ViT-B-32** | Zero-shot classification | LAION-trained variant |
| **BLIP** | Image captioning | Natural language descriptions |
| **DETR** | Object detection | Bounding boxes & labels |

### API Capabilities

- 🔗 **URL Classification** - Classify images from any accessible URL
- 📤 **File Upload** - Direct image file classification
- 📊 **CSV Batch Processing** - Process multiple images from CSV
- 🔍 **Firebase URL Validation** - Smart validation for Firebase Storage URLs
- 📝 **Detailed Responses** - Class, confidence, description, and failure reasons

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
├─────────────────────────────────────────────────────────────┤
│  Endpoints: /classify, /classify-csv, /health               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  EnsembleClassifier                          │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│ CLIP B/32│ CLIP L/14│ OpenCLIP │   BLIP   │      DETR      │
│ (OpenAI) │ (OpenAI) │ (LAION)  │(Caption) │  (Detection)   │
└──────────┴──────────┴──────────┴──────────┴────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              GPU/CPU Processing (PyTorch)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for all models)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mohit1053/classify-image-objects.git
   cd classify-image-objects
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

5. **Run the server**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8001
   ```

---

## 💻 Usage

### Single Image Classification

```bash
# Classify image from URL
curl -X POST "http://localhost:8001/classify?image_url=https://example.com/image.jpg"
```

### File Upload

```bash
curl -X POST "http://localhost:8001/classify" \
  -F "file=@/path/to/image.jpg"
```

### Batch CSV Processing

```bash
curl -X POST "http://localhost:8001/classify-csv" \
  -F "file=@images.csv"
```

### Response Format

```json
{
  "image_url": "https://example.com/image.jpg",
  "predicted_class": "Metal Scrap",
  "confidence": 0.87,
  "blip_description": "a pile of metal scraps and recyclable materials"
}
```

---

## 📚 API Documentation

Once the server is running, access the interactive documentation:

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/classify` | Classify single image (URL or upload) |
| `POST` | `/classify-csv` | Batch classify from CSV |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/docs` | Swagger documentation |

---

## 🚀 Deployment

### AWS EC2 Deployment

Detailed deployment instructions are available in [DEPLOYMENT.md](DEPLOYMENT.md).

**Quick Start:**

```bash
# Upload files to EC2
scp -i your-key.pem -r . ubuntu@your-ec2-ip:/home/ubuntu/classify_image_objects/

# Connect and deploy
ssh -i your-key.pem ubuntu@your-ec2-ip
cd ~/classify_image_objects
chmod +x deploy.sh start.sh stop.sh
./deploy.sh
```

### Available Scripts

| Script | Purpose |
|--------|---------|
| `deploy.sh` | Full deployment setup |
| `start.sh` | Start the API server |
| `stop.sh` | Stop the API server |
| `quick-deploy.sh` | Quick restart deployment |

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `env.example`:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8001
MAX_FILE_SIZE_MB=50
LOG_LEVEL=INFO

# AI Model Configuration
DEVICE=cuda              # 'cuda' or 'cpu'
CONFIDENCE_THRESHOLD=0.7

# Optional Settings
MODEL_CACHE_DIR=/opt/models
DEBUG=false
```

### Classification Categories

The system classifies images into pickup/scrap categories defined in `file1.py`. Current supported classes include various types of scrap materials and recyclable items.

---

## 📁 Project Structure

```
classify-image-objects/
├── .github/
│   ├── workflows/
│   │   └── ci.yml              # CI/CD pipeline
│   ├── ISSUE_TEMPLATE.md
│   └── PULL_REQUEST_TEMPLATE.md
├── main.py                     # FastAPI application
├── file1.py                    # EnsembleClassifier & model logic
├── requirements.txt            # Python dependencies
├── env.example                 # Environment template
├── .env.example                # Environment template (alias)
├── .gitignore                  # Git ignore rules
├── DEPLOYMENT.md               # EC2 deployment guide
├── CONTRIBUTING.md             # Contribution guidelines
├── CHANGELOG.md                # Version history
├── LICENSE                     # MIT License
├── README.md                   # This file
├── deploy.sh                   # Deployment script
├── start.sh                    # Start server script
├── stop.sh                     # Stop server script
├── quick-deploy.sh             # Quick deployment script
├── nginx.conf                  # Nginx configuration
├── supervisor.conf             # Supervisor configuration
├── image-classification-api.service  # Systemd service
└── pickup_items_csv.csv        # Sample data
```

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

### Quick Start

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Image Classification API**

- GitHub: [@Mohit1053](https://github.com/Mohit1053)

---

## 🙏 Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## ⚠️ Security Notice

**Never commit:**
- `.pem` files (SSH keys)
- `.env` files (credentials)
- API keys or tokens

Always use `.env.example` as a template and keep actual credentials secure.

---

<p align="center">
  Made with ❤️ for intelligent image classification
</p>


