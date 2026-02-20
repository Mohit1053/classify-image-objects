# Changelog

All notable changes to the Image Classification API will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Add authentication and rate limiting
- Docker containerization support
- API documentation improvements

## [1.0.0] - 2025-12-30

### Added
- Initial release of Image Classification API
- FastAPI-based REST API endpoints
- Ensemble classifier using multiple CLIP models
  - OpenAI CLIP ViT-B/32
  - OpenAI CLIP ViT-L/14
  - OpenCLIP ViT-B-32
- BLIP model for image captioning
- DETR model for object detection
- CSV batch processing support
- Firebase Storage URL validation
- EC2 deployment scripts and configuration
- Nginx reverse proxy configuration
- Supervisor process management
- Systemd service configuration

### Security
- Environment variable configuration for sensitive data
- .gitignore to prevent credential exposure

### Documentation
- Comprehensive deployment guide
- API usage examples
- Installation instructions
