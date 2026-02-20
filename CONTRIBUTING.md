# Contributing to Image Classification API

Thank you for your interest in contributing to the Image Classification API! This document provides guidelines for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/classify-image-objects.git
   cd classify-image-objects
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/classify-image-objects.git
   ```

## How to Contribute

### Reporting Bugs

- Check existing issues first
- Use the issue template
- Include steps to reproduce
- Provide system information (OS, Python version, GPU info)

### Suggesting Features

- Open an issue with the `enhancement` label
- Describe the feature and use case
- Explain why it would be beneficial

### Code Contributions

1. Create a feature branch
2. Make your changes
3. Write/update tests
4. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp env.example .env
# Edit .env with your configuration

# Run development server
uvicorn main:app --reload --port 8001
```

### Running Tests

```bash
pytest tests/ -v --cov=.
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints
- Maximum line length: 100 characters
- Use meaningful variable/function names

### Documentation

- Add docstrings to all functions
- Update README for new features
- Include usage examples

### Example Function

```python
def classify_image(image_url: str, threshold: float = 0.7) -> dict:
    """
    Classify an image from URL using ensemble models.
    
    Args:
        image_url: URL of the image to classify
        threshold: Confidence threshold (0.0 to 1.0)
    
    Returns:
        Dictionary containing predicted class and confidence
    
    Raises:
        ValueError: If image URL is invalid
        HTTPException: If image cannot be loaded
    """
    pass
```

## Commit Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

### Examples

```
feat(api): add batch classification endpoint
fix(classifier): handle malformed image URLs
docs(readme): update installation instructions
test(api): add integration tests for CSV upload
```

## Pull Request Process

1. **Update documentation** for any changes
2. **Add tests** for new functionality
3. **Ensure tests pass** locally
4. **Update CHANGELOG.md**
5. **Fill out PR template** completely

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No secrets or credentials committed

## Questions?

Open an issue for any questions or concerns. Thank you for contributing!
