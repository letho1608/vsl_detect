# Vietnamese Sign Language Detection - Makefile

.PHONY: help install install-dev clean test lint format build docker run

# Default target
help:
	@echo "Vietnamese Sign Language Detection - Development Commands"
	@echo "========================================================"
	@echo "install      Install package dependencies"
	@echo "install-dev  Install development dependencies" 
	@echo "clean        Clean build artifacts and cache"
	@echo "test         Run tests"
	@echo "lint         Run code linting"
	@echo "format       Format code with black"
	@echo "build        Build package"
	@echo "docker       Build Docker image"
	@echo "run          Run application"

# Install dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r config/requirements_detailed/dev.txt
	pip install -e .

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete

# Run tests
test:
	python -m pytest tests/ -v

# Run linting
lint:
	flake8 vsl/
	mypy vsl/

# Format code
format:
	black vsl/
	black tests/

# Build package
build: clean
	python dev/setup.py sdist bdist_wheel

# Build Docker image
docker:
	docker build -f dev/Dockerfile -t vsl-detect .

# Run application
run:
	python main.py

# Development server
dev:
	python main.py --debug

# Optimize data
optimize:
	python tools/optimize_all.py

# Train model
train:
	python vsl/training.py