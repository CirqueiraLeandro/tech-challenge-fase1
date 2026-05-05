.PHONY: help install dev test lint format run-api run-notebook clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make dev           - Install development dependencies"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run linting with ruff"
	@echo "  make format        - Format code with black and isort"
	@echo "  make run-api       - Start FastAPI server"
	@echo "  make run-notebook  - Start Jupyter notebook"
	@echo "  make clean         - Clean temporary files"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=html

smoke-test:
	pytest tests/ -v -m "smoke"

schema-test:
	pytest tests/ -v -m "schema"

lint:
	ruff check src/ tests/ --show-source

format:
	black src/ tests/ notebooks/
	isort src/ tests/ notebooks/

run-api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

run-notebook:
	jupyter notebook notebooks/

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/ .ruff_cache/
