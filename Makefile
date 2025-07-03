.PHONY: help install install-dev test test-cov lint format type-check clean build upload upload-test run-example

help:
	@echo "Available commands:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install the package in development mode with dev dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting (flake8)"
	@echo "  format       Format code (black)"
	@echo "  type-check   Run type checking (mypy)"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build the package"
	@echo "  upload       Upload to PyPI"
	@echo "  upload-test  Upload to Test PyPI"
	@echo "  run-example  Run the basic usage example"

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

test-cov:
	pytest --cov=toksum --cov-report=html --cov-report=term-missing

lint:
	flake8 toksum tests examples

format:
	black toksum tests examples

type-check:
	mypy toksum

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	twine upload dist/*

upload-test: build
	twine upload --repository testpypi dist/*

run-example:
	python examples/basic_usage.py

# Development workflow
dev-setup: install-dev
	@echo "Development environment set up successfully!"

dev-check: format lint type-check test
	@echo "All development checks passed!"
