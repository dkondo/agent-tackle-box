.PHONY: all format lint lint_package lint_tests test tests typecheck clean build help

# Default target
all: format lint test

# Format code with ruff
format:
	uv run --group lint ruff format .
	uv run --group lint ruff check --fix .

# Lint code with ruff
lint:
	uv run --group lint ruff check .
	uv run --group lint ruff format --check .

# Lint package code (excluding tests)
lint_package:
	uv run --group lint ruff check agent_debugger examples
	uv run --group lint ruff format --check agent_debugger examples

# Lint tests only
lint_tests:
	uv run --group lint ruff check tests
	uv run --group lint ruff format --check tests

# Run unit tests
test tests:
	uv run --group test pytest tests -v

# Type checking
typecheck:
	uv run --group typing mypy agent_debugger

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Build package
build:
	uv build

# Help
help:
	@echo "Available targets:"
	@echo "  format           - Format code with ruff"
	@echo "  lint             - Lint code with ruff"
	@echo "  lint_package     - Lint package code (excluding tests)"
	@echo "  lint_tests       - Lint tests only"
	@echo "  test / tests     - Run unit tests"
	@echo "  typecheck        - Run mypy type checking"
	@echo "  clean            - Remove build artifacts"
	@echo "  build            - Build package"
