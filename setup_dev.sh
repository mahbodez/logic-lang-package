#!/bin/bash

# Development setup script for logic-lang package

set -e

echo "Setting up development environment for logic-lang package..."

# Install build tools
echo "Installing build tools..."
pip install --upgrade pip setuptools wheel build twine

# Install package in development mode with dev dependencies
echo "Installing package in development mode..."
pip install -e ".[dev]"

# Run tests to verify installation
echo "Running tests..."
python -m pytest tests/ -v

echo ""
echo "Development environment setup complete!"
echo "You can now:"
echo "1. Run tests: python -m pytest tests/"
echo "2. Build package: ./build.sh"
echo "3. Upload to PyPI: python -m twine upload dist/*"
