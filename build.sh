#!/bin/bash

# Build script for logic-lang package

set -e

echo "Building logic-lang package..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/

# Build the package
echo "Building package..."
python -m build

echo "Build complete! Distribution files are in dist/"
echo "Contents of dist/:"
ls -la dist/

echo ""
echo "To upload to PyPI:"
echo "1. Test upload: python -m twine upload --repository testpypi dist/*"
echo "2. Production upload: python -m twine upload dist/*"
