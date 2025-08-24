# Logic Lang Package - Publishing Guide

## Package Structure

Your `logic-lang` package has been successfully converted into a proper Python package named `logic_lang` with the following structure:

```
logic-lang-package/
├── logic_lang/           # Main package directory
│   ├── __init__.py       # Package initialization with exports
│   ├── ast_nodes.py      # AST node definitions
│   ├── exceptions.py     # Custom exceptions
│   ├── interpreter.py    # Rule interpreter
│   ├── loss.py          # Loss functions
│   ├── parser.py        # Rule parser
│   └── soft_logic.py    # Soft logic implementations
├── tests/               # Test suite
│   ├── __init__.py
│   └── test_basic.py    # Basic functionality tests
├── dist/               # Built packages (created by build process)
│   ├── logic_lang-0.1.0-py3-none-any.whl
│   └── logic_lang-0.1.0.tar.gz
├── pyproject.toml      # Modern Python packaging configuration
├── LICENSE             # MIT license
├── MANIFEST.in         # Include/exclude files in distribution
├── README.md           # Package documentation
├── build.sh           # Build script
└── setup_dev.sh       # Development setup script
```

## Package Information

- **Name**: `logic-lang` (PyPI name, installed as `logic_lang`)
- **Version**: 0.1.2
- **License**: MIT
- **Dependencies**: PyTorch >= 1.9.0, NumPy >= 1.20.0
- **Python**: >= 3.8

## Installation

Once published, users can install your package with:

```bash
pip install logic-lang
```

And use it in Python:

```python
from logic_lang import RuleParser, RuleInterpreter

# Create parser and interpreter
parser = RuleParser()
interpreter = RuleInterpreter()

# Parse and execute rules
rules = '''
expect mass_L, mc_L
define findings_L = mass_L | mc_L
constraint exactly_one(findings_L) weight=1.0
'''

ast = parser.parse(rules)
constraints = interpreter.execute(rules, features)
```

## Publishing to PyPI

### Prerequisites

1. **Create PyPI account**: Register at https://pypi.org/account/register/
2. **Install twine**: `pip install twine`
3. **Get API token**: Go to https://pypi.org/manage/account/token/ and create a new API token

### Publishing Steps

#### 1. Test on TestPyPI (Recommended First)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ logic-lang
```

#### 2. Publish to Production PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*
```

You'll be prompted for your username (use `__token__`) and password (use your API token).

### Automation with GitHub Actions

Consider setting up automated publishing with GitHub Actions. Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## Development Workflow

### Local Development

```bash
# Clone your repository
git clone <your-repo-url>
cd logic-lang-package

# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run linting
black logic_lang/
flake8 logic_lang/
mypy logic_lang/
```

### Making Updates

1. **Update version** in `pyproject.toml`
2. **Add new features** and corresponding tests
3. **Update README.md** with new documentation
4. **Build and test locally**:
   ```bash
   ./build.sh
   python -m pytest tests/ -v
   ```
5. **Publish new version**

## Repository Setup

To complete the package setup:

1. **Initialize git repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: logic-lang package"
   ```

2. **Create GitHub repository** and push:
   ```bash
   git remote add origin https://github.com/yourusername/logic-lang.git
   git branch -M main
   git push -u origin main
   ```

3. **Update URLs in pyproject.toml** with your actual GitHub repository URLs

## Package Features

Your package provides:

- **RuleParser**: Parse rule scripts into AST
- **RuleInterpreter**: Execute parsed rules to generate constraints
- **Soft Logic Operations**: Various constraint types and semantics
- **Loss Functions**: Integration with PyTorch for training
- **AST Nodes**: Complete AST representation for logic language
- **Exception Handling**: Comprehensive error types for rule processing

## Next Steps

1. **Update author information** in `pyproject.toml`
2. **Set up proper GitHub repository** with your URLs
3. **Add more comprehensive tests**
4. **Consider adding documentation** with Sphinx
5. **Add continuous integration** with GitHub Actions
6. **Create example notebooks** for users

Your package is now ready for publishing to PyPI! 🎉
