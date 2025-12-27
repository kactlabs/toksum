# toksum Documentation

This directory contains the Sphinx documentation for the toksum library.

## Building the Documentation

### Prerequisites

Install the required dependencies:

```bash
pip install sphinx sphinx-rtd-theme
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

### Building HTML Documentation

To build the HTML documentation:

```bash
cd docs
make html
```

The generated HTML files will be in `_build/html/`. Open `_build/html/index.html` in your browser to view the documentation.

### Building Other Formats

Sphinx supports multiple output formats:

```bash
make pdf      # PDF (requires LaTeX)
make epub     # EPUB
make man      # Manual pages
make text     # Plain text
make help     # Show all available formats
```

### Cleaning Build Files

To clean the build directory:

```bash
make clean
```

## Documentation Structure

- `index.rst` - Main documentation index
- `api/` - API reference documentation
  - `core.rst` - Core module documentation
  - `cli.rst` - CLI module documentation  
  - `exceptions.rst` - Exceptions documentation
- `examples.rst` - Usage examples
- `changelog.rst` - Version history
- `conf.py` - Sphinx configuration
- `requirements.txt` - Documentation dependencies

## Updating Documentation

The documentation is automatically generated from docstrings in the Python code. To update:

1. Update docstrings in the Python files
2. Rebuild the documentation with `make html`
3. Review the generated HTML files

## Sphinx Configuration

The documentation uses:

- **Theme**: sphinx-rtd-theme (Read the Docs theme)
- **Extensions**: autodoc, autosummary, napoleon, viewcode, intersphinx
- **Docstring Style**: Google/NumPy style with Napoleon extension
- **Auto-generation**: Enabled for API documentation

## Viewing Documentation

After building, you can view the documentation by opening `_build/html/index.html` in your web browser, or serve it locally:

```bash
cd _build/html
python -m http.server 8000
```

Then visit http://localhost:8000 in your browser.