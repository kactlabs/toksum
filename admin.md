# Admin Notes

## Documentation

### Serve docs locally
```bash
python build_docs.py serve
```

### Build docs
```bash
python build_docs.py build
```

### Build and serve
```bash
python build_docs.py build-serve
```

### Clean build
```bash
python build_docs.py clean
```

## Development

### Test docs build
```bash
cd docs && make html
```

### Direct sphinx build
```bash
sphinx-build -b html docs docs/_build/html
```

### Version info
- Current: 0.7.0
- Files: `toksum/__init__.py`, `pyproject.toml`, `docs/conf.py`

## Quick checks
- Docs build: ✅
- Version sync: ✅
- Examples work: ✅