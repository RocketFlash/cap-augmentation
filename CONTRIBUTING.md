# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Tests

Run the full test suite:

```bash
pytest
```

Check formatting:

```bash
black --check src tests dataset
```

Run the optional Torchvision integration tests:

```bash
python -m pip install -e ".[test,torchvision]"
pytest tests/test_wrappers.py
```

Before opening a pull request, also check for whitespace errors:

```bash
git diff --check
```

## Dataset Scripts

The `dataset/` directory is intentionally kept as clone-only tooling. It is not
installed as part of the `cap_augmentation` Python package, so run those scripts
from the repository root after installing the `dataset` extra.
