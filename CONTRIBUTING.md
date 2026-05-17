# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
# Upgrade the seeded build tools — `python -m venv` ships an old setuptools
# that has known CVEs (pip-audit will flag it). The package itself builds
# with setuptools>=68 (see pyproject.toml [build-system]); this just brings
# the dev venv's seeded versions in line.
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[test]"
```

## Tests

Run the full test suite:

```bash
pytest
```

Check formatting and lint:

```bash
black --check src tests dataset_tools
ruff check src tests dataset_tools
```

Both run in CI and must pass for merge.

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

The `dataset_tools/` directory is intentionally kept as clone-only tooling. It is not
installed as part of the `cap_augmentation` Python package, so run those scripts
from the repository root after installing the `dataset` extra.
