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
python -m pip install -e ".[test,dev]"
```

`[test]` installs the runtime extras the tests need; `[dev]` adds the
contributor toolchain (black, ruff, build). End users who only want the
runtime should install `pip install "cap-augmentation[all]"` — that
deliberately omits the dev tools.

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

To also exercise the Torchvision wrappers locally, install them alongside
`[test,dev]` and re-run the full suite (no need to target the wrapper
test file specifically — `pytest` picks it up either way and the imports
skip cleanly when torchvision isn't installed):

```bash
python -m pip install -e ".[test,dev,torchvision]"
pytest
```

To see line + branch coverage like CI does:

```bash
pytest --cov --cov-report=term-missing
```

Before opening a pull request, also check for whitespace errors:

```bash
git diff --check
```

## Dataset Scripts

The `dataset_tools/` directory is intentionally kept as clone-only tooling. It is not
installed as part of the `cap_augmentation` Python package, so run those scripts
from the repository root after installing the `dataset` extra.

## Releasing

1. Make sure CHANGELOG.md has a `## X.Y.Z` heading (plain version, no
   brackets or dates — the `publish.yml` release-notes extractor parses
   `## ` + the version number after stripping the leading `v` from the
   tag). Add the section before bumping `version` in `pyproject.toml`.
2. `git tag -a vX.Y.Z -m "..."` on `main`, then `git push --tags`.
3. The `publish.yml` workflow builds, OIDC-publishes to PyPI, and
   creates (or updates) a GitHub Release with notes extracted from
   CHANGELOG.md. Both steps are idempotent.
