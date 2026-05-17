# Agent guide

This file is for AI coding agents (Claude Code, Cursor, etc.) working on this
repo. Humans should start with `README.md` and `CONTRIBUTING.md`.

## What this is

`cap-augmentation` is a published Python library on PyPI
(<https://pypi.org/project/cap-augmentation/>) implementing "cut and paste"
augmentation for object detection / instance + semantic segmentation, with an
optional BEV (bird's-eye-view) mode for camera-coordinate placement.

The repository ships **two distinct things**:

1. **The installable library** under `src/cap_augmentation/` — what users get
   when they `pip install cap-augmentation`. This is the *only* code in the
   wheel. Treat it as a public API.
2. **Repo-only dataset tooling** under `dataset_tools/` — scripts for
   generating PNG cutouts from Cityscapes/CityPersons and VinBigData. **Not
   installed** with the wheel. Run from a clone with `pip install -e ".[dataset]"`.

## Quick start (do this before making changes)

```bash
# Use uv (already present on dev machines) for fast ephemeral venvs
uv venv .venv-dev --python 3.11
uv pip install --python .venv-dev/bin/python -e ".[test,torchvision]"
.venv-dev/bin/pytest            # 28 tests; all must pass
.venv-dev/bin/black --check src tests dataset_tools
```

`.venv-*` is gitignored; clean up with `rm -rf .venv-dev` when done.

## Directory map

```
src/cap_augmentation/        ← THE installed package (only thing in the wheel)
  __init__.py                ← public exports; __version__ via importlib.metadata
  cap_aug.py                 ← CapAug, CapAugMulticlass, resize_keep_ar
  utils.py                   ← viz helpers (draw_bboxes, show_image, …)
  bev/
    __init__.py              ← BEV + matrix helpers
    bev_transform.py         ← BEV class; defaults inlined (no separate config)
    default_calibration.yaml ← PACKAGE DATA — must ship in the wheel
  wrappers/
    __init__.py              ← lazy re-exports
    albu.py                  ← CapAlbumentations (requires [albumentations] extra)
    tv.py                    ← CapTorchvision (requires [torchvision] extra)
    image_mask_transform.py  ← ImageMaskTransform adapter

dataset_tools/               ← REPO-ONLY scripts, never installed
  cityscapes/{generate_dataset,filter_dataset,config,run.sh}
  vinbig/{generate_dataset,generate_analytics,config}.py

tests/                       ← flat; one file per logical unit
examples/notebooks/          ← user-facing demos
examples/images/             ← demo input images (referenced by README + notebooks)
data/                        ← (gitignored) script outputs go here

.github/workflows/
  test.yml                   ← pytest + black on push/PR, 3.9 / 3.11 / 3.12
  publish.yml                ← on v* tag push: build + OIDC publish to PyPI
```

## Public API surface (DO NOT BREAK without a major-version bump)

```python
from cap_augmentation import (
    CapAug,                # core augmenter
    CapAugMulticlass,      # multi-class compositor
    CapAlbumentations,     # requires [albumentations]
    CapTorchvision,        # requires [torchvision]
    ImageMaskTransform,    # per-object transform adapter
    resize_keep_ar,        # AR-preserving resize helper
    __version__,           # from installed metadata
)
from cap_augmentation.bev import BEV
```

These names are **canonical**. The package was renamed from `CAP_AUG`-style
SCREAMING_SNAKE_CASE in v0.2.0 with **no backward-compat aliases**. Do not
re-introduce them.

## Conventions that aren't obvious from the code

- **PEP 8 everywhere.** Class names = `CapWords`, functions = `snake_case`,
  constants = `UPPER_SNAKE`. Recently enforced — don't suggest reverting to
  legacy names even if old issues/PRs reference them.
- **No backward-compatibility shims.** When refactoring, delete old names
  rather than aliasing them. Version bumps signal the break.
- **Lazy wrapper imports.** `wrappers/__init__.py` uses `__getattr__` to
  import `albu` / `tv` only on demand. This keeps the core import cheap and
  defers the optional-extras failure until the user actually touches the
  wrapper. Preserve this pattern when adding wrappers.
- **`black --check` is a CI gate** with `target-version = ["py39"]`. Format
  before committing: `black src tests dataset_tools`.
- **Tests must pass on Python 3.9 / 3.11 / 3.12** — no walrus-in-comprehension
  3.10+ syntax, no `match` statements, no `X | Y` type unions outside string
  annotations (use `Optional[X]` / `Union[X, Y]` instead).
- **Notebook outputs are not committed.** If you re-run a notebook, strip
  outputs (`jupyter nbconvert --clear-output --inplace`) before committing.

## Don't-do list (real things people have tried)

| Don't | Because |
|---|---|
| Move `bev/default_calibration.yaml` outside `src/cap_augmentation/` | It's bundled as `[tool.setuptools.package-data]` and looked up via `Path(__file__).parent`. Outside the package = not in the wheel = broken default. |
| Re-add `src/cap_augmentation/bev/config.py` | Deleted in 0.2.0; defaults are inlined into `bev_transform.py` (`_DEFAULT_CAMERA_INFO`, `_DEFAULT_PIX_PER_METER`, `_DEFAULT_CALIB_PATH`). One source of truth. |
| Add `dataset_tools/` to the installed package | It's intentionally repo-only. The `[tool.setuptools.packages.find]` block uses `include = ["cap_augmentation*"]` to scope discovery. |
| Hardcode private dataset paths in `dataset_tools/*/config.py` | Use `None` + a startup assertion in the script (see `vinbig/config.py` for the pattern). |
| Add a `configs/` top-level folder | We considered this and rejected it. YAML must stay package-local; per-script configs stay next to their scripts. |
| Use `CAP_AUG` or `CAP_Albu` in new code | Removed in 0.2.0, no aliases. |

## Release workflow

1. Make changes on a feature branch; PR to `main`.
2. After merge, bump `version` in `pyproject.toml` and add a `CHANGELOG.md`
   entry.
3. `git tag -a vX.Y.Z -m "..."` on main, `git push --tags`.
4. The `publish.yml` workflow auto-publishes to PyPI via OIDC trusted
   publishing (no API tokens stored as secrets). The action passes
   `skip-existing: true` so re-tagging is a safe no-op.
5. Verify with `pip install cap-augmentation==X.Y.Z` from a clean venv.

If trusted publishing isn't yet configured on PyPI, manual fallback is
`twine upload dist/*` with `~/.pypirc` containing a token under `[pypi]`.

## Common change recipes

**Adding a public class to the core**
1. Implement in `src/cap_augmentation/cap_aug.py` (or a new module).
2. Add to the `from .cap_aug import …` line in `src/cap_augmentation/__init__.py`.
3. Add to the `__all__` list.
4. Add a test in `tests/test_cap_aug.py`.
5. Document in README under "Public API".

**Adding a new optional integration wrapper**
1. Add the dep to a new extra in `pyproject.toml` `[project.optional-dependencies]`.
2. Create `src/cap_augmentation/wrappers/<name>.py` with the class.
3. Add a lazy entry in `wrappers/__init__.py`'s `__getattr__` (follow the
   `albu.py` / `tv.py` pattern).
4. Re-export the class name from `src/cap_augmentation/__init__.py`'s
   `__getattr__` and `__all__`.
5. Add tests with `pytest.importorskip(...)` so they skip when the extra
   isn't installed (see `tests/test_wrappers.py`).

**Renaming a public symbol**
- This is breaking. Bump the minor version (0.X.0 → 0.X+1.0) and document in
  CHANGELOG. Don't add a backward-compat alias — past refactors deliberately
  rejected that pattern.

## External references

- PyPI project: <https://pypi.org/project/cap-augmentation/>
- DOI (concept): <https://zenodo.org/badge/latestdoi/328174810>
- Source paper: Ghiasi et al. 2020, "Simple Copy-Paste is a Strong Data
  Augmentation Method for Instance Segmentation", <https://arxiv.org/abs/2012.07177>
- Cityscapes dataset: <https://www.cityscapes-dataset.com/>
- VinBigData dataset: <https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection>
