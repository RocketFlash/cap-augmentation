# Changelog

## 0.2.2

### Bug fixes
- `CapAug.paste_object` no longer crashes when an `object_transforms` callable
  crops or resizes the input. Previously the source dimensions were captured
  before the transform ran, so the post-transform mask was sliced with stale
  ROI bounds and OpenCV failed with a size-mismatch assertion. The transform
  is now applied first, and shape-changing transforms that return mismatched
  image and mask sizes raise a clear `ValueError`.
- Returned bounding boxes are now tight against the visible (alpha > 0) region
  of the pasted object rather than the source canvas. PNGs with transparent
  padding (e.g. a 20×20 file containing a 10×10 visible object) used to yield
  a box covering the full canvas; the box now matches the pixels that
  actually changed in the destination. Behaviour is unchanged for fully
  opaque sources.
- Returned mask is sliced to the same tight region as the new bbox, so
  multiclass aggregation and instance-mask blits stay aligned.

### Dev / CI
- `CONTRIBUTING.md` now tells contributors to also upgrade `setuptools` and
  `wheel` when bootstrapping the dev venv. `python -m venv` seeds a pinned
  old setuptools that `pip-audit` flags for known CVEs; the project build
  itself uses `setuptools>=68` (see `pyproject.toml [build-system]`) so this
  only affects the dev environment, but it removes the noise.
- Added `ruff` lint to the project and CI (`tool.ruff` config in
  `pyproject.toml`, `ruff check` step in `.github/workflows/test.yml`,
  `ruff>=0.6` in the `[test]` and `[all]` extras). Auto-fixed 21 findings:
  removed 2 genuinely unused imports
  (`dataset_tools/cityscapes/filter_dataset.py:Path`,
  `dataset_tools/vinbig/generate_dataset.py:os`), modernised
  `super(Cls, self).__init__` and `class Foo(object)`, sorted imports, and
  dropped redundant `# coding: utf-8` declarations.

## 0.2.1

- Expose `cap_augmentation.__version__` (read from installed package metadata via `importlib.metadata`).
- Make `.github/workflows/publish.yml` idempotent by passing `skip-existing: true` to the PyPI publish action; re-tagging an already-published version no longer fails the workflow.

## 0.2.0

### Breaking changes
- Renamed public classes to PEP 8: `CAP_AUG` → `CapAug`, `CAP_AUG_Multiclass` → `CapAugMulticlass`, `CAP_Albu`/`CAP_Albumentations` → `CapAlbumentations`, `CAP_TorchVision` → `CapTorchvision`. No backward-compatible aliases.
- Renamed `dataset/` → `dataset_tools/` to disambiguate from the runtime `data/` directory.
- Renamed `dataset/vb/` → `dataset_tools/vinbig/` (long form matches the upstream VinBigData dataset name).
- Renamed wrapper modules: `wrappers/albumentations.py` → `wrappers/albu.py`, `wrappers/torchvision.py` → `wrappers/tv.py`, `wrappers/generic.py` → `wrappers/image_mask_transform.py`. Public imports from `cap_augmentation` are unchanged.
- Renamed packaged calibration: `bev/camera_intrinsic_params.yaml` → `bev/default_calibration.yaml`.
- Moved demo assets: top-level `example_images/` → `examples/images/`.
- Renamed demo notebooks: `test_generation.ipynb` → `bev_and_pedestrians_demo.ipynb`, `test_generation_vbd.ipynb` → `vinbig_demo.ipynb`.
- Renamed Cityscapes runner: `generate_and_filter_dataset.sh` → `run.sh`.

### Repository
- Renamed GitHub repository `CAP_augmentation` → `cap-augmentation` to align with the PyPI distribution name. The old URL redirects.

### Internal
- Removed dead `src/cap_augmentation/bev/config.py`; inlined the default camera intrinsics/extrinsics into `bev/bev_transform.py`.
- Split `tests/test_dataset_scripts.py` into `tests/test_dataset_cityscapes.py` and `tests/test_dataset_vinbig.py`.
- Replaced the original author's hardcoded VinBig paths in `dataset_tools/vinbig/config.py` with `None` plus startup assertions.
- Added `.` to `pytest.pythonpath` so `dataset_tools.*` imports resolve in tests.
- Updated README, CONTRIBUTING, and the CI workflow to reflect every rename.

## 0.1.0

- Restructured the project into an installable `src/cap_augmentation` package.
- Added tests for the augmentation core, BEV helpers, dataset scripts, and wrappers.
- Added optional Albumentations and Torchvision integrations.
- Moved example notebooks under `examples/notebooks/`.
- Reduced repository size by stripping notebook outputs and replacing large example PNGs.
