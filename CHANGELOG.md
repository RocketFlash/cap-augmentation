# Changelog

<!--
Header format: `## X.Y.Z` (plain version, no brackets or dates). The
publish workflow extracts release notes by matching `## ` + the tag
version with the leading `v` stripped. Keep-a-Changelog style headers
like `## [0.3.0] - 2026-05-17` will silently produce empty notes.
-->

## 0.4.1

### New features
- `max_overlap` parameter on `CapAug`: skip pastes whose tight bbox
  exceeds the given IoU with any already-accepted bbox. Useful for
  dense placement where overlapping pastes would corrupt detection
  ground truth. Default `None` keeps the prior unrestricted behavior.

### Tests & CI
- Promoted Albumentations and Torchvision `DeprecationWarning` and
  `PendingDeprecationWarning` to errors in `pyproject.toml`
  `filterwarnings`. Future upstream deprecations will fail the suite
  immediately rather than slipping through.

### Docs
- Corrected the 0.4.0 `CapAlbumentations.always_apply` note: the
  field was removed by Albumentations **2.0.0**, not a hypothetical
  3.x. We had already required `>=2.0.8`, so the wrapper carried the
  alias longer than the underlying library supported it.

## 0.4.0

### Breaking changes
- `CapAlbumentations.always_apply` was removed. The kwarg had been a
  deprecated alias for `p=1.0` since 0.2.x; Albumentations itself
  removed the field in **2.0.0** (we already require `>=2.0.8`), so
  the wrapper was kept around longer than the underlying library
  supported it. Passing `always_apply=...` now raises a `TypeError`
  with a migration hint pointing at `p=1.0`.

### Bug fixes
- `object_transforms` callables that return a 4-channel image now raise
  a clear `ValueError` instead of silently broadcasting a (H, W) alpha
  against a (H, W, 4) source in the composite. The contract (alpha
  travels via the separate `mask` argument, not the image one) is now
  explicit in the error message.

### Performance
- `probability_map` is now normalised once and cached on the `CapAug`
  instance instead of being re-summed and re-divided on every call. For
  a 1000×1000 map that's ~1 MB of per-call busywork avoided in tight
  training loops. The cached value is invalidated only by constructing
  a new `CapAug` — replace the array, don't mutate it in place.

### Validation
- `probability_map` inputs with `ndim != 2` now raise a clear
  `ValueError` instead of failing downstream in `np.random.choice` with
  an opaque shape mismatch.

## 0.3.1

### Tests & CI
- Added end-to-end coverage for `image_format='rgb'`, the `s_range`
  scale path (when `h_range` is None), and `CapAugMulticlass` composed
  with `bev_transform` — three paths the audit flagged as untested.
- Added `tests/test_notebooks.py`: parses each `examples/notebooks/*.ipynb`,
  compiles every code cell (catches API-rename drift), and asserts cell
  outputs are stripped per repo convention.
- CI now runs on macOS-latest and Windows-latest (Python 3.12) in
  addition to the Linux 3.10 / 3.11 / 3.12 / 3.13 matrix.
- Coverage reporting via `pytest-cov` on the Linux/3.12 job, with the
  `coverage.xml` artifact uploaded for inspection.

### Packaging
- Split `[all]` into runtime extras and a new `[dev]` extra. Previously
  `pip install "cap-augmentation[all]"` leaked black/ruff/build into
  user environments; `[all]` is now runtime-only, and contributors
  install `pip install -e ".[test,dev]"` to get the CI toolchain.
- `[test]` no longer carries black/ruff (moved to `[dev]`).

### Docs
- Documented the default BEV calibration YAML (source camera + FOV +
  "placeholder, replace for production") inline in the YAML and in the
  README's BEV section.
- CONTRIBUTING.md gained a Releasing section and a coverage-reporting
  command; pinned the CHANGELOG header format so `publish.yml`'s
  release-notes extractor doesn't silently produce empty notes.

## 0.3.0

### Breaking changes
- Dropped Python 3.9 support; minimum is now Python 3.10 (3.9 reached EOL
  in October 2025). CI matrix is now 3.10 / 3.11 / 3.12 / 3.13.
- The default soft-alpha composite now honors intermediate alpha values
  on source PNGs. For sources with anti-aliased edges (most real
  cutouts), pasted objects blend smoothly into the destination instead
  of being hard-thresholded by the previous bitwise composite. Outputs
  are bit-identical for sources with binary alpha (alpha ∈ {0, 255}),
  which is what `dataset_tools/cityscapes` produces.

### New features
- `CapAug(..., rng=42)` accepts an int seed or `numpy.random.Generator`
  for local, reproducible randomness — no more seeding both `random.seed`
  and `np.random.seed` globally. `rng=None` (default) preserves the
  legacy global-state behavior.
- `CapAug(..., cache_size=...)` caches decoded source PNGs. Default is
  unbounded; set `0` to disable, or `N` for an LRU cap. Eliminates the
  per-paste `cv2.imread` cost that dominated training-loop wall time.
- Public type annotations on `CapAug`, `CapAugMulticlass`,
  `resize_keep_ar`, `ImageMaskTransform`, and `__version__`. Ships
  `py.typed` so mypy/pyright honor them.
- New `OpaqueSourceWarning` (exported) fires once per source path when
  CapAug detects a grayscale, 3-channel, or fully-opaque source — these
  silently paste the full rectangle as an "object", which is almost
  always a user error.

### Bug fixes
- `_align_columns` preserved float padding even when both inputs were
  integer, silently upcasting box arrays. Padding now uses
  `np.result_type(*inputs)` so homogeneous integer inputs stay integer.
- Pixel-mode now rejects non-integer ranges with an explicit error
  pointing at `normalized_range=True` or `bev_transform=BEV(...)`.
  Previously `np.random.randint` silently truncated floats — passing
  `(0.0, 1.0)` produced all-zero placements with no feedback.
- `_match_histogram` no longer routes through a misleading
  `cv2.COLOR_BGR2BGRA` constant + redundant `bitwise_and`; the RGBA
  array is reassembled directly from numpy slices.

### Docs
- README gained sections on reproducibility (`rng=`), the source-image
  cache, `blending_coeff` semantics (now a "ghost factor" over soft
  alpha), and the `OpaqueSourceWarning` rationale.
- `CapTorchvision` docstring now spells out the target merge rules
  (when boxes / labels / masks / semantic_mask are appended vs. dropped
  vs. created from scratch). Locked with a regression test.

## 0.2.3

Documentation-only release.

- Reformatted all Python code blocks in `README.md` to PEP 8 / black-compatible
  style (kwargs on their own lines under multi-line calls, consistent list
  spacing `[a, b]`, double-quoted strings, stdlib imports before third-party).
- Fixed wrapper-install instructions: now show `pip install
  "cap-augmentation[albumentations]"` instead of the editable `pip install -e
  ".[albumentations]"`, matching the canonical install section.
- BEV usage section explicitly states that `x_range` / `y_range` / `z_range` /
  `h_range` are interpreted in metres when `bev_transform` is set (was
  implicit before).
- Marked `albu_transforms` parameter as a deprecated alias of
  `object_transforms` in the README to match the code's docstring.
- Removed a duplicate `### Usage with multiple classes` heading; fixed typos
  ("cold be found", "cutted").
- `.gitignore`: added `results/` for ad-hoc augmentation outputs.

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
