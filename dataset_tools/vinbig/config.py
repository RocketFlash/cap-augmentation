from pathlib import Path

from easydict import EasyDict

REPO_ROOT = Path(__file__).resolve().parents[2]

"""
    images_path - directory of VinBigData PNG images (must be set before running)
    annotations_csv_path - path to the train fold-annotated CSV (must be set before running)
    save_dir - where per-class crops + analytics will be written
    del_if_exist - delete save directory if it's already exists
    fold_idx - validation fold to exclude (None to use all folds)
"""

data_generation = EasyDict(
    dict(
        images_path=None,
        annotations_csv_path=None,
        save_dir=REPO_ROOT / "data/vinbig_dataset",
        del_if_exist=False,
        fold_idx=0,
    )
)
