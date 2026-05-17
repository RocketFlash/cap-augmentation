from easydict import EasyDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

"""
    dataset_root - dataset path
    save_dir - save directory path
"""

data_generation = EasyDict(
    dict(
        images_path=Path("/dataset/kitti_format/vbd/png_keep_ratio/train/"),
        annotations_csv_path=Path(
            "/root/workdir/vbd/vbd/splits/train_detection_folds.csv"
        ),
        save_dir=REPO_ROOT / "data/vinbig_dataset",
        del_if_exist=False,
        fold_idx=0,
    )
)
