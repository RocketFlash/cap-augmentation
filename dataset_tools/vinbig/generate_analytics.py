__author__ = "RocketFlash: https://github.com/RocketFlash"

from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .config import data_generation as cfg
except ImportError:
    from config import data_generation as cfg


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


CLASSES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]


def heatmap2d(arr: np.ndarray, save_path: str, title: str):
    plt.figure()
    plt.imshow(arr, cmap="viridis")
    plt.colorbar()
    plt.suptitle(title)
    plt.savefig(save_path)


def generate_bboxes_distribution(
    annotations_csv_path, save_dir, fold_idx=None, output_img_size=1000
):
    """Write per-class probability maps and metadata.

    The ``.npy`` metadata files contain Python dictionaries, so consumers must
    load them with ``allow_pickle=True``. Only load files from trusted sources.
    """
    df = pd.read_csv(annotations_csv_path)
    if fold_idx is not None:
        df = df[df.fold != fold_idx]

    gb = df.groupby("image_id")

    total_counter = {}
    bboxes_distributions = {}

    for name, group in tqdm(gb, total=len(gb)):
        group.reset_index(inplace=True)

        for index, row in group.iterrows():
            if float(row["x_max"]) <= float(row["x_min"]) or float(
                row["y_max"]
            ) <= float(row["y_min"]):
                continue

            class_id = str(row["class_id"])
            if class_id not in total_counter:
                total_counter[class_id] = AverageMeter()
                bboxes_distributions[class_id] = np.zeros(
                    (output_img_size, output_img_size)
                )

            total_counter[class_id].update(float(row["h_norm"]))
            x_min, x_max, y_min, y_max = [
                int(float(val) * output_img_size)
                for val in [
                    row["x_min_norm"],
                    row["x_max_norm"],
                    row["y_min_norm"],
                    row["y_max_norm"],
                ]
            ]
            bboxes_distributions[class_id][y_min:y_max, x_min:x_max] += 1

    for class_id, bboxes_distribution in bboxes_distributions.items():
        total_sum = np.sum(bboxes_distribution)
        bboxes_distribution_norm = bboxes_distribution / total_sum

        heatmap2d(
            bboxes_distribution_norm,
            save_dir / f"{class_id}.png",
            CLASSES[int(class_id)],
        )

        to_save_dict = {
            "probability_map": bboxes_distribution_norm,
            "mean_h": total_counter[class_id].avg,
            "n_bboxes": total_counter[class_id].count,
        }

        with open(save_dir / f"{class_id}.npy", "wb") as f:
            np.save(f, to_save_dict)


if __name__ == "__main__":

    if cfg.annotations_csv_path is None:
        raise SystemExit(
            "Set annotations_csv_path in dataset_tools/vinbig/config.py "
            "before running this script."
        )

    ANNOTATIONS_CSV_PATH = cfg.annotations_csv_path
    SAVE_DIR = cfg.save_dir / "analytics/"

    if cfg.del_if_exist:
        if SAVE_DIR.exists():
            rmtree(SAVE_DIR)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    generate_bboxes_distribution(
        annotations_csv_path=cfg.annotations_csv_path,
        save_dir=SAVE_DIR,
        fold_idx=cfg.fold_idx,
    )
