# coding: utf-8
__author__ = "RocketFlash: https://github.com/RocketFlash"

import os
import cv2
import pandas as pd
from tqdm import tqdm
from shutil import rmtree

try:
    from .config import data_generation as cfg
except ImportError:
    from config import data_generation as cfg


def generate_object_dataset_vinbig(
    annotations_csv_path, images_path, save_dir, fold_idx=None
):
    df = pd.read_csv(annotations_csv_path)
    if fold_idx is not None:
        df = df[df.fold != fold_idx]

    gb = df.groupby("image_id")
    extension = ".png"
    total_counter = {}

    for name, group in tqdm(gb, total=len(gb)):
        group.reset_index(inplace=True)
        image = cv2.imread(str(images_path / f"{name}{extension}"))
        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {images_path / f'{name}{extension}'}"
            )

        counter = {}

        for index, row in group.iterrows():
            if float(row["x_max"]) <= float(row["x_min"]) or float(
                row["y_max"]
            ) <= float(row["y_min"]):
                continue
            class_img = image[
                int(row["y_min"]) : int(row["y_max"]),
                int(row["x_min"]) : int(row["x_max"]),
            ]

            class_id = str(row["class_id"])
            if class_id not in counter:
                counter[class_id] = 0

            if class_id not in total_counter:
                total_counter[class_id] = 0
                class_save_dir = save_dir / class_id
                class_save_dir.mkdir(exist_ok=True)

            counter[class_id] += 1
            total_counter[class_id] += 1

            save_img_path = (
                save_dir / class_id / f"{name}_{class_id}_{counter[class_id]}.png"
            )
            cv2.imwrite(str(save_img_path), class_img)


if __name__ == "__main__":

    if cfg.images_path is None or cfg.annotations_csv_path is None:
        raise SystemExit(
            "Set images_path and annotations_csv_path in dataset_tools/vinbig/config.py "
            "before running this script."
        )

    print("START DATASET GENERATION")

    IMAGES_PATH = cfg.images_path
    ANNOTATIONS_CSV_PATH = cfg.annotations_csv_path
    SAVE_DIR = cfg.save_dir

    if cfg.del_if_exist:
        if SAVE_DIR.exists():
            rmtree(SAVE_DIR)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    generate_object_dataset_vinbig(
        annotations_csv_path=cfg.annotations_csv_path,
        images_path=IMAGES_PATH,
        save_dir=SAVE_DIR,
        fold_idx=cfg.fold_idx,
    )
    print("DATASET GENERATED!")
