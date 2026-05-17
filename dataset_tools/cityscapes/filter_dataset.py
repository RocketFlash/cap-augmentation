__author__ = "RocketFlash: https://github.com/RocketFlash"

import argparse
import os
from shutil import copyfile, rmtree

from scipy.io import loadmat
from tqdm import tqdm

try:
    from .config import data_filtering as cfg
except ImportError:
    from config import data_filtering as cfg

"""
    class_label =0: ignore regions (fake humans, e.g. people on posters, reflections etc.)
    class_label =1: pedestrians
    class_label =2: riders
    class_label =3: sitting persons
    class_label =4: other persons with unusual postures
    class_label =5: group of people
"""

NAME_TO_ID = {
    "ignore": 0,
    "pedestrian": 1,
    "rider": 2,
    "sitting": 3,
    "unusual": 4,
    "group": 5,
}


def filter_data(
    mat_file_path,
    allowed_classes=("pedestrian",),
    allowed_viz_area_ratio=0.8,
    min_h=0,
    min_w=0,
    max_h=10000,
    max_w=10000,
):

    postfix_image = "leftImg8bit.png"
    mat_file = loadmat(mat_file_path, mat_dtype=True)
    mat_file_name = mat_file_path.stem
    aligned_key = f"{mat_file_name}_aligned"
    if aligned_key not in mat_file:
        available_keys = sorted(k for k in mat_file.keys() if not k.startswith("__"))
        raise KeyError(
            f"Expected key {aligned_key!r} in {mat_file_path}; "
            f"available keys: {available_keys}"
        )
    mat = mat_file[aligned_key][0]
    allowed_classes_ids = [NAME_TO_ID[allowed_cl] for allowed_cl in allowed_classes]

    filtered_file_names = []

    for img_idx in tqdm(range(len(mat)), position=0):
        img_anno = mat[img_idx][0, 0]

        img_name_with_ext = img_anno[1][0]
        bboxes = img_anno[2]

        for bbox in bboxes:
            class_label, x1, y1, w, h, instance_id, x1_vis, y1_vis, w_vis, h_vis = bbox
            if class_label in allowed_classes_ids:
                area_tot = w * h
                area_viz = w_vis * h_vis
                area_ratio = area_viz / area_tot
                if area_ratio >= allowed_viz_area_ratio:
                    if (
                        w_vis >= min_w
                        and h_vis >= min_h
                        and w_vis <= max_w
                        and h_vis <= max_h
                    ):
                        filtered_file_name = img_name_with_ext.replace(
                            postfix_image, str(int(instance_id))
                        )
                        filtered_file_names.append(filtered_file_name)

    return filtered_file_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook", action="store_true", help="true if run it in jupyter notebook"
    )
    args = parser.parse_args()

    if args.notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    print("START DATASET FILTERING")
    if cfg.dataset_type == "citypersons":
        DATASET_ROOT = cfg.dataset_root
        FILTERED_SAVE_PATH = cfg.save_dir

        CITYPERSONS_ROOT = cfg.annos_path
        CITYPERSONS_ANNS_PATH = CITYPERSONS_ROOT / "annotations"

        SPLIT_DIRS = cfg.split_dirs
        ALLOWED_CLASSES = cfg.class_types

        CITYPERSONS_MAT_PATHS = [
            CITYPERSONS_ANNS_PATH / f"anno_{sd}.mat" for sd in SPLIT_DIRS
        ]

        if cfg.del_if_exist and FILTERED_SAVE_PATH.exists():
            rmtree(FILTERED_SAVE_PATH)
        os.makedirs(FILTERED_SAVE_PATH, exist_ok=True)
        copied_files_cntr = 0

        for CITYPERSONS_MAT_PATH in tqdm(
            CITYPERSONS_MAT_PATHS, position=0, desc="Total splits"
        ):
            data_names = filter_data(
                CITYPERSONS_MAT_PATH,
                allowed_classes=ALLOWED_CLASSES,
                allowed_viz_area_ratio=cfg.viz_ratio,
                min_h=cfg.min_h,
                min_w=cfg.min_w,
            )
            for image_name in tqdm(data_names, position=1):
                src_image_file_path = DATASET_ROOT / f"{image_name}.png"
                dst_image_file_path = FILTERED_SAVE_PATH / f"{image_name}.png"
                if src_image_file_path.is_file():
                    copyfile(src_image_file_path, dst_image_file_path)
                    copied_files_cntr += 1
        print(f"Total number of copied files: {copied_files_cntr}")
    print("DATASET FILTERED!")
