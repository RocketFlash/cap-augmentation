__author__ = "RocketFlash: https://github.com/RocketFlash"

import argparse
import os
from shutil import rmtree

import cv2
import numpy as np
from tqdm import tqdm

try:
    from .config import data_generation as cfg
except ImportError:
    from config import data_generation as cfg

NAME_TO_ID = {"pedestrian": 24}


def generate_object_dataset_cityscapes(
    annotations_path,
    images_path,
    save_dir,
    split_dirs=("train",),
    object_name="pedestrian",
):
    """
    Function description
    """
    postfix_image = "leftImg8bit"
    postfix_label = "gtFine_labelIds"
    postfix_instance = "gtFine_instanceIds"
    obj_id = NAME_TO_ID[object_name]

    for split_dir in tqdm(split_dirs, position=0, desc="Total splits"):
        SPLIT_ANNOS = annotations_path / split_dir
        SPLIT_IMAGES = images_path / split_dir

        CITY_DIRS = SPLIT_ANNOS.glob("*/")
        CITY_NAMES = [city_path.name for city_path in CITY_DIRS]

        for city_name in tqdm(
            CITY_NAMES, position=1, desc=f"{split_dir:5} split cities: "
        ):
            city_images_path = SPLIT_IMAGES / city_name
            city_annos_path = SPLIT_ANNOS / city_name
            city_images = city_images_path.glob("*.png")

            for city_image in tqdm(
                list(city_images), position=2, desc=f"{city_name:15} city images: "
            ):
                mask_label_path = city_annos_path / city_image.name.replace(
                    postfix_image, postfix_label
                )
                mask_instance_path = city_annos_path / city_image.name.replace(
                    postfix_image, postfix_instance
                )

                image = cv2.imread(str(city_image))
                mask_label = cv2.imread(str(mask_label_path), 0)
                mask_instance = cv2.imread(str(mask_instance_path), cv2.IMREAD_ANYDEPTH)
                if image is None or mask_label is None or mask_instance is None:
                    raise FileNotFoundError(
                        f"Could not read image or masks for {city_image}"
                    )

                mask_label_obj = np.zeros_like(mask_label)
                mask_instance_obj = np.zeros_like(mask_instance)
                mask_label_obj[mask_label == obj_id] = 1

                mask_instance_obj = cv2.bitwise_and(
                    mask_instance, mask_instance, mask=mask_label_obj
                )
                instance_ids = np.unique(mask_instance_obj)
                instance_ids = instance_ids[instance_ids != 0]

                for instance_id in instance_ids:
                    mask_instance_obj_curr = np.zeros_like(
                        mask_instance_obj, dtype=np.uint8
                    )
                    mask_instance_obj_curr[mask_instance_obj == instance_id] = 255
                    image_instance = cv2.bitwise_and(
                        image, image, mask=mask_instance_obj_curr
                    )

                    blob_mask = np.where(mask_instance_obj_curr > 0)
                    mask_ys, mask_xs = blob_mask[0], blob_mask[1]

                    blob_x1, blob_x2 = min(mask_xs), max(mask_xs) + 1
                    blob_y1, blob_y2 = min(mask_ys), max(mask_ys) + 1

                    obj_region_image = image_instance[
                        blob_y1:blob_y2, blob_x1:blob_x2, :
                    ]
                    obj_region_mask = mask_instance_obj_curr[
                        blob_y1:blob_y2, blob_x1:blob_x2
                    ]

                    n_nonzero = len(obj_region_mask[obj_region_mask > 0])
                    n_total = obj_region_mask.shape[0] * obj_region_mask.shape[1]

                    if n_nonzero / n_total > 0.5:
                        obj_region_image_path = save_dir / city_image.name.replace(
                            postfix_image, f"{instance_id}"
                        )

                        obj_rgba = cv2.cvtColor(obj_region_image, cv2.COLOR_BGR2BGRA)
                        obj_rgba[:, :, 3] = obj_region_mask

                        cv2.imwrite(str(obj_region_image_path), obj_rgba)


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
    print("START DATASET GENERATION")
    if cfg.dataset_type == "cityscapes":
        DATASET_ROOT = cfg.dataset_root
        ANNOTATIONS_PATH = DATASET_ROOT / "gtFine/"
        IMAGES_PATH = DATASET_ROOT / "leftImg8bit/"

        SAVE_DIR = cfg.save_dir

        if cfg.del_if_exist and SAVE_DIR.exists():
            rmtree(SAVE_DIR)
        os.makedirs(SAVE_DIR, exist_ok=True)

        SPLIT_DIRS = cfg.split_dirs

        generate_object_dataset_cityscapes(
            annotations_path=ANNOTATIONS_PATH,
            images_path=IMAGES_PATH,
            save_dir=SAVE_DIR,
            split_dirs=SPLIT_DIRS,
            object_name=cfg.obj_type,
        )
    print("DATASET GENERATED!")
