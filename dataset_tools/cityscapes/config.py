from pathlib import Path

from easydict import EasyDict

REPO_ROOT = Path(__file__).resolve().parents[2]

"""
    dataset_root - dataset path
    dataset_type - dataset type
    save_dir - save directory path
    split_dirs - split directories to use from [train, val]
    obj_type - type of the object to parse
"""

data_generation = EasyDict(
    dict(
        dataset_root=REPO_ROOT / "data/cityscapes",
        dataset_type="cityscapes",
        save_dir=REPO_ROOT / "data/human_dataset",
        split_dirs=["train", "val"],
        obj_type="pedestrian",
        del_if_exist=False,
    )
)


"""
    dataset_root - path to generated dataset
    dataset_type - dataset type
    annos_path - path to annotations (path to citipersons dataset)
    save_dir - save directory path
    split_dirs - split directories to use from [train, val]
    class_types - allowed class types
    viz_ratio - allowed vizible part ratio
    min_h - min number of pixels in height
    min_w - min number of pixels in width
    max_h - max number of pixels in height
    max_w - max number of pixels in width
    del_if_exist - delete save directory if it's already exists
"""

data_filtering = EasyDict(
    dict(
        dataset_root=REPO_ROOT / "data/human_dataset",
        dataset_type="citypersons",
        annos_path=REPO_ROOT / "data/CityPersons",
        save_dir=REPO_ROOT / "data/human_dataset_filtered",
        split_dirs=["train", "val"],
        class_types=["pedestrian"],
        viz_ratio=0.8,
        min_h=100,
        min_w=0,
        max_h=10000,
        max_w=10000,
        del_if_exist=False,
    )
)
