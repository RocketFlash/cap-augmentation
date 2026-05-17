import cv2
import numpy as np
import pytest

from dataset_tools.cityscapes import filter_dataset as city_filter
from dataset_tools.cityscapes.generate_dataset import generate_object_dataset_cityscapes


def test_cityscapes_generator_cuts_rgba_objects_with_bgr_color(tmp_path):
    annotations_path = tmp_path / "gtFine"
    images_path = tmp_path / "leftImg8bit"
    save_dir = tmp_path / "objects"
    city_annos = annotations_path / "train" / "city"
    city_images = images_path / "train" / "city"
    city_annos.mkdir(parents=True)
    city_images.mkdir(parents=True)
    save_dir.mkdir()

    image_name = "city_000000_000000_leftImg8bit.png"
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2:6, 3:7] = [10, 20, 30]
    label = np.zeros((8, 8), dtype=np.uint8)
    label[2:6, 3:7] = 24
    instance = np.zeros((8, 8), dtype=np.uint16)
    instance[2:6, 3:7] = 1000

    cv2.imwrite(str(city_images / image_name), image)
    cv2.imwrite(
        str(city_annos / image_name.replace("leftImg8bit", "gtFine_labelIds")),
        label,
    )
    cv2.imwrite(
        str(city_annos / image_name.replace("leftImg8bit", "gtFine_instanceIds")),
        instance,
    )

    generate_object_dataset_cityscapes(
        annotations_path=annotations_path,
        images_path=images_path,
        save_dir=save_dir,
        split_dirs=["train"],
    )

    output = cv2.imread(
        str(save_dir / image_name.replace("leftImg8bit", "1000")), cv2.IMREAD_UNCHANGED
    )
    assert output.shape == (4, 4, 4)
    assert output[0, 0, :3].tolist() == [10, 20, 30]
    assert output[:, :, 3].min() == 255


def test_citypersons_filter_data_filters_by_class_visibility_and_size(
    monkeypatch, tmp_path
):
    class FakeEntry:
        def __init__(self, record):
            self.record = record

        def __getitem__(self, key):
            assert key == (0, 0)
            return self.record

    class FakeMat:
        def __init__(self, record):
            self.record = record

        def __len__(self):
            return 1

        def __getitem__(self, key):
            if key == 0:
                return FakeEntry(self.record)
            raise IndexError

    class FakeAligned:
        def __init__(self, record):
            self.record = record

        def __getitem__(self, key):
            if key == 0:
                return FakeMat(self.record)
            raise IndexError

    bboxes = np.array(
        [
            [1, 0, 0, 10, 20, 123, 0, 0, 10, 20],
            [2, 0, 0, 10, 20, 999, 0, 0, 10, 20],
            [1, 0, 0, 10, 20, 555, 0, 0, 1, 1],
        ],
        dtype=float,
    )
    record = [None, np.array(["city_000000_000000_leftImg8bit.png"]), bboxes]
    root = FakeAligned(record)

    monkeypatch.setattr(
        city_filter,
        "loadmat",
        lambda path, mat_dtype=True: {"anno_train_aligned": root},
    )

    names = city_filter.filter_data(
        tmp_path / "anno_train.mat",
        allowed_classes=["pedestrian"],
        allowed_viz_area_ratio=0.8,
        min_h=10,
    )

    assert names == ["city_000000_000000_123"]


def test_citypersons_filter_data_reports_missing_aligned_key(monkeypatch, tmp_path):
    monkeypatch.setattr(
        city_filter,
        "loadmat",
        lambda path, mat_dtype=True: {"unexpected_key": np.array([])},
    )

    with pytest.raises(KeyError, match="anno_train_aligned"):
        city_filter.filter_data(tmp_path / "anno_train.mat")
