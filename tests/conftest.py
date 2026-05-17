import random

import cv2
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def deterministic_random():
    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def destination_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def make_source_image(tmp_path):
    def _make_source_image(
        name="object.png", color=(10, 20, 200), size=(20, 10), alpha=255
    ):
        h, w = size
        image = np.zeros((h, w, 4), dtype=np.uint8)
        image[:, :, :3] = color
        image[:, :, 3] = alpha
        path = tmp_path / name
        cv2.imwrite(str(path), image)
        return path

    return _make_source_image
