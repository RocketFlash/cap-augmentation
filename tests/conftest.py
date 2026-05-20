import random
import warnings

import cv2
import numpy as np
import pytest

from cap_augmentation import OpaqueSourceWarning


@pytest.fixture(autouse=True)
def deterministic_random():
    random.seed(0)
    np.random.seed(0)


@pytest.fixture(autouse=True)
def _silence_opaque_source_warnings():
    """Most fixtures intentionally use fully opaque sources to focus on the
    paste mechanics. Suppress the OpaqueSourceWarning by default so it
    doesn't drown logs; tests that exercise the warning opt in via
    pytest.warns().
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OpaqueSourceWarning)
        yield


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
