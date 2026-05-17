import numpy as np

from cap_augmentation.utils import draw_bboxes, show_image_and_masks


def test_draw_bboxes_supports_class_last_boxes(destination_image):
    boxes = np.array([[10, 20, 30, 40, 5]])

    result, mask = draw_bboxes(destination_image, boxes)

    assert mask is None
    assert result.shape == destination_image.shape
    assert result.sum() > 0


def test_draw_bboxes_overlays_instance_masks(destination_image):
    mask = np.zeros(destination_image.shape[:2], dtype=np.uint8)
    mask[10:20, 10:20] = 1

    result, result_mask = draw_bboxes(
        destination_image, np.array([[10, 10, 20, 20]]), mask=mask
    )

    assert result.shape == destination_image.shape
    assert result_mask.shape == destination_image.shape
    assert result_mask.sum() > 0


def test_show_image_and_masks_accepts_semantic_masks(monkeypatch, destination_image):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    mask = np.zeros(destination_image.shape[:2], dtype=np.uint8)
    mask[10:20, 10:20] = 2

    show_image_and_masks(
        destination_image,
        destination_image,
        mask,
        destination_image,
        is_mask_semantic=True,
    )
