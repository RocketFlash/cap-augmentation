import albumentations as A
import cv2
import numpy as np
import pytest

from cap_augmentation import (
    CapAlbumentations,
    CapAug,
    CapAugMulticlass,
    ImageMaskTransform,
    resize_keep_ar,
)


def test_resize_keep_ar_uses_height_or_scale(make_source_image):
    image = np.zeros((20, 10, 4), dtype=np.uint8)

    by_height = resize_keep_ar(image, height=40)
    by_scale = resize_keep_ar(image, scale=0.5)

    assert by_height.shape[:2] == (40, 20)
    assert by_scale.shape[:2] == (10, 5)


def test_pixel_generation_returns_boxes_masks_and_image(
    destination_image, make_source_image
):
    source = make_source_image()
    aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        random_v_flip=False,
    )

    result, boxes, semantic_mask, instance_mask = aug(destination_image)

    assert boxes.tolist() == [[45, 60, 55, 80]]
    assert semantic_mask.sum() == 200
    assert instance_mask.sum() == 200
    assert result[70, 50].tolist() == [10, 20, 200]


def test_coordinate_formats_and_empty_outputs(destination_image, make_source_image):
    source = make_source_image()

    xywh_aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        coords_format="xywh",
    )
    _, xywh_boxes, _, _ = xywh_aug(destination_image)
    assert xywh_boxes.tolist() == [[45, 60, 10, 20]]

    yolo_aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        coords_format="yolo",
    )
    _, yolo_boxes, _, _ = yolo_aug(destination_image)
    np.testing.assert_allclose(yolo_boxes, [[0.5, 0.7, 0.1, 0.2]])

    offscreen_aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[-200, -199],
        y_range=[-200, -199],
        random_h_flip=False,
        coords_format="yolo",
    )
    _, empty_boxes, semantic_mask, instance_mask = offscreen_aug(destination_image)
    assert empty_boxes.shape == (0, 4)
    assert semantic_mask.sum() == 0
    assert instance_mask.sum() == 0


def test_normalized_range_and_probability_map(destination_image, make_source_image):
    source = make_source_image()

    normalized_aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[0.2, 0.2],
        x_range=[0.5, 0.5],
        y_range=[0.8, 0.8],
        normalized_range=True,
        random_h_flip=False,
    )
    _, normalized_boxes, _, _ = normalized_aug(destination_image)
    assert normalized_boxes.tolist() == [[45, 60, 55, 80]]

    with pytest.warns(DeprecationWarning):
        alias_aug = CapAug(
            [source],
            n_objects_range=[1, 1],
            h_range=[0.2, 0.2],
            x_range=[0.5, 0.5],
            y_range=[0.8, 0.8],
            normilized_range=True,
            random_h_flip=False,
        )
    _, alias_boxes, _, _ = alias_aug(destination_image)
    assert alias_boxes.tolist() == [[45, 60, 55, 80]]

    probability_map = np.array([[0, 0], [0, 10]], dtype=float)
    probability_aug = CapAug(
        [source],
        probability_map=probability_map,
        n_objects_range=[1, 1],
        h_range=[0.2, 0.2],
        random_h_flip=False,
    )
    _, probability_boxes, _, _ = probability_aug(destination_image)
    assert probability_boxes.tolist() == [[45, 30, 55, 50]]


def test_class_idx_blending_histogram_and_object_transforms(
    destination_image, make_source_image
):
    source = make_source_image(color=(20, 20, 20))
    destination_image[:] = 100
    transforms = A.Compose([A.InvertImg(p=1.0)])

    aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        histogram_matching=True,
        blending_coeff=0.5,
        class_idx=3,
        albu_transforms=transforms,
    )
    result, boxes, semantic_mask, _ = aug(destination_image)

    assert boxes.shape == (1, 5)
    assert boxes[0, -1] == 3
    assert semantic_mask.sum() == 200
    assert not np.array_equal(result[70, 50], destination_image[70, 50])


def test_object_transforms_accept_tuple_adapters(destination_image, make_source_image):
    source = make_source_image()

    def brighten(image, mask):
        return np.full_like(image, 120), mask

    aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        object_transforms=ImageMaskTransform(brighten),
    )

    result, boxes, semantic_mask, _ = aug(destination_image)

    assert boxes.tolist() == [[45, 60, 55, 80]]
    assert semantic_mask.sum() == 200
    assert result[70, 50].tolist() == [120, 120, 120]


def test_rejects_duplicate_object_transform_aliases(make_source_image):
    source = make_source_image()

    with pytest.raises(ValueError):
        CapAug(
            [source],
            albu_transforms=lambda **kwargs: kwargs,
            object_transforms=lambda **kwargs: kwargs,
        )


def test_rgb_grayscale_source_loads_as_four_channels(tmp_path):
    path = tmp_path / "gray.png"
    cv2.imwrite(str(path), np.full((4, 4), 17, dtype=np.uint8))
    aug = CapAug([path], image_format="rgb")

    selected = aug.select_image([path], 0)

    assert selected.shape == (4, 4, 4)
    assert selected[0, 0].tolist() == [17, 17, 17, 255]


def test_shape_changing_object_transform_does_not_crash(
    destination_image, make_source_image
):
    """Regression: object_transforms that crop/resize used to crash because
    src_h/src_w were captured before the transform ran but the mask ROI was
    sliced after it.
    """
    source = make_source_image()  # 20x10 alpha PNG

    def crop_to_half(image, mask):
        h, w = image.shape[:2]
        return image[: h // 2, : w // 2], mask[: h // 2, : w // 2]

    aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        random_v_flip=False,
        object_transforms=ImageMaskTransform(crop_to_half),
    )

    result, boxes, semantic_mask, _ = aug(destination_image)

    assert boxes.shape == (1, 4)
    box = boxes[0].tolist()
    # Object cropped to 10x5; placement uses post-transform dims so the box
    # is small and the result image was modified inside that box.
    assert box[2] - box[0] == 5 and box[3] - box[1] == 10
    assert semantic_mask.sum() == 50
    assert not np.array_equal(result[box[1] : box[3], box[0] : box[2]], 0)


def test_object_transform_with_mismatched_shapes_raises(
    destination_image, make_source_image
):
    """A transform that returns image and mask with different H/W should fail
    with a clear error instead of a downstream OpenCV assertion.
    """
    source = make_source_image()

    def bad_transform(image, mask):
        return image[:10, :5], mask  # image shrunk, mask untouched -> mismatch

    aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        object_transforms=ImageMaskTransform(bad_transform),
    )

    with pytest.raises(ValueError, match="same height/width"):
        aug(destination_image)


def test_soft_alpha_edges_blend_with_destination(tmp_path, destination_image):
    """Regression: a source with anti-aliased edges (intermediate alpha) used
    to be hard-thresholded by the bitwise composite. After the soft-alpha
    fix, edge pixels must be a smooth blend of source and destination.
    """
    src = np.zeros((20, 20, 4), dtype=np.uint8)
    src[:, :, :3] = (10, 20, 200)
    # Linear alpha ramp 0..255 across the width — every column has a
    # different mix of src and dst expected.
    for x in range(20):
        src[:, x, 3] = int(round(x * 255 / 19))
    src_path = tmp_path / "soft.png"
    cv2.imwrite(str(src_path), src)
    destination_image[:] = 200

    aug = CapAug(
        [src_path],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        random_v_flip=False,
    )
    result, *_ = aug(destination_image)

    # Sample the same row at three columns that map to alpha ≈ 0.25, 0.5, 0.75.
    # Object spans dst x = 40..60 (centered at 50). Inside-src x = dst_x - 40.
    row = result[70]
    for dst_x, src_x in [(45, 5), (50, 10), (55, 15)]:
        alpha = round(src_x * 255 / 19) / 255.0
        expected = np.clip(
            np.array((10, 20, 200), dtype=float) * alpha
            + np.array((200, 200, 200), dtype=float) * (1.0 - alpha),
            0,
            255,
        ).astype(np.uint8)
        np.testing.assert_allclose(row[dst_x], expected, atol=1)


def test_binary_alpha_compositing_is_unchanged(make_source_image, destination_image):
    """Sanity: for fully opaque (binary alpha 255) sources, the soft-alpha
    composite must produce the exact pixels the old bitwise composite did.
    Locks the behavior the README documents.
    """
    source = make_source_image()  # 20x10 fully opaque (alpha=255)
    destination_image[:] = 50
    aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
    )
    result, *_ = aug(destination_image)

    # Inside the box, pixels are exactly the source color.
    assert result[70, 50].tolist() == [10, 20, 200]
    # Outside the box, pixels are exactly the destination color.
    assert result[0, 0].tolist() == [50, 50, 50]


def test_alpha_padded_source_yields_tight_bbox(tmp_path, destination_image):
    """Regression: a PNG with transparent padding around the visible object
    used to return a bbox covering the full canvas. The bbox must match the
    visible alpha region instead.
    """
    # 20x20 canvas with a 10x10 fully opaque region centered (5..15, 5..15).
    src = np.zeros((20, 20, 4), dtype=np.uint8)
    src[5:15, 5:15, :3] = (10, 20, 200)
    src[5:15, 5:15, 3] = 255
    src_path = tmp_path / "padded.png"
    cv2.imwrite(str(src_path), src)

    aug = CapAug(
        [src_path],
        n_objects_range=[1, 1],
        h_range=[20, 21],  # canvas height
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        random_v_flip=False,
    )
    _, boxes, semantic_mask, _ = aug(destination_image)

    # Old (canvas-based) box would be [40, 60, 60, 80] (20x20 canvas).
    # Tight (alpha-based) box must match where the visible 10x10 pixels
    # actually landed in dst: canvas top-left = (40, 60), inner alpha
    # offset (5, 5), so [45, 65, 55, 75].
    assert boxes.tolist() == [[45, 65, 55, 75]]
    assert semantic_mask.sum() == 100  # exactly the 10x10 visible region


def test_missing_source_image_raises(destination_image, tmp_path):
    aug = CapAug(
        [tmp_path / "missing.png"],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
    )

    with pytest.raises(FileNotFoundError):
        aug(destination_image)


class FakeBEV:
    def meters_to_pixels(self, points):
        return np.array([[30, 80], [70, 80]], dtype=float)

    def calculate_dist_meters(self, points):
        return np.array([10, 20], dtype=float)

    def get_height_in_pixels(self, height, distance):
        del distance
        return height


def test_bev_generation_sorts_points_objects_and_z_offsets(
    destination_image, make_source_image
):
    near_source = make_source_image("near.png", color=(10, 0, 0))
    far_source = make_source_image("far.png", color=(0, 20, 0))
    aug = CapAug(
        [near_source, far_source],
        bev_transform=FakeBEV(),
        objects_idxs=[0, 1],
        random_h_flip=False,
    )

    points = np.array([[0, 10, 1], [0, 20, 3]], dtype=float)
    heights = np.array([20, 20])
    result, boxes, _, _ = aug.generate_objects_coord(
        destination_image, points, heights=heights, scales=None
    )

    assert boxes.tolist() == [[65, 57, 75, 77], [25, 59, 35, 79]]
    assert result[70, 70].tolist() == [0, 20, 0]
    assert result[70, 30].tolist() == [10, 0, 0]


def test_multiclass_keeps_class_ids_when_some_augmenters_are_skipped(destination_image):
    class DummyAug:
        def __init__(self, x1):
            self.x1 = x1

        def __call__(self, image):
            semantic_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            semantic_mask[1:3, self.x1 : self.x1 + 2] = 1
            instance_mask = semantic_mask.copy()
            boxes = np.array([[self.x1, 1, self.x1 + 2, 3]])
            return image, boxes, semantic_mask, instance_mask

    multiclass = CapAugMulticlass(
        [DummyAug(1), DummyAug(5)],
        probabilities=[0.0, 1.0],
        class_idxs=[2, 7],
    )

    _, boxes, semantic_mask, instance_masks = multiclass(destination_image)

    assert boxes.tolist() == [[5, 1, 7, 3, 7]]
    assert set(np.unique(semantic_mask)) == {0, 7}
    assert len(instance_masks) == 1


def test_multiclass_aligns_boxes_with_existing_class_columns(destination_image):
    class DummyAug:
        def __init__(self, boxes):
            self.boxes = np.asarray(boxes)

        def __call__(self, image):
            semantic_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            instance_mask = semantic_mask.copy()
            return image, self.boxes, semantic_mask, instance_mask

    multiclass = CapAugMulticlass(
        [DummyAug([[1, 1, 3, 3, 99]]), DummyAug([[5, 1, 7, 3]])],
        probabilities=[1.0, 1.0],
        class_idxs=[2, 7],
    )

    _, boxes, _, _ = multiclass(destination_image)

    assert boxes.tolist() == [[1, 1, 3, 3, 2], [5, 1, 7, 3, 7]]


def test_cap_albu_adds_image_mask_and_bboxes(destination_image, make_source_image):
    source = make_source_image()
    transform = A.Compose(
        [
            CapAlbumentations(
                p=1.0,
                source_images=[source],
                n_objects_range=[1, 1],
                h_range=[20, 21],
                x_range=[50, 51],
                y_range=[80, 81],
                random_h_flip=False,
                class_idx=4,
            )
        ],
        bbox_params=A.BboxParams(format="pascal_voc"),
    )
    mask = np.zeros(destination_image.shape[:2], dtype=np.uint8)

    transformed = transform(
        image=destination_image,
        bboxes=np.array([[1, 1, 3, 3, 1]], dtype=float),
        mask=mask,
    )

    assert len(transformed["bboxes"]) == 2
    assert np.array(transformed["mask"]).sum() == 200
    generated_bbox = np.asarray(transformed["bboxes"][-1])
    np.testing.assert_allclose(generated_bbox[:4], [45, 60, 55, 80])
    assert generated_bbox[4] == 4
