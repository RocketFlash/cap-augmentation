import random
import warnings

import albumentations as A
import cv2
import numpy as np
import pytest

from cap_augmentation import (
    CapAlbumentations,
    CapAug,
    CapAugMulticlass,
    ImageMaskTransform,
    OpaqueSourceWarning,
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

    with pytest.warns(OpaqueSourceWarning, match="grayscale"):
        selected = aug.select_image([path], 0)

    assert selected.shape == (4, 4, 4)
    assert selected[0, 0].tolist() == [17, 17, 17, 255]


def test_source_without_alpha_emits_warning_once(tmp_path):
    path = tmp_path / "opaque.png"
    cv2.imwrite(str(path), np.full((4, 4, 3), 99, dtype=np.uint8))
    aug = CapAug([path])

    with pytest.warns(OpaqueSourceWarning, match="no alpha channel"):
        aug.select_image([path], 0)

    # Repeated reads of the same source must not re-warn — noisy in training loops.
    with warnings.catch_warnings():
        warnings.simplefilter("error", OpaqueSourceWarning)
        aug.select_image([path], 0)


def test_fully_opaque_alpha_emits_warning(tmp_path):
    path = tmp_path / "opaque_alpha.png"
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., :3] = 50
    rgba[..., 3] = 255  # alpha present in file, but every pixel fully opaque
    cv2.imwrite(str(path), rgba)
    aug = CapAug([path])

    with pytest.warns(OpaqueSourceWarning, match="fully opaque"):
        aug.select_image([path], 0)


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


def test_object_transform_returning_four_channels_raises(
    destination_image, make_source_image
):
    """CapAug separates RGB from alpha before invoking the transform. A
    transform that hands back a 4-channel image would broadcast a (H, W)
    alpha against a (H, W, 4) src downstream and produce silently wrong
    pixels — surface the contract instead.
    """
    source = make_source_image()

    def alpha_in_image(image, mask):
        rgba = np.dstack([image, mask])  # 4 channels — wrong by contract
        return rgba, mask

    aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        object_transforms=ImageMaskTransform(alpha_in_image),
    )

    with pytest.raises(ValueError, match="3-channel image"):
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


def test_max_overlap_zero_skips_all_overlapping_pastes(
    make_source_image, destination_image
):
    """With max_overlap=0, two pastes whose proposed bboxes overlap each
    other at all must result in only one accepted paste — the first one
    wins, subsequent overlapping pastes are rolled back.
    """
    source = make_source_image()  # 20x10 fully opaque
    # All pastes land at the same (x_coord, y_coord), so every candidate
    # after the first overlaps the accepted box at IoU == 1.
    aug = CapAug(
        [source],
        n_objects_range=[5, 5],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        max_overlap=0.0,
    )
    _, boxes, semantic_mask, _ = aug(destination_image)

    assert boxes.shape == (1, 4)
    # The semantic mask must reflect a single 20×10 paste only.
    assert int(semantic_mask.sum()) == 200


def test_max_overlap_default_off_preserves_existing_behavior(
    make_source_image, destination_image
):
    """Regression guard: with max_overlap=None (default), all candidates
    paste even if they fully overlap. Locks the pre-0.5 behavior.
    """
    source = make_source_image()
    aug = CapAug(
        [source],
        n_objects_range=[3, 3],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
    )
    _, boxes, *_ = aug(destination_image)
    assert boxes.shape[0] == 3


def test_max_overlap_partial_allows_some_overlap(make_source_image, destination_image):
    """max_overlap=0.5 must accept pastes that overlap at IoU < 0.5 and
    reject pastes that overlap more. Two pastes at IoU ~= 0.33 (shared
    bottom-half band) should both be accepted.
    """
    source = make_source_image()  # 20x10
    aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        max_overlap=0.5,
    )
    # Place two objects deterministically via generate_objects_coord. With
    # 20×10 sources foot-anchored at (50, 80) and (50, 90), the second box
    # is [45, 70, 55, 90] and the first is [45, 60, 55, 80]. They share a
    # height-10 strip → IoU = 0 (touching but not overlapping). Stress the
    # accept side first.
    points = np.array([[50, 80], [50, 90]], dtype=float)
    heights = np.array([20, 20], dtype=int)
    _, boxes, *_ = aug.generate_objects_coord(
        destination_image, points, heights, scales=None
    )
    assert boxes.shape[0] == 2


def test_max_overlap_validates_range(make_source_image):
    source = make_source_image()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CapAug([source], max_overlap=1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CapAug([source], max_overlap=-0.1)


def test_probability_map_normalization_is_cached(make_source_image, destination_image):
    """Regression: probability_map was re-summed and re-divided on every
    aug() call. For a 1000x1000 map that's ~1 MB of busywork per training
    step. Verify the normalised flat array is computed once and reused.
    """
    source = make_source_image()
    probability_map = np.zeros((50, 50), dtype=float)
    probability_map[40, 25] = 1.0
    aug = CapAug(
        [source],
        probability_map=probability_map,
        n_objects_range=[1, 1],
        h_range=[0.2, 0.2],
        random_h_flip=False,
    )

    aug(destination_image)
    cached_first = aug._normalized_probability_map
    assert cached_first is not None

    aug(destination_image)
    cached_second = aug._normalized_probability_map
    # `is`: must be the exact same tuple, not a freshly recomputed one.
    assert cached_second is cached_first


def test_probability_map_rejects_non_2d_input(make_source_image, destination_image):
    source = make_source_image()
    aug = CapAug(
        [source],
        probability_map=np.zeros((3, 3, 3)),
        n_objects_range=[1, 1],
        h_range=[0.2, 0.2],
    )
    with pytest.raises(ValueError, match="must be 2D"):
        aug(destination_image)


def test_source_image_cache_avoids_repeated_disk_reads(
    monkeypatch, make_source_image, destination_image
):
    source = make_source_image()
    aug = CapAug(
        [source],
        n_objects_range=[3, 3],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
    )

    reads = []
    real_imread = cv2.imread

    def counting_imread(path, *args, **kwargs):
        reads.append(path)
        return real_imread(path, *args, **kwargs)

    monkeypatch.setattr(cv2, "imread", counting_imread)
    aug(destination_image)
    aug(destination_image)

    assert reads.count(str(source)) == 1


def test_cache_size_zero_disables_caching(
    monkeypatch, make_source_image, destination_image
):
    source = make_source_image()
    aug = CapAug(
        [source],
        n_objects_range=[2, 2],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        cache_size=0,
    )

    reads = []
    real_imread = cv2.imread

    def counting_imread(path, *args, **kwargs):
        reads.append(path)
        return real_imread(path, *args, **kwargs)

    monkeypatch.setattr(cv2, "imread", counting_imread)
    aug(destination_image)

    assert reads.count(str(source)) == 2


def test_cache_evicts_least_recently_used(make_source_image, destination_image):
    sources = [make_source_image(name=f"s{i}.png") for i in range(3)]
    aug = CapAug(
        sources,
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        cache_size=2,
    )

    for idx in [0, 1, 2, 1]:
        aug.select_image(sources, idx)

    assert (str(sources[0]), aug.image_format) not in aug._image_cache
    assert {key[0] for key in aug._image_cache} == {str(sources[1]), str(sources[2])}


def test_histogram_matching_changes_pasted_pixels(make_source_image, destination_image):
    """Lock the histogram-matching behavior: when the source and destination
    histograms differ, the pasted pixels should *not* be the raw source
    colour. The old test only checked shape/class; this proves the matching
    actually runs end-to-end.
    """
    source = make_source_image(color=(20, 20, 20))
    destination_image[:] = 200  # bright BG forces histogram matching to shift src
    aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        histogram_matching=True,
    )
    result, *_ = aug(destination_image)

    pasted = result[70, 50].tolist()
    # The original source color was (20, 20, 20); after matching against a
    # uniform 200 destination the values must shift away from 20.
    assert pasted != [20, 20, 20]
    assert max(pasted) > 100


def test_pixel_mode_rejects_float_ranges(make_source_image, destination_image):
    """Regression: numpy.random.randint silently truncates floats, so a user
    who passes (0.5, 0.8) intending normalized coordinates would get all-zero
    placements. Validate up front and point at the right mode flag.
    """
    source = make_source_image()
    aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[0.5, 0.8],
        y_range=[0.5, 0.8],
    )

    with pytest.raises(ValueError, match="normalized_range=True"):
        aug(destination_image)


def test_pixel_mode_accepts_int_typed_floats(make_source_image, destination_image):
    """Whole-number floats like 50.0 are user-friendly; treat them as valid
    pixel coordinates rather than rejecting on dtype alone.
    """
    source = make_source_image()
    aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20.0, 21.0],
        x_range=[50.0, 51.0],
        y_range=[80.0, 81.0],
        random_h_flip=False,
    )
    _, boxes, *_ = aug(destination_image)
    assert boxes.shape == (1, 4)


def test_rng_seed_is_reproducible_without_global_seeding(
    make_source_image, destination_image
):
    """Two CapAug instances seeded with the same int must produce
    bit-identical results regardless of the global random/numpy state.
    """
    source = make_source_image()

    def run_with_seed(seed):
        # Deliberately scramble global state between calls.
        random.seed(12345)
        np.random.seed(67890)
        aug = CapAug(
            [source],
            n_objects_range=[2, 4],
            h_range=[15, 25],
            x_range=[10, 90],
            y_range=[40, 95],
            rng=seed,
        )
        return aug(destination_image)

    img_a, boxes_a, sem_a, _ = run_with_seed(42)
    img_b, boxes_b, sem_b, _ = run_with_seed(42)
    img_c, boxes_c, sem_c, _ = run_with_seed(7)

    np.testing.assert_array_equal(img_a, img_b)
    np.testing.assert_array_equal(boxes_a, boxes_b)
    np.testing.assert_array_equal(sem_a, sem_b)

    # Different seed → different result (sanity check).
    assert not np.array_equal(boxes_a, boxes_c) or not np.array_equal(img_a, img_c)


def test_rng_accepts_numpy_generator(make_source_image, destination_image):
    source = make_source_image()
    aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        rng=np.random.default_rng(11),
    )
    _, boxes, *_ = aug(destination_image)
    assert boxes.shape == (1, 4)


def test_rng_rejects_invalid_type(make_source_image):
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        CapAug([make_source_image()], rng="not-a-seed")


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


def test_image_format_rgb_swaps_channels_end_to_end(tmp_path):
    """End-to-end: with image_format='rgb', cv2 reads source PNGs as BGR
    and the loader must swap to RGB before pasting. The pasted pixel
    must therefore match the RGB-interpreted source colour.
    """
    # Write a 4-channel source. cv2.imwrite serialises BGRA, so the bytes
    # on disk are (B=10, G=20, R=200, A=255).
    src = np.zeros((20, 10, 4), dtype=np.uint8)
    src[:, :, :3] = (10, 20, 200)
    src[:, :, 3] = 255
    src_path = tmp_path / "rgb_src.png"
    cv2.imwrite(str(src_path), src)

    dst = np.zeros((100, 100, 3), dtype=np.uint8)
    aug = CapAug(
        [src_path],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        image_format="rgb",
    )
    result, *_ = aug(dst)

    # Under image_format='rgb' the loader swaps B↔R, so the in-memory
    # pixel is (200, 20, 10) — the same triplet the user would write to
    # an RGB-shaped destination buffer.
    assert result[70, 50].tolist() == [200, 20, 10]


def test_s_range_scales_source_image(make_source_image, destination_image):
    """When h_range is None, CapAug picks a scale from s_range and resizes
    the source by that factor. Source is 20x10; scale 2.0 → object footprint
    is 40x20 px.
    """
    source = make_source_image()
    aug = CapAug(
        [source],
        n_objects_range=[1, 1],
        h_range=None,
        s_range=(2.0, 2.0),
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
    )
    _, boxes, _, _ = aug(destination_image)

    x1, y1, x2, y2 = boxes[0].tolist()
    assert (x2 - x1, y2 - y1) == (20, 40)  # 10*2, 20*2


def test_multiclass_with_bev_transform(destination_image, make_source_image):
    """CapAugMulticlass should compose per-class augmenters that use
    bev_transform. Regression net: previously this combination had no
    coverage, despite being the headline 3D-aware feature.
    """
    near_source = make_source_image("ped.png", color=(10, 0, 0))
    far_source = make_source_image("car.png", color=(0, 20, 0))

    bev = FakeBEV()
    pedestrians = CapAug(
        [near_source],
        bev_transform=bev,
        objects_idxs=[0, 0],
        random_h_flip=False,
    )
    cars = CapAug(
        [far_source],
        bev_transform=bev,
        objects_idxs=[0, 0],
        random_h_flip=False,
    )

    # Bypass random sampling by calling generate_objects_coord directly
    # via a thin adapter exposed as __call__.
    class FixedPlacement:
        def __init__(self, inner):
            self._inner = inner

        def __call__(self, image):
            points = np.array([[0, 10, 1], [0, 20, 3]], dtype=float)
            heights = np.array([20, 20], dtype=float)
            return self._inner.generate_objects_coord(image, points, heights, None)

    multiclass = CapAugMulticlass(
        [FixedPlacement(pedestrians), FixedPlacement(cars)],
        probabilities=[1.0, 1.0],
        class_idxs=[1, 2],
    )
    _, boxes, sem, instance_masks = multiclass(destination_image)

    assert boxes.shape == (4, 5)
    assert set(boxes[:, 4].astype(int).tolist()) == {1, 2}
    assert set(np.unique(sem)) <= {0, 1, 2}
    assert len(instance_masks) == 2


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


def test_multiclass_padding_preserves_integer_dtype(destination_image):
    """Regression: zero-padding used np.zeros() (float64 default), so
    appending a 5-col int array to a 6-col int array upcast both to float
    and broke `.tolist()` integer comparisons downstream.
    """

    class DummyAug:
        def __init__(self, boxes):
            self.boxes = np.asarray(boxes, dtype=np.int64)

        def __call__(self, image):
            semantic_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            return image, self.boxes, semantic_mask, semantic_mask.copy()

    # First augmenter returns 6 columns (with extra metadata col), second 5.
    multiclass = CapAugMulticlass(
        [DummyAug([[1, 1, 3, 3, 99, 42]]), DummyAug([[5, 1, 7, 3, 0]])],
        probabilities=[1.0, 1.0],
        class_idxs=[2, 7],
    )
    _, boxes, _, _ = multiclass(destination_image)

    assert boxes.dtype.kind in ("i", "u")
    assert boxes.tolist() == [[1, 1, 3, 3, 2, 42], [5, 1, 7, 3, 7, 0]]


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
