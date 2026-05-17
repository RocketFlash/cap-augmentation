__author__ = "RocketFlash: https://github.com/RocketFlash"

import random
import warnings
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_COORD_FORMATS = {"xyxy", "xywh", "yolo"}


class OpaqueSourceWarning(UserWarning):
    """Emitted when a source image lacks a meaningful transparency mask.

    The pasted "object" then covers the entire source rectangle, which is
    rarely what the user wants. Filter or silence this warning if the
    behavior is intentional (e.g. pasting fully opaque sprites).
    """


def _as_scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def resize_keep_ar(image, height=500, scale=None):
    if scale is not None:
        return cv2.resize(image, None, fx=float(scale), fy=float(scale))

    height = max(1, int(round(_as_scalar(height))))
    r = height / float(image.shape[0])
    width = max(1, int(round(r * image.shape[1])))
    return cv2.resize(image, (width, height))


def _apply_image_mask_transform(transform, image, mask):
    transformed = transform(image=image, mask=mask)
    if isinstance(transformed, dict):
        if "image" not in transformed or "mask" not in transformed:
            raise ValueError(
                "object transform dict output must contain 'image' and 'mask'"
            )
        return transformed["image"], transformed["mask"]
    if isinstance(transformed, tuple) and len(transformed) == 2:
        return transformed
    raise TypeError(
        "object transform must return {'image': ..., 'mask': ...} or (image, mask)"
    )


def _align_columns(*arrays):
    max_cols = max(array.shape[1] for array in arrays)
    aligned = []
    for array in arrays:
        if array.shape[1] < max_cols:
            pad = np.zeros((len(array), max_cols - array.shape[1]))
            array = np.c_[array, pad]
        aligned.append(array)
    return aligned


def _with_class_column(coords, class_idx):
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 4:
        raise ValueError(
            "augmentation coordinates must be a 2D array with at least four columns"
        )

    if coords.shape[1] == 4:
        return np.c_[coords, class_idx * np.ones(len(coords))]

    coords = coords.copy()
    coords[:, 4] = class_idx
    return coords


class CapAugMulticlass:
    """
    cap_augs - list of cap augmentations for each class
    probabilities - list of probabilities for each augmentation
    class_idxs - class indexes
    """

    def __init__(self, cap_augs, probabilities, class_idxs):
        self.cap_augs = cap_augs
        self.probabilities = probabilities
        self.class_idxs = class_idxs
        if not (
            len(self.cap_augs) == len(self.probabilities)
            and len(self.cap_augs) == len(self.class_idxs)
        ):
            raise ValueError(
                "cap_augs, probabilities, and class_idxs must have equal length"
            )

    def __call__(self, image):
        return self.generate_objects(image)

    def generate_objects(self, image):
        result_image = image.copy()
        total_result_coords = []
        total_instance_masks = []
        result_sem_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for cap_aug, p, class_idx in zip(
            self.cap_augs, self.probabilities, self.class_idxs
        ):
            if float(p) >= float(np.random.uniform(0, 1)):
                result_image, result_coords, semantic_mask, instance_mask = cap_aug(
                    result_image
                )
                if len(result_coords) > 0:
                    total_result_coords.append(
                        _with_class_column(result_coords, class_idx)
                    )
                result_sem_mask[semantic_mask == 1] = class_idx
                total_instance_masks.append(instance_mask)

        if total_result_coords:
            total_result_coords = np.vstack(_align_columns(*total_result_coords))
        else:
            total_result_coords = np.empty((0, 5), dtype=float)

        return result_image, total_result_coords, result_sem_mask, total_instance_masks


class CapAug:
    """
    source_images - list of image paths
    bev_transform - bird's eye view transformation
    probability_map - mask with probability values
    mean_h_norm - mean normalized height
    n_objects_range - [min, max] number of objects
    s_range - range of scales of original image size
    h_range - range of object heights
              if bev_transform is not None range in meters, else in pixels
    x_range - if bev_transform is None -> image coordinates in pixels [int, int]
              else -> camera coordinate system in meters [float, float]
    y_range - if bev_transform is None -> image coordinates in pixels [int, int]
              else -> camera coordinate system in meters [float, float]
    z_range - if bev_transform is None -> image coordinates in pixels [int, int]
              else -> camera coordinate system in meters [float, float]
    objects_idxs - object indexes from dataset to paste [idx1, idx2, ...]
    random_h_flip - source image random horizontal flip
    random_v_flip - source image random vertical flip
    histogram_matching - apply histogram matching
    hm_offset - histogram matching offset
    blending_coeff - coefficient of image blending inside the pasted mask
    image_format - color image format: {bgr, rgb}
    coords_format - output coordinates format: {xyxy, xywh, yolo}
    normalized_range - range in normalized image coordinates [0, 1]
    normilized_range - deprecated alias for normalized_range
    class_idx - class id appended to result boxes: [x1, y1, x2, y2, class_idx]
    object_transforms - callable applied to pasted object image and alpha mask.
                        Must accept image=..., mask=... and return either
                        {'image': image, 'mask': mask} or (image, mask).
    albu_transforms - deprecated alias for object_transforms
    cache_size - number of decoded source images to keep in memory. None (the
                 default) caches every source. Set to 0 to disable caching
                 and re-read from disk on every paste. Cached images are
                 returned by reference; do not mutate them in place.
    """

    def __init__(
        self,
        source_images,
        bev_transform=None,
        probability_map=None,
        mean_h_norm=None,
        n_objects_range=(1, 6),
        h_range=None,
        s_range=(0.5, 1.5),
        x_range=(200, 500),
        y_range=(100, 300),
        z_range=(0, 0),
        objects_idxs=None,
        random_h_flip=True,
        random_v_flip=False,
        histogram_matching=False,
        hm_offset=200,
        image_format="bgr",
        coords_format="xyxy",
        normilized_range=False,
        blending_coeff=0,
        class_idx=None,
        albu_transforms=None,
        object_transforms=None,
        normalized_range=None,
        cache_size=None,
    ):
        if coords_format not in SUPPORTED_COORD_FORMATS:
            raise ValueError(
                f"coords_format must be one of {sorted(SUPPORTED_COORD_FORMATS)}"
            )
        if image_format not in {"bgr", "rgb"}:
            raise ValueError("image_format must be 'bgr' or 'rgb'")
        if len(source_images) == 0:
            raise ValueError("source_images must contain at least one image path")

        self.source_images = list(source_images)
        self.bev_transform = bev_transform
        if normalized_range is not None:
            if normilized_range not in (False, normalized_range):
                raise ValueError(
                    "Use either normalized_range or normilized_range, not both"
                )
            normilized_range = normalized_range
        elif normilized_range:
            warnings.warn(
                "normilized_range is deprecated; use normalized_range instead",
                DeprecationWarning,
                stacklevel=2,
            )

        self.n_objects_range = tuple(n_objects_range)
        self.s_range = tuple(s_range)
        self.h_range = h_range
        self.x_range = tuple(x_range)
        self.y_range = tuple(y_range)
        self.z_range = tuple(z_range)
        self.objects_idxs = objects_idxs
        self.random_h_flip = random_h_flip
        self.random_v_flip = random_v_flip
        self.image_format = image_format
        self.coords_format = coords_format
        self.normalized_range = bool(normilized_range)
        self.normilized_range = self.normalized_range
        self.probability_map = probability_map
        self.mean_h_norm = mean_h_norm
        self.histogram_matching = histogram_matching
        self.hm_offset = hm_offset
        self.blending_coeff = blending_coeff
        self.class_idx = class_idx
        if albu_transforms is not None and object_transforms is not None:
            raise ValueError(
                "Use either object_transforms or albu_transforms, not both"
            )
        self.object_transforms = (
            object_transforms if object_transforms is not None else albu_transforms
        )
        self.albu_transforms = self.object_transforms

        if cache_size is not None and int(cache_size) < 0:
            raise ValueError("cache_size must be None, 0, or a positive integer")
        self.cache_size = None if cache_size is None else int(cache_size)
        self._image_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._warned_opaque: dict[str, bool] = {}

    def __call__(self, image):
        return self.generate_objects(image)

    def generate_objects(self, image):
        n_objects = random.randint(*self.n_objects_range)
        heights = None
        scales = None

        if self.probability_map is not None:
            probability_map = np.asarray(self.probability_map, dtype=float)
            p_h, p_w = probability_map.shape
            prob_map_1d = probability_map.reshape(-1)
            prob_sum = prob_map_1d.sum()
            if prob_sum <= 0:
                raise ValueError("probability_map must contain a positive sum")
            prob_map_1d = prob_map_1d / prob_sum
            select_indexes = np.random.choice(
                np.arange(prob_map_1d.size), n_objects, p=prob_map_1d
            )
            points = np.array(
                [
                    [(select_idx % p_w) / p_w, (select_idx // p_w) / p_h]
                    for select_idx in select_indexes
                ],
                dtype=float,
            )

            if self.mean_h_norm is not None:
                heights = np.random.uniform(
                    low=self.mean_h_norm * 0.98,
                    high=self.mean_h_norm * 1.02,
                    size=n_objects,
                )
            elif self.h_range is not None:
                heights = np.random.uniform(
                    low=self.h_range[0], high=self.h_range[1], size=n_objects
                )
        elif self.bev_transform is not None:
            points = np.random.uniform(
                low=[self.x_range[0], self.y_range[0], self.z_range[0]],
                high=[self.x_range[1], self.y_range[1], self.z_range[1]],
                size=(n_objects, 3),
            )
            if self.h_range is not None:
                heights = np.random.uniform(
                    low=self.h_range[0], high=self.h_range[1], size=n_objects
                )
            else:
                heights = np.random.uniform(low=0.5, high=1.5, size=n_objects)
        elif self.normalized_range:
            points = np.random.uniform(
                low=[self.x_range[0], self.y_range[0]],
                high=[self.x_range[1], self.y_range[1]],
                size=(n_objects, 2),
            )
            if self.h_range is not None:
                heights = np.random.uniform(
                    low=self.h_range[0], high=self.h_range[1], size=n_objects
                )
        else:
            points = np.random.randint(
                low=[self.x_range[0], self.y_range[0]],
                high=[self.x_range[1], self.y_range[1]],
                size=(n_objects, 2),
            )
            if self.h_range is not None:
                heights = np.random.randint(
                    low=self.h_range[0], high=self.h_range[1], size=n_objects
                )

        if heights is None:
            scales = np.random.uniform(
                low=self.s_range[0], high=self.s_range[1], size=n_objects
            )

        return self.generate_objects_coord(image, points, heights, scales)

    def generate_objects_coord(self, image, points, heights, scales):
        """
        points - numpy array of coordinates in pixels, normalized coordinates,
                 or meters depending on configuration.
        """
        points = np.asarray(points)
        n_objects = points.shape[0]

        if self.objects_idxs is None:
            objects_idxs = np.array(
                [
                    random.randint(0, len(self.source_images) - 1)
                    for _ in range(n_objects)
                ]
            )
        else:
            objects_idxs = np.asarray(self.objects_idxs)

        if len(objects_idxs) != n_objects:
            raise ValueError("objects_idxs length must match points length")

        image_dst = image.copy()
        dst_h, dst_w = image_dst.shape[:2]
        coords_all = []

        if heights is not None:
            heights = np.asarray(heights).reshape(-1)
        if scales is not None:
            scales = np.asarray(scales).reshape(-1)

        z_offsets = np.zeros(n_objects, dtype=float)
        distances = None
        if self.bev_transform is not None and n_objects > 0:
            meter_points = points
            points_pixels = self.bev_transform.meters_to_pixels(meter_points)
            distances = self.bev_transform.calculate_dist_meters(meter_points)
            d_sorted_idxs = np.argsort(distances)[::-1]
            distances = distances[d_sorted_idxs]
            points = points_pixels[d_sorted_idxs]
            objects_idxs = objects_idxs[d_sorted_idxs]
            if meter_points.shape[1] > 2:
                z_offsets = meter_points[:, 2][d_sorted_idxs]
            if heights is not None:
                heights = heights[d_sorted_idxs]
            elif scales is not None:
                scales = scales[d_sorted_idxs]

        semantic_mask = np.zeros((dst_h, dst_w), dtype=np.uint8)
        instance_mask = np.zeros((dst_h, dst_w), dtype=np.uint8)

        for idx, object_idx in enumerate(objects_idxs):
            point = points[idx]
            height = _as_scalar(heights[idx]) if heights is not None else None
            scale = _as_scalar(scales[idx]) if scales is not None else None
            image_src = self.select_image(self.source_images, int(object_idx))

            if self.probability_map is not None or self.normalized_range:
                x_coord, y_coord = int(point[0] * dst_w), int(point[1] * dst_h)
                if height is not None:
                    height *= dst_h
                image_src = resize_keep_ar(image_src, height=height, scale=scale)
            else:
                x_coord, y_coord = int(point[0]), int(point[1])
                if self.bev_transform is not None:
                    z_offset = _as_scalar(z_offsets[idx])
                    distance = _as_scalar(distances[idx])
                    height_pixels = self.bev_transform.get_height_in_pixels(
                        height, distance
                    )
                    height_w_offset_pixels = self.bev_transform.get_height_in_pixels(
                        z_offset + height, distance
                    )
                    pixels_offset = height_w_offset_pixels - height_pixels
                    y_coord -= int(pixels_offset)
                    image_src = resize_keep_ar(image_src, height=height_pixels)
                else:
                    image_src = resize_keep_ar(image_src, height=height, scale=scale)

            if self.histogram_matching:
                image_src = self._match_histogram(image, image_src, x_coord, y_coord)

            image_dst, coords, mask = self.paste_object(
                image_dst,
                image_src,
                x_coord,
                y_coord,
                self.random_h_flip,
                self.random_v_flip,
            )
            if coords:
                coords_all.append(coords)
                x1, y1, x2, y2 = coords
                curr_mask = (mask > 0).astype(np.uint8)
                curr_mask_ins = curr_mask * (idx + 1)

                roi_mask_sem = semantic_mask[y1:y2, x1:x2]
                roi_mask_ins = instance_mask[y1:y2, x1:x2]

                mask_inv = cv2.bitwise_not(curr_mask * 255)

                img_sem_bg = cv2.bitwise_and(roi_mask_sem, roi_mask_sem, mask=mask_inv)
                img_ins_bg = cv2.bitwise_and(roi_mask_ins, roi_mask_ins, mask=mask_inv)

                semantic_mask[y1:y2, x1:x2] = cv2.add(img_sem_bg, curr_mask)
                instance_mask[y1:y2, x1:x2] = cv2.add(img_ins_bg, curr_mask_ins)

        coords_all = self._format_coords(coords_all, dst_w, dst_h)

        if self.class_idx is not None:
            coords_all = np.c_[coords_all, self.class_idx * np.ones(len(coords_all))]

        return image_dst, coords_all, semantic_mask, instance_mask

    def _format_coords(self, coords_all, dst_w, dst_h):
        if len(coords_all) == 0:
            dtype = float if self.coords_format == "yolo" else int
            return np.empty((0, 4), dtype=dtype)

        coords_all = np.asarray(coords_all, dtype=float)

        if self.coords_format == "yolo":
            x = coords_all.copy()
            widths = coords_all[:, 2] - coords_all[:, 0]
            heights = coords_all[:, 3] - coords_all[:, 1]
            x[:, 0] = ((coords_all[:, 0] + coords_all[:, 2]) / 2.0) / dst_w
            x[:, 1] = ((coords_all[:, 1] + coords_all[:, 3]) / 2.0) / dst_h
            x[:, 2] = widths / dst_w
            x[:, 3] = heights / dst_h
            return x

        if self.coords_format == "xywh":
            x = coords_all.copy()
            x[:, 2] = coords_all[:, 2] - coords_all[:, 0]
            x[:, 3] = coords_all[:, 3] - coords_all[:, 1]
            return x.astype(int)

        return coords_all.astype(int)

    def _match_histogram(self, image, image_src, x_coord, y_coord):
        try:
            from skimage import exposure
        except ImportError as exc:
            raise ImportError(
                "histogram_matching=True requires scikit-image. "
                'Install it with: pip install "cap-augmentation[histogram]"'
            ) from exc

        dst_h, dst_w = image.shape[:2]
        image_ref = image[
            max(0, y_coord - self.hm_offset) : min(y_coord + self.hm_offset, dst_h),
            max(0, x_coord - self.hm_offset) : min(x_coord + self.hm_offset, dst_w),
            :,
        ]
        if image_ref.size == 0:
            return image_src

        mask_src = image_src[:, :, 3]
        matched = exposure.match_histograms(
            image_src[:, :, :3], image_ref, channel_axis=-1
        )
        matched = np.clip(matched, 0, 255).astype(image_src.dtype)
        image_src = cv2.bitwise_and(matched, matched, mask=mask_src)
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2BGRA)
        image_src[:, :, 3] = mask_src
        return image_src

    def select_image(self, source_images, object_idx):
        source_image_path = Path(source_images[object_idx])
        cache_key = (str(source_image_path), self.image_format)
        if self.cache_size != 0 and cache_key in self._image_cache:
            self._image_cache.move_to_end(cache_key)
            return self._image_cache[cache_key]

        image_src = cv2.imread(str(source_image_path), cv2.IMREAD_UNCHANGED)
        if image_src is None:
            raise FileNotFoundError(f"Could not read source image: {source_image_path}")

        no_alpha_reason = None
        if image_src.ndim == 2:
            no_alpha_reason = "grayscale"
            code = (
                cv2.COLOR_GRAY2RGBA
                if self.image_format == "rgb"
                else cv2.COLOR_GRAY2BGRA
            )
            image_src = cv2.cvtColor(image_src, code)
        elif image_src.shape[2] == 3:
            no_alpha_reason = "no alpha channel"
            code = (
                cv2.COLOR_BGR2RGBA if self.image_format == "rgb" else cv2.COLOR_BGR2BGRA
            )
            image_src = cv2.cvtColor(image_src, code)
        elif image_src.shape[2] == 4 and self.image_format == "rgb":
            image_src = cv2.cvtColor(image_src, cv2.COLOR_BGRA2RGBA)
        elif image_src.shape[2] != 4:
            raise ValueError(f"Unsupported image channel count: {image_src.shape[2]}")

        if no_alpha_reason is None and image_src[:, :, 3].min() == 255:
            no_alpha_reason = "fully opaque"

        if no_alpha_reason is not None and not self._warned_opaque.get(
            str(source_image_path)
        ):
            warnings.warn(
                f"Source {source_image_path} has {no_alpha_reason}; pasting "
                "the full image rectangle as an object. CapAug expects PNGs "
                "with a transparency mask defining the visible object.",
                OpaqueSourceWarning,
                stacklevel=2,
            )
            self._warned_opaque[str(source_image_path)] = True

        if self.cache_size != 0:
            image_src.setflags(write=False)
            self._image_cache[cache_key] = image_src
            if self.cache_size is not None:
                while len(self._image_cache) > self.cache_size:
                    self._image_cache.popitem(last=False)

        return image_src

    def paste_object(
        self,
        image_dst,
        image_src,
        x_coord,
        y_coord,
        random_h_flip=True,
        random_v_flip=False,
    ):
        if random_h_flip and random.uniform(0, 1) > 0.5:
            image_src = cv2.flip(image_src, 1)
        if random_v_flip and random.uniform(0, 1) > 0.5:
            image_src = cv2.flip(image_src, 0)

        mask_src = image_src[:, :, 3]
        rgb_img = image_src[:, :, :3]

        # Apply object-level transforms before reading dimensions: a transform
        # may crop or resize the image/mask, so size-derived ROI bounds must be
        # computed from the post-transform shapes.
        if self.object_transforms is not None:
            rgb_img, mask_src = _apply_image_mask_transform(
                self.object_transforms, rgb_img, mask_src
            )
            if rgb_img.shape[:2] != mask_src.shape[:2]:
                raise ValueError(
                    "object_transforms must return image and mask with the same "
                    f"height/width; got image {rgb_img.shape[:2]} and mask "
                    f"{mask_src.shape[:2]}"
                )

        src_h, src_w = mask_src.shape[:2]
        dst_h, dst_w = image_dst.shape[:2]
        x_offset = int(round(x_coord - src_w / 2))
        y_offset = int(round(y_coord - src_h))

        y1, y2 = max(y_offset, 0), min(y_offset + src_h, dst_h)
        x1, x2 = max(x_offset, 0), min(x_offset + src_w, dst_w)
        if x1 >= x2 or y1 >= y2:
            return image_dst, [], None

        y1_m, y2_m = y1 - y_offset, y2 - y_offset
        x1_m, x2_m = x1 - x_offset, x2 - x_offset

        src_roi = rgb_img[y1_m:y2_m, x1_m:x2_m]
        dst_roi = image_dst[y1:y2, x1:x2]
        mask_roi = mask_src[y1_m:y2_m, x1_m:x2_m]

        # Soft-alpha composite. For binary masks (alpha ∈ {0, 255}), this
        # produces the same pixels as the old bitwise path; for masks with
        # intermediate alpha values (anti-aliased cutouts), edges blend
        # instead of being hard-thresholded.
        alpha = (mask_roi.astype(np.float32) / 255.0)[..., None]
        src_f = src_roi.astype(np.float32)
        dst_f = dst_roi.astype(np.float32)

        if self.blending_coeff > 0:
            # Ghost effect: inside the mask region, blend src with dst at
            # `blending_coeff`. Soft alpha still controls the boundary.
            src_f = src_f * float(self.blending_coeff) + dst_f * (
                1.0 - float(self.blending_coeff)
            )

        out_img = src_f * alpha + dst_f * (1.0 - alpha)
        image_dst[y1:y2, x1:x2] = np.clip(out_img, 0, 255).astype(np.uint8)

        # Tighten the returned bbox to the visible (alpha>0) pixels of the
        # pasted ROI translated into destination coords, so PNGs with
        # transparent padding don't yield boxes covering empty canvas. The
        # mask is sliced to the same tight region so callers can blit it
        # into instance/semantic masks at the bbox coordinates.
        ys, xs = np.where(mask_roi > 0)
        if ys.size == 0:
            return image_dst, [], mask_roi
        y_min, y_max = int(ys.min()), int(ys.max()) + 1
        x_min, x_max = int(xs.min()), int(xs.max()) + 1
        tight_box = [x1 + x_min, y1 + y_min, x1 + x_max, y1 + y_max]
        tight_mask = mask_roi[y_min:y_max, x_min:x_max]
        return image_dst, tight_box, tight_mask
