"""Albumentations integration for CAP augmentation."""

import albumentations as A
import cv2
import numpy as np

from ..cap_aug import CapAug, _align_columns


class CapAlbumentations(A.DualTransform):
    """Albumentations DualTransform wrapper around CapAug.

    Albumentations calls ``apply``, ``apply_to_mask``, and ``apply_to_bboxes``
    sequentially on the same transform instance. This wrapper stores that
    per-call state on the instance, so do not share one transform object across
    concurrent threads.

    Pass ``p`` to control how often the augmentation fires. The legacy
    ``always_apply`` keyword (removed in Albumentations 3.x) was dropped
    in cap-augmentation 0.4.0 — use ``p=1.0`` instead.
    """

    def __init__(self, p=0.5, **kwargs):
        kwargs = dict(kwargs)
        if "always_apply" in kwargs:
            raise TypeError(
                "always_apply was removed in cap-augmentation 0.4.0 (and in "
                "Albumentations 3.x). Use p=1.0 to always apply the transform."
            )
        if kwargs.get("coords_format", "xyxy") != "xyxy":
            raise ValueError("CapAlbumentations requires CapAug coords_format='xyxy'")
        kwargs["coords_format"] = "xyxy"
        super().__init__(p=p)

        self.cap_aug = CapAug(**kwargs)
        self.cap_image = None
        self.cap_bboxes = np.empty((0, 4), dtype=float)
        self.cap_mask_sem = None
        self.cap_mask_ins = None
        self.img_h = None
        self.img_w = None

    @staticmethod
    def get_class_fullname():
        return "CapAlbumentations"

    def get_transform_init_args_names(self):
        return ()

    def apply(self, image, **params):
        result_image, result_coords, semantic_mask, instance_mask = self.cap_aug(image)
        self.cap_image = result_image
        self.cap_bboxes = np.asarray(result_coords, dtype=float)
        self.cap_mask_sem = semantic_mask
        self.cap_mask_ins = instance_mask
        self.img_h, self.img_w = self.cap_image.shape[:2]
        return result_image

    def apply_to_mask(self, mask, **params):
        if self.cap_mask_sem is None:
            return mask
        return cv2.bitwise_or(mask, self.cap_mask_sem)

    def apply_to_bboxes(self, bboxes, **params):
        bboxes = np.asarray(bboxes, dtype=float)
        cap_bboxes = self._normalized_cap_bboxes()

        if len(bboxes) > 0 and len(cap_bboxes) > 0:
            bboxes, cap_bboxes = _align_columns(bboxes, cap_bboxes)
            return np.concatenate((bboxes, cap_bboxes), axis=0)
        if len(cap_bboxes) > 0:
            return cap_bboxes
        return bboxes

    def _normalized_cap_bboxes(self):
        if self.cap_bboxes is None or len(self.cap_bboxes) == 0:
            return np.empty((0, 4), dtype=float)

        norm_cap_bboxes = self.cap_bboxes.copy().astype(float)
        norm_cap_bboxes[:, [0, 2]] /= self.img_w
        norm_cap_bboxes[:, [1, 3]] /= self.img_h
        return norm_cap_bboxes

    def apply_to_keypoints(self, keypoints, **params):
        return keypoints
