"""Albumentations integration for CAP augmentation."""

import warnings

import albumentations as A
import cv2
import numpy as np

from ..cap_aug import CAP_AUG, _align_columns


class CAP_Albu(A.DualTransform):
    """Albumentations DualTransform wrapper around CAP_AUG.

    Albumentations calls ``apply``, ``apply_to_mask``, and ``apply_to_bboxes``
    sequentially on the same transform instance. This wrapper stores that
    per-call state on the instance, so do not share one transform object across
    concurrent threads.
    """

    def __init__(self, p=0.5, always_apply=False, **kwargs):
        kwargs = dict(kwargs)
        if kwargs.get("coords_format", "xyxy") != "xyxy":
            raise ValueError("CAP_Albu requires CAP_AUG coords_format='xyxy'")
        kwargs["coords_format"] = "xyxy"
        if always_apply:
            warnings.warn(
                "always_apply is deprecated; use p=1.0 instead",
                DeprecationWarning,
                stacklevel=2,
            )
        super(CAP_Albu, self).__init__(p=1.0 if always_apply else p)

        self.cap_aug = CAP_AUG(**kwargs)
        self.cap_image = None
        self.cap_bboxes = np.empty((0, 4), dtype=float)
        self.cap_mask_sem = None
        self.cap_mask_ins = None
        self.img_h = None
        self.img_w = None

    @staticmethod
    def get_class_fullname():
        return "CAP_Albu"

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
