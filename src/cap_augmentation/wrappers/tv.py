"""Torchvision v2 style wrapper for CAP augmentation."""

from __future__ import annotations

import numpy as np

from ..cap_aug import CapAug


class CapTorchvision:
    """Apply CAP augmentation to a torchvision image and target dictionary.

    The wrapper expects absolute XYXY boxes. If the target uses
    ``torchvision.tv_tensors.BoundingBoxes``, the original bounding box format
    and canvas size are preserved.

    Target merge rules (intentional, to avoid changing the target schema):

    * ``boxes`` — always appended (the wrapper assumes the target carries
      detection targets if it has any boxes at all).
    * ``labels`` — appended if present in the input target. Requires either
      ``class_idx=...`` on the wrapper or class ids carried in the fifth
      column of the generated boxes. If labels are present and neither
      source of class ids exists, a ValueError is raised.
    * ``masks`` — appended if ``masks`` is already in the target. If the
      target has boxes but no ``masks`` key, generated instance masks are
      dropped, even with ``output_masks=True``. This avoids silently
      growing the target schema mid-pipeline (a downstream transform that
      doesn't know about masks would crash or silently corrupt them).
    * ``masks`` (no existing boxes) — created from scratch if
      ``output_masks=True``.
    * ``semantic_mask`` — merged with ``torch.maximum`` if present.
    """

    def __init__(self, output_masks=True, **kwargs):
        kwargs = dict(kwargs)
        self.class_idx = kwargs.pop("class_idx", None)
        if kwargs.get("coords_format", "xyxy") != "xyxy":
            raise ValueError("CapTorchvision requires CapAug coords_format='xyxy'")
        kwargs["coords_format"] = "xyxy"
        kwargs.setdefault("image_format", "rgb")

        self.output_masks = output_masks
        self.cap_aug = CapAug(**kwargs)

    def __call__(self, image, target=None):
        deps = _load_torchvision()
        np_image, restore_image = _image_to_numpy(image, deps)

        result_image, cap_boxes, semantic_mask, instance_mask = self.cap_aug(np_image)
        result_image = np.ascontiguousarray(result_image)
        restored_image = restore_image(result_image)

        if target is None:
            return restored_image

        result_target = self._merge_target(
            target,
            cap_boxes,
            semantic_mask,
            instance_mask,
            result_image.shape[:2],
            deps,
        )
        return restored_image, result_target

    def _merge_target(
        self,
        target,
        cap_boxes,
        semantic_mask,
        instance_mask,
        canvas_size,
        deps,
    ):
        result = dict(target)
        cap_boxes, generated_labels = self._split_generated_boxes_and_labels(cap_boxes)

        existing_boxes = target.get("boxes")
        existing_box_count = (
            _first_dim(existing_boxes) if existing_boxes is not None else 0
        )
        result["boxes"] = _append_boxes(existing_boxes, cap_boxes, canvas_size, deps)

        if "labels" in target:
            if generated_labels is None and len(cap_boxes) > 0:
                raise ValueError("class_idx is required when target contains labels")
            result["labels"] = _append_labels(target["labels"], generated_labels, deps)
        elif generated_labels is not None and existing_box_count == 0:
            result["labels"] = _new_labels(generated_labels, deps)

        generated_masks = _instance_masks(instance_mask, len(cap_boxes))
        if "masks" in target:
            result["masks"] = _append_masks(target["masks"], generated_masks, deps)
        elif self.output_masks and existing_box_count == 0:
            result["masks"] = _new_masks(generated_masks, deps)

        if "semantic_mask" in target:
            result["semantic_mask"] = _merge_semantic_mask(
                target["semantic_mask"], semantic_mask, deps
            )

        return result

    def _split_generated_boxes_and_labels(self, cap_boxes):
        cap_boxes = np.asarray(cap_boxes, dtype=np.float32)
        if cap_boxes.size == 0:
            return np.empty((0, 4), dtype=np.float32), None
        if cap_boxes.ndim == 1:
            cap_boxes = cap_boxes.reshape(1, -1)

        if cap_boxes.shape[1] > 4:
            generated_labels = cap_boxes[:, 4].astype(np.int64)
            cap_boxes = cap_boxes[:, :4]
        elif self.class_idx is not None:
            generated_labels = np.full(len(cap_boxes), self.class_idx, dtype=np.int64)
        else:
            generated_labels = None

        return cap_boxes.astype(np.float32), generated_labels


def _load_torchvision():
    try:
        import torch
        from torchvision import tv_tensors
        from torchvision.ops import box_convert
    except ImportError as exc:
        raise ImportError(
            "CapTorchvision requires the optional torchvision extra. "
            'Install it with: pip install "cap-augmentation[torchvision]"'
        ) from exc
    return torch, tv_tensors, box_convert


def _image_to_numpy(image, deps):
    torch, tv_tensors, _ = deps

    if isinstance(image, np.ndarray):
        original_dtype = image.dtype
        np_image = _to_uint8_hwc(image)

        def restore(result):
            return _restore_numpy(result, original_dtype)

        return np_image, restore

    image_module = _pil_image_module()
    if image_module is not None and isinstance(image, image_module.Image):
        np_image = _to_uint8_hwc(np.asarray(image.convert("RGB")))

        def restore(result):
            return image_module.fromarray(result)

        return np_image, restore

    if isinstance(image, torch.Tensor):
        is_tv_image = isinstance(image, tv_tensors.Image)
        tensor = image.as_subclass(torch.Tensor) if is_tv_image else image
        tensor_cpu = tensor.detach().cpu()
        channel_first = _is_channel_first(tensor_cpu)
        hwc = (
            tensor_cpu.permute(1, 2, 0).numpy() if channel_first else tensor_cpu.numpy()
        )
        np_image = _to_uint8_hwc(hwc)
        original_dtype = tensor.dtype
        original_device = tensor.device
        original_is_float = tensor.is_floating_point()

        def restore(result):
            restored = _restore_tensor(
                result,
                torch,
                original_dtype,
                original_device,
                original_is_float,
                channel_first,
            )
            if is_tv_image:
                return tv_tensors.Image(restored)
            return restored

        return np_image, restore

    raise TypeError(
        "image must be a numpy array, PIL image, torch.Tensor, or tv_tensors.Image"
    )


def _pil_image_module():
    try:
        from PIL import Image
    except ImportError:
        return None
    return Image


def _is_channel_first(tensor):
    if tensor.ndim != 3:
        raise ValueError("Torchvision images must be 3D tensors")
    return tensor.shape[0] in {1, 3, 4} and tensor.shape[-1] not in {1, 3, 4}


def _to_uint8_hwc(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("CapTorchvision expects HWC images with exactly 3 channels")
    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 1.0
        scale = 255.0 if max_value <= 1.0 else 1.0
        image = image * scale
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _restore_numpy(image, dtype):
    if np.issubdtype(dtype, np.floating):
        return (image.astype(np.float32) / 255.0).astype(dtype)
    return image.astype(dtype, copy=False)


def _restore_tensor(image, torch, dtype, device, is_float, channel_first):
    if is_float:
        array = image.astype(np.float32) / 255.0
    else:
        array = image
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if channel_first:
        tensor = tensor.permute(2, 0, 1).contiguous()
    return tensor.to(device=device, dtype=dtype)


def _append_boxes(existing_boxes, cap_boxes, canvas_size, deps):
    torch, tv_tensors, box_convert = deps
    if existing_boxes is None:
        return torch.as_tensor(cap_boxes, dtype=torch.float32)

    if isinstance(existing_boxes, tv_tensors.BoundingBoxes):
        raw_boxes = existing_boxes.as_subclass(torch.Tensor)
        box_format = _format_name(existing_boxes.format)
        existing_xyxy = _convert_boxes(raw_boxes, box_format, "XYXY", box_convert)
        cap_tensor = torch.as_tensor(
            cap_boxes, dtype=raw_boxes.dtype, device=raw_boxes.device
        )
        merged_xyxy = torch.cat([existing_xyxy, cap_tensor], dim=0)
        merged = _convert_boxes(merged_xyxy, "XYXY", box_format, box_convert)
        return tv_tensors.BoundingBoxes(
            merged,
            format=existing_boxes.format,
            canvas_size=getattr(existing_boxes, "canvas_size", canvas_size),
        )

    if isinstance(existing_boxes, torch.Tensor):
        cap_tensor = torch.as_tensor(
            cap_boxes, dtype=existing_boxes.dtype, device=existing_boxes.device
        )
        return torch.cat([existing_boxes, cap_tensor], dim=0)

    existing_np = np.asarray(existing_boxes)
    cap_np = cap_boxes.astype(existing_np.dtype if existing_np.size else np.float32)
    return np.concatenate([existing_np, cap_np], axis=0)


def _convert_boxes(boxes, in_format, out_format, box_convert):
    if in_format == out_format:
        return boxes
    return box_convert(boxes, in_fmt=in_format.lower(), out_fmt=out_format.lower())


def _format_name(box_format):
    return str(box_format).split(".")[-1].upper()


def _append_labels(existing_labels, generated_labels, deps):
    if generated_labels is None or len(generated_labels) == 0:
        return existing_labels

    torch, _, _ = deps
    if isinstance(existing_labels, torch.Tensor):
        new_labels = torch.as_tensor(
            generated_labels, dtype=existing_labels.dtype, device=existing_labels.device
        )
        return torch.cat([existing_labels, new_labels], dim=0)

    if isinstance(existing_labels, np.ndarray):
        return np.concatenate(
            [existing_labels, generated_labels.astype(existing_labels.dtype)]
        )

    return list(existing_labels) + generated_labels.tolist()


def _new_labels(generated_labels, deps):
    torch, _, _ = deps
    return torch.as_tensor(generated_labels, dtype=torch.int64)


def _instance_masks(instance_mask, box_count):
    h, w = instance_mask.shape[:2]
    mask_ids = [mask_id for mask_id in np.unique(instance_mask) if mask_id > 0]
    masks = [(instance_mask == mask_id).astype(np.uint8) for mask_id in mask_ids]
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, h, w), dtype=np.uint8)

    if len(masks) < box_count:
        padding = np.zeros((box_count - len(masks), h, w), dtype=np.uint8)
        masks = np.concatenate([masks, padding], axis=0)
    return masks[:box_count]


def _append_masks(existing_masks, generated_masks, deps):
    if len(generated_masks) == 0:
        return existing_masks

    torch, tv_tensors, _ = deps
    if isinstance(existing_masks, tv_tensors.Mask):
        raw_masks = existing_masks.as_subclass(torch.Tensor)
        new_masks = torch.as_tensor(
            generated_masks, dtype=raw_masks.dtype, device=raw_masks.device
        )
        return tv_tensors.Mask(torch.cat([_ensure_nhw(raw_masks), new_masks], dim=0))

    if isinstance(existing_masks, torch.Tensor):
        new_masks = torch.as_tensor(
            generated_masks, dtype=existing_masks.dtype, device=existing_masks.device
        )
        return torch.cat([_ensure_nhw(existing_masks), new_masks], dim=0)

    existing_np = _ensure_nhw(np.asarray(existing_masks))
    return np.concatenate(
        [existing_np, generated_masks.astype(existing_np.dtype)], axis=0
    )


def _new_masks(generated_masks, deps):
    torch, _, _ = deps
    return torch.as_tensor(generated_masks, dtype=torch.uint8)


def _ensure_nhw(masks):
    if masks.ndim == 2:
        return masks[None, ...]
    return masks


def _merge_semantic_mask(existing_mask, generated_mask, deps):
    torch, tv_tensors, _ = deps
    if isinstance(existing_mask, tv_tensors.Mask):
        raw_mask = existing_mask.as_subclass(torch.Tensor)
        generated = torch.as_tensor(
            generated_mask, dtype=raw_mask.dtype, device=raw_mask.device
        )
        return tv_tensors.Mask(torch.maximum(raw_mask, generated))

    if isinstance(existing_mask, torch.Tensor):
        generated = torch.as_tensor(
            generated_mask, dtype=existing_mask.dtype, device=existing_mask.device
        )
        return torch.maximum(existing_mask, generated)

    return np.maximum(existing_mask, generated_mask.astype(existing_mask.dtype))


def _first_dim(value):
    if value is None:
        return 0
    return int(value.shape[0]) if hasattr(value, "shape") else len(value)
