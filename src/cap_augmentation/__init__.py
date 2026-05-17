"""Cut-and-paste augmentation utilities."""

from .cap_aug import CAP_AUG, CAP_AUG_Multiclass, resize_keep_ar

__all__ = [
    "CAP_AUG",
    "CAP_AUG_Multiclass",
    "CAP_Albu",
    "CAP_TorchVision",
    "ImageMaskTransform",
    "resize_keep_ar",
]


def __getattr__(name):
    if name in {"CAP_Albu", "CAP_TorchVision", "ImageMaskTransform"}:
        from . import wrappers

        return getattr(wrappers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
