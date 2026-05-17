"""Cut-and-paste augmentation utilities."""

from .cap_aug import CapAug, CapAugMulticlass, resize_keep_ar

__all__ = [
    "CapAug",
    "CapAugMulticlass",
    "CapAlbumentations",
    "CapTorchvision",
    "ImageMaskTransform",
    "resize_keep_ar",
]


def __getattr__(name):
    if name in {"CapAlbumentations", "CapTorchvision", "ImageMaskTransform"}:
        from . import wrappers

        return getattr(wrappers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
