"""Optional integrations with third-party augmentation libraries."""

from .image_mask_transform import ImageMaskTransform

__all__ = [
    "CapAlbumentations",
    "CapTorchvision",
    "ImageMaskTransform",
]


def __getattr__(name):
    if name == "CapAlbumentations":
        from .albu import CapAlbumentations

        return CapAlbumentations
    if name == "CapTorchvision":
        from .tv import CapTorchvision

        return CapTorchvision
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
