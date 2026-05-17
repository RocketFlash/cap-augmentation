"""Optional integrations with third-party augmentation libraries."""

from .generic import ImageMaskTransform

__all__ = [
    "CAP_Albu",
    "CAP_Albumentations",
    "CAP_TorchVision",
    "ImageMaskTransform",
]


def __getattr__(name):
    if name in {"CAP_Albu", "CAP_Albumentations"}:
        from .albumentations import CAP_Albu

        return CAP_Albu
    if name == "CAP_TorchVision":
        from .torchvision import CAP_TorchVision

        return CAP_TorchVision
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
