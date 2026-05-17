"""Cut-and-paste augmentation utilities."""

from importlib.metadata import PackageNotFoundError, version as _version

from .cap_aug import CapAug, CapAugMulticlass, resize_keep_ar

try:
    __version__ = _version("cap-augmentation")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "CapAug",
    "CapAugMulticlass",
    "CapAlbumentations",
    "CapTorchvision",
    "ImageMaskTransform",
    "__version__",
    "resize_keep_ar",
]


def __getattr__(name):
    if name in {"CapAlbumentations", "CapTorchvision", "ImageMaskTransform"}:
        from . import wrappers

        return getattr(wrappers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
