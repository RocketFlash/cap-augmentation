"""Cut-and-paste augmentation utilities."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .blending import seamless_blend
from .cap_aug import CapAug, CapAugMulticlass, OpaqueSourceWarning, resize_keep_ar

try:
    __version__: str = _version("cap-augmentation")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "CapAug",
    "CapAugMulticlass",
    "CapAlbumentations",
    "CapTorchvision",
    "ImageMaskTransform",
    "OpaqueSourceWarning",
    "__version__",
    "resize_keep_ar",
    "seamless_blend",
]


def __getattr__(name):
    if name in {"CapAlbumentations", "CapTorchvision", "ImageMaskTransform"}:
        from . import wrappers

        return getattr(wrappers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
