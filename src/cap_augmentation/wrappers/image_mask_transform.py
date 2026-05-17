"""Small adapters shared by optional wrapper integrations."""

from __future__ import annotations

from typing import Any, Callable


class ImageMaskTransform:
    """Adapt a simple image/mask callable to CapAug's object transform hook."""

    def __init__(self, transform: Callable[..., Any]) -> None:
        self.transform = transform

    def __call__(self, image: Any, mask: Any) -> dict[str, Any]:
        result = self.transform(image, mask)
        if isinstance(result, dict):
            return result
        if isinstance(result, tuple) and len(result) == 2:
            image, mask = result
            return {"image": image, "mask": mask}
        raise TypeError(
            "transform must return {'image': ..., 'mask': ...} or (image, mask)"
        )
