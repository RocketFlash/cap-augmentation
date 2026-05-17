"""Small adapters shared by optional wrapper integrations."""


class ImageMaskTransform:
    """Adapt a simple image/mask callable to CAP_AUG's object transform hook."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, mask):
        result = self.transform(image, mask)
        if isinstance(result, dict):
            return result
        if isinstance(result, tuple) and len(result) == 2:
            image, mask = result
            return {"image": image, "mask": mask}
        raise TypeError(
            "transform must return {'image': ..., 'mask': ...} or (image, mask)"
        )
