from __future__ import annotations

__author__ = "RocketFlash: https://github.com/RocketFlash"

from pathlib import Path

import cv2
import numpy as np

PathLike = str | Path

_MODE_FLAGS = {
    "normal": cv2.NORMAL_CLONE,
    "mixed": cv2.MIXED_CLONE,
    "monochrome": cv2.MONOCHROME_TRANSFER,
}


def seamless_blend(
    foreground_path: PathLike,
    background_path: PathLike,
    center_x: int,
    center_y: int,
    output_path: PathLike,
    mode: str = "normal",
) -> None:
    """Blend a transparent PNG foreground into a background via Poisson editing.

    Uses ``cv2.seamlessClone`` so the foreground's color and lighting adapt
    to the destination, unlike a straight alpha composite. The foreground
    must be a 4-channel image; its alpha channel is reused as the clone
    mask. If the foreground is larger than the background it is scaled down
    (aspect-ratio preserved) to fit, because ``cv2.seamlessClone`` segfaults
    on a source rectangle that overflows the destination.

    ``mode`` selects the OpenCV clone flag:

    - ``"normal"`` (default, ``cv2.NORMAL_CLONE``) — preserves source
      gradients, reconstructs absolute color from the seam. Best when the
      source and destination already have similar color/lighting.
    - ``"mixed"`` (``cv2.MIXED_CLONE``) — at each pixel keeps whichever of
      source or destination has the stronger gradient. Use this when the
      source would otherwise wash out against a textured destination (e.g.
      pasting people onto gravel), at the cost of some destination texture
      bleeding through the silhouette.
    - ``"monochrome"`` (``cv2.MONOCHROME_TRANSFER``) — transfers only the
      source luminance and re-colors with the destination palette. Rarely
      what you want for compositing distinct objects.
    """
    if mode not in _MODE_FLAGS:
        raise ValueError(f"mode must be one of {sorted(_MODE_FLAGS)}; got {mode!r}")
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"Could not read background image: {background_path}")

    foreground = cv2.imread(str(foreground_path), cv2.IMREAD_UNCHANGED)
    if foreground is None:
        raise FileNotFoundError(f"Could not read foreground image: {foreground_path}")
    if foreground.ndim != 3 or foreground.shape[2] != 4:
        raise ValueError(
            "Foreground must be a 4-channel image with an alpha channel; "
            f"got shape {foreground.shape}"
        )

    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]
    if fg_h > bg_h or fg_w > bg_w:
        scale = min(bg_h / fg_h, bg_w / fg_w)
        new_w = max(1, int(round(fg_w * scale)))
        new_h = max(1, int(round(fg_h * scale)))
        foreground = cv2.resize(
            foreground, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

    source = np.ascontiguousarray(foreground[:, :, :3])
    alpha = foreground[:, :, 3]
    mask = cv2.merge([alpha, alpha, alpha])

    center = (int(center_x), int(center_y))
    result = cv2.seamlessClone(source, background, mask, center, _MODE_FLAGS[mode])

    if not cv2.imwrite(str(output_path), result):
        raise OSError(f"Failed to write output image: {output_path}")
