import cv2
import numpy as np
import pytest

from cap_augmentation import seamless_blend


@pytest.fixture
def background_path(tmp_path):
    path = tmp_path / "background.png"
    bg = np.full((100, 100, 3), 80, dtype=np.uint8)
    cv2.imwrite(str(path), bg)
    return path


def test_seamless_blend_writes_composite(tmp_path, background_path):
    # Foreground with internal gradient — Poisson cloning preserves gradients,
    # so the blended region must visibly differ from the flat gray background.
    fg = np.zeros((20, 20, 4), dtype=np.uint8)
    fg[..., 0] = np.linspace(0, 255, 20, dtype=np.uint8)[None, :]
    fg[..., 2] = np.linspace(0, 255, 20, dtype=np.uint8)[:, None]
    fg[..., 3] = 255
    fg_path = tmp_path / "gradient.png"
    cv2.imwrite(str(fg_path), fg)
    out = tmp_path / "blended.png"

    seamless_blend(str(fg_path), str(background_path), 50, 50, str(out))

    result = cv2.imread(str(out))
    assert result is not None
    assert result.shape == (100, 100, 3)
    background = cv2.imread(str(background_path))
    assert not np.array_equal(result, background)


def test_seamless_blend_scales_oversized_foreground(
    tmp_path, make_source_image, background_path
):
    fg = make_source_image(size=(200, 200))
    out = tmp_path / "blended.png"

    seamless_blend(str(fg), str(background_path), 50, 50, str(out))

    result = cv2.imread(str(out))
    assert result.shape == (100, 100, 3)


def test_seamless_blend_rejects_foreground_without_alpha(tmp_path, background_path):
    fg_path = tmp_path / "opaque.png"
    cv2.imwrite(str(fg_path), np.full((20, 20, 3), 200, dtype=np.uint8))
    out = tmp_path / "blended.png"

    with pytest.raises(ValueError, match="4-channel"):
        seamless_blend(str(fg_path), str(background_path), 50, 50, str(out))


def test_seamless_blend_mixed_mode_differs_from_normal(tmp_path):
    # MIXED_CLONE keeps whichever of source/destination has the stronger
    # gradient per pixel, so it only diverges from NORMAL_CLONE when the
    # destination has texture. Use a checkerboard background to ensure that.
    rng = np.random.default_rng(0)
    bg = rng.integers(40, 200, size=(100, 100, 3), dtype=np.uint8)
    bg_path = tmp_path / "noise.png"
    cv2.imwrite(str(bg_path), bg)

    fg = np.zeros((40, 40, 4), dtype=np.uint8)
    fg[..., 0] = np.linspace(0, 255, 40, dtype=np.uint8)[None, :]
    fg[..., 2] = np.linspace(0, 255, 40, dtype=np.uint8)[:, None]
    fg[..., 3] = 255
    fg_path = tmp_path / "gradient.png"
    cv2.imwrite(str(fg_path), fg)

    normal_out = tmp_path / "normal.png"
    mixed_out = tmp_path / "mixed.png"
    seamless_blend(str(fg_path), str(bg_path), 50, 50, str(normal_out), mode="normal")
    seamless_blend(str(fg_path), str(bg_path), 50, 50, str(mixed_out), mode="mixed")

    normal = cv2.imread(str(normal_out))
    mixed = cv2.imread(str(mixed_out))
    assert not np.array_equal(normal, mixed)


def test_seamless_blend_rejects_unknown_mode(
    tmp_path, make_source_image, background_path
):
    fg = make_source_image(size=(20, 20))
    with pytest.raises(ValueError, match="mode must be one of"):
        seamless_blend(
            str(fg), str(background_path), 50, 50, str(tmp_path / "x.png"), mode="bogus"
        )


def test_seamless_blend_raises_on_missing_files(tmp_path, make_source_image):
    fg = make_source_image()
    out = tmp_path / "blended.png"

    with pytest.raises(FileNotFoundError):
        seamless_blend(str(fg), str(tmp_path / "missing.png"), 50, 50, str(out))

    with pytest.raises(FileNotFoundError):
        seamless_blend(
            str(tmp_path / "missing.png"),
            str(make_source_image(name="bg.png", size=(80, 80))),
            40,
            40,
            str(out),
        )
