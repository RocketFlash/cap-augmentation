import numpy as np

from cap_augmentation.bev import (
    BEV,
    calculate_BEV_H,
    get_BEV_H,
    get_RX,
    get_RY,
    get_RZ,
    get_T,
)


def test_rotation_and_translation_matrix_shapes():
    for matrix in [get_RX(10), get_RY(20), get_RZ(30), get_T(1, 2, 3)]:
        assert matrix.shape == (4, 4)
        np.testing.assert_allclose(matrix[3], [0, 0, 0, 1])


def test_default_bev_loads_packaged_calibration_and_transforms_image():
    bev = BEV()
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    transformed = bev(image)

    assert bev.H.shape == (3, 3)
    assert transformed.shape == (bev.output_h, bev.output_w, 3)


def test_meter_pixel_round_trip_is_stable_for_ground_points():
    bev = BEV()
    points_meters = np.array([[0, 15, 0], [5, 25, 0], [-5, 20, 0]], dtype=float)

    pixels = bev.meters_to_pixels(points_meters)
    round_trip = bev.pixels_to_meters(pixels)

    np.testing.assert_allclose(round_trip, points_meters[:, :2], atol=1e-6)


def test_calculate_bev_h_respects_pix_per_meter_argument():
    _, calib_matrices = get_BEV_H()
    camera_info = {
        "pitch": -2,
        "yaw": 0,
        "roll": 0,
        "tx": 0,
        "ty": 5,
        "tz": 0,
        "output_w": 1000,
        "output_h": 1000,
    }

    h_20 = calculate_BEV_H(calib_matrices, camera_info, pix_per_meter=20)
    h_80 = calculate_BEV_H(calib_matrices, camera_info, pix_per_meter=80)

    assert h_20.shape == (3, 3)
    assert h_80.shape == (3, 3)
    assert not np.allclose(h_20, h_80)
