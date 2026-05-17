import cv2
import numpy as np
import yaml

from . import config as cfg


def get_RX(pitch_angle):
    pitch_angle = (np.pi / 180) * pitch_angle
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(pitch_angle), -np.sin(pitch_angle), 0],
            [0, np.sin(pitch_angle), np.cos(pitch_angle), 0],
            [0, 0, 0, 1],
        ]
    )


def get_RY(yaw_angle):
    yaw_angle = (np.pi / 180) * yaw_angle
    return np.array(
        [
            [np.cos(yaw_angle), 0, np.sin(yaw_angle), 0],
            [0, 1, 0, 0],
            [-np.sin(yaw_angle), 0, np.cos(yaw_angle), 0],
            [0, 0, 0, 1],
        ]
    )


def get_RZ(roll_angle):
    roll_angle = (np.pi / 180) * roll_angle
    return np.array(
        [
            [np.cos(roll_angle), -np.sin(roll_angle), 0, 0],
            [np.sin(roll_angle), np.cos(roll_angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def get_T(vtx, vty, vtz):
    return np.array([[1, 0, 0, vtx], [0, 1, 0, vty], [0, 0, 1, vtz], [0, 0, 0, 1]])


def calculate_BEV_H(calib_params, camera_info=None, pix_per_meter=None):
    if camera_info is None:
        camera_info = cfg.camera_info[cfg.camera_name]
    if pix_per_meter is None:
        pix_per_meter = cfg.pix_per_meter
    output_w = camera_info["output_w"]
    output_h = camera_info["output_h"]

    RX = get_RX(camera_info["pitch"])
    RY = get_RY(camera_info["yaw"])
    RZ = get_RZ(camera_info["roll"])
    T = get_T(camera_info["tx"], camera_info["ty"], camera_info["tz"])

    camera2xyz = get_RX(90) @ get_RZ(180)
    camera2loco = camera2xyz @ RZ @ RY @ RX @ T

    ex_loco = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    camera2loco = ex_loco @ camera2loco

    R = camera2loco[:3, :3]
    T = camera2loco[:3, 3]

    K = calib_params["camera_matrix"]
    H = np.zeros((3, 3))
    H[:, :2] = (K @ R.T)[:3, :2]
    H[:, 2] = -K @ R.T @ T

    H_inv = np.linalg.inv(H)
    image2ground = H_inv

    meters_to_pix = np.array(
        [[0, -pix_per_meter, output_w * 0.5], [-pix_per_meter, 0, output_h], [0, 0, 1]]
    )
    image2ground = meters_to_pix @ image2ground

    return image2ground


def get_BEV_H(camera_info=None, calib_yaml_path=None, pix_per_meter=None):
    """Load camera intrinsics and build the image-to-BEV homography.

    The homography uses the camera matrix only. Input images are expected to be
    rectified already; distortion coefficients in ROS-style calibration YAMLs
    are intentionally ignored.
    """
    if calib_yaml_path is None:
        calib_yaml_path = cfg.DEFAULT_CALIB_PATH
    calib_matrices = {}

    with open(calib_yaml_path) as file:
        calibration_params = yaml.safe_load(file)

        param = calibration_params["camera_matrix"]
        matrix = np.array(param["data"]).reshape((param["rows"], param["cols"]))
        calib_matrices["camera_matrix"] = matrix

    H = calculate_BEV_H(calib_matrices, camera_info, pix_per_meter=pix_per_meter)
    return H, calib_matrices


class BEV(object):
    def __init__(self, camera_info=None, calib_yaml_path=None, pix_per_meter=None):
        if pix_per_meter is None:
            pix_per_meter = cfg.pix_per_meter
        self.H, self.calib_matrices = get_BEV_H(
            camera_info,
            calib_yaml_path,
            pix_per_meter=pix_per_meter,
        )
        self.inv_H = np.linalg.inv(self.H)
        self.pixels_per_meter = pix_per_meter
        if camera_info is None:
            camera_info = cfg.camera_info[cfg.camera_name]
        self.output_w = camera_info["output_w"]
        self.output_h = camera_info["output_h"]

        self.f_x = self.calib_matrices["camera_matrix"][0, 0]
        self.f_y = self.calib_matrices["camera_matrix"][1, 1]

    def transform(self, img):
        transformed_img = cv2.warpPerspective(
            img, self.H, (self.output_w, self.output_h)
        )
        return transformed_img

    def calculate_dist_meters(self, p_meters):
        dists = np.sqrt(
            p_meters[:, 0] * p_meters[:, 0] + p_meters[:, 1] * p_meters[:, 1]
        )
        return dists

    def calculate_dist(self, points_bev):
        new_p_centered = np.array([self.output_w / 2, self.output_h]) - points_bev
        new_p_meters = new_p_centered / self.pixels_per_meter
        dists = self.calculate_dist_meters(new_p_meters)
        return dists

    def calculate_dist_bev(self, points):
        points_bev = self.points_to_bev(points)
        return self.calculate_dist(points_bev)

    def points_to_bev(self, points):
        points_ex = np.ones((points.shape[0], points.shape[1] + 1))
        points_ex[:, :2] = points
        new_p = self.H @ points_ex.T
        new_p = new_p.T
        new_p /= new_p[:, 2:]
        new_p = new_p[:, :2]

        return new_p

    def bev_to_points(self, points):
        points_ex = np.ones((points.shape[0], points.shape[1] + 1))
        points_ex[:, :2] = points

        new_p = self.inv_H @ points_ex.T
        new_p = new_p.T
        new_p /= new_p[:, 2:]
        new_p = new_p[:, :2]

        return new_p

    def pixels_to_meters(self, points):
        points_bev = self.points_to_bev(points)
        new_p_centered = np.array([self.output_w / 2, self.output_h]) - points_bev
        new_p_meters = new_p_centered / self.pixels_per_meter
        return new_p_meters

    def meters_to_pixels(self, points):
        new_p_pixels = self.pixels_per_meter * points
        new_p_uncentered = (
            np.array([self.output_w / 2, self.output_h]) - new_p_pixels[:, :2]
        )
        new_p_pixels = self.bev_to_points(new_p_uncentered)

        return new_p_pixels

    def get_height_in_pixels(self, height, distance):
        return self.f_y * height / distance

    def __call__(self, img):
        return self.transform(img)
