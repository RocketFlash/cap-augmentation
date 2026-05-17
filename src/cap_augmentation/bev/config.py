from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CALIB_PATH = PACKAGE_DIR / "camera_intrinsic_params.yaml"

camera_name = "camera0"

# Path to intrinsic camera parameters file
calib_file_path = {"camera0": DEFAULT_CALIB_PATH}

# Extrinsic camera parameters
camera_info = {
    "camera0": {
        "pitch": -2,
        "yaw": 0,
        "roll": 0,
        "tx": 0,
        "ty": 5,
        "tz": 0,
        "output_w": 1000,
        "output_h": 1000,
    }
}

pix_per_meter = 20
