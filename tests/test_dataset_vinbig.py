import cv2
import numpy as np
import pandas as pd

from dataset_tools.vinbig.generate_analytics import generate_bboxes_distribution
from dataset_tools.vinbig.generate_dataset import generate_object_dataset_vinbig


def test_vinbig_generator_crops_class_directories(tmp_path):
    images_path = tmp_path / "images"
    save_dir = tmp_path / "objects"
    images_path.mkdir()
    save_dir.mkdir()

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[2:6, 1:5] = [3, 4, 5]
    cv2.imwrite(str(images_path / "img1.png"), image)
    csv_path = tmp_path / "ann.csv"
    pd.DataFrame(
        [
            {
                "image_id": "img1",
                "fold": 0,
                "x_min": 1,
                "y_min": 2,
                "x_max": 5,
                "y_max": 6,
                "class_id": 2,
            }
        ]
    ).to_csv(csv_path, index=False)

    generate_object_dataset_vinbig(csv_path, images_path, save_dir, fold_idx=None)

    output = cv2.imread(str(save_dir / "2" / "img1_2_1.png"))
    assert output.shape == (4, 4, 3)
    assert output[0, 0].tolist() == [3, 4, 5]


def test_vinbig_analytics_writes_probability_map_and_metadata(tmp_path):
    csv_path = tmp_path / "ann.csv"
    save_dir = tmp_path / "analytics"
    save_dir.mkdir()
    pd.DataFrame(
        [
            {
                "image_id": "img1",
                "fold": 0,
                "x_min": 10,
                "y_min": 10,
                "x_max": 20,
                "y_max": 20,
                "x_min_norm": 0.1,
                "y_min_norm": 0.1,
                "x_max_norm": 0.3,
                "y_max_norm": 0.3,
                "h_norm": 0.2,
                "class_id": 0,
            }
        ]
    ).to_csv(csv_path, index=False)

    generate_bboxes_distribution(csv_path, save_dir, fold_idx=None, output_img_size=10)

    saved = np.load(save_dir / "0.npy", allow_pickle=True).item()
    assert saved["n_bboxes"] == 1
    assert saved["mean_h"] == 0.2
    np.testing.assert_allclose(saved["probability_map"].sum(), 1.0)
    assert (save_dir / "0.png").exists()
