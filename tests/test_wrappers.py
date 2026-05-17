import numpy as np
import pytest

from cap_augmentation.wrappers import CAP_Albu, CAP_Albumentations, CAP_TorchVision


def test_wrapper_exports_keep_albumentations_alias():
    assert CAP_Albumentations is CAP_Albu


def test_torchvision_wrapper_merges_tensor_targets(make_source_image):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    source = make_source_image(color=(200, 20, 10))
    image = torch.zeros((3, 100, 100), dtype=torch.uint8)
    masks = torch.zeros((1, 100, 100), dtype=torch.uint8)
    masks[0, 1:3, 1:3] = 1
    target = {
        "boxes": torch.tensor([[1, 1, 3, 3]], dtype=torch.float32),
        "labels": torch.tensor([9], dtype=torch.int64),
        "masks": masks,
    }
    transform = CAP_TorchVision(
        source_images=[source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        class_idx=4,
    )

    result_image, result_target = transform(image, target)

    assert result_image.shape == image.shape
    assert result_image[:, 70, 50].tolist() == [10, 20, 200]
    torch.testing.assert_close(
        result_target["boxes"],
        torch.tensor([[1, 1, 3, 3], [45, 60, 55, 80]], dtype=torch.float32),
    )
    assert result_target["labels"].tolist() == [9, 4]
    assert result_target["masks"].shape == (2, 100, 100)
    assert int(result_target["masks"][1].sum().item()) == 200


def test_torchvision_wrapper_preserves_tv_tensors(make_source_image):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torchvision import tv_tensors

    source = make_source_image(color=(200, 20, 10))
    image = tv_tensors.Image(torch.zeros((3, 100, 100), dtype=torch.uint8))
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor([[1, 1, 3, 3]], dtype=torch.float32),
        format="XYXY",
        canvas_size=(100, 100),
    )
    target = {
        "boxes": boxes,
        "labels": torch.tensor([2], dtype=torch.int64),
        "semantic_mask": torch.zeros((100, 100), dtype=torch.uint8),
    }
    transform = CAP_TorchVision(
        source_images=[source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        class_idx=4,
    )

    result_image, result_target = transform(image, target)

    assert isinstance(result_image, tv_tensors.Image)
    assert isinstance(result_target["boxes"], tv_tensors.BoundingBoxes)
    assert result_target["boxes"].canvas_size == (100, 100)
    assert str(result_target["boxes"].format).endswith("XYXY")
    assert result_target["labels"].tolist() == [2, 4]
    assert int(result_target["semantic_mask"].sum().item()) == 200


def test_torchvision_wrapper_handles_numpy_images(make_source_image):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    source = make_source_image(color=(200, 20, 10))
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    transform = CAP_TorchVision(
        source_images=[source],
        n_objects_range=[1, 1],
        h_range=[20, 21],
        x_range=[50, 51],
        y_range=[80, 81],
        random_h_flip=False,
        class_idx=4,
    )

    result_image = transform(image)

    assert isinstance(result_image, np.ndarray)
    assert result_image[70, 50].tolist() == [10, 20, 200]
