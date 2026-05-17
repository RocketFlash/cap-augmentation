__author__ = "RocketFlash: https://github.com/RocketFlash"

import cv2
import numpy as np


def _pyplot():
    try:
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Visualization helpers require matplotlib. Install it with: "
            'pip install "cap-augmentation[viz]"'
        ) from exc
    return plt


def show_image(image):
    plt = _pyplot()
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("image")
    plt.show()


def show_image_and_masks(
    image: np.ndarray,
    result_image: np.ndarray,
    result_mask: np.ndarray,
    result_image_anno: np.ndarray,
    is_mask_semantic: bool = False,
):

    plt = _pyplot()
    f, axarr = plt.subplots(2, 2, figsize=(25, 15))

    axarr[0, 0].imshow(image)
    axarr[0, 1].imshow(result_image)
    if is_mask_semantic:
        result_mask_viz = np.zeros_like(image)
        ids = np.unique(result_mask)
        ids = ids[ids != 0]
        for id_i in ids:
            id_color = list(np.random.choice(range(256), size=3))
            result_mask_viz[result_mask == id_i] = id_color
        result_mask = result_mask_viz

    axarr[1, 0].imshow(result_mask)
    axarr[1, 1].imshow(result_image_anno)

    axarr[0, 0].set_aspect("auto")
    axarr[1, 0].set_aspect("auto")
    axarr[0, 1].set_aspect("auto")
    axarr[1, 1].set_aspect("auto")

    axarr[0, 0].set_xticks([])
    axarr[0, 0].set_yticks([])

    axarr[0, 1].set_xticks([])
    axarr[0, 1].set_yticks([])

    axarr[1, 0].set_xticks([])
    axarr[1, 0].set_yticks([])

    axarr[1, 1].set_xticks([])
    axarr[1, 1].set_yticks([])

    f.subplots_adjust(wspace=0, hspace=0)

    plt.show()


def draw_bboxes(image, bboxes, mask=None, color=(255, 0, 0)):

    result_image = image.copy()
    result_mask = np.zeros_like(result_image)

    if mask is not None:
        ids = np.unique(mask)
        ids = ids[ids != 0]
        for id_i in ids:
            id_color = list(np.random.choice(range(256), size=3))
            result_mask[mask == id_i] = id_color
        cv2.addWeighted(result_mask, 0.4, result_image, 1, 0, result_image)

    for bbox in bboxes:
        if len(bbox) == 5:
            x1, y1, x2, y2 = [int(b) for b in bbox[:4]]
            label = int(bbox[4])
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                result_image,
                str(label),
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.rectangle(
                result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2
            )

    if mask is None:
        return result_image, None
    else:
        return result_image, result_mask
