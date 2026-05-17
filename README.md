# "Cut and paste" augmentation

[![DOI](https://zenodo.org/badge/328174810.svg)](https://zenodo.org/badge/latestdoi/328174810)

Repository contains easy to use Python implementation of "Cut and paste" augmentation for object detection and instance and semantic segmentations. The main idea was taken from [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://arxiv.org/pdf/2012.07177v1.pdf) and supplemented by the ability to add objects in 3D in the camera coordinate system using a Bird's Eye View Transformation (BEV). Optional wrappers are available for [Albumentations](https://github.com/albumentations-team/albumentations) and Torchvision.

<figure>
  <img src="./examples/images/all.jpg"></img>
</figure>

## Installation

The package is published on PyPI:

```bash
pip install cap-augmentation
```

Optional integrations are installed as extras:

```bash
pip install "cap-augmentation[albumentations]"   # CapAlbumentations wrapper
pip install "cap-augmentation[torchvision]"      # CapTorchvision wrapper
pip install "cap-augmentation[histogram]"        # histogram_matching=True support
pip install "cap-augmentation[viz]"              # visualization helpers (matplotlib)
pip install "cap-augmentation[dataset]"          # dependencies for dataset_tools/ scripts
```

To install several extras at once:

```bash
pip install "cap-augmentation[albumentations,torchvision,histogram,viz]"
```

### From source (for development)

Clone the repository and install in editable mode with the test extras:

```bash
git clone https://github.com/RocketFlash/cap-augmentation.git
cd cap-augmentation
pip install -e ".[test,torchvision]"
pytest
```

## Public API

```python
from cap_augmentation import (
    CapAug,              # core cut-and-paste augmenter
    CapAugMulticlass,    # combine per-class CapAug instances
    CapAlbumentations,   # Albumentations DualTransform wrapper
    CapTorchvision,      # torchvision v2-style wrapper
    ImageMaskTransform,  # adapter for per-object (image, mask) callables
    resize_keep_ar,      # aspect-ratio-preserving resize helper
)
```

The wrapper classes require their respective extras (`albumentations`,
`torchvision`).

## Example of usage

All examples are shown in [examples/notebooks/bev_and_pedestrians_demo.ipynb](https://github.com/RocketFlash/cap-augmentation/blob/main/examples/notebooks/bev_and_pedestrians_demo.ipynb)
(BEV / pixel coordinates / multi-class) and
[examples/notebooks/vinbig_demo.ipynb](https://github.com/RocketFlash/cap-augmentation/blob/main/examples/notebooks/vinbig_demo.ipynb)
(VinBigData chest X-rays).


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Rmln475YERs5ZIp3_jDKTV8JEfk_qDdy?usp=sharing)

### Usage in pixel coordinates

```python
from cap_augmentation import CapAug
import cv2

SOURCE_IMAGES = ['list/', 'of/', 'paths/', 'to/', 'the/', 'source/', 'image/', 'files']
##### For example a list of paths to images can be set like this #####
# DATASET_ROOT = Path('data/human_dataset_filtered/')
# SOURCE_IMAGES = sorted(list(DATASET_ROOT.glob('*.png')))
######################################################################

image = cv2.imread('path/to/the/destination/image')

cap_aug = CapAug(SOURCE_IMAGES, n_objects_range=[10,20],
                                       h_range=[100,101],
                                       x_range=[500, 1500],
                                       y_range=[600 ,1000],
                                       coords_format='xyxy') # xyxy, xywh or yolo
result_image, bboxes_coords, semantic_mask, instance_mask = cap_aug(image)
```

### Usage in camera coordinate system (all values are in meters)

When using bev transformation it is necessary to set range values in meters.

```python
from cap_augmentation import CapAug
from cap_augmentation.bev import BEV
import cv2

SOURCE_IMAGES = ['list/', 'of/', 'paths/', 'to/', 'the/', 'source/', 'image/', 'files']

image = cv2.imread('path/to/the/destination/image')

# Extrinsic camera parameters
camera_info = {'pitch' : -2 ,
               'yaw' : 0 ,
               'roll' : 0 ,
               'tx' : 0,
               'ty' : 5,
               'tz' : 0,
               'output_w': 1000, # output bev image shape
               'output_h': 1000}
calib_yaml_path=None # path to intrinsic parameters (see example in src/cap_augmentation/bev/default_calibration.yaml)
                     # if calib_yaml_path is None, intrinsic params will be loaded from the packaged default

bev_transform = BEV(camera_info=camera_info,
                    calib_yaml_path=calib_yaml_path)

cap_aug = CapAug(SOURCE_IMAGES, bev_transform=bev_transform,
                                              n_objects_range=[30,50],
                                              h_range=[2.0, 2.5],
                                              x_range=[-25, 25],
                                              y_range=[0 ,100],
                                              z_range=[0 ,2],
                                              coords_format='yolo') # xyxy, xywh or yolo
result_image, bboxes_coords, semantic_mask, instance_mask = cap_aug(image)
```

### Multi-class usage

`CapAugMulticlass` runs several `CapAug` instances (one per class) and merges
their boxes/masks, tagging each generated box with its class id.

```python
from cap_augmentation import CapAug, CapAugMulticlass

cap_augs = [
    CapAug(PEDESTRIAN_IMAGES, n_objects_range=[5, 10], h_range=[80, 120],
           x_range=[0, 1920], y_range=[400, 1000]),
    CapAug(CAR_IMAGES, n_objects_range=[2, 5], h_range=[60, 100],
           x_range=[0, 1920], y_range=[400, 1000]),
]
cap_multiclass = CapAugMulticlass(
    cap_augs=cap_augs,
    probabilities=[1.0, 0.7],
    class_idxs=[1, 2],
)
result_image, boxes_with_class, semantic_mask, instance_masks = cap_multiclass(image)
```

### Usage with albumentations

Install the optional Albumentations integration first:

```bash
pip install -e ".[albumentations]"
```

```python

from cap_augmentation import CapAlbumentations
import albumentations as A

transform = A.Compose([
    CapAlbumentations(p=1,
                      source_images=SOURCE_IMAGES,
                      n_objects_range=[10,20],
                      h_range=[100,101],
                      x_range=[500, 1500],
                      y_range=[600 ,1000],
                      class_idx=1),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomRain(p=1.0, blur_value=3)
], bbox_params=A.BboxParams(format='pascal_voc'))

```

Do not share one `CapAlbumentations` instance across concurrent threads;
Albumentations calls image, mask, and bounding-box hooks sequentially on the
same transform object.

### Usage with torchvision

The Torchvision integration follows the detection target style used by
`torchvision.transforms.v2`: images can be tensors, `tv_tensors.Image`, PIL
images, or numpy arrays, and targets are dictionaries with `boxes`, `labels`,
and optionally `masks`.

```python
from cap_augmentation import CapTorchvision

transform = CapTorchvision(
    source_images=SOURCE_IMAGES,
    n_objects_range=[10, 20],
    h_range=[100, 101],
    x_range=[500, 1500],
    y_range=[600, 1000],
    class_idx=1,
)

image, target = transform(image, target)
```

### Object-level transforms

`CapAug` can also transform each pasted object before it is inserted. Existing
Albumentations callables still work through the `albu_transforms` argument; new
code can use the library-neutral `object_transforms` argument.

`histogram_matching=True` requires the `histogram` extra.

```python
from cap_augmentation import CapAug, ImageMaskTransform

def object_transform(image, mask):
    return image, mask

cap_aug = CapAug(
    SOURCE_IMAGES,
    object_transforms=ImageMaskTransform(object_transform),
)
```

### Usage with multiple classes
Example of usage cold be found in [examples/notebooks/bev_and_pedestrians_demo.ipynb](https://github.com/RocketFlash/cap-augmentation/blob/main/examples/notebooks/bev_and_pedestrians_demo.ipynb)

## Data preparation

Any png images with transparency are suitable for inserting objects for object detection or instance segmentation. It is possible to generate own dataset of png images with transparency by cutting images from various segmentation datasets. An example of preparing such a dataset for insertion is shown below.

The `dataset_tools/` scripts are repository tools, not part of the installed
Python package. Run them from a cloned repository after installing the
`dataset` extra (`pip install -e ".[dataset]"`).

### Generate pedestrians dataset from CityScapes and CityPersons

Put [Cityscapes](https://www.cityscapes-dataset.com/) and [CityPersons](https://github.com/cvgroup-njust/CityPersons) datasets in ./data folder. Edit parameters in dataset_tools/cityscapes/config.py if you want and then just run:

```bash
./dataset_tools/cityscapes/run.sh
```

This script will create a dataset of png images cutted and filtered in the data/human_dataset_filtered folder or in the folder that you specified in the dataset_tools/cityscapes/config.py file.

Another option is to run python scripts manually step by step. First, we need to create .png files of people using instance masks from cityscapes dataset:

```bash
python dataset_tools/cityscapes/generate_dataset.py
```

Next, we need to filter images to remove too small or too cropped (only a small part of the body is visible) images:

```bash
python dataset_tools/cityscapes/filter_dataset.py
```

Now the dataset for insertion is available in ./data/human_dataset_filtered.

### Generate medical-imaging dataset from VinBigData

[VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection)
is a public dataset of chest X-rays annotated with bounding boxes for 14
thoracic abnormality classes. The `dataset_tools/vinbig/` scripts crop each annotated
bounding box into a per-class PNG library suitable for cut-and-paste insertion,
and (optionally) compute per-class spatial distributions used as
`probability_map` / `mean_h_norm` inputs for `CapAug`.

Edit paths in `dataset_tools/vinbig/config.py` to point at the VinBigData PNG
images and the annotation CSV, then run:

```bash
# Crop annotated boxes into data/vinbig_dataset/<class_id>/<image_id>_<class>_<n>.png
python dataset_tools/vinbig/generate_dataset.py

# Save per-class probability maps + bbox stats to data/vinbig_dataset/analytics/<class_id>.npy
python dataset_tools/vinbig/generate_analytics.py
```

The analytics files are saved as Python dictionaries pickled inside `.npy`
files. They are intended to be loaded with
`np.load(path, allow_pickle=True).item()` — only load files from a trusted
source, since pickled objects can execute arbitrary code on load. An end-to-end
example is in
[examples/notebooks/vinbig_demo.ipynb](https://github.com/RocketFlash/cap-augmentation/blob/main/examples/notebooks/vinbig_demo.ipynb).
