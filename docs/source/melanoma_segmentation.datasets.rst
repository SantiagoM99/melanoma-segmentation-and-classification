Melanoma Segmentation Dataset handling (melanoma\_segmentation.datasets)
=======================================

This subpackage provides utilities for handling datasets related to skin lesion segmentation. It includes modules for managing datasets, splitting data into training/validation/testing sets, and applying augmentations.

Skin Lesion Segmentation Dataset Module
---------------------------------------

This module defines a custom dataset class for handling skin lesion images and their corresponding masks.
It is designed to be compatible with PyTorch's Dataset class, and it supports the use of Albumentations
for image augmentation. 

Classes
^^^^^^^^
- **SkinLesionDataset**: A dataset class for loading and processing skin lesion images and segmentation masks.

Usage
^^^^^
The `SkinLesionDataset` class is used to load images and their associated masks, apply optional augmentations,
and return them in a format compatible with PyTorch models.

.. automodule:: melanoma_segmentation.datasets.data
   :members:
   :undoc-members:
   :show-inheritance:

Image Dataset Utility Module
----------------------------

This module provides a utility class for handling datasets consisting of images and their corresponding ground truth masks. 
It is designed to facilitate dataset organization, path retrieval, and consistency checks between images and masks.

Classes
^^^^^^^^
- **ImageDataset**: A class for managing image datasets and performing basic checks and retrieval operations.

Usage
^^^^^
The `ImageDataset` class is used to initialize a dataset directory, retrieve paths for images and ground truth masks, 
and ensure consistency in the number of samples. Example:

.. code-block:: python

    dataset = ImageDataset(base_dir="path/to/dataset", image_folder="images", gt_folder="masks")
    image_paths, mask_paths = dataset.get_image_and_gt_paths()
    dataset.check_dimensions()

.. automodule:: melanoma_segmentation.datasets.image_data
   :members:
   :undoc-members:
   :show-inheritance:

Data Splitting Utility Module
-----------------------------

This module provides a utility class for splitting datasets into training, validation, and testing sets.
It ensures that the image and ground truth paths are split consistently, supporting reproducibility with a fixed random state.

Classes
^^^^^^^^
- **DataSplitter**: A class for splitting datasets into training, validation, and testing subsets.

Usage
^^^^^
The `DataSplitter` class is used to divide a dataset of images and corresponding ground truth masks into 
train, validation, and test sets based on specified ratios. Example:

.. code-block:: python

    splitter = DataSplitter(
        image_paths=image_paths,
        gt_paths=gt_paths,
        split_train=0.7,
        split_val=0.2,
        split_test=0.1
    )
    splitter.split_data()
    splitter.print_split()

.. automodule:: melanoma_segmentation.datasets.split_data
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: melanoma_segmentation.datasets
   :members:
   :undoc-members:
   :show-inheritance:
