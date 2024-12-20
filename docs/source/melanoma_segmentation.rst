Melanoma Segmentation (melanoma_segmentation)
==============================

The `melanoma_segmentation` package provides a comprehensive suite of tools for developing, training, and evaluating 
segmentation models tailored for melanoma detection and skin lesion analysis. It integrates utilities for managing datasets, 
model architectures, evaluation metrics, and result visualization, offering a complete pipeline for medical image segmentation.

Subpackages
-----------

- **Configs**:
  Centralizes project configuration settings, including paths, model details, and training parameters, to streamline management and reproducibility.

- **Datasets**:
  Offers tools to handle image and mask datasets, including classes for loading data, managing paths, applying transformations, and splitting datasets into training, validation, and testing subsets.

- **Models**:
  Implements state-of-the-art segmentation architectures, such as U-Net, Attention U-Net, R2U-Net, R2AttUNet, and TransUNet, designed to achieve high accuracy in medical image segmentation tasks.

- **Results**:
  Includes utilities for evaluating model performance with metrics like Dice coefficient, IoU, accuracy, and recall, as well as visualization tools for overlaying predictions and comparing results.

- **Utils**:
  Provides additional tools, such as dataset preparation utilities, to facilitate the integration of various components of the pipeline.

.. toctree::
   :maxdepth: 4

   melanoma_segmentation.configs
   melanoma_segmentation.datasets
   melanoma_segmentation.models
   melanoma_segmentation.results
   melanoma_segmentation.utils

Module contents
---------------

.. automodule:: melanoma_segmentation
   :members:
   :undoc-members:
   :show-inheritance:
