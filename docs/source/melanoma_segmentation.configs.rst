Melanoma Segmentation Configurations (melanoma\_segmentation.configs)
======================================

This subpackage contains configuration utilities for the skin lesion segmentation project. It includes a centralized configuration module to manage all essential project parameters, such as dataset paths, model settings, and training configurations.

Configuration Module for Skin Lesion Segmentation Project
---------------------------------------------------------

This module defines the configuration settings for the skin lesion segmentation project. The settings include 
paths to data directories, model details, training parameters, and device configuration.

Attributes
^^^^^^^^^^
- **CONFIG**: A dictionary containing configuration settings for the project.
    - **base_dir**: Base directory containing the data.
    - **image_folder**: Folder containing the images.
    - **gt_folder**: Folder containing the ground truth masks.
    - **model_name**: Name of the segmentation model.
    - **split_train**: Ratio of training data.
    - **split_val**: Ratio of validation data.
    - **split_test**: Ratio of testing data.
    - **image_size**: Size of the input images.
    - **batch_size**: Batch size for training.
    - **model_path**: Path to save the trained model.
    - **device**: Device to run the model on.

Usage
^^^^^
The `CONFIG` dictionary is used to centralize all configuration parameters, making it easier to manage and update settings 
across the project. Example:

.. code-block:: python

    from melanoma_segmentation.configs.config_setting import CONFIG

    print(f"Base directory: {CONFIG['base_dir']}")
    print(f"Model will run on: {CONFIG['device']}")

.. automodule:: melanoma_segmentation.configs.config_setting
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: melanoma_segmentation.configs
   :members:
   :undoc-members:
   :show-inheritance:
