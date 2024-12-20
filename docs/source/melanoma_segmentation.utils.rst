Melanoma Segmentation Util Functions (melanoma\_segmentation.utils)
====================================

This subpackage provides utility functions to streamline dataset preparation and other tasks necessary for 
training and evaluating segmentation models.

Dataset Preparation Utility
---------------------------

This module provides a utility function to prepare datasets for training, validation, and testing. 
It handles data retrieval, splitting, and transformation using configurations provided in a dictionary.

Functions
^^^^^^^^^
- **prepare_datasets**: Prepares and returns the training, validation, and testing datasets with appropriate transformations.

Usage
^^^^^
To prepare datasets using a configuration dictionary:

.. code-block:: python

    from melanoma_segmentation.utils.preparation_tools import prepare_datasets

    config = {
        "base_dir": "data",
        "image_folder": "images",
        "gt_folder": "masks",
        "split_train": 0.7,
        "split_val": 0.2,
        "split_test": 0.1,
        "image_size": 256,
    }

    train_dataset, val_dataset, test_dataset = prepare_datasets(config)

.. automodule:: melanoma_segmentation.utils.preparation_tools
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: melanoma_segmentation.utils
   :members:
   :undoc-members:
   :show-inheritance:
