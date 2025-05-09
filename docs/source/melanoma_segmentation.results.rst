Melanoma Segmentation Evaluation and Visualization (melanoma\_segmentation.results)
======================================

This subpackage provides tools for evaluating and visualizing the performance of segmentation models. It includes utilities for calculating metrics, identifying performance extremes, and creating visual overlays of segmentation results.

Model Evaluation Utility for Image Segmentation
-----------------------------------------------

This module provides an `Evaluator` class for assessing the performance of segmentation models using metrics such as 
Dice coefficient, IoU, accuracy, and recall. It also tracks the images with the highest and lowest scores for each metric.

Class
^^^^^
- **Evaluator**: Evaluates a PyTorch model on a test dataset using common segmentation metrics.

Usage
^^^^^
To evaluate a trained segmentation model:

.. code-block:: python

    from melanoma_segmentation.results.evaluator import Evaluator
    from torch.utils.data import DataLoader

    # Assume `model` is a trained PyTorch segmentation model and `test_dataset` is a dataset for evaluation.
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    evaluator = Evaluator(model_path="path_to_model.pth", model=model, test_dataloader=test_loader, device="cuda")

    avg_dice, avg_iou, avg_accuracy, avg_recall, max_indices, min_indices = evaluator.evaluate()

    print(f"Average Dice: {avg_dice}")
    print(f"Average IoU: {avg_iou}")
    print(f"Average Accuracy: {avg_accuracy}")
    print(f"Average Recall: {avg_recall}")

.. automodule:: melanoma_segmentation.results.evaluator
   :members:
   :undoc-members:
   :show-inheritance:

Visualization Utility for Segmentation Models
---------------------------------------------

This module provides functions for visualizing segmentation results, including the input image, ground truth mask, 
and model predictions. It also supports overlaying masks on images with customizable colors and transparency.

Functions
^^^^^^^^^
- **overlay_mask**: Overlays a binary mask on an image with a specified color and transparency.
- **plot_img_mask_pred**: Visualizes the image, ground truth mask, and optionally the model's prediction.

Usage
^^^^^
To visualize an image with its ground truth and predicted mask:

.. code-block:: python

    from melanoma_segmentation.results.plots import plot_img_mask_pred
    from torch.utils.data import DataLoader

    # Assume `dataset` is a PyTorch dataset containing images and masks, and `model` is a trained segmentation model.
    plot_img_mask_pred(dataset=dataset, index=10, plot_pred=True, model=model, device="cuda")

To overlay a mask on an image:

.. code-block:: python

    from melanoma_segmentation.results.plots import overlay_mask
    import numpy as np

    # Example usage
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
    color = [255, 0, 0]  # Red color
    overlaid_image = overlay_mask(image, mask, color)

.. automodule:: melanoma_segmentation.results.plots
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: melanoma_segmentation.results
   :members:
   :undoc-members:
   :show-inheritance:
