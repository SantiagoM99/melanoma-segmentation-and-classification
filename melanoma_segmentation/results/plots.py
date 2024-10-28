import random
import torch
import matplotlib.pyplot as plt


def plot_img_mask_pred(dataset, index=None, plot_pred=False, model=None, device="cpu"):
    """
    Plot the image, mask, and prediction from a given dataset.

    This function visualizes the original image, ground truth mask, and (optionally) the predicted mask 
    using a trained model. The function displays the image, ground truth mask, and model predictions side by side.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset containing images and masks in the format (image, mask).
    index : int, optional
        Index of the image to plot. If not provided, a random index is selected. Default is None.
    plot_pred : bool, optional
        Flag indicating whether to plot the model's prediction. Default is False.
    model : torch.nn.Module, optional
        The model used for generating predictions. Required if `plot_pred` is True. Default is None.
    device : str, optional
        Device to use for generating predictions (e.g., "cpu" or "cuda"). Default is "cpu".

    References
    ----------
    - Title: ISIC 2016 Lesion Segmentation Challenge
    - Author: an-eve
    - Repository: https://github.com/an-eve/ISIC-2016-lesion-segmentation-challenge/blob/main/helper_plotting.py

    Examples
    --------
    >>> # Plot image and mask from the dataset
    >>> plot_img_mask_pred(dataset, index=5)

    >>> # Plot image, mask, and prediction using a trained model
    >>> plot_img_mask_pred(dataset, index=5, plot_pred=True, model=trained_model, device="cuda")
    """

    if not index:
        index = random.randint(0, len(dataset) - 1)

    image = dataset[index][0].permute(1, 2, 0)
    mask = dataset[index][1].permute(1, 2, 0)

    if plot_pred:
        img_to_pred = dataset[index][0].unsqueeze(0).to(device)
        pred = model(img_to_pred)
        pred = pred.squeeze(0).cpu().detach().permute(1, 2, 0)
        pred[pred < 0] = 0
        pred[pred > 0] = 1

        # Plot the image
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Image")

        # Plot the mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Mask")

        # Plot the predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap="gray")
        plt.title("Prediction")

        # Show the plots
        plt.tight_layout()
        plt.show()

    else:
        # Plot the image
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Image")

        # Plot the mask
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Mask")

        # Show the plots
        plt.tight_layout()
        plt.show()
