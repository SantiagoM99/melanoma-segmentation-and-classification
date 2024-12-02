import random
import matplotlib.pyplot as plt
import numpy as np
import torch

def overlay_mask(image, mask, color, alpha=0.5, border_only=False):
        """
        Overlay a mask on an image with a given color and transparency.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.
        mask : numpy.ndarray
            Binary mask to overlay.
        color : list
            RGB color to use for the overlay.
        alpha : float, optional
            Transparency of the overlay. Default is 0.5.
        border_only : bool, optional
            If True, only the border of the mask will be shown. Default is False.

        Returns
        -------
        numpy.ndarray
            Image with the overlay applied
        """
        overlay = image.copy()
        if border_only:
            from scipy.ndimage import binary_dilation
            mask_border = binary_dilation(mask) & ~mask
            overlay[mask_border > 0] = color
        else:
            for c in range(3):  # Apply alpha transparency
                overlay[..., c] = (
                    overlay[..., c] * (1 - alpha) + color[c] * alpha * mask
                )
        # Ensure values remain in valid range
        return np.clip(overlay, 0, 255).astype(np.uint8)

def plot_img_mask_pred(dataset, index=None, plot_pred=False, model=None, device="cpu", comparison=False):
    """
    Plot the image, ground truth mask, and prediction from a given dataset.

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
    comparison : bool, optional
        If True, only the image with both masks (ground truth and predicted) as borders will be shown. Default is False.
  
    """
    if index is None:
        index = random.randint(0, len(dataset) - 1)

    # Get image and mask
    image = dataset[index][0].permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())  # Scale to [0, 1]
    image = (image * 255).astype(np.uint8)  # Convert to [0, 255] integers
    mask = dataset[index][1].permute(1, 2, 0).cpu().numpy().squeeze()

    # Generate prediction if needed
    if plot_pred:
        img_to_pred = dataset[index][0].unsqueeze(0).to(device)
        pred = model(img_to_pred).squeeze(0).cpu().detach().permute(1, 2, 0).numpy().squeeze()
        pred_binary = (pred > 0.5).astype(np.uint8)  # Convert predictions to binary mask
    else:
        pred_binary = None

    # Define colors for ground truth and prediction
    gt_color = [5, 72, 161]
    pred_color = [173, 0, 4]

    if comparison and plot_pred:
        # Show comparison image with both borders
        comparison_image = overlay_mask(image, mask, gt_color, border_only=True)
        comparison_image = overlay_mask(comparison_image, pred_binary, pred_color, border_only=True)

        plt.figure(figsize=(8, 8))
        plt.imshow(comparison_image)
        plt.axis("off")

        # Add legend for the colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=np.array(gt_color) / 255, edgecolor='black', label='Ground Truth'),
            Patch(facecolor=np.array(pred_color) / 255, edgecolor='black', label='Prediction')
        ]
        plt.title("Image with Ground Truth and Prediction Segmentation")
        plt.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
        plt.tight_layout()
        plt.show()
        return

    if plot_pred:
        # Create overlays for ground truth and prediction
        image_with_gt = overlay_mask(image, mask, gt_color, alpha=0.5)
        image_with_pred = overlay_mask(image, pred_binary, pred_color, alpha=0.5)

        # Plot the images
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(image_with_gt)
        plt.title("Image with Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(image_with_pred)
        plt.title("Image with Prediction")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
    else:
        # Show only image and ground truth overlay
        image_with_gt = overlay_mask(image, mask, gt_color, alpha=0.5)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(image_with_gt)
        plt.title("Image with Ground Truth")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        plt.imsave("segmented_img.png", image_with_gt)
