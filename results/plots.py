import random
import torch
import matplotlib.pyplot as plt


def plot_img_mask_pred(dataset, index=None, plot_pred=False, model=None, device="cuda"):
    """
    author: an-eve
    Plot the image, mask, and prediction
    repository: https://github.com/an-eve/ISIC-2016-lesion-segmentation-challenge/blob/main/helper_plotting.py
    """
    if not index:
        index = random.randint(0, len(dataset) - 1)

    image = dataset[index][0].permute(1, 2, 0)
    mask = dataset[index][1].permute(1, 2, 0)

    if plot_pred:
        img_to_pred = dataset[index][0].unsqueeze(0).type(torch.float32).to(device)
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
