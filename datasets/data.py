import os
import torch
from PIL import Image
import numpy as np
import albumentations as A
from torch.utils.data import Dataset


class SkinLesionDataset(Dataset):
    """
    Custom dataset class for skin lesion segmentation.

    Attributes
    ----------
    img_dir : list
        List of image file paths.
    mask_dir : list
        List of mask file paths.
    transform : albumentations.Compose
        Image augmentation pipeline.    
    """
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        # Load image and mask
        img_name = self.img_dir[idx]
        mask_name = self.mask_dir[idx]

        # Open image and mask
        image = Image.open(img_name)
        mask = Image.open(mask_name)

        # Transform into array
        img_np = np.array(image)
        msk_np = np.array(mask)
        msk_np[msk_np == 255.0] = 1.0

        if self.transform:
            # For reference (C,H,W) represent (channel, height, width)
            transformed = self.transform(image=img_np, mask=msk_np)
            img_np = transformed["image"]
            msk_np = transformed["mask"]
            # Add channel dimension to make it compatible with PyTorch (C, H, W)
            msk_np = torch.unsqueeze(msk_np, 0)

        return img_np, msk_np
