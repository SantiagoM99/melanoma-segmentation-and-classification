import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(phase, image_size):
    """
    Get the appropriate transforms for the specified phase.

    This function returns a set of transformations for image preprocessing based on the specified phase.
    During training, data augmentation techniques such as horizontal flipping are applied, whereas in the 
    testing/validation phase, only resizing and normalization are performed.

    Parameters
    ----------
    phase : str
        The phase of the transformation. Must be one of 'train' or 'test'.
    image_size : int
        The size to which the images will be resized (height and width will be equal).

    Returns
    -------
    A.Compose
        A composition of transformations from the Albumentations library.

    Raises
    ------
    ValueError
        If an unsupported phase is specified.

    Examples
    --------
    >>> train_transforms = get_transforms(phase="train", image_size=256)
    >>> val_transforms = get_transforms(phase="test", image_size=256)
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),
                ToTensorV2(),
            ]
        )
    elif phase == "test":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),
                ToTensorV2(),
            ]
        )
    elif phase == "train_v2":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(p=0.75),
                A.OneOf([
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                    ], p=0.6),
                A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
                A.CLAHE(clip_limit=4.0, p=0.7),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, border_mode=0, p=0.6),
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(
            f"Phase {phase} is not supported. Supported phases are 'train' and 'test'."
        )
