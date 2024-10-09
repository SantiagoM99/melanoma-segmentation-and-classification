import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(phase, image_size):
    """
    Get the appropriate transforms for the specified phase.

    :param phase: The phase of the transformation, either "train" or "val".
    :return: The transforms for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.RGBShift(p=0.2),
                A.Blur(p=0.2),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    elif phase == "test":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(
            f"Phase {phase} is not supported. Supported phases are 'train' and 'test'."
        )
