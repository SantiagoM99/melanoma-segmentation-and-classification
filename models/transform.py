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
    else:
        raise ValueError(
            f"Phase {phase} is not supported. Supported phases are 'train' and 'test'."
        )
