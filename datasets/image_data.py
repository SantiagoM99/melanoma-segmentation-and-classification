import os


class ImageDataset:
    def __init__(self, base_dir, image_folder, gt_folder):
        """
        Initialize the ImageDataset class with base directory, image folder, and ground truth folder.

        :param base_dir: The base directory where images and ground truth data are stored.
        :param image_folder: The subdirectory for images.
        :param gt_folder: The subdirectory for ground truth masks.
        """
        self.base_dir = base_dir
        self.image_folder = image_folder
        self.gt_folder = gt_folder

        self.image_dir = ""
        self.gt_dir = ""

    def get_image_and_gt_paths(self):
        """
        Retrieve and store the paths to the images and ground truth masks.
        """
        # Combine the base directory with subdirectories for images and ground truth
        self.image_dir = os.path.join(self.base_dir, self.image_folder)
        self.gt_dir = os.path.join(self.base_dir, self.gt_folder)
        # Get sorted paths for images and ground truth masks
        image_paths = sorted(
            [
                os.path.join(self.image_dir, fname)
                for fname in os.listdir(self.image_dir)
                if fname.endswith(".jpg")
            ]
        )
        gt_paths = sorted(
            [
                os.path.join(self.gt_dir, fname)
                for fname in os.listdir(self.gt_dir)
                if fname.endswith(".png")
            ]
        )
        return image_paths, gt_paths

    def check_dimensions(self):
        """
        Print the number of image and ground truth samples and check if they match.
        """

        if len(self.image_paths) == len(self.gt_paths):
            print(
                f"The number of image and mask samples match. Total samples: {len(self.image_paths)}"
            )
        else:
            print(
                f"The number of image and mask samples do not match. "
                f"Images: {len(self.image_paths)}, Masks: {len(self.gt_paths)}"
            )
