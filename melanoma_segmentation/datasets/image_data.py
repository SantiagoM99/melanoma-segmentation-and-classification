import os


class ImageDataset:
    def __init__(self, base_dir, image_folder, gt_folder):
        """
        Initialize the ImageDataset class with base directory, image folder, and ground truth folder.

        Attributes
        ----------
        base_dir : str
            Base directory containing the image and ground truth folders.
        image_folder : str
            Folder containing the images.
        gt_folder : str
            Folder containing the ground truth masks.
        image_dir : str
            Directory containing the images.
        gt_dir : str
            Directory containing the ground truth masks.
        image_paths : list
            List of image file paths.
        gt_paths : list
            List of ground truth file paths.

        Methods
        -------
        get_image_and_gt_paths()
            Retrieve and store the paths to the images and ground truth masks.
        check_dimensions()
            Print the number of image and ground truth samples and check if they match
        """
        self.base_dir = base_dir
        self.image_folder = image_folder
        self.gt_folder = gt_folder

        self.image_dir = ""
        self.gt_dir = ""

    def get_image_and_gt_paths(self):
        """
        Retrieve and store the paths to the images and ground truth masks.

        Returns
        -------
        tuple
            A tuple containing the image and ground truth paths.
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
