
from datasets.image_data import ImageDataset
from datasets.data import SkinLesionDataset
from datasets.split_data import DataSplitter
from models.transform import get_transforms


def prepare_datasets(config, train_transform_type="train"):
    """
    Prepare datasets for training, validation, and testing based on the provided configuration.
    
    Args:
        config (dict): Configuration dictionary containing paths, model settings, and split ratios.
        train_transform_type (str): Specifies the type of transformation to use for training (default: "train").
        
    Returns:
        tuple: A tuple containing train, validation, and test datasets.
    """
    # Unpack the configuration settings
    base_dir = config["base_dir"]
    image_folder = config["image_folder"]
    gt_folder = config["gt_folder"]
    split_train = config["split_train"]
    split_val = config["split_val"]
    split_test = config["split_test"]
    image_size = config["image_size"]
    # Retrieve the image and ground truth paths
    dataset_paths = ImageDataset(base_dir, image_folder, gt_folder)
    print("Retrieving image and ground truth paths...")
    
    image_paths, gt_paths = dataset_paths.get_image_and_gt_paths()

    # Split the data into training, validation, and testing sets (Paths)
    splitter = DataSplitter(image_paths, gt_paths, split_train, split_val, split_test)
    
    img_train_p, img_val_p, img_test_p, gt_train_p, gt_val_p, gt_test_p = splitter.split_data()

    # Create the train, validation, and test datasets based on the paths
    train_dataset = SkinLesionDataset(
        img_train_p, gt_train_p, transform=get_transforms(train_transform_type, image_size)
    )
    val_dataset = SkinLesionDataset(
        img_val_p, gt_val_p, transform=get_transforms("test", image_size)
    )
    test_dataset = SkinLesionDataset(
        img_test_p, gt_test_p, transform=get_transforms("test", image_size)
    )

    # Return the datasets
    return train_dataset, val_dataset, test_dataset