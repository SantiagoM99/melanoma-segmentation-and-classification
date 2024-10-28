import torch

"""
Configuration settings for the skin lesion segmentation project.

Attributes
----------
CONFIG : dict
    Dictionary containing configuration settings for the project.
    - base_dir (str): Base directory containing the data.
    - image_folder (str): Folder containing the images.
    - gt_folder (str): Folder containing the ground truth masks.
    - model_name (str): Name of the segmentation model.
    - split_train (float): Ratio of training data.
    - split_val (float): Ratio of validation data.
    - split_test (float): Ratio of testing data.
    - image_size (int): Size of the input images.
    - batch_size (int): Batch size for training.
    - model_path (str): Path to save the trained model.
    - device (torch.device): Device to run the model on.
"""
CONFIG = {
    "base_dir": "data",
    "image_folder": "ISIC-2017_Training_Data",
    "gt_folder": "ISIC-2017_Training_Part1_GroundTruth",
    "model_name": "unet",
    "split_train": 0.8,
    "split_val": 0.1,
    "split_test": 0.1,
    "image_size": 512,
    "batch_size": 16,
    "model_path": "results/saved_models/",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
