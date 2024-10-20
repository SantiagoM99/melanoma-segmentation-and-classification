import torch

# Configuration settings
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
    "model_path": "results/",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
