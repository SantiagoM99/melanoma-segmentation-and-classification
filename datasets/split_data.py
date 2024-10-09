# implement a class to split data into train, validation and test sets

import os
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, image_paths, gt_paths, split_train, split_val, split_test):
        """
        Initialize the DataSplitter class with image paths, ground truth paths, and split ratios.

        :param image_paths: The paths to the images.
        :param gt_paths: The paths to the ground truth masks.
        :param split_train: The ratio of training data.
        :param split_val: The ratio of validation data.
        :param split_test: The ratio of testing data.
        """
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.split_train = split_train
        self.split_val = split_val
        self.split_test = split_test

        self.image_train = []
        self.image_val = []
        self.image_test = []
        self.gt_train = []
        self.gt_val = []
        self.gt_test = []

    def split_data(self):
        """
        Split the image and ground truth paths into training, validation, and testing sets.
        """
        # Split the image and ground truth paths into training and testing sets
        image_train, image_test, gt_train, gt_test = train_test_split(
            self.image_paths, self.gt_paths, test_size=self.split_test, random_state=42
        )
        # Split the training set into training and validation sets
        image_train, image_val, gt_train, gt_val = train_test_split(
            image_train,
            gt_train,
            test_size=self.split_val,
            random_state=42,
        )

        self.image_train = image_train
        self.image_val = image_val
        self.image_test = image_test
        self.gt_train = gt_train
        self.gt_val = gt_val
        self.gt_test = gt_test

        return (
            self.image_train,
            self.image_val,
            self.image_test,
            self.gt_train,
            self.gt_val,
            self.gt_test,
        )

    def print_split(self):
        """
        Print the number of samples in each split.
        """
        print(f"Training samples: {len(self.image_train)}")
        print(f"Validation samples: {len(self.image_val)}")
        print(f"Testing samples: {len(self.image_test)}")

    def get_splits(self):
        """
        Return the training, validation, and testing sets.
        """
        return (
            self.image_train,
            self.image_val,
            self.image_test,
            self.gt_train,
            self.gt_val,
            self.gt_test,
        )
