import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


class Evaluator:
    """
    Class to evaluate a segmentation model using metrics such as Dice coefficient, IoU, precision, and recall.
    
    Attributes:
        model (nn.Module): The PyTorch model to evaluate.
        test_dataloader (DataLoader): The DataLoader providing test images and masks.
        device (torch.device): The device to run the evaluation on.
    """

    def __init__(self, model_path, model, test_dataloader, device):
        """
        Initialize the Evaluator with a model, test dataloader, and device.

        Args:
            model_path (str): Path to the trained model file.
            model (nn.Module): The PyTorch model to evaluate.
            test_dataloader (DataLoader): Dataloader for test data.
            device (torch.device): Device on which to perform evaluation (e.g., "cuda" or "cpu").
        """
        self.model = model
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()  # Set model to evaluation mode
        self.test_dataloader = test_dataloader
        self.device = device

    def dice_coefficient(self, preds, targets, smooth=1e-6):
        """
        Compute the Dice coefficient between predicted and target masks.

        Args:
            preds (torch.Tensor): Predicted mask.
            targets (torch.Tensor): Ground truth mask.
            smooth (float, optional): Smoothing constant to avoid division by zero. Defaults to 1e-6.

        Returns:
            float: Dice coefficient.
        """
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item()

    def iou_score(self, preds, targets, smooth=1e-6):
        """
        Compute the Intersection over Union (IoU) between predicted and target masks.

        Args:
            preds (torch.Tensor): Predicted mask.
            targets (torch.Tensor): Ground truth mask.
            smooth (float, optional): Smoothing constant to avoid division by zero. Defaults to 1e-6.

        Returns:
            float: IoU score.
        """
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()

    def precision(self, preds, targets, smooth=1e-6):
        """
        Compute the precision between predicted and target masks.

        Args:
            preds (torch.Tensor): Predicted mask.
            targets (torch.Tensor): Ground truth mask.
            smooth (float, optional): Smoothing constant to avoid division by zero. Defaults to 1e-6.

        Returns:
            float: Precision score.
        """
        true_positives = (preds * targets).sum()
        predicted_positives = preds.sum()
        precision = (true_positives + smooth) / (predicted_positives + smooth)
        return precision.item()

    def recall(self, preds, targets, smooth=1e-6):
        """
        Compute the recall between predicted and target masks.

        Args:
            preds (torch.Tensor): Predicted mask.
            targets (torch.Tensor): Ground truth mask.
            smooth (float, optional): Smoothing constant to avoid division by zero. Defaults to 1e-6.

        Returns:
            float: Recall score.
        """
        true_positives = (preds * targets).sum()
        actual_positives = targets.sum()
        recall = (true_positives + smooth) / (actual_positives + smooth)
        return recall.item()

    def evaluate(self):
        """
        Evaluate the model on the test dataset using Dice, IoU, precision, and recall.
        
        Also tracks the indices of the images with the highest and lowest scores for each metric.

        For each metric, the indices are stored in the following order:
        - Dice: max_indices[0], min_indices[0]
        - IoU: max_indices[1], min_indices[1]
        - Precision: max_indices[2], min_indices[2]
        - Recall: max_indices[3], min_indices[3]

        Returns:
            tuple: A tuple containing:
                - avg_dice (float): Average Dice coefficient.
                - avg_iou (float): Average IoU score.
                - avg_precision (float): Average precision score.
                - avg_recall (float): Average recall score.
                - max_indices (list): Indices of the images with the highest Dice, IoU, precision, and recall.
                - min_indices (list): Indices of the images with the lowest Dice, IoU, precision, and recall.
        """
        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []
        indices = []  # Track image indices

        # To track max/min Dice, IoU, precision, and recall scores
        max_values = [-1] * 4
        min_values = [float("inf")] * 4
        max_indices = [0] * 4
        min_indices = [0] * 4

        with torch.no_grad():
            for idx, (images, masks) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device).float()

                # Forward pass: get predictions
                outputs = self.model(images)
                preds = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
                preds = (preds > 0.5).float()  # Thresholding to get binary predictions

                # Calculate metrics
                dice = self.dice_coefficient(preds, masks)
                iou = self.iou_score(preds, masks)
                precision = self.precision(preds, masks)
                recall = self.recall(preds, masks)

                # Store the metrics
                dice_scores.append(dice)
                iou_scores.append(iou)
                precision_scores.append(precision)
                recall_scores.append(recall)
                indices.append(idx)  # Track the index of the image

                # Update max/min values for each metric
                if dice > max_values[0]:
                    max_values[0] = dice
                    max_indices[0] = idx
                if dice < min_values[0]:
                    min_values[0] = dice
                    min_indices[0] = idx

                if iou > max_values[1]:
                    max_values[1] = iou
                    max_indices[1] = idx
                if iou < min_values[1]:
                    min_values[1] = iou
                    min_indices[1] = idx

                if precision > max_values[2]:
                    max_values[2] = precision
                    max_indices[2] = idx
                if precision < min_values[2]:
                    min_values[2] = precision
                    min_indices[2] = idx

                if recall > max_values[3]:
                    max_values[3] = recall
                    max_indices[3] = idx
                if recall < min_values[3]:
                    min_values[3] = recall
                    min_indices[3] = idx

        # Print summary statistics
        print(f"Total number of images evaluated: {len(dice_scores)}")
        print(f"Range of Dice scores: {min(dice_scores)} - {max(dice_scores)}")
        print(f"Range of IoU scores: {min(iou_scores)} - {max(iou_scores)}")
        print(f"Range of Precision scores: {min(precision_scores)} - {max(precision_scores)}")
        print(f"Range of Recall scores: {min(recall_scores)} - {max(recall_scores)}")

        # Compute average metrics
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)

        # Return the average metrics and the indices for the highest/lowest scores
        return avg_dice, avg_iou, avg_precision, avg_recall, max_indices, min_indices
