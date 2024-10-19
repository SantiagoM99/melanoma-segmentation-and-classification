import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


class Evaluator:
    def __init__(self, model_path, model, test_dataloader, device):
        self.model = model
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()  # Set model to evaluation mode
        self.test_dataloader = test_dataloader
        self.device = device

    def dice_coefficient(self, preds, targets, smooth=1e-6):
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item()

    def iou_score(self, preds, targets, smooth=1e-6):
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()

    def precision(self, preds, targets, smooth=1e-6):
        true_positives = (preds * targets).sum()
        predicted_positives = preds.sum()
        precision = (true_positives + smooth) / (predicted_positives + smooth)
        return precision.item()

    def recall(self, preds, targets, smooth=1e-6):
        true_positives = (preds * targets).sum()
        actual_positives = targets.sum()
        recall = (true_positives + smooth) / (actual_positives + smooth)
        return recall.item()

    def evaluate(self):
        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []

        with torch.no_grad():
            for images, masks in self.test_dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device).float()

                # Forward pass: get predictions
                outputs = self.model(images)
                preds = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
                preds = (preds > 0.5).float()  # Thresholding to get binary predictions

                # Calculate metrics for each sample in the batch
                dice = self.dice_coefficient(preds, masks)
                iou = self.iou_score(preds, masks)
                precision = self.precision(preds, masks)
                recall = self.recall(preds, masks)

                # Store the scores
                dice_scores.append(dice)
                iou_scores.append(iou)
                precision_scores.append(precision)
                recall_scores.append(recall)

        # Compute average metrics
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)

        return avg_dice, avg_iou, avg_precision, avg_recall
