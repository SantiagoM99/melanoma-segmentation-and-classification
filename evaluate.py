from configs.config_setting import CONFIG
from models.unet import UNet
from models.attention_unet import AttUNet
from models.trans_unet import TransUNet
from utils.preparation_tools import prepare_datasets
from results.evaluator import Evaluator
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from results.plots import plot_img_mask_pred
import torch

# Configuration
base_dir = CONFIG["base_dir"]
image_folder = CONFIG["image_folder"]
gt_folder = CONFIG["gt_folder"]
model_name = CONFIG["model_name"]
split_train = CONFIG["split_train"]  
split_val = CONFIG["split_val"]
split_test = CONFIG["split_test"]
image_size = CONFIG["image_size"]
batch_size = CONFIG["batch_size"]
model_path = CONFIG["model_path"]+"unet_128.pth"
device = CONFIG["device"]

CONFIG_FINAL = CONFIG.copy()
CONFIG_FINAL["image_size"] = 128

train_dataset, val_dataset, test_dataset = prepare_datasets(CONFIG_FINAL, train_transform_type="train")

model = DataParallel(UNet())
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)


test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


evaluator = Evaluator(model_path, model, test_dataloader, device)

# Evaluate the model
avg_dice, avg_iou, avg_accuracy, avg_recall, max_indices, min_indices = evaluator.evaluate()

# Print results
print("\n----- Evaluation Results -----")
print(f"Average Dice Coefficient: {avg_dice:.4f}")
print(f"Average IoU Score: {avg_iou:.4f}")
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average Recall: {avg_recall:.4f}")

# Plot the image with the highest and lowest Dice score
print("\nPlotting the image with the highest Dice score:")
plot_img_mask_pred(test_dataset, index=max_indices[0], plot_pred=True, model=model, device=device)

print("\nPlotting the image with the lowest Dice score:")
plot_img_mask_pred(test_dataset, index=min_indices[0], plot_pred=True, model=model, device=device)


