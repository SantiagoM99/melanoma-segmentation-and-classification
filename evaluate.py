from configs.config_setting import CONFIG
from datasets.image_data import ImageDataset
from datasets.data import SkinLesionDataset
from datasets.split_data import DataSplitter
from models.transform import get_transforms
from models.unet import UNet
# from models.attention_unet import AttUNet
from models.trans_unet import TransUNet
from utils import prepare_datasets
from results.plots import plot_img_mask_pred
from results.evaluator import Evaluator
from torch.utils.data import DataLoader
from torch.nn import DataParallel


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
model_path = CONFIG["model_path"]
device = CONFIG["device"]

train_dataset, val_dataset, test_dataset = prepare_datasets(CONFIG, train_transform_type="train")


test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



model = DataParallel(UNet())


evaluator = Evaluator(model_path+"unet.pth", model, test_dataloader, device)


avg_dice, avg_iou, avg_precision, avg_recall = evaluator.evaluate()

print("\n----- Evaluation Results -----")
print(f"Average Dice Coefficient: {avg_dice:.4f}")
print(f"Average IoU Score: {avg_iou:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
