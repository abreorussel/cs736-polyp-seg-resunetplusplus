import os
import argparse
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from dataset.polyps_dataloader import PolypsDataset
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
from utils.hparams import HParam
from utils import metrics

from dataset.polyps_dataloader import *

# Define test directories according to your train.py
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'new_data/Kvasir-SEG')
TEST_DIR = os.path.join(DATA_DIR, 'test')
TEST_IMGS_DIR = os.path.join(TEST_DIR, 'images')
TEST_LABELS_DIR = os.path.join(TEST_DIR, 'masks')


def evaluate(model, test_loader, device, result_folder):
    """
    Evaluates the model on the test dataset. Computes dice coefficient,
    pixel accuracy, and saves a combined image (input, ground truth, prediction)
    for each test sample into result_folder.
    """
    model.eval()
    dice_sum = 0.0
    pixel_correct = 0
    pixel_total = 0
    num_samples = 0

    os.makedirs(result_folder, exist_ok=True)

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs = data["image"].to(device)  # shape: [B, C, H, W]
            labels = data["mask"].to(device)     # shape: [B, 1, H, W]

            # Forward pass (apply sigmoid to get probability between 0 and 1)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)

            # Binarize predictions with threshold=0.5
            preds = (outputs > 0.6).float()

            # Update dice coefficient (using your metrics function)
            # Here we assume metrics.dice_coeff accepts (prediction, ground_truth)
            dice = metrics.dice_coeff(preds, labels)
            # Multiply by batch size to later compute the average
            dice_sum += dice * inputs.size(0)
            num_samples += inputs.size(0)

            # Compute pixel accuracy (all channels)
            pixel_correct += (preds == labels).float().sum().item()
            pixel_total += torch.numel(labels)

            # Save results: for each sample, create a combined image:
            for i in range(inputs.size(0)):
                # Bring tensor to cpu and convert to numpy array
                input_np = inputs[i].cpu().numpy()         # [C, H, W]
                label_np = labels[i].cpu().numpy()           # [1, H, W]
                pred_np = preds[i].cpu().numpy()             # [1, H, W]

                # For input: convert from tensor (C,H,W) in [0,1] to (H,W,C) in [0,255]
                input_img = (np.transpose(input_np, (1, 2, 0)) * 255).astype(np.uint8)

                # For label and prediction: squeeze channel and scale to [0,255]
                label_img = (label_np.squeeze(0) * 255).astype(np.uint8)
                pred_img = (pred_np.squeeze(0) * 255).astype(np.uint8)

                # Optionally, convert label and prediction to 3-channel images for visualization
                label_img_color = cv2.cvtColor(label_img, cv2.COLOR_GRAY2BGR)
                pred_img_color = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)

                # Combine images side-by-side: input | ground truth | prediction
                # combined = np.hstack((input_img, label_img_color, pred_img_color))
                # Define unique output filename
                out_filename = os.path.join(result_folder, f"sample_{batch_idx * inputs.size(0) + i}.png")
                cv2.imwrite(out_filename, pred_img_color)

    avg_dice = dice_sum / num_samples
    accuracy = pixel_correct / pixel_total

    return avg_dice, accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluation for Polyp Segmentation")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to YAML config file")
    parser.add_argument('-m', '--model', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('-o', '--output', type=str, default="results", help="Folder to store result images")
    parser.add_argument('--device', type=str, default="cuda:3" if torch.cuda.is_available() else "cpu", help="Computation device")
    args = parser.parse_args()

    # Load hyperparameters from config file
    hp = HParam(args.config)

    device = args.device

    # Build model based on configuration
    if hp.RESNET_PLUS_PLUS:
        model = ResUnetPlusPlus(3).to(device)
        print("RESEUNET ++ !!!!!!!!!!!!!")

    else:
        model = ResUnet(3, 64).to(device)

    # Load model checkpoint
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.model, checkpoint.get("epoch", "N/A")))
    else:
        raise ValueError("Checkpoint not found: {}".format(args.model))

    # Define test transforms (similar to validation transforms in train.py)
    # test_transform = v2.Compose([
    #     v2.ToImage(),                       # converts numpy arrays (or tensors) to PIL Image
    #     v2.ToDtype(torch.float32, scale=True),  # converts to float and scales [0,255] -> [0,1]
    # ])
    test_transform = v2.Compose([
        GrayscaleNormalization(mean=0.5, std=0.5),
        ToTensor(),
    ])

    # Create test dataset and dataloader
    test_dataset = PolypsDataset(TEST_IMGS_DIR, TEST_LABELS_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, num_workers=2, shuffle=False)

    # Evaluate the model on test data
    avg_dice, accuracy = evaluate(model, test_loader, device, args.output)

    print(f"Test Dice Coefficient: {avg_dice:.4f}")
    print(f"Test Pixel Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()



'''
python eval.py  --config "configs/polyps.yaml"  -m "checkpoints/default/default_checkpoint_16000.pt"
'''