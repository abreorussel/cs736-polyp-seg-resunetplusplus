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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'new_data/Kvasir-SEG')
TEST_DIR = os.path.join(DATA_DIR, 'test')
TEST_IMGS_DIR = os.path.join(TEST_DIR, 'images')
TEST_LABELS_DIR = os.path.join(TEST_DIR, 'masks')


def construct_overlay_image(original_img, segmented_img, gt_img , crt_id, results_dir):

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    prediction_color = [58, 140, 255]  # Blue 
    gt_color = [47, 237, 57]  # Green  
    intersection_color = [236, 255, 88]  # Yellow 

    colored_prediction = np.zeros_like(original_img)
    colored_gt = np.zeros_like(original_img)
    colored_intersection = np.zeros_like(original_img)

    prediction_mask = segmented_img == 255
    gt_mask = gt_img == 255
    intersection_mask = np.logical_and(prediction_mask, gt_mask)

    colored_prediction[prediction_mask] = prediction_color 
    colored_gt[gt_mask] = gt_color 
    colored_intersection[intersection_mask] = intersection_color

    overlay_img = original_img.copy()

    overlay_img[prediction_mask] = cv2.addWeighted(original_img[prediction_mask], 0.6, colored_prediction[prediction_mask], 0.6, 0)
    overlay_img[gt_mask] = cv2.addWeighted(original_img[gt_mask], 0.6, colored_gt[gt_mask], 0.6, 0)
    overlay_img[intersection_mask] = cv2.addWeighted(original_img[intersection_mask], 0.6, colored_intersection[intersection_mask], 0.6, 0)

    cv2.imwrite(os.path.join(results_dir, f'overlay_{crt_id}.png'), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

def evaluate(model, test_loader, device, result_folder):

    model.eval()
    dice_sum = 0.0
    hd95_sum = 0.0
    pixel_correct = 0
    pixel_total = 0
    num_samples = 0

    os.makedirs(result_folder, exist_ok=True)

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs = data["image"].to(device)  # shape: [B, C, H, W]
            labels = data["mask"].to(device)     # shape: [B, 1, H, W]

            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)

            preds = (outputs > 0.6).float()

            # preds : [8, 1, 256, 256]
            dice = metrics.dice_coeff(preds, labels)
            hd95 = metrics.hd95_batch(preds, labels)

            dice_sum += dice * inputs.size(0)
            hd95_sum += hd95 * inputs.size(0)
            num_samples += inputs.size(0)

            pixel_correct += (preds == labels).float().sum().item()
            pixel_total += torch.numel(labels)

            for i in range(inputs.size(0)):
                label_np = labels[i].cpu().numpy()           # [1, H, W]
                pred_np = preds[i].cpu().numpy()             # [1, H, W]

                input_img = to_numpy(denormalization(inputs[i], mean=0.5, std=0.5))

                label_img = (label_np.squeeze(0) * 255).astype(np.uint8)
                pred_img = (pred_np.squeeze(0) * 255).astype(np.uint8)

                label_img_color = cv2.cvtColor(label_img, cv2.COLOR_GRAY2BGR)
                pred_img_color = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
                combined = np.hstack((input_img.squeeze(), label_img_color, pred_img_color))
                
                out_filename = os.path.join(result_folder, f"sample_{batch_idx * inputs.size(0) + i}.png")
                cv2.imwrite(out_filename, combined)
                
                # print(f"INPUT : {input_img.shape}\nSEGMENTED : {pred_img.shape}\nGT : {label_img.shape}")
                construct_overlay_image(original_img=input_img, segmented_img=pred_img, gt_img=label_img, crt_id=f"{batch_idx}_{i}", results_dir=result_folder)

    avg_dice = dice_sum / num_samples
    avg_hd95 = hd95_sum / num_samples
    accuracy = pixel_correct / pixel_total

    return avg_dice, avg_hd95, accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluation for Polyp Segmentation")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to YAML config file")
    parser.add_argument('-m', '--model', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('-o', '--output', type=str, default="results_resunetpp", help="Folder to store result images")
    parser.add_argument('--device', type=str, default="cuda:3" if torch.cuda.is_available() else "cpu", help="Computation device")
    args = parser.parse_args()

    hp = HParam(args.config)

    device = args.device

    if hp.RESNET_PLUS_PLUS:
        model = ResUnetPlusPlus(3).to(device)
        print("RESEUNET ++ !!!!!!!!!!!!!")
    else:
        model = ResUnet(3).to(device)
        print("RESUNET !!!!!!!!!")

    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.model, checkpoint.get("epoch", "N/A")))
    else:
        raise ValueError("Checkpoint not found: {}".format(args.model))

    test_transform = v2.Compose([
        GrayscaleNormalization(mean=0.5, std=0.5),
        ToTensor(),
    ])

    test_dataset = PolypsDataset(TEST_IMGS_DIR, TEST_LABELS_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, num_workers=2, shuffle=False)

    test_data_num = len(test_dataset)
    test_batch_num = int(np.ceil(test_data_num / hp.batch_size))

    avg_dice, avg_hd95, accuracy = evaluate(model, test_loader, device, args.output)

    print(f"Test Dice Coefficient: {avg_dice:.4f}")
    print(f"Test HD95: {avg_hd95:.4f}")
    print(f"Test Pixel Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()



'''
python eval.py  --config "configs/polyps.yaml"  -m "checkpoints/default/default_checkpoint_52000.pt"

---------
RESULTS
---------

Resunet : 50000.pt
Test Dice Coefficient: 0.7742
Test HD95: 29.6213
Test Pixel Accuracy: 0.9172

Resunet++ : 32000.pt
Test Dice Coefficient: 0.8064
Test HD95: 24.9032
Test Pixel Accuracy: 0.9267


'''