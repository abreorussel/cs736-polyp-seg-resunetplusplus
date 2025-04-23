import warnings

warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from utils.hparams import HParam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import dataloader
from utils import metrics
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
from utils.logger import MyWriter
import torch
import argparse
import os
from torchvision.transforms import v2

from dataset.polyps_dataloader import *
from dataset.dataloader import *


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ )))
DATA_DIR = os.path.join(ROOT_DIR, 'new_data/Kvasir-SEG')

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_IMGS_DIR = os.path.join(TRAIN_DIR, 'images')
VAL_IMGS_DIR = os.path.join(VAL_DIR, 'images')
TEST_IMGS_DIR = os.path.join(TEST_DIR, 'images')

TRAIN_LABELS_DIR = os.path.join(TRAIN_DIR, 'masks')
VAL_LABELS_DIR = os.path.join(VAL_DIR, 'masks')
TEST_LABELS_DIR = os.path.join(TEST_DIR, 'masks')


def main(hp, num_epochs, resume, name, device='cpu'):
    checkpoint_dir = "{}/{}".format(hp.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs("{}/{}".format(hp.log, name), exist_ok=True)
    writer = MyWriter("{}/{}".format(hp.log, name))

    if hp.RESNET_PLUS_PLUS:
        model = ResUnetPlusPlus(3).to(device)
        print("Loading RESUNET++ model !!!!!!!!!!!\n")
    else:
        model = ResUnet(3).to(device)

    criterion = metrics.BCEDiceLoss()
    # criterion = metrics.BCEDiceLossWithLogits()

    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

    # decay LR
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6, verbose=True)

    # starting params
    best_loss = 999
    start_epoch = 0
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)

            start_epoch = checkpoint["epoch"]

            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_transform = transforms.Compose([
        GrayscaleNormalization(mean=0.5, std=0.5),
        ToTensor(),
    ])
    val_transform = transforms.Compose([
        GrayscaleNormalization(mean=0.5, std=0.5),
        ToTensor(),
    ])

    train_dataset = PolypsDataset(TRAIN_IMGS_DIR, TRAIN_LABELS_DIR, transform=train_transform)
    val_dataset = PolypsDataset(VAL_IMGS_DIR, VAL_LABELS_DIR, transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=hp.batch_size, num_workers=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hp.batch_size, num_workers=2, shuffle=False)

    step = 0
    for epoch in range(start_epoch, num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        train_acc = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()

        loader = tqdm(train_dataloader, desc="training")
        for idx, data in enumerate(loader):

            inputs = data["image"].to(device)
            labels = data["mask"].to(device)

            optimizer.zero_grad()

            # forward
            # prob_map = model(inputs) 
            # outputs = (prob_map > 0.3).float()
            outputs = model(inputs)

            # outputs = torch.nn.functional.sigmoid(outputs)

            # print(f"MODEL O/P shape : {outputs.shape}, labels : {labels.shape}")
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            train_loss.update(loss.data.item(), outputs.size(0))

            if step % hp.validation_interval == 0:
                valid_metrics = validation(
                    val_dataloader, model, criterion, writer, step, device
                )
                lr_scheduler.step(valid_metrics["valid_loss"])
                save_path = os.path.join(
                    checkpoint_dir, "%s_checkpoint_%04d.pt" % (name, step)
                )
                
                best_loss = min(valid_metrics["valid_loss"], best_loss)
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "arch": "ResUnet",
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                print("Saved checkpoint to: %s" % save_path)

            step += 1


def validation(valid_loader, model, criterion, logger, step, device='cpu'):

    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    model.eval()

    for idx, data in enumerate(tqdm(valid_loader, desc="validation")):

        inputs = data["image"].to(device)
        labels = data["mask"].to(device)

        # forward
        # prob_map = model(inputs)
        # outputs = (prob_map > 0.3).float()
        outputs = model(inputs)
        # outputs = torch.nn.functional.sigmoid(outputs)

        loss = criterion(outputs, labels)

        valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        valid_loss.update(loss.data.item(), outputs.size(0))

    print("Validation Loss: {:.4f} Acc: {:.4f}".format(valid_loss.avg, valid_acc.avg))
    model.train()
    return {"valid_loss": valid_loss.avg, "valid_acc": valid_acc.avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )
    parser.add_argument(
        "--epochs",
        default=75,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--name", default="default", type=str, help="Experiment name")
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu", help="Device to use for training")

    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = "".join(f.readlines())

    main(hp, num_epochs=args.epochs, resume=args.resume, name=args.name, device=args.device)



'''
python train.py --name "default" --config "configs/polyps.yaml" --epochs 100 
'''


'''
    IMAGE SIZE : [3, 256, 256]
    MASK SIZE  : [1, 256, 256]

'''
