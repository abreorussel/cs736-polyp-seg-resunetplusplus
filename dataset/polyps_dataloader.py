import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
import cv2
import torch
import random
import glob

from skimage import io
from utils.polyps_augmentation import *


class PolypsDataset(Dataset):
    def __init__(self, imgs_dir, labels_dir, transform=None):
        self.imgs = sorted(glob.glob(os.path.join(imgs_dir, '*.jpg')))
        self.masks = sorted(glob.glob(os.path.join(labels_dir, '*.jpg')))
        self.transform = transform

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        # print(self.imgs[idx])
        image = cv2.imread(self.imgs[idx], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        # image = io.imread(self.imgs[idx])
        # mask = io.imread(self.masks[idx])
        # sample = {
        #     "img" : image,
        #     "mask" : mask[:, :, np.newaxis]
        # }
        sample = {
            "img" : image,
            "mask" : mask
        }
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        # print(f"POLYPS DATASET : IMAGE : {sample['img'].shape}  ,   MASK : {sample['mask'].shape}")
        return sample
        

class GrayscaleNormalization:
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
        
    def __call__(self, data):
        image, mask = data['img'], data['mask']
        image = (image - self.mean) / self.std
        sample = {
            "img" : image,
            "mask" : mask
        }
        return sample
    

class ToTensor:
    def __call__(self, data):
        img, mask = data['img'], data['mask']

        # print(f"image : {img.shape} label : {mask.shape}")
        
        img = img.transpose(2, 0, 1).astype(np.float32)  # torch 의 경우 (C, H, W)
        mask = mask.transpose(2, 0, 1).astype(np.float32)
        
        sample = {
            'img': torch.from_numpy(img),
            'mask': torch.from_numpy(mask),
        }
        return sample


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, mask = data["img"], data["mask"]
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size)
        sample = {
            "img" : image,
            "mask" : mask
        }
        # print("RESIZE")
        # print(sample['mask'].shape)
        return sample

class RandomCrop:
    def __init__(self, crop_size, size):
        self.crop_size = crop_size
        self.size = size

    def __call__(self, data):
        image, mask = data["img"], data["mask"]
        image, mask =  random_crop(image, mask, self.crop_size, self.size)
        sample = {
            "img" : image,
            "mask" : mask
        }
        return sample

class HorizontalFlip:
    def __init__(self, p=0.5, size=(256, 256)):
        self.p = p
        self.size = size

    def __call__(self, data):
        image, mask = data["img"], data["mask"]
        if random.random() < self.p:
            image, mask =  horizontal_flip(image, mask, self.size)
        sample = {
            "img" : image,
            "mask" : mask
        }
        return sample
    
class VerticalFlip:
    def __init__(self, p=0.5, size=(256, 256)):
        self.p = p
        self.size = size

    def __call__(self, data):
        image, mask = data["img"], data["mask"]
        if random.random() < self.p:
            image, mask =  vertical_flip(image, mask, self.size)
        sample = {
            "img" : image,
            "mask" : mask
        }
        return sample
    
class RandomRotation:
    def __init__(self, p=0.5, size=(256, 256)):
        self.p = p
        self.size = size

    def __call__(self, data):
        image, mask = data["img"], data["mask"]
        if random.random() < self.p:
            image, mask =  random_rotation(image, mask, self.size)
        sample = {
            "img" : image,
            "mask" : mask
        }
        return sample

class BrightnessAugment:
    def __init__(self, factor_range=(0.3, 0.9), size=(256, 256)):
        self.factor_range = factor_range
        self.size = size

    def __call__(self, data):
        image, mask = data["img"], data["mask"]
        factor = random.uniform(*self.factor_range)
        image, mask =  brightness_augment(image, mask, factor)
        sample = {
            "img" : image,
            "mask" : mask
        }
        return sample


def denormalization(data, mean, std):
    return (data * std) + mean

    
     