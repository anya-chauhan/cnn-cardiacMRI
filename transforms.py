from torchvision import transforms
import random
from torchvision.transforms import functional as TF
import torch

def get_train_transforms(mean, std):
    return transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply([lambda img: TF.adjust_contrast(img, contrast_factor=random.uniform(0.8, 1.2))], p=0.5),
        transforms.RandomApply([lambda img: TF.adjust_brightness(img, brightness_factor=random.uniform(0.8, 1.2))], p=0.5),
        transforms.RandomApply([lambda img: TF.gaussian_blur(img, kernel_size=(5, 9), sigma=(0.1, 5))], p=0.3),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

def get_val_test_transforms(mean, std):
    return transforms.Compose([
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

def calculate_stats(dataset):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for image, _ in dataset:
        mean += image.mean(dim=[1, 2])
        std += image.std(dim=[1, 2])
    mean /= len(dataset)
    std /= len(dataset)
    return mean, std
