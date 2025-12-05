import random
import torch
import torchvision.transforms.functional as F


class TrainTransform:
    """
    Resize → RandomCrop → RandomFlip → ToTensor
    """
    def __init__(self, size=256):
        self.size = size

    def __call__(self, img_in, img_gt):
        # Random horizontal flip
        if random.random() < 0.5:
            img_in  = F.hflip(img_in)
            img_gt = F.hflip(img_gt)

        # Resize
        img_in  = F.resize(img_in, (self.size, self.size))
        img_gt = F.resize(img_gt, (self.size, self.size))

        # ToTensor
        img_in  = F.to_tensor(img_in)
        img_gt = F.to_tensor(img_gt)

        return img_in, img_gt


class TestTransform:
    """Just resize + ToTensor"""
    def __init__(self, size=256):
        self.size = size

    def __call__(self, img_in, img_gt):
        img_in  = F.resize(img_in, (self.size, self.size))
        img_gt = F.resize(img_gt, (self.size, self.size))

        img_in  = F.to_tensor(img_in)
        img_gt = F.to_tensor(img_gt)
        return img_in, img_gt
