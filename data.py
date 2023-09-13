import torch
import torchvision.transforms as transforms
from PIL import ImageFilter, Image
from torch.utils.data import Dataset
import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import logging
logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

logger = logging.getLogger("lightning.pytorch.core")

def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    logger.info('Making data transforms')

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform_list = []
    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class FaceDataset(Dataset):
    def __init__(self, cfg, train):
        self.transform = make_transforms(
            crop_size=cfg.data.crop_size,
            crop_scale=cfg.data.crop_scale,
            gaussian_blur=cfg.data.use_gaussian_blur,
            horizontal_flip=cfg.data.use_horizontal_flip,
            color_distortion=cfg.data.use_color_distortion,
            color_jitter=cfg.data.color_jitter_strength)

        self.files = pd.read_csv(cfg.data.csv_dataset)
        train_ds, test_ds = train_test_split(self.files, test_size=0.01)
        self.files = train_ds if train else test_ds
        logger.info(f'No. of images loaded: {len(self.files)}')
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files.iloc[idx].file)
        img = self.transform(img)
        return img, 0 # Dummy Label