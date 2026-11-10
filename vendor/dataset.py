#!/usr/bin/env python3
import os
import glob
import random
from typing import List
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def list_normal_images(root: str, category: str, split: str='train') -> List[str]:
    good_dir = os.path.join(root, category, split, 'good')
    if not os.path.isdir(good_dir):
        raise FileNotFoundError(f"Could not find '{good_dir}'. Check --data_root and category name. (splicing_connectors is the folder name for 'connectors'.)")
    files = [p for p in glob.glob(os.path.join(good_dir, '*')) if p.lower().endswith(IMG_EXTS)]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No images found in '{good_dir}'.")
    return files

class MVTecLOCONormal(Dataset):

    def _init_(self, root: str, category: str, resolution: int=512, fraction: float=1.0, augment: bool=True, seed: int=0):
        self.files = list_normal_images(root, category, split='train')
        if 0 < fraction < 1.0:
            rng = random.Random(seed)
            k = max(1, int(round(len(self.files) * fraction)))
            self.files = rng.sample(self.files, k)
        self.resolution = resolution
        if augment:
            self.transform = T.Compose([T.Resize(int(resolution * 1.15)), T.RandomCrop(resolution), T.RandomRotation(15), T.RandomHorizontalFlip(0.5), T.ToTensor(), T.Normalize([0.5], [0.5])])
        else:
            self.transform = T.Compose([T.Resize(resolution), T.CenterCrop(resolution), T.ToTensor(), T.Normalize([0.5], [0.5])])

    def _len_(self):
        return len(self.files)

    def _getitem_(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return {'pixel_values': self.transform(img)}

def reference_images(root: str, category: str, resolution: int=256, max_images: int=200) -> torch.Tensor:
    files = list_normal_images(root, category, split='train')[:max_images]
    tf = T.Compose([T.Resize(resolution), T.CenterCrop(resolution), T.ToTensor()])
    imgs = [tf(Image.open(f).convert('RGB')) for f in files]
    return torch.stack(imgs, dim=0)
