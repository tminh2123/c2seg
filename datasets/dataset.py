"""
Dataset for C2Seg segmentation task.
Expects folder structure:
    root/
      images/  *.png | *.jpg
      masks/   *.png          (binary, same stem as image)
      texts.json              {"image_filename": "text description", ...}
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random


class C2SegDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        image_size: int = 224,
        augment: bool   = False,
    ):
        self.root   = Path(root_dir)
        self.size   = image_size
        self.augment = augment

        # load text mapping
        text_file = self.root / "texts.json"
        with open(text_file) as f:
            self.texts: dict = json.load(f)

        # collect samples
        img_dir  = self.root / "images"
        mask_dir = self.root / "masks"
        self.samples = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            mask_path = mask_dir / (img_path.stem + '.png')
            if not mask_path.exists():
                continue
            text = self.texts.get(img_path.name, "")
            self.samples.append((img_path, mask_path, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, text = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')

        image, mask = self._transform(image, mask)
        return image, mask, text

    def _transform(self, image, mask):
        # resize
        image = TF.resize(image, [self.size, self.size], interpolation=Image.Resampling.BILINEAR)
        mask  = TF.resize(mask,  [self.size, self.size], interpolation=Image.Resampling.NEAREST)

        if self.augment:
            # random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)
            # random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)
            # random rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask  = TF.rotate(mask,  angle)

        # to tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        mask = mask.unsqueeze(0)   # [1, H, W]
        return image, mask
