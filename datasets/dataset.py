"""
Dataset for C2Seg segmentation task.
Supports both folder structures:
    1. QaTa-COV19 style:
        root/
          images/   *.png | *.jpg
          masks/    mask_*.png
          texts.json

    2. MosMedData+/LViT style:
        root/
          img/       *.png | *.tif
          labelcol/  *.png
          *_text.xlsx
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
        augment: bool = False,
    ):
        self.root = Path(root_dir)
        self.size = image_size
        self.augment = augment

        # Find image and mask directories
        self.img_dir = self._find_dir(['images', 'img'])
        self.mask_dir = self._find_dir(['masks', 'labelcol'])
        
        # Load text mapping (supports both json and xlsx)
        self.texts = self._load_texts()

        # Collect samples
        self.samples = []
        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
                continue
            mask_path = self._find_mask(img_path.stem)
            if mask_path is None:
                continue
            text = self._get_text(img_path.name, mask_path.name)
            self.samples.append((img_path, mask_path, text))

    def _find_dir(self, candidates: list) -> Path:
        """Find first existing directory from candidates."""
        for name in candidates:
            path = self.root / name
            if path.exists():
                return path
        raise FileNotFoundError(f"None of {candidates} found in {self.root}")

    def _find_mask(self, stem: str) -> Path:
        """Find mask file with various naming conventions."""
        # Try: stem.png, mask_stem.png, stem.tif, etc.
        for prefix in ['', 'mask_']:
            for ext in ['.png', '.tif', '.tiff']:
                path = self.mask_dir / f"{prefix}{stem}{ext}"
                if path.exists():
                    return path
        return None

    def _load_texts(self) -> dict:
        """Load text annotations from json or xlsx."""
        # Try JSON first
        json_file = self.root / "texts.json"
        if json_file.exists():
            with open(json_file) as f:
                return json.load(f)
        
        # Try Excel (LViT format)
        xlsx_files = list(self.root.glob("*_text.xlsx")) + list(self.root.glob("*_text.xls"))
        if xlsx_files:
            import pandas as pd
            df = pd.read_excel(xlsx_files[0])
            texts = {}
            for _, row in df.iterrows():
                # Column names: "Image" and "Description" (LViT format)
                img_name = str(row.iloc[0])  # First column is image name
                desc = str(row.iloc[1]) if len(row) > 1 else ""  # Second column is description
                texts[img_name] = desc.strip()
            return texts
        
        # No text file found - return empty dict (will use empty strings)
        print(f"Warning: No text annotations found in {self.root}")
        return {}

    def _get_text(self, img_name: str, mask_name: str) -> str:
        """Get text for image, trying multiple key formats."""
        # Try different key formats
        for key in [img_name, mask_name, f"mask_{img_name}", img_name.replace('.png', ''), mask_name.replace('.png', '')]:
            if key in self.texts:
                return self.texts[key]
        return ""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, text = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image, mask = self._transform(image, mask)
        return image, mask, text

    def _transform(self, image, mask):
        # resize
        image = TF.resize(image, [self.size, self.size], interpolation=Image.Resampling.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size], interpolation=Image.Resampling.NEAREST)

        if self.augment:
            # random 90/180/270 rotation + random flip (like ref/c2seg)
            if random.random() > 0.5:
                image, mask = self._random_rot_flip(image, mask)
            # random small-angle rotation [-30, 30] (like ref/c2seg)
            if random.random() > 0.5:
                angle = random.uniform(-30, 30)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # to tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        mask = mask.unsqueeze(0)  # [1, H, W]
        return image, mask

    def _random_rot_flip(self, image, mask):
        """Random rotation by 0/90/180/270 + random flip (matches ref/c2seg)"""
        # Random 90-degree rotation
        k = random.randint(0, 3)
        for _ in range(k):
            image = TF.rotate(image, 90)
            mask = TF.rotate(mask, 90)

        # Random flip (horizontal or vertical)
        if random.random() > 0.5:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            else:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
        return image, mask
