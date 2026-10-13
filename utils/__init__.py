import random
import numpy as np
import torch
import yaml
from pathlib import Path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, strict=False):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=strict)
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', 0), ckpt.get('best_metric', 0.0)
