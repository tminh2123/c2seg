"""
Segmentation metrics: Dice, mIoU, HD95, ASSD
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """pred/target: [B, 1, H, W] float tensors (probs after sigmoid)."""
    # Model output already has sigmoid, so just threshold directly
    pred = (pred > threshold).float()
    flat_p = pred.flatten(1)
    flat_t = target.flatten(1)
    intersection = (flat_p * flat_t).sum(1)
    score = (2 * intersection + 1) / (flat_p.sum(1) + flat_t.sum(1) + 1)
    return score.mean().item() * 100


def miou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """pred/target: [B, 1, H, W] float tensors (probs after sigmoid)."""
    pred = (pred > threshold).float()
    flat_p = pred.flatten(1)
    flat_t = target.flatten(1)
    intersection = (flat_p * flat_t).sum(1)
    union = flat_p.sum(1) + flat_t.sum(1) - intersection
    iou = (intersection + 1) / (union + 1)
    return iou.mean().item() * 100


def _surface_distances(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """Compute surface distances between two binary masks."""
    pred_border = pred_bin ^ _erode(pred_bin)
    gt_border   = gt_bin   ^ _erode(gt_bin)

    dt_pred = distance_transform_edt(~pred_bin)
    dt_gt   = distance_transform_edt(~gt_bin)

    sds_pred2gt = dt_gt[pred_border]
    sds_gt2pred = dt_pred[gt_border]
    return sds_pred2gt, sds_gt2pred


def _erode(arr: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    return binary_erosion(arr)


def hd95(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """pred/target: [B, 1, H, W] float tensors (probs after sigmoid)."""
    pred_bin = (pred > threshold).cpu().numpy().astype(bool)
    gt_bin   = target.cpu().numpy().astype(bool)

    scores = []
    for p, g in zip(pred_bin, gt_bin):
        p, g = p[0], g[0]
        if p.sum() == 0 or g.sum() == 0:
            continue
        d1, d2 = _surface_distances(p, g)
        if len(d1) == 0 or len(d2) == 0:
            continue
        all_d = np.concatenate([d1, d2])
        scores.append(np.percentile(all_d, 95))
    return float(np.mean(scores)) if scores else 0.0


def assd(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """pred/target: [B, 1, H, W] float tensors (probs after sigmoid)."""
    pred_bin = (pred > threshold).cpu().numpy().astype(bool)
    gt_bin   = target.cpu().numpy().astype(bool)

    scores = []
    for p, g in zip(pred_bin, gt_bin):
        p, g = p[0], g[0]
        if p.sum() == 0 or g.sum() == 0:
            continue
        d1, d2 = _surface_distances(p, g)
        if len(d1) == 0 or len(d2) == 0:
            continue
        scores.append((d1.mean() + d2.mean()) / 2)
    return float(np.mean(scores)) if scores else 0.0
