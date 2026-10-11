"""
Losses for C2Seg:
  - CaCLLoss: Cluster-aware Contrastive Learning loss (Stage I)
  - BCEDiceLoss: segmentation loss (Stage II)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CaCLLoss(nn.Module):
    """
    Cluster-aware Contrastive Learning loss.
    Converts intra-batch text similarity into soft labels to suppress false negatives.

    Args:
        temperature: tau for cross-modal logits (default 0.07)
        eps: soft-label mixing weight (default 0.8)
        lang_encoder: frozen language encoder to compute text embeddings
    """

    def __init__(self, temperature: float = 0.07, eps: float = 0.8):
        super().__init__()
        self.tau = temperature
        self.eps = eps

    def forward(self, v: torch.Tensor, l: torch.Tensor, l_raw: torch.Tensor):
        """
        v:     [B, d]  L2-normalised image embeddings
        l:     [B, d]  L2-normalised sentence embeddings (projected)
        l_raw: [B, d]  L2-normalised sentence embeddings from frozen encoder
                       (used to build soft-label matrix M)
        """
        B = v.shape[0]

        # --- build soft labels from text-text similarity ---
        M = torch.mm(l_raw, l_raw.t())          # [B, B] cosine similarity

        # row-mean debias + clip negatives
        mu = M.mean(dim=1, keepdim=True)
        M_deb = (M - mu).clamp(min=0)           # [B, B]

        # soft targets via temperature softmax
        Y_hat = F.softmax(M_deb / self.tau, dim=1)   # [B, B]

        # mix with hard self-positive
        I = torch.eye(B, device=v.device)
        Y = self.eps * Y_hat + (1 - self.eps) * I    # [B, B]

        # --- cross-modal logits ---
        S = torch.mm(v, l.t()) / self.tau            # [B, B]

        P_v2l = F.log_softmax(S,   dim=1)            # image -> text
        P_l2v = F.log_softmax(S.t(), dim=1)          # text  -> image

        loss = -(Y * P_v2l).sum(dim=1).mean() \
               -(Y.t() * P_l2v).sum(dim=1).mean()
        return loss * 0.5


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss for binary segmentation.
    
    Expects input to be probabilities (after sigmoid), NOT logits.
    """

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_w = bce_weight
        # Use BCELoss since model output already has sigmoid
        self.bce = nn.BCELoss()

    def forward(self, probs: torch.Tensor, targets: torch.Tensor):
        """
        probs:   [B, 1, H, W]  probability output (after sigmoid in model)
        targets: [B, 1, H, W]  binary mask float
        """
        # Clamp to avoid log(0)
        probs = probs.clamp(1e-7, 1 - 1e-7)
        
        bce_loss = self.bce(probs, targets)

        flat_p = probs.flatten(1)
        flat_t = targets.flatten(1)
        intersection = (flat_p * flat_t).sum(1)
        dice_loss = 1 - (2 * intersection + 1) / (flat_p.sum(1) + flat_t.sum(1) + 1)
        dice_loss = dice_loss.mean()

        return self.bce_w * bce_loss + (1 - self.bce_w) * dice_loss
