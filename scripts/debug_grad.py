#!/usr/bin/env python3
"""Debug soft labels on real batch from dataset."""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')

from torch.utils.data import DataLoader
from datasets.c2seg_dataset import C2SegDataset
from models.c2seg import C2Seg


def analyze_batch(data_root: str, batch_size: int = 256):
    print("=" * 60)
    print(f"REAL BATCH SOFT LABEL ANALYSIS (B={batch_size})")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = C2SegDataset(
        root_dir=f"{data_root}/train",
        image_size=224,
        augment=False
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Load model (just need language encoder)
    model = C2Seg(
        in_channels=3,
        num_classes=1,
        clip_model="openai/clip-vit-base-patch32",
        d_t=512
    ).to(device)
    
    # Get one batch
    images, masks, texts = next(iter(loader))
    images = images.to(device)
    
    print(f"\nBatch texts (first 10):")
    for i, t in enumerate(texts[:10]):
        print(f"  [{i}] {t[:60]}...")
    
    # Get embeddings
    with torch.no_grad():
        _, sent_emb = model.lang_enc(texts, device)
        l_raw = F.normalize(sent_emb, dim=-1)
    
    # Compute similarity matrix M
    M = torch.mm(l_raw, l_raw.t()).cpu()
    
    print(f"\n--- Text Similarity Matrix M [{batch_size}x{batch_size}] ---")
    print(f"Diagonal (self-sim): all 1.0")
    print(f"Off-diagonal mean: {(M - torch.diag(M.diag())).mean():.4f}")
    print(f"Off-diagonal std:  {(M - torch.diag(M.diag())).std():.4f}")
    print(f"Off-diagonal min:  {(M - torch.eye(batch_size) * 999).min():.4f}")
    print(f"Off-diagonal max:  {(M - torch.eye(batch_size) * 999).max():.4f}")
    
    # Debias
    mu = M.mean(dim=1, keepdim=True)
    M_deb = (M - mu).clamp(min=0)
    
    print(f"\n--- Debiased M ---")
    print(f"Non-zero count: {(M_deb > 0).sum().item()}/{batch_size*batch_size}")
    print(f"Non-zero ratio: {(M_deb > 0).float().mean().item()*100:.1f}%")
    print(f"Mean of non-zeros: {M_deb[M_deb > 0].mean().item():.4f}" if (M_deb > 0).any() else "N/A")
    
    # Soft labels Y_hat
    tau = 0.07
    Y_hat = F.softmax(M_deb / tau, dim=1)
    
    print(f"\n--- Soft Labels Y_hat (τ={tau}) ---")
    print(f"Diagonal mean: {Y_hat.diag().mean():.4f}")
    print(f"Off-diagonal mean: {(Y_hat - torch.diag(Y_hat.diag())).mean():.6f}")
    print(f"Max per row (should be high if good): {Y_hat.max(dim=1)[0].mean():.4f}")
    print(f"Entropy per row (low=peaked, high=uniform): {-(Y_hat * Y_hat.log().clamp(min=-100)).sum(dim=1).mean():.2f}")
    print(f"  (Uniform entropy = {torch.log(torch.tensor(batch_size)).item():.2f})")
    
    # Final Y
    eps = 0.8
    I = torch.eye(batch_size)
    Y = eps * Y_hat + (1 - eps) * I
    
    print(f"\n--- Final Soft Labels Y (ρ={eps}) ---")
    print(f"Diagonal mean: {Y.diag().mean():.4f}")
    print(f"Off-diagonal mean: {(Y - torch.diag(Y.diag())).mean():.6f}")
    
    # Check unique texts
    unique_texts = len(set(texts))
    print(f"\n--- Text Diversity ---")
    print(f"Unique texts in batch: {unique_texts}/{batch_size}")
    print(f"Diversity: {unique_texts/batch_size*100:.1f}%")
    
    # Count exact duplicates
    from collections import Counter
    text_counts = Counter(texts)
    print(f"Most common text appears: {text_counts.most_common(1)[0][1]} times")
    
    # Visualize small portion of M
    print(f"\n--- Sample of M (top-left 8x8) ---")
    print("     ", end="")
    for j in range(8):
        print(f"  [{j}]  ", end="")
    print()
    for i in range(8):
        print(f"[{i}] ", end="")
        for j in range(8):
            print(f" {M[i,j]:.3f} ", end="")
        print()


if __name__ == '__main__':
    data_root = sys.argv[1] if len(sys.argv) > 1 else "../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    analyze_batch(data_root, batch_size)
