#!/usr/bin/env python3
"""Debug CaCL loss to check if it's working correctly."""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')

from models.losses import CaCLLoss


def test_cacl_loss():
    print("=" * 60)
    print("DEBUG CaCL LOSS")
    print("=" * 60)
    
    B = 8  # small batch for testing
    d = 256
    
    # Create random normalized embeddings
    torch.manual_seed(42)
    v = F.normalize(torch.randn(B, d), dim=-1)
    l = F.normalize(torch.randn(B, d), dim=-1)
    l_raw = F.normalize(torch.randn(B, 512), dim=-1)
    
    print(f"\nInput shapes:")
    print(f"  v (image emb):    {v.shape}")
    print(f"  l (text emb):     {l.shape}")
    print(f"  l_raw (raw text): {l_raw.shape}")
    
    # Test loss
    cacl = CaCLLoss(temperature=0.07, eps=0.8)
    loss = cacl(v, l, l_raw)
    print(f"\nRandom embeddings loss: {loss.item():.4f}")
    print(f"  Expected: ~{torch.log(torch.tensor(B)).item():.2f} (log(B)={B})")
    
    # Test with aligned embeddings (v ≈ l)
    v_aligned = l.clone()
    loss_aligned = cacl(v_aligned, l, l_raw)
    print(f"\nAligned embeddings loss: {loss_aligned.item():.4f}")
    print(f"  Expected: << random loss (should be near 0)")
    
    # Check gradients
    v.requires_grad_(True)
    l.requires_grad_(True)
    loss = cacl(v, l, l_raw)
    loss.backward()
    
    print(f"\nGradient check:")
    print(f"  v.grad exists: {v.grad is not None}")
    print(f"  l.grad exists: {l.grad is not None}")
    print(f"  v.grad norm: {v.grad.norm().item():.6f}")
    print(f"  l.grad norm: {l.grad.norm().item():.6f}")
    
    # Test soft label matrix
    print("\n" + "=" * 60)
    print("SOFT LABEL ANALYSIS")
    print("=" * 60)
    
    M = torch.mm(l_raw, l_raw.t())
    print(f"\nText similarity matrix M:")
    print(f"  Shape: {M.shape}")
    print(f"  Diagonal (self-sim): {M.diag()}")
    print(f"  Off-diagonal mean: {(M - torch.diag(M.diag())).mean():.4f}")
    
    mu = M.mean(dim=1, keepdim=True)
    M_deb = (M - mu).clamp(min=0)
    print(f"\nDebiased M:")
    print(f"  After debias, non-zero count: {(M_deb > 0).sum().item()}/{B*B}")
    
    Y_hat = F.softmax(M_deb / 0.07, dim=1)
    I = torch.eye(B)
    Y = 0.8 * Y_hat + 0.2 * I
    
    print(f"\nSoft labels Y:")
    print(f"  Diagonal mean: {Y.diag().mean():.4f}")
    print(f"  Off-diagonal mean: {(Y - torch.diag(Y.diag())).mean():.4f}")
    print(f"  Row sums (should be ~1): {Y.sum(dim=1)}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if loss_aligned.item() < loss.item() * 0.5:
        print("✅ Loss decreases when embeddings are aligned - loss function works!")
    else:
        print("❌ Loss doesn't decrease properly - BUG in loss function!")
    
    if v.grad is not None and v.grad.norm() > 0:
        print("✅ Gradients flow correctly")
    else:
        print("❌ Gradient issue!")


def test_model_forward():
    """Test if model produces correct embeddings."""
    print("\n" + "=" * 60)
    print("DEBUG MODEL FORWARD")
    print("=" * 60)
    
    from models.c2seg import C2Seg
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = C2Seg(
        in_channels=3,
        num_classes=1,
        clip_model="openai/clip-vit-base-patch32",
        d_t=512
    ).to(device)
    
    # Test forward
    B = 4
    images = torch.randn(B, 3, 224, 224).to(device)
    texts = ["Bilateral pulmonary infection"] * B
    
    v, l, l_raw = model.get_contrastive_embeddings(images, texts)
    
    print(f"\nModel output shapes:")
    print(f"  v: {v.shape}")
    print(f"  l: {l.shape}")
    print(f"  l_raw: {l_raw.shape}")
    
    print(f"\nNormalization check (should be ~1.0):")
    print(f"  v norms: {v.norm(dim=-1)}")
    print(f"  l norms: {l.norm(dim=-1)}")
    print(f"  l_raw norms: {l_raw.norm(dim=-1)}")
    
    # Check if same text gives same l_raw
    print(f"\nl_raw similarity for same text (should be 1.0):")
    print(f"  {torch.mm(l_raw, l_raw.t()).diag()}")
    
    # Check trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for n, p in model.named_parameters() if 'lang_enc' in n)
    print(f"\nParameters:")
    print(f"  Total trainable: {trainable:,}")
    print(f"  Frozen (lang_enc): {frozen:,}")


if __name__ == '__main__':
    test_cacl_loss()
    test_model_forward()
