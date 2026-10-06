"""
C2Seg: Consistency-enhanced Two-stage Segmentation
Paper: "Towards Text-Mask Consistency in Medical Image Segmentation" (ICLR 2026)

Architecture:
  Stage I  - Cluster-aware Contrastive Learning (CaCL)
  Stage II - Dual encoder + BCAM + K-Gate + U-Net decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer


# ---------------------------------------------------------------------------
# KAN layer (lightweight: learnable B-spline via linear + activation approx)
# ---------------------------------------------------------------------------
class KANLayer(nn.Module):
    """Simplified KAN: two linear layers with learnable nonlinearity via SiLU."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# Visual Encoder: 3 Conv blocks + 2 KAN layers, returns multi-scale features
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VisualEncoder(nn.Module):
    """
    Produces 4 multi-scale feature maps V1..V4 (progressively downsampled).
    Channels: 64 -> 128 -> 256 -> 512 (after KAN stages).
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # 3 CNN stages
        self.enc1 = ConvBlock(in_channels, 64)   # H x W
        self.enc2 = ConvBlock(64, 128)            # H/2 x W/2
        self.enc3 = ConvBlock(128, 256)           # H/4 x W/4
        self.pool = nn.MaxPool2d(2)

        # 2 KAN stages (operate channel-wise via 1x1-style reshape)
        self.kan4 = self._make_kan_stage(256, 512)  # H/8 x W/8
        self.kan5 = self._make_kan_stage(512, 512)  # H/16 x W/16

    def _make_kan_stage(self, in_ch, out_ch):
        return nn.Sequential(
            nn.MaxPool2d(2),
            # channel-mix via KAN (reshape to apply over channels)
            _SpatialKAN(in_ch, out_ch),
        )

    def forward(self, x):
        v1 = self.enc1(x)                  # [B, 64,  H,   W]
        v2 = self.enc2(self.pool(v1))      # [B, 128, H/2, W/2]
        v3 = self.enc3(self.pool(v2))      # [B, 256, H/4, W/4]
        v4 = self.kan4(v3)                 # [B, 512, H/8, W/8]
        v5 = self.kan5(v4)                 # [B, 512, H/16,W/16]

        # global image vector
        v_global = F.adaptive_avg_pool2d(v5, 1).flatten(1)  # [B, 512]
        return [v1, v2, v3, v4, v5], v_global


class _SpatialKAN(nn.Module):
    """Apply KAN across channels at each spatial location."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.kan = KANLayer(in_ch, out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)   # [B*H*W, C]
        x = self.kan(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x


# ---------------------------------------------------------------------------
# Language Encoder wrapper (frozen CLIP)
# ---------------------------------------------------------------------------
class LanguageEncoder(nn.Module):
    def __init__(self, clip_model: str = "openai/clip-vit-base-patch32", max_length: int = 77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        self.encoder   = CLIPTextModel.from_pretrained(clip_model)
        self.max_length = max_length
        # freeze
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, texts: list, device):
        tokens = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        ).to(device)
        out = self.encoder(**tokens)
        token_emb = out.last_hidden_state          # [B, N, d_t]
        sent_emb  = out.pooler_output              # [B, d_t]
        return token_emb, sent_emb                 # d_t = 512 for ViT-B/32


# ---------------------------------------------------------------------------
# Per-scale language projection
# ---------------------------------------------------------------------------
class LangProjection(nn.Module):
    def __init__(self, d_t: int, channels: list):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d_t, c) for c in channels])

    def forward(self, token_emb):
        # token_emb: [B, N, d_t]
        return [proj(token_emb) for proj in self.projs]  # list of [B, N, C_ω]


# ---------------------------------------------------------------------------
# BCAM: Bidirectional Complementary Attention Module
# ---------------------------------------------------------------------------
class BCAM(nn.Module):
    def __init__(self, vis_ch: int, lang_ch: int):
        super().__init__()
        d = vis_ch
        self.v_key   = KANLayer(vis_ch,  d)
        self.v_value = KANLayer(vis_ch,  d)
        self.l_key   = KANLayer(lang_ch, d)
        self.l_value = KANLayer(lang_ch, d)
        self.scale   = d ** -0.5

    def forward(self, V, L_w):
        """
        V:   [B, C, H, W]  visual features
        L_w: [B, N, C]     language tokens (projected to same C)
        Returns F_v, F_l both [B, C, H, W]
        """
        B, C, H, W = V.shape
        P = H * W
        N = L_w.shape[1]

        # flatten spatial
        V_flat = V.flatten(2).permute(0, 2, 1)   # [B, P, C]

        # KAN keys & values
        Vk = self.v_key(V_flat)                   # [B, P, C]
        Vv = self.v_value(V_flat)                 # [B, P, C]
        Lk = self.l_key(L_w)                      # [B, N, C]
        Lv = self.l_value(L_w)                    # [B, N, C]

        # attention score A: [B, P, N]
        A = torch.bmm(Vk, Lk.transpose(1, 2)) * self.scale

        # --- vision-dominant path ---
        # each pixel gathers from language tokens
        F_v = torch.bmm(F.softmax(A, dim=2), Lv)  # [B, P, C]

        # --- language-dominant path (Eq. 6) ---
        # for each token n: softmax over spatial P, then element-wise multiply with V_value
        # F_l = (1/N) * sum_n [ Softmax_P(A^T[n,:]) ⊗ V_value ]
        A_T = A.transpose(1, 2)              # [B, N, P]
        w   = F.softmax(A_T, dim=2)         # [B, N, P], each token's spatial attention
        # w[:,n,:] shape [B,P], Vv shape [B,P,C] → element-wise then sum over N
        # einsum: b n p, b p c -> b p c  (weighted sum over N)
        F_l = torch.einsum('bnp,bpc->bpc', w, Vv) / N  # [B, P, C]

        # reshape back to spatial
        F_v = F_v.permute(0, 2, 1).reshape(B, C, H, W)
        F_l = F_l.permute(0, 2, 1).reshape(B, C, H, W)
        return F_v, F_l


# ---------------------------------------------------------------------------
# K-Gate: KAN-based Attention Gating
# ---------------------------------------------------------------------------
class KGate(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        # g = tanh(KAN2(ReLU(KAN1(F)))) per Eq. 7
        self.gate_v = nn.Sequential(KANLayer(ch, ch), nn.ReLU(), KANLayer(ch, ch), nn.Tanh())
        self.gate_l = nn.Sequential(KANLayer(ch, ch), nn.ReLU(), KANLayer(ch, ch), nn.Tanh())
        self.fuse   = nn.Conv2d(ch * 2, ch, 1)

    def forward(self, F_v, F_l):
        B, C, H, W = F_v.shape
        P = H * W

        def gate(module, feat):
            # reshape to [B*P, C], apply gate, reshape back
            f = feat.flatten(2).permute(0, 2, 1).reshape(-1, C)
            g = module(f).reshape(B, P, C).permute(0, 2, 1).reshape(B, C, H, W)
            return feat * g  # element-wise reweighting

        Fv_g = gate(self.gate_v, F_v)
        Fl_g = gate(self.gate_l, F_l)
        return self.fuse(torch.cat([Fv_g, Fl_g], dim=1))


# ---------------------------------------------------------------------------
# Decoder block (U-Net style)
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Full C2Seg model
# ---------------------------------------------------------------------------
class C2Seg(nn.Module):
    # visual encoder channel sizes per scale
    VIS_CHANNELS = [64, 128, 256, 512, 512]

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        clip_model: str  = "openai/clip-vit-base-patch32",
        d_t: int         = 512,
    ):
        super().__init__()
        self.visual_enc = VisualEncoder(in_channels)
        self.lang_enc   = LanguageEncoder(clip_model)
        self.lang_proj  = LangProjection(d_t, self.VIS_CHANNELS)

        # BCAM at each of the 5 scales
        self.bcam = nn.ModuleList([
            BCAM(c, c) for c in self.VIS_CHANNELS
        ])
        self.kgate = nn.ModuleList([
            KGate(c) for c in self.VIS_CHANNELS
        ])

        # Decoder: upsample from v5 -> v4 -> v3 -> v2 -> v1 -> output
        self.dec4 = DecoderBlock(512, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec2 = DecoderBlock(128, 128, 64)
        self.dec1 = DecoderBlock(64,  64,  32)
        self.head = nn.Conv2d(32, num_classes, 1)

        # projection from sentence embedding to global visual for CaCL
        self.img_proj  = nn.Linear(512, 256)
        self.text_proj = nn.Linear(d_t,  256)

    def encode(self, images, texts):
        """Extract visual & language features."""
        feats, v_global = self.visual_enc(images)
        token_emb, sent_emb = self.lang_enc(texts, images.device)
        lang_per_scale = self.lang_proj(token_emb)  # list of [B, N, C_ω]
        return feats, v_global, token_emb, sent_emb, lang_per_scale

    def forward(self, images, texts):
        feats, v_global, _, _, lang_per_scale = self.encode(images, texts)

        # fuse at each scale with BCAM + K-Gate
        fused = []
        for i, (V, L_w) in enumerate(zip(feats, lang_per_scale)):
            F_v, F_l = self.bcam[i](V, L_w)
            fused.append(self.kgate[i](F_v, F_l))

        # U-Net decoder with skip connections from fused features
        x = fused[4]                              # [B, 512, H/16, W/16]
        x = self.dec4(x, fused[3])               # [B, 256, H/8,  W/8]
        x = self.dec3(x, fused[2])               # [B, 128, H/4,  W/4]
        x = self.dec2(x, fused[1])               # [B, 64,  H/2,  W/2]
        x = self.dec1(x, fused[0])               # [B, 32,  H,    W]
        return self.head(x)                       # [B, num_classes, H, W]

    def get_contrastive_embeddings(self, images, texts):
        """For CaCL pretraining: lightweight forward — only visual global vec + sent_emb."""
        # only need v_global, skip BCAM/lang_proj overhead
        feats, v_global = self.visual_enc(images)
        with torch.no_grad():
            _, sent_emb = self.lang_enc(texts, images.device)
        v     = F.normalize(self.img_proj(v_global), dim=-1)
        l     = F.normalize(self.text_proj(sent_emb), dim=-1)
        l_raw = F.normalize(sent_emb.detach().float(), dim=-1)
        return v, l, l_raw
