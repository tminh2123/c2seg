"""
C2Seg: Consistency-enhanced Two-stage Segmentation
Paper: "Towards Text-Mask Consistency in Medical Image Segmentation" (ICLR 2026)

Architecture follows ref/c2seg implementation:
  - Visual Encoder: 5 DownBlocks (Conv only, no KAN)
  - BCAM: Bidirectional Cross Attention with Sinusoidal2D + KCGate
  - Decoder: UpBlockAttention with PixLevelModule
  - KAN only used in KCGate (gating mechanism)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

# Import KAN implementation
from .kan import KANLinear


# ---------------------------------------------------------------------------
# Simple KAN wrapper for KCGate
# ---------------------------------------------------------------------------
class SimpleKANLayer(nn.Module):
    """Simple KANLinear wrapper for gating."""
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kan = KANLinear(
            in_dim, out_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=[-1, 1],
        )

    def forward(self, x):
        if x.dim() == 3:
            B, N, C = x.shape
            x = x.reshape(B * N, C)
            x = self.kan(x)
            x = x.reshape(B, N, self.out_dim)
            return x
        return self.kan(x)


class MLPLayer(nn.Module):
    """MLP replacement for ablation study."""
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        if x.dim() == 3:
            B, N, C = x.shape
            x = x.reshape(B * N, C)
            x = self.mlp(x)
            x = x.reshape(B, N, self.out_dim)
            return x
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Sinusoidal 2D Position Embedding (from ref/c2seg)
# ---------------------------------------------------------------------------
class Sinusoidal2DPositionEmbedding(nn.Module):
    """
    2D sinusoidal position embedding for image features.
    Generates deterministic positional encodings of shape [1, C, H, W].
    """
    def __init__(self, num_channels: int, height: int, width: int):
        super().__init__()
        assert num_channels % 4 == 0, "Embedding dim must be divisible by 4 for 2D encoding"
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.register_buffer("pos_encoding", self._build_sincos_pos_encoding())

    def _build_sincos_pos_encoding(self) -> torch.Tensor:
        pe = torch.zeros(self.num_channels, self.height, self.width)

        y_pos = torch.arange(self.height, dtype=torch.float32).unsqueeze(1).repeat(1, self.width)
        x_pos = torch.arange(self.width, dtype=torch.float32).unsqueeze(0).repeat(self.height, 1)

        half_dim = self.num_channels // 2
        div_term = torch.exp(torch.arange(0, half_dim, 2).float() * -(math.log(10000.0) / half_dim))

        pe[0::4, :, :] = torch.sin(y_pos[None, :, :] * div_term[:, None, None])
        pe[1::4, :, :] = torch.cos(y_pos[None, :, :] * div_term[:, None, None])
        pe[2::4, :, :] = torch.sin(x_pos[None, :, :] * div_term[:, None, None])
        pe[3::4, :, :] = torch.cos(x_pos[None, :, :] * div_term[:, None, None])

        return pe.unsqueeze(0)  # [1, C, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_encoding


# ---------------------------------------------------------------------------
# Pixel Level Attention Module (from ref/c2seg)
# ---------------------------------------------------------------------------
class PixLevelModule(nn.Module):
    """
    Pixel-Level Attention Module for skip connections.
    Combines average-pooled and max-pooled features to generate pixel-wise attention.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 2):
        super().__init__()
        self.conv_avg = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_max = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(3 * reduction_ratio, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_feat = torch.mean(self.conv_avg(x), dim=1, keepdim=True)
        max_feat = torch.max(self.conv_max(x), dim=1, keepdim=True).values

        combined = torch.cat([avg_feat, max_feat, avg_feat + max_feat], dim=1)
        out = combined.permute(0, 2, 3, 1)
        out = self.bottleneck(out)
        out = out.permute(0, 3, 1, 2)

        return x * out


# ---------------------------------------------------------------------------
# KCGate: KAN-based Gating (from ref/c2seg BCAM)
# ---------------------------------------------------------------------------
class KCGate(nn.Module):
    """
    KAN-based Channel Gate.
    g = tanh(KAN(ReLU(KAN(x))))
    """
    def __init__(self, in_channels: int, grid_size: int = 5, spline_order: int = 3, use_kan: bool = True):
        super().__init__()
        Layer = SimpleKANLayer if use_kan else MLPLayer
        self.gate1 = Layer(in_channels, in_channels, grid_size, spline_order)
        self.gate2 = Layer(in_channels, in_channels, grid_size, spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] or [B, HW, C]
        Returns:
            gated output [B, N, C]
        """
        g = self.gate1(x)
        g = F.relu(g)
        g = self.gate2(g)
        g = torch.tanh(g)
        return x * g


# ---------------------------------------------------------------------------
# BCAM: Bidirectional Cross Attention Module (from ref/c2seg)
# ---------------------------------------------------------------------------
class BCAModule(nn.Module):
    """
    Bidirectional Cross Attention (BCA) module.
    Performs cross-attention between image and text features.
    """
    def __init__(
        self, 
        img_size: int, 
        in_channels: int, 
        num_heads: int = 8, 
        dropout: float = 0.0,
        grid_size: int = 5,
        spline_order: int = 3,
        use_kan: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_heads = num_heads

        self.pos_embed = Sinusoidal2DPositionEmbedding(in_channels, img_size, img_size)

        # Projection layers (Conv1d like ref)
        self.v_key = self._make_conv(in_channels, in_channels)
        self.v_value = self._make_conv(in_channels, in_channels)
        self.l_key = self._make_conv(512, in_channels)  # CLIP text dim = 512
        self.l_value = self._make_conv(512, in_channels)

        # KAN-based gating
        self.gate_v = KCGate(in_channels, grid_size, spline_order, use_kan)
        self.gate_l = KCGate(in_channels, grid_size, spline_order, use_kan)

        # Output projection
        self.project = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm1d(in_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    @staticmethod
    def _make_conv(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.InstanceNorm1d(out_ch)
        )

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat: [B, C, H, W]
            text_feat: [B, N, C_txt] where C_txt=512

        Returns:
            Fused features [B, C, H, W]
        """
        b, c, h, w = img_feat.shape
        hw = h * w
        n_t = text_feat.size(1)

        # Flatten image features and add position encoding
        img_flat = img_feat.flatten(2)  # [B, C, HW]
        v = self.pos_embed(img_feat).flatten(2)  # [B, C, HW]
        l = text_feat.transpose(1, 2)  # [B, C_txt, N]

        # Keys and values
        v_key, v_value = self.v_key(v), self.v_value(v)  # [B, C, HW]
        l_key, l_value = self.l_key(l), self.l_value(l)  # [B, C, N]

        # Reshape for multi-head attention
        head_dim = self.in_channels // self.num_heads
        v_key = v_key.view(b, self.num_heads, head_dim, hw).permute(0, 1, 3, 2)      # [B, H, HW, D]
        l_key = l_key.view(b, self.num_heads, head_dim, n_t)                          # [B, H, D, N]
        v_value = v_value.view(b, self.num_heads, head_dim, hw).permute(0, 1, 3, 2)  # [B, H, HW, D]
        l_value = l_value.view(b, self.num_heads, head_dim, n_t).permute(0, 1, 3, 2) # [B, H, N, D]

        # Attention maps
        sim_map = torch.matmul(v_key, l_key) * (self.in_channels ** -0.5)  # [B, H, HW, N]
        attn_v2l = F.softmax(sim_map, dim=-1)                               # [B, H, HW, N]
        attn_l2v = F.softmax(sim_map.transpose(-1, -2), dim=-1)             # [B, H, N, HW]

        # Cross attention
        v_out = torch.matmul(attn_v2l, l_value)                             # [B, H, HW, D]
        v_out = v_out.permute(0, 2, 1, 3).reshape(b, hw, c)                 # [B, HW, C]
        a_v = self.gate_v(v_out).permute(0, 2, 1)                           # [B, C, HW]

        l_out = torch.matmul(attn_l2v, v_value)                             # [B, H, N, D]
        l_out = l_out.permute(0, 2, 1, 3).reshape(b, n_t, c)                # [B, N, C]
        # Broadcast l_out to spatial: average over text tokens
        l_out_spatial = l_out.mean(dim=1, keepdim=True).expand(-1, hw, -1)  # [B, HW, C]
        a_l = self.gate_l(l_out_spatial).permute(0, 2, 1)                   # [B, C, HW]

        # Fuse and project
        fused = self.project(img_flat + a_v * a_l).view(b, c, h, w)

        return fused


# ---------------------------------------------------------------------------
# Basic Conv blocks (from ref/c2seg)
# ---------------------------------------------------------------------------
class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


def conv_stack(in_channels, out_channels, num_layers):
    """Stack of Conv-BN-ReLU layers"""
    layers = [ConvBNReLU(in_channels, out_channels)]
    layers += [ConvBNReLU(out_channels, out_channels) for _ in range(num_layers - 1)]
    return nn.Sequential(*layers)


class DownBlock(nn.Module):
    """Down-sampling block with stacked conv layers"""
    def __init__(self, in_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = conv_stack(in_channels, out_channels, num_layers)

    def forward(self, x):
        return self.convs(x)


class UpBlockAttention(nn.Module):
    """Up-sampling block with PixLevelModule attention on skip connection"""
    def __init__(self, skip_channels, up_channels, out_channels, num_layers=2):
        super().__init__()
        self.att = PixLevelModule(skip_channels)
        self.up = nn.ConvTranspose2d(up_channels, up_channels, kernel_size=2, stride=2)
        self.convs = conv_stack(skip_channels + up_channels, out_channels, num_layers)

    def forward(self, skip_x, x):
        skip_att = self.att(skip_x)
        up_x = self.up(x)
        # Handle size mismatch
        if up_x.shape[2:] != skip_att.shape[2:]:
            up_x = F.interpolate(up_x, size=skip_att.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([skip_att, up_x], dim=1)
        return self.convs(out)


# ---------------------------------------------------------------------------
# Language Encoder wrapper (frozen CLIP)
# ---------------------------------------------------------------------------
class LanguageEncoder(nn.Module):
    def __init__(self, clip_model: str = "openai/clip-vit-base-patch32", max_length: int = 18):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        self.encoder = CLIPTextModel.from_pretrained(clip_model)
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
        token_emb = out.last_hidden_state  # [B, N, 512]
        sent_emb = out.pooler_output       # [B, 512]
        return token_emb, sent_emb


# ---------------------------------------------------------------------------
# Full C2Seg model (MBINet architecture from ref/c2seg)
# ---------------------------------------------------------------------------
class C2Seg(nn.Module):
    """
    Multiscale Bidirectional Interaction Network (MBINet).
    Following ref/c2seg architecture exactly.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 224,
        clip_model: str = "openai/clip-vit-base-patch32",
        d_t: int = 512,
        max_token_len: int = 18,
        base_channel: int = 64,
        channel_scale: float = 1.0,
        num_heads: int = 8,
        dropout: float = 0.5,
        use_text: bool = True,
        use_bcam: bool = True,
        use_kan: bool = True,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        deep_supervision: bool = False,
        # Unused params for compatibility
        use_kgate: bool = True,
        drop_path_rate: float = 0.,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.use_text = use_text
        self.use_bcam = use_bcam
        
        # Scale base channel
        base_ch = max(16, int(base_channel * channel_scale))
        img_scales = [img_size, img_size//2, img_size//4, img_size//8, img_size//16]

        # Vision encoder (Conv only, no KAN)
        self.pool = nn.MaxPool2d(2, 2)
        self.down0 = DownBlock(in_channels, base_ch)
        self.down1 = DownBlock(base_ch, base_ch * 2)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)
        self.down4 = DownBlock(base_ch * 8, base_ch * 8)

        # Vision decoder with PixLevelModule
        self.up3 = UpBlockAttention(base_ch * 8, base_ch * 8, base_ch * 4)
        self.up2 = UpBlockAttention(base_ch * 4, base_ch * 4, base_ch * 2)
        self.up1 = UpBlockAttention(base_ch * 2, base_ch * 2, base_ch)
        self.up0 = UpBlockAttention(base_ch, base_ch, base_ch)

        # Language encoder (frozen CLIP)
        if use_text:
            self.lang_enc = LanguageEncoder(clip_model, max_length=max_token_len)

            # Cross-modal fusion (BCAM at 4 scales: v1, v2, v3, v4)
            if use_bcam:
                self.fuse1 = BCAModule(img_scales[1], base_ch, num_heads, dropout, 
                                       kan_grid_size, kan_spline_order, use_kan)
                self.fuse2 = BCAModule(img_scales[2], base_ch * 2, num_heads, dropout,
                                       kan_grid_size, kan_spline_order, use_kan)
                self.fuse3 = BCAModule(img_scales[3], base_ch * 4, num_heads, dropout,
                                       kan_grid_size, kan_spline_order, use_kan)
                self.fuse4 = BCAModule(img_scales[4], base_ch * 8, num_heads, dropout,
                                       kan_grid_size, kan_spline_order, use_kan)

        # Output heads
        self.out0 = nn.Sequential(nn.Conv2d(base_ch, num_classes, 1), nn.Sigmoid())
        
        if self.deep_supervision:
            self.out1 = nn.Sequential(nn.Conv2d(base_ch, num_classes, 1), nn.Sigmoid())
            self.out2 = nn.Sequential(nn.Conv2d(base_ch * 2, num_classes, 1), nn.Sigmoid())
            self.out3 = nn.Sequential(nn.Conv2d(base_ch * 4, num_classes, 1), nn.Sigmoid())
            self.out4 = nn.Sequential(nn.Conv2d(base_ch * 8, num_classes, 1), nn.Sigmoid())

        # Projections for CaCL pretraining
        self.img_proj = nn.Linear(base_ch * 8, 256)
        self.text_proj = nn.Linear(d_t, 256)
        
        # Store for param counting
        self.base_ch = base_ch

    def forward(self, images, texts=None):
        # Encode text if provided
        l = None
        if self.use_text and texts is not None:
            token_emb, _ = self.lang_enc(texts, images.device)
            l = token_emb  # [B, N, 512]

        # Encode image (vision stream)
        v0 = self.down0(images)              # [B, C, 224, 224]
        v1 = self.down1(self.pool(v0))       # [B, 2C, 112, 112]
        v2 = self.down2(self.pool(v1))       # [B, 4C, 56, 56]
        v3 = self.down3(self.pool(v2))       # [B, 8C, 28, 28]
        v4 = self.down4(self.pool(v3))       # [B, 8C, 14, 14]

        # Cross-modal fusion + decoder
        if self.use_text and self.use_bcam and l is not None:
            f4 = self.fuse4(v4, l)
            
            up3 = self.up3(v3, f4)
            f3 = self.fuse3(up3, l)
            
            up2 = self.up2(v2, f3)
            f2 = self.fuse2(up2, l)
            
            up1 = self.up1(v1, f2)
            f1 = self.fuse1(up1, l)
            
            up0 = self.up0(v0, f1)
        else:
            # No text fusion
            up3 = self.up3(v3, v4)
            up2 = self.up2(v2, up3)
            up1 = self.up1(v1, up2)
            up0 = self.up0(v0, up1)
            f1, f2, f3, f4 = up1, up2, up3, v4

        out0 = self.out0(up0)

        if self.deep_supervision:
            return (out0,
                    self.out1(f1),
                    self.out2(f2),
                    self.out3(f3),
                    self.out4(f4))
        return out0

    def get_contrastive_embeddings(self, images, texts):
        """For CaCL pretraining."""
        if not self.use_text:
            raise RuntimeError("Cannot use get_contrastive_embeddings when use_text=False")

        # Forward through encoder only
        v0 = self.down0(images)
        v1 = self.down1(self.pool(v0))
        v2 = self.down2(self.pool(v1))
        v3 = self.down3(self.pool(v2))
        v4 = self.down4(self.pool(v3))

        # Global image vector
        v_global = F.adaptive_avg_pool2d(v4, 1).flatten(1)  # [B, 8*base_ch]

        with torch.no_grad():
            _, sent_emb = self.lang_enc(texts, images.device)

        v = F.normalize(self.img_proj(v_global), dim=-1)
        l = F.normalize(self.text_proj(sent_emb), dim=-1)
        l_raw = F.normalize(sent_emb.detach().float(), dim=-1)
        
        return v, l, l_raw
