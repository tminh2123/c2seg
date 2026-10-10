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

# Import real KAN implementation
from .kan import KANLinear


# ---------------------------------------------------------------------------
# KAN layer wrapper for easy use
# ---------------------------------------------------------------------------
class KANLayer(nn.Module):
    """
    Wrapper for KANLinear with proper initialization for C2Seg.
    Handles both 2D [batch, features] and 3D [batch, seq, features] inputs.
    """
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kan = KANLinear(
            in_dim, 
            out_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=[-1, 1],
        )

    def forward(self, x):
        # Handle 3D input [B, N, C] -> flatten -> KAN -> unflatten
        if x.dim() == 3:
            B, N, C = x.shape
            x = x.reshape(B * N, C)
            x = self.kan(x)
            x = x.reshape(B, N, self.out_dim)
            return x
        # 2D input [batch, features]
        return self.kan(x)


class MLPLayer(nn.Module):
    """
    MLP replacement for KANLayer - for ablation study.
    Same interface as KANLayer but uses Linear layers.
    """
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, spline_order: int = 3):
        # grid_size and spline_order ignored, kept for API compatibility
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        # Handle 3D input [B, N, C]
        if x.dim() == 3:
            B, N, C = x.shape
            x = x.reshape(B * N, C)
            x = self.mlp(x)
            x = x.reshape(B, N, self.out_dim)
            return x
        # 2D input [batch, features]
        return self.mlp(x)


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
    Produces 5 multi-scale feature maps V1..V5 (progressively downsampled).
    Default channels: 64 -> 128 -> 256 -> 512 -> 512
    First 3 stages are CNN, last 2 are KAN/MLP (as per paper Figure 2).
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        kan_grid_size: int = 5, 
        kan_spline_order: int = 3,
        channels: list = None,  # Custom channel dims [c1, c2, c3, c4, c5]
        use_kan: bool = True,   # False to use MLP instead
    ):
        super().__init__()
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        self.use_kan = use_kan
        
        # Select layer type
        Layer = KANLayer if use_kan else MLPLayer
        
        # Default or custom channels
        if channels is None:
            channels = [64, 128, 256, 512, 512]
        c1, c2, c3, c4, c5 = channels
        
        # 3 CNN stages
        self.enc1 = ConvBlock(in_channels, c1)   # H x W
        self.enc2 = ConvBlock(c1, c2)            # H/2 x W/2
        self.enc3 = ConvBlock(c2, c3)            # H/4 x W/4
        self.pool = nn.MaxPool2d(2)

        # 2 KAN/MLP stages (operate channel-wise via 1x1-style reshape)
        self.stage4 = self._make_stage(c3, c4, Layer)  # H/8 x W/8
        self.stage5 = self._make_stage(c4, c5, Layer)  # H/16 x W/16
        
        self.out_channels = c5  # For global vector dim

    def _make_stage(self, in_ch, out_ch, Layer):
        return nn.Sequential(
            nn.MaxPool2d(2),
            _SpatialLayer(in_ch, out_ch, self.kan_grid_size, self.kan_spline_order, Layer),
        )

    def forward(self, x):
        v1 = self.enc1(x)                  # [B, c1, H,   W]
        v2 = self.enc2(self.pool(v1))      # [B, c2, H/2, W/2]
        v3 = self.enc3(self.pool(v2))      # [B, c3, H/4, W/4]
        v4 = self.stage4(v3)               # [B, c4, H/8, W/8]
        v5 = self.stage5(v4)               # [B, c5, H/16,W/16]

        # global image vector
        v_global = F.adaptive_avg_pool2d(v5, 1).flatten(1)  # [B, c5]
        return [v1, v2, v3, v4, v5], v_global


class _SpatialLayer(nn.Module):
    """Apply KAN/MLP across channels at each spatial location."""
    def __init__(self, in_ch, out_ch, grid_size: int = 5, spline_order: int = 3, Layer=KANLayer):
        super().__init__()
        self.layer = Layer(in_ch, out_ch, grid_size, spline_order)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)   # [B*H*W, C]
        x = self.layer(x)
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
    def __init__(self, vis_ch: int, lang_ch: int, grid_size: int = 5, spline_order: int = 3, use_kan: bool = True):
        super().__init__()
        d = vis_ch
        Layer = KANLayer if use_kan else MLPLayer
        self.v_key   = Layer(vis_ch,  d, grid_size, spline_order)
        self.v_value = Layer(vis_ch,  d, grid_size, spline_order)
        self.l_key   = Layer(lang_ch, d, grid_size, spline_order)
        self.l_value = Layer(lang_ch, d, grid_size, spline_order)
        self.scale   = d ** -0.5

    def forward(self, V, L_w):
        """
        V:   [B, C, H, W]  visual features
        L_w: [B, N, C]     language tokens (projected to same C)
        Returns F_v, F_l both [B, C, H, W]
        
        Paper equations:
        - Eq. 4: A = (1/√d) * V_key @ L_key^T  ∈ ℝ^(P×N)
        - Eq. 5: F_v = Softmax_N(A) @ L_value  ∈ ℝ^(P×C)
        - Eq. 6: F_l = (1/N) * Σ_n [ Softmax_P(A^T[n,:]) ⊙ V_value ]  ∈ ℝ^(P×C)
        
        Memory-efficient: uses einsum to avoid creating [B, N, P, C] tensor
        """
        B, C, H, W = V.shape
        P = H * W
        N = L_w.shape[1]

        # Flatten spatial: V [B, C, H, W] -> V_flat [B, P, C]
        V_flat = V.flatten(2).permute(0, 2, 1)   # [B, P, C]

        # KAN projections for keys & values
        Vk = self.v_key(V_flat)                   # [B, P, d]
        Vv = self.v_value(V_flat)                 # [B, P, C]
        Lk = self.l_key(L_w)                      # [B, N, d]
        Lv = self.l_value(L_w)                    # [B, N, C]

        # Eq. 4: Attention scores A = (1/√d) * V_key @ L_key^T
        A = torch.bmm(Vk, Lk.transpose(1, 2)) * self.scale  # [B, P, N]

        # --- Eq. 5: Vision-dominant path ---
        # F_v = Softmax_N(A) @ L_value
        attn_v = F.softmax(A, dim=2)              # [B, P, N] softmax over N
        F_v = torch.bmm(attn_v, Lv)               # [B, P, C]

        # --- Eq. 6: Language-dominant path (memory efficient) ---
        # F_l = (1/N) * Σ_n [ Softmax_P(A^T[n,:]) ⊙ V_value ]
        # Instead of expanding to [B, N, P, C], use einsum:
        # attn_l[b,n,p] * Vv[b,p,c] -> sum over n -> [b,p,c]
        A_T = A.transpose(1, 2)                   # [B, N, P]
        attn_l = F.softmax(A_T, dim=2)            # [B, N, P] softmax over P
        
        # einsum: 'bnp,bpc->bpc' means: for each (b,p,c), sum over n: attn_l[b,n,p] * Vv[b,p,c]
        # This is equivalent to: (attn_l.sum(dim=1).unsqueeze(-1) * Vv) but we need weighted sum
        # Correct form: for each position p, weight Vv[p] by attention from all N tokens
        # F_l[b,p,c] = (1/N) * sum_n(attn_l[b,n,p]) * Vv[b,p,c]
        # But that's wrong - we need element-wise per n then sum
        # 
        # Actually Eq.6: F_l = (1/N) * Σ_n [ w_n ⊙ V_value ] where w_n = softmax_P(A^T[n,:])
        # This means: weighted average of V_value, where weight at position p is avg of attn from all tokens
        # F_l[b,p,c] = (1/N) * Σ_n attn_l[b,n,p] * Vv[b,p,c]
        #            = Vv[b,p,c] * (1/N) * Σ_n attn_l[b,n,p]
        #            = Vv[b,p,c] * mean_n(attn_l[b,n,p])
        attn_l_mean = attn_l.mean(dim=1)          # [B, P] - average attention across all N tokens
        F_l = Vv * attn_l_mean.unsqueeze(-1)      # [B, P, C]

        # Reshape back to spatial [B, C, H, W]
        F_v = F_v.permute(0, 2, 1).reshape(B, C, H, W)
        F_l = F_l.permute(0, 2, 1).reshape(B, C, H, W)
        return F_v, F_l


# ---------------------------------------------------------------------------
# K-Gate: KAN-based Attention Gating
# ---------------------------------------------------------------------------
class KGate(nn.Module):
    """
    KAN/MLP-based Attention Gating (Section 3.4, Eq. 7)
    
    Each path: Layer → ReLU → Layer → Tanh
    
    g_v = tanh(Layer(ReLU(Layer(F_v))))
    g_l = tanh(Layer(ReLU(Layer(F_l))))
    F_v^g = F_v ⊙ g_v,  F_l^g = F_l ⊙ g_l
    F_out = Conv1x1([F_v^g || F_l^g])
    """
    def __init__(self, ch: int, grid_size: int = 5, spline_order: int = 3, use_kan: bool = True):
        super().__init__()
        Layer = KANLayer if use_kan else MLPLayer
        
        # LayerNorm for stability
        self.norm_v = nn.LayerNorm(ch)
        self.norm_l = nn.LayerNorm(ch)
        
        # Gating: Layer → ReLU → Layer → Tanh (as per paper Figure 2)
        self.gate_v = nn.Sequential(
            Layer(ch, ch, grid_size, spline_order),
            nn.ReLU(),
            Layer(ch, ch, grid_size, spline_order),
            nn.Tanh()
        )
        self.gate_l = nn.Sequential(
            Layer(ch, ch, grid_size, spline_order),
            nn.Tanh()
        )
        
        # 1x1 conv for channel alignment after concatenation
        self.fuse = nn.Conv2d(ch * 2, ch, 1)

    def forward(self, F_v, F_l):
        """
        F_v, F_l: [B, C, H, W] from BCAM
        Returns: F_out [B, C, H, W]
        
        Process per-sample to reduce memory (avoid B*H*W explosion)
        """
        B, C, H, W = F_v.shape
        
        g_v_list = []
        g_l_list = []
        
        for b in range(B):
            # Single sample: [H*W, C]
            fv = F_v[b].permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
            fl = F_l[b].permute(1, 2, 0).reshape(-1, C)
            
            # Normalize
            fv_norm = self.norm_v(fv)
            fl_norm = self.norm_l(fl)
            
            # KAN gating
            gv = self.gate_v(fv_norm)  # [H*W, C]
            gl = self.gate_l(fl_norm)
            
            g_v_list.append(gv.reshape(H, W, C).permute(2, 0, 1))  # [C, H, W]
            g_l_list.append(gl.reshape(H, W, C).permute(2, 0, 1))
        
        g_v = torch.stack(g_v_list, dim=0)  # [B, C, H, W]
        g_l = torch.stack(g_l_list, dim=0)
        
        # Element-wise gating
        Fv_g = F_v * g_v
        Fl_g = F_l * g_l
        
        # Concatenate and fuse
        F_out = self.fuse(torch.cat([Fv_g, Fl_g], dim=1))
        
        return F_out


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

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        clip_model: str  = "openai/clip-vit-base-patch32",
        d_t: int         = 512,
        use_text: bool   = True,  # Set False to disable BCAM/K-Gate (ablation)
        use_bcam: bool   = True,  # Set False to skip BCAM (ablation)
        use_kgate: bool  = True,  # Set False to skip K-Gate (ablation)
        use_kan: bool    = True,  # Set False to use MLP instead of KAN (ablation)
        kan_grid_size: int = 5,   # U-KAN default
        kan_spline_order: int = 3,  # U-KAN default (cubic spline)
        channel_scale: float = 1.0,  # Scale factor for channels (0.5 = half, 0.25 = quarter)
    ):
        super().__init__()
        self.use_text = use_text
        self.use_bcam = use_bcam
        self.use_kgate = use_kgate
        self.use_kan = use_kan
        
        # Scale channels (KAN can use narrower networks)
        base_channels = [64, 128, 256, 512, 512]
        self.VIS_CHANNELS = [max(16, int(c * channel_scale)) for c in base_channels]
        self.BCAM_CHANNELS = self.VIS_CHANNELS[1:]  # Skip first scale
        
        self.visual_enc = VisualEncoder(
            in_channels, kan_grid_size, kan_spline_order, 
            channels=self.VIS_CHANNELS,
            use_kan=use_kan,
        )
        
        if use_text:
            self.lang_enc   = LanguageEncoder(clip_model)
            # Language projection for all 5 scales (needed for indexing)
            self.lang_proj  = LangProjection(d_t, self.VIS_CHANNELS)

            # BCAM at 4 scales (v2, v3, v4, v5 - skip v1) as per paper Figure 2
            if use_bcam:
                self.bcam = nn.ModuleList([
                    BCAM(c, c, kan_grid_size, kan_spline_order, use_kan=use_kan) for c in self.BCAM_CHANNELS
                ])
            
            # K-Gate at 4 scales (same as BCAM)
            if use_kgate:
                self.kgate = nn.ModuleList([
                    KGate(c, kan_grid_size, kan_spline_order, use_kan=use_kan) for c in self.BCAM_CHANNELS
                ])
            
            # Simple fusion layer when not using K-Gate
            if use_bcam and not use_kgate:
                self.simple_fuse = nn.ModuleList([
                    nn.Conv2d(c * 2, c, 1) for c in self.BCAM_CHANNELS
                ])

        # Decoder channels also scaled
        dec_ch = self.VIS_CHANNELS
        self.dec4 = DecoderBlock(dec_ch[4], dec_ch[3], dec_ch[2])
        self.dec3 = DecoderBlock(dec_ch[2], dec_ch[2], dec_ch[1])
        self.dec2 = DecoderBlock(dec_ch[1], dec_ch[1], dec_ch[0])
        self.dec1 = DecoderBlock(dec_ch[0], dec_ch[0], max(16, dec_ch[0]//2))
        self.head = nn.Conv2d(max(16, dec_ch[0]//2), num_classes, 1)

        # projection from sentence embedding to global visual for CaCL
        self.img_proj  = nn.Linear(dec_ch[4], 256)
        self.text_proj = nn.Linear(d_t, 256)

    def encode(self, images, texts):
        """Extract visual & language features."""
        feats, v_global = self.visual_enc(images)
        if self.use_text:
            token_emb, sent_emb = self.lang_enc(texts, images.device)
            lang_per_scale = self.lang_proj(token_emb)  # list of [B, N, C_ω]
            return feats, v_global, token_emb, sent_emb, lang_per_scale
        return feats, v_global, None, None, None

    def forward(self, images, texts=None):
        feats, v_global, _, _, lang_per_scale = self.encode(images, texts)

        if self.use_text and lang_per_scale is not None:
            # feats[0] (v1) passes through without BCAM/K-Gate
            # feats[1:5] (v2-v5) go through BCAM + K-Gate
            fused = [feats[0]]  # v1 unchanged
            
            for i, (V, L_w) in enumerate(zip(feats[1:], lang_per_scale[1:])):
                # i=0 → feats[1], i=1 → feats[2], etc.
                if self.use_bcam:
                    # BCAM produces F_v (vision-dominant) and F_l (language-dominant)
                    F_v, F_l = self.bcam[i](V, L_w)
                    
                    if self.use_kgate:
                        # K-Gate fuses F_v and F_l with KAN-based gating
                        fused.append(self.kgate[i](F_v, F_l))
                    else:
                        # Simple concat + conv fusion (no K-Gate)
                        fused.append(self.simple_fuse[i](torch.cat([F_v, F_l], dim=1)))
                else:
                    # No BCAM - just use visual features
                    fused.append(V)
        else:
            # No text fusion - just use visual features directly
            fused = feats

        # U-Net decoder with skip connections from fused features
        x = fused[4]                              # [B, 512, H/16, W/16]
        x = self.dec4(x, fused[3])               # [B, 256, H/8,  W/8]
        x = self.dec3(x, fused[2])               # [B, 128, H/4,  W/4]
        x = self.dec2(x, fused[1])               # [B, 64,  H/2,  W/2]
        x = self.dec1(x, fused[0])               # [B, 32,  H,    W]
        return self.head(x)                       # [B, num_classes, H, W]

    def get_contrastive_embeddings(self, images, texts):
        """For CaCL pretraining: lightweight forward — only visual global vec + sent_emb."""
        if not self.use_text:
            raise RuntimeError("Cannot use get_contrastive_embeddings when use_text=False")
        
        # only need v_global, skip BCAM/lang_proj overhead
        feats, v_global = self.visual_enc(images)
        with torch.no_grad():
            _, sent_emb = self.lang_enc(texts, images.device)
        v     = F.normalize(self.img_proj(v_global), dim=-1)
        l     = F.normalize(self.text_proj(sent_emb), dim=-1)
        l_raw = F.normalize(sent_emb.detach().float(), dim=-1)
        return v, l, l_raw
