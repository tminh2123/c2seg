"""
Standard U-Net for binary segmentation baseline.
No text input - pure visual segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture.
    
    Encoder: 4 downsampling blocks (64 -> 128 -> 256 -> 512)
    Bottleneck: 1024 channels
    Decoder: 4 upsampling blocks with skip connections
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)  # 512 + 512 skip
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)   # 256 + 256 skip
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)   # 128 + 128 skip
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)    # 64 + 64 skip
        
        # Output
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x, texts=None):
        """
        Forward pass. `texts` argument is ignored (for compatibility with C2Seg dataset).
        """
        # Encoder
        e1 = self.enc1(x)                    # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))        # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))        # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))        # [B, 512, H/8, W/8]
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))   # [B, 1024, H/16, W/16]
        
        # Decoder with skip connections
        d4 = self.up4(b)                     # [B, 512, H/8, W/8]
        d4 = torch.cat([d4, e4], dim=1)      # [B, 1024, H/8, W/8]
        d4 = self.dec4(d4)                   # [B, 512, H/8, W/8]
        
        d3 = self.up3(d4)                    # [B, 256, H/4, W/4]
        d3 = torch.cat([d3, e3], dim=1)      # [B, 512, H/4, W/4]
        d3 = self.dec3(d3)                   # [B, 256, H/4, W/4]
        
        d2 = self.up2(d3)                    # [B, 128, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)      # [B, 256, H/2, W/2]
        d2 = self.dec2(d2)                   # [B, 128, H/2, W/2]
        
        d1 = self.up1(d2)                    # [B, 64, H, W]
        d1 = torch.cat([d1, e1], dim=1)      # [B, 128, H, W]
        d1 = self.dec1(d1)                   # [B, 64, H, W]
        
        return self.head(d1)                 # [B, num_classes, H, W]


if __name__ == '__main__':
    # Test
    model = UNet(in_channels=3, num_classes=1)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
