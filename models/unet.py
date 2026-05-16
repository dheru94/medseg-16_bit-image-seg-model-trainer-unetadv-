# models/unet.py
"""
Classic UNet — Ronneberger et al. 2015
Supports any number of input channels and output classes.

Improvements for defect segmentation:
- Dropout2d at bottleneck to prevent background overfitting
- Slower BatchNorm momentum for small batch sizes (batch=2)
- Gradient checkpointing support for large images
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv -> BN -> ReLU blocks."""
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            # momentum=0.01 gives slower running stats — better for batch_size=2
            nn.BatchNorm2d(mid_ch, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool then DoubleConv — encoder step."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample then DoubleConv — decoder step."""
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 to match x2 spatial size (handles odd input dimensions)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    """1x1 conv to map features to n_classes."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Full UNet with optional bottleneck dropout.

    Args:
        n_channels:       Number of input image channels (1 for grayscale, 4 for RGBA).
        n_classes:        Number of output segmentation classes.
        bilinear:         Use bilinear upsampling instead of transposed conv.
        dropout:          Dropout probability at bottleneck (0.0 to disable).
                          Helps prevent background overfitting in defect segmentation.
    """
    def __init__(self, n_channels: int = 1, n_classes: int = 4,
                 bilinear: bool = False, dropout: float = 0.5):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear
        f      = 64   # base feature count
        factor = 2 if bilinear else 1

        # ── Encoder ───────────────────────────────────────────────────────────
        self.inc   = DoubleConv(n_channels, f)
        self.down1 = Down(f,     f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16 // factor)

        # Bottleneck dropout — randomly zeros feature maps to force model to
        # learn defect features instead of relying on background shortcut
        self.bottleneck_drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up1  = Up(f * 16, f * 8  // factor, bilinear)
        self.up2  = Up(f * 8,  f * 4  // factor, bilinear)
        self.up3  = Up(f * 4,  f * 2  // factor, bilinear)
        self.up4  = Up(f * 2,  f,                bilinear)
        self.outc = OutConv(f, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Bottleneck dropout
        x5 = self.bottleneck_drop(x5)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        return self.outc(x)

    def use_checkpointing(self):
        """
        Enable gradient checkpointing to save VRAM at cost of ~20% speed.
        Call after model creation if running out of GPU memory.
        """
        self.inc   = torch.utils.checkpoint.checkpoint_wrapper(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint_wrapper(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint_wrapper(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint_wrapper(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint_wrapper(self.down4)