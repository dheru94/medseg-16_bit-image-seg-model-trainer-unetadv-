# models/deeplab.py
"""
DeepLab v3+ — Chen et al. 2018
Uses atrous (dilated) convolutions + ASPP for large receptive field.
Best for satellite imagery and large-scale structures.
Built from scratch — no torchvision backbone dependency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1,
                 dilation=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding * dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """Simple residual block with optional dilation."""
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        self.conv1    = ConvBnRelu(in_ch, out_ch, stride=stride, dilation=dilation,
                                   padding=dilation)
        self.conv2    = ConvBnRelu(out_ch, out_ch, dilation=dilation, padding=dilation)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        return F.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class Encoder(nn.Module):
    """Lightweight encoder with low-level and high-level features."""
    def __init__(self, in_ch):
        super().__init__()
        # Entry flow
        self.layer0 = nn.Sequential(
            ConvBnRelu(in_ch, 32, kernel=3, stride=2, padding=1),
            ConvBnRelu(32, 64, kernel=3, stride=1, padding=1),
        )
        self.layer1 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
        )
        # Low-level features saved here (for decoder skip connection)
        self.layer2 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
        )
        # High-level features with dilation
        self.layer3 = nn.Sequential(
            ResBlock(256, 512, dilation=2),
            ResBlock(512, 512, dilation=2),
            ResBlock(512, 512, dilation=4),
        )

    def forward(self, x):
        x  = self.layer0(x)
        x  = self.layer1(x)
        low_level = self.layer2(x)   # (B, 256, H/8, W/8) — skip connection
        x  = self.layer3(low_level)  # (B, 512, H/8, W/8)
        return x, low_level


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — captures multi-scale context."""
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.conv1   = ConvBnRelu(in_ch, out_ch, kernel=1, padding=0)
        self.conv6   = ConvBnRelu(in_ch, out_ch, dilation=6,  padding=6)
        self.conv12  = ConvBnRelu(in_ch, out_ch, dilation=12, padding=12)
        self.conv18  = ConvBnRelu(in_ch, out_ch, dilation=18, padding=18)
        self.pool    = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = ConvBnRelu(out_ch * 5, out_ch, kernel=1, padding=0)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        size = x.shape[2:]
        p1   = self.conv1(x)
        p6   = self.conv6(x)
        p12  = self.conv12(x)
        p18  = self.conv18(x)
        pg   = F.interpolate(self.pool(x), size=size,
                             mode='bilinear', align_corners=False)
        x    = torch.cat([p1, p6, p12, p18, pg], dim=1)
        return self.dropout(self.project(x))


class Decoder(nn.Module):
    """Decoder with skip connection from low-level features."""
    def __init__(self, n_classes, low_level_ch=256):
        super().__init__()
        self.low_proj = ConvBnRelu(low_level_ch, 48, kernel=1, padding=0)
        self.refine   = nn.Sequential(
            ConvBnRelu(256 + 48, 256),
            ConvBnRelu(256, 256),
            nn.Conv2d(256, n_classes, 1),
        )

    def forward(self, x, low_level, input_size):
        low_level = self.low_proj(low_level)
        x = F.interpolate(x, size=low_level.shape[2:],
                          mode='bilinear', align_corners=False)
        x = torch.cat([x, low_level], dim=1)
        x = self.refine(x)
        return F.interpolate(x, size=input_size,
                             mode='bilinear', align_corners=False)


class DeepLabV3Plus(nn.Module):
    def __init__(self, n_channels: int = 4, n_classes: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes

        self.encoder = Encoder(n_channels)
        self.aspp    = ASPP(in_ch=512, out_ch=256)
        self.decoder = Decoder(n_classes, low_level_ch=256)

    def forward(self, x):
        input_size    = x.shape[2:]
        x, low_level  = self.encoder(x)
        x             = self.aspp(x)
        return self.decoder(x, low_level, input_size)

    def use_checkpointing(self):
        import logging
        logging.warning('use_checkpointing not implemented for DeepLab — reduce batch size instead.')
