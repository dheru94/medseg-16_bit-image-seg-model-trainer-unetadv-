# models/unetpp.py
"""
UNet++ — Zhou et al. 2018
Nested skip connections for better fine-structure segmentation.
Better than UNet for small structures like defects.

Improvements for defect segmentation:
- Dropout2d at bottleneck to prevent background overfitting
- Slower BatchNorm momentum for small batch sizes (batch=2)
- Deep supervision enabled by default for better gradient flow to rare classes
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    """Single Conv -> BN -> ReLU block."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # momentum=0.01 — slower running stats, better for batch_size=2
            nn.BatchNorm2d(out_ch, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """Two consecutive ConvBnRelu blocks."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPP(nn.Module):
    """
    UNet++ with nested skip connections.

    Args:
        n_channels:        Number of input image channels.
        n_classes:         Number of output segmentation classes.
        deep_supervision:  Average outputs from all decoder heads.
                           Recommended True during training — better gradient
                           flow helps rare defect classes learn faster.
        dropout:           Dropout probability at bottleneck (0.0 to disable).
    """
    def __init__(self, n_channels: int = 1, n_classes: int = 4,
                 deep_supervision: bool = True, dropout: float = 0.5):
        super().__init__()
        self.n_channels       = n_channels
        self.n_classes        = n_classes
        self.deep_supervision = deep_supervision
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ── Encoder ───────────────────────────────────────────────────────────
        self.conv0_0 = DoubleConv(n_channels,   nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        # Bottleneck dropout — forces model to learn defect features
        # instead of relying on background-only shortcut
        self.bottleneck_drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        # ── Nested decoder nodes ──────────────────────────────────────────────
        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        # ── Output heads ──────────────────────────────────────────────────────
        if deep_supervision:
            # Four decoder heads — averaged for final prediction
            # Each head gets direct gradient signal — helps rare class learning
            self.final1 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder ───────────────────────────────────────────────────────────
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Bottleneck dropout
        x4_0 = self.bottleneck_drop(x4_0)

        # ── Nested decoder ────────────────────────────────────────────────────
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        # ── Output ────────────────────────────────────────────────────────────
        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return (out1 + out2 + out3 + out4) / 4
        else:
            return self.final(x0_4)

    def use_checkpointing(self):
        """
        Gradient checkpointing is not supported for UNet++ due to
        the complex nested skip connections. Reduce batch size or
        img_size instead if running out of VRAM.
        """
        logging.warning(
            'use_checkpointing() not supported for UNet++ — '
            'try reducing --batch-size or --img-size instead.'
        )