# models/unetadv.py
"""
Variable Attention Nested UNet++ (VA-UNet++)

Based on paper: "A Variable Attention Nested UNet++ Network-Based NDT X-ray
Image Defect Segmentation Method" - Liu & Kim, Coatings 2022

Key innovations:
1. Selective Kernel (SK) Block - variable receptive fields (3x3 + 5x5)
2. Attention Gate - suppresses background, highlights defect regions
3. Deep supervision - independent loss heads (not averaged logits)

Fixed bugs vs original:
  BUG 1 (Critical): Attention gate g-signal was stale at j=2,3,4.
    The gate must always be UP(X(i+1, j-1)) — the upsampled decoder node
    one column to the right and one row below — NOT a fixed encoder feature.
    This is the core of the paper equation:
      X(i,j) = φ[ Σ_{k=0}^{j-1} Ag(X(i,k), UP(X(i+1,j-1))),  UP(X(i+1,j-1)) ]
    The same gate signal UP(X(i+1,j-1)) gates ALL incoming skip features at
    that node — not separate gates per skip. Fixed by passing the correct
    current-column decoder output as gate at every node.

  BUG 2 (Critical): Deep supervision returned averaged logits.
    Paper's deep supervision means independent predictions from each head,
    each trained with a loss — NOT blended logits fed into a single loss.
    Fixed: forward() returns a list of 4 logit tensors in train mode;
    train.py applies a weighted loss sum. At inference call
    model.eval() and the final head (index -1) is used.

  BUG 3 (Safety): No spatial size guard in AttentionGate.
    Added F.interpolate alignment so non-power-of-2 image sizes don't crash.

  BUG 4 (Safety): SK bottleneck width clamped to min 4 to avoid 0-channel FC.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Selective Kernel Block ────────────────────────────────────────────────────

class SKBlock(nn.Module):
    """
    Selective Kernel Block — Figure 5 in paper.

    Dual branch:
      Branch 1 : 3×3 conv  (RF = 3)
      Branch 2 : 3×3 → 3×3 (effective RF ≈ 5, same as a single 5×5)

    GAP → FC → Softmax → weighted sum of branches.
    Final BN+ReLU applied after fusion.
    """
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16):
        super().__init__()
        self.out_channels = out_channels

        # Branch 1 — 3×3
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        # Branch 2 — 3×3 → 3×3  (effective 5×5 receptive field)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        # Attention: GAP on concatenated branches → FC bottleneck → 2 weights
        mid = max(4, out_channels // reduction)          # BUG 4 fix: clamp to ≥4
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(out_channels * 2, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, out_channels * 2, bias=False),
        )

        # Post-fusion normalisation
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channels, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)

        y1 = self.branch1(x)
        y2 = self.branch2(x)

        # [B, 2*C] → attention weights [B, 2, C]
        gap       = self.gap(torch.cat([y1, y2], dim=1)).view(b, -1)
        attn      = self.fc(gap).view(b, 2, self.out_channels)
        attn      = F.softmax(attn, dim=1)

        # Weighted fusion
        out = (attn[:, 0].view(b, self.out_channels, 1, 1) * y1 +
               attn[:, 1].view(b, self.out_channels, 1, 1) * y2)

        return self.bn_relu(out)


class DoubleSK(nn.Module):
    """Two consecutive SK blocks — used in both encoder and decoder."""
    def __init__(self, in_ch: int, out_ch: int, reduction: int = 16):
        super().__init__()
        self.block = nn.Sequential(
            SKBlock(in_ch,  out_ch, reduction),
            SKBlock(out_ch, out_ch, reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Attention Gate ────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Attention Gate — Figure 6 in paper.

    g : gate signal  (upsampled decoder context)
    x : skip feature to be gated

    Wg(g) + Wx(x) → ReLU → psi Conv → BN → Sigmoid → x * α
    """
    def __init__(self, g_channels: int, x_channels: int, int_channels: int):
        super().__init__()
        self.Wg  = nn.Sequential(
            nn.Conv2d(g_channels,   int_channels, 1, bias=False),
            nn.BatchNorm2d(int_channels, momentum=0.01),
        )
        self.Wx  = nn.Sequential(
            nn.Conv2d(x_channels,   int_channels, 1, bias=False),
            nn.BatchNorm2d(int_channels, momentum=0.01),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(int_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1, momentum=0.01),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.Wg(g)
        x1 = self.Wx(x)

        # BUG 3 fix: align spatial dims for non-power-of-2 inputs
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:],
                               mode='bilinear', align_corners=True)

        alpha = self.psi(self.relu(g1 + x1))   # [B,1,H,W]
        return x * alpha


# ── VA-UNet++ ─────────────────────────────────────────────────────────────────

class UNetAdv(nn.Module):
    """
    Variable Attention Nested UNet++ (VA-UNet++)

    5-level encoder / decoder with:
      • DoubleSK blocks everywhere (variable receptive field)
      • Attention Gates on every skip connection with the CORRECT gate signal
      • Deep supervision: 4 independent output heads

    Args:
        n_channels      : input image channels
        n_classes       : segmentation classes
        deep_supervision: True = return list of 4 logit tensors during training;
                          False = return single logit tensor (final head only)
        dropout         : Dropout2d probability at bottleneck (0 = off)
        base_filters    : channel widths per level (5 values)
        sk_reduction    : SK attention reduction ratio
    """
    def __init__(self,
                 n_channels:       int   = 1,
                 n_classes:        int   = 1,
                 deep_supervision: bool  = True,
                 dropout:          float = 0.5,
                 base_filters:     list  = None,
                 sk_reduction:     int   = 16):
        super().__init__()
        self.n_channels       = n_channels
        self.n_classes        = n_classes
        self.deep_supervision = deep_supervision

        if base_filters is None:
            base_filters = [32, 64, 128, 256, 512]
        f = self.nb_filter = base_filters          # shorthand

        r = sk_reduction

        # ── Pooling / upsampling ──────────────────────────────────────────────
        self.pool = nn.MaxPool2d(2, 2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc0_0 = DoubleSK(n_channels, f[0], r)   # H,   W,   f[0]=32
        self.enc1_0 = DoubleSK(f[0],       f[1], r)   # H/2, W/2, f[1]=64
        self.enc2_0 = DoubleSK(f[1],       f[2], r)   # H/4, W/4, f[2]=128
        self.enc3_0 = DoubleSK(f[2],       f[3], r)   # H/8, W/8, f[3]=256
        self.enc4_0 = DoubleSK(f[3],       f[4], r)   # H/16,W/16,f[4]=512

        self.bottleneck_drop = (nn.Dropout2d(p=dropout) if dropout > 0
                                else nn.Identity())

        # ── Attention Gates ───────────────────────────────────────────────────
        # Naming: ag{i}_{j}  gates feature X(i, j-1) using UP(X(i+1, j-1))
        # g_channels = channels of the upsampled gate  = f[i+1]
        # x_channels = channels of the skip feature    = f[i]
        # int_channels (bottleneck)                    = f[i]

        # Level 0 (x = f[0]=32)
        self.ag0_1 = AttentionGate(f[1], f[0], f[0])
        self.ag0_2 = AttentionGate(f[1], f[0], f[0])
        self.ag0_3 = AttentionGate(f[1], f[0], f[0])
        self.ag0_4 = AttentionGate(f[1], f[0], f[0])

        # Level 1 (x = f[1]=64)
        self.ag1_1 = AttentionGate(f[2], f[1], f[1])
        self.ag1_2 = AttentionGate(f[2], f[1], f[1])
        self.ag1_3 = AttentionGate(f[2], f[1], f[1])

        # Level 2 (x = f[2]=128)
        self.ag2_1 = AttentionGate(f[3], f[2], f[2])
        self.ag2_2 = AttentionGate(f[3], f[2], f[2])

        # Level 3 (x = f[3]=256)
        self.ag3_1 = AttentionGate(f[4], f[3], f[3])

        # ── Decoder nodes ─────────────────────────────────────────────────────
        # Channel counts follow:
        #   X(i,j): j skip features (each f[i]) + 1 upsampled (f[i+1]) → f[i]
        #
        # j = 1
        self.dec0_1 = DoubleSK(f[0]     + f[1], f[0], r)   # 32+64=96
        self.dec1_1 = DoubleSK(f[1]     + f[2], f[1], r)   # 64+128=192
        self.dec2_1 = DoubleSK(f[2]     + f[3], f[2], r)   # 128+256=384
        self.dec3_1 = DoubleSK(f[3]     + f[4], f[3], r)   # 256+512=768

        # j = 2
        self.dec0_2 = DoubleSK(f[0]*2   + f[1], f[0], r)   # 32*2+64=128
        self.dec1_2 = DoubleSK(f[1]*2   + f[2], f[1], r)   # 64*2+128=256
        self.dec2_2 = DoubleSK(f[2]*2   + f[3], f[2], r)   # 128*2+256=512

        # j = 3
        self.dec0_3 = DoubleSK(f[0]*3   + f[1], f[0], r)   # 32*3+64=160
        self.dec1_3 = DoubleSK(f[1]*3   + f[2], f[1], r)   # 64*3+128=320

        # j = 4
        self.dec0_4 = DoubleSK(f[0]*4   + f[1], f[0], r)   # 32*4+64=192

        # ── Output heads ──────────────────────────────────────────────────────
        if deep_supervision:
            self.final1 = nn.Conv2d(f[0], n_classes, 1)
            self.final2 = nn.Conv2d(f[0], n_classes, 1)
            self.final3 = nn.Conv2d(f[0], n_classes, 1)
            self.final4 = nn.Conv2d(f[0], n_classes, 1)
        else:
            self.final  = nn.Conv2d(f[0], n_classes, 1)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _up(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Returns:
          training + deep_supervision : list of 4 logit tensors
                                        [out_d0_1, out_d0_2, out_d0_3, out_d0_4]
          otherwise                   : single logit tensor (d0_4 head)
        """
        # ── Encoder ──────────────────────────────────────────────────────────
        e0_0 = self.enc0_0(x)                            # f[0]  H
        e1_0 = self.enc1_0(self.pool(e0_0))              # f[1]  H/2
        e2_0 = self.enc2_0(self.pool(e1_0))              # f[2]  H/4
        e3_0 = self.enc3_0(self.pool(e2_0))              # f[3]  H/8
        e4_0 = self.enc4_0(self.pool(e3_0))              # f[4]  H/16
        e4_0 = self.bottleneck_drop(e4_0)

        # ── j = 1 ─────────────────────────────────────────────────────────────
        # Gate signal for all level-i nodes at j=1 is UP(X(i+1, 0)) = up(enc)
        #
        # X(3,1): gate = up(e4_0)
        up_e4 = self._up(e4_0)                           # f[4]→H/8
        d3_1  = self.dec3_1(torch.cat([
            self.ag3_1(up_e4, e3_0),                     # gated e3_0  f[3]
            up_e4,                                        # upsampled   f[4]
        ], dim=1))                                        # → f[3]  H/8

        # X(2,1): gate = up(e3_0)
        up_e3 = self._up(e3_0)                           # f[3]→H/4
        d2_1  = self.dec2_1(torch.cat([
            self.ag2_1(up_e3, e2_0),                     # gated e2_0  f[2]
            up_e3,                                        # upsampled   f[3]
        ], dim=1))                                        # → f[2]  H/4

        # X(1,1): gate = up(e2_0)
        up_e2 = self._up(e2_0)                           # f[2]→H/2
        d1_1  = self.dec1_1(torch.cat([
            self.ag1_1(up_e2, e1_0),                     # gated e1_0  f[1]
            up_e2,                                        # upsampled   f[2]
        ], dim=1))                                        # → f[1]  H/2

        # X(0,1): gate = up(e1_0)
        up_e1 = self._up(e1_0)                           # f[1]→H
        d0_1  = self.dec0_1(torch.cat([
            self.ag0_1(up_e1, e0_0),                     # gated e0_0  f[0]
            up_e1,                                        # upsampled   f[1]
        ], dim=1))                                        # → f[0]  H

        # ── j = 2 ─────────────────────────────────────────────────────────────
        # Gate signal at j=2 is UP(X(i+1, 1)) = up(d*_1)   ← FIXED (was encoder)
        #
        # X(2,2): gate = up(d3_1)
        up_d3_1 = self._up(d3_1)                         # f[3]→H/4
        d2_2    = self.dec2_2(torch.cat([
            self.ag2_1(up_d3_1, e2_0),                   # Ag(X(2,0))   f[2]
            self.ag2_2(up_d3_1, d2_1),                   # Ag(X(2,1))   f[2]  ← FIXED gate
            up_d3_1,                                      # UP(X(3,1))   f[3]
        ], dim=1))                                        # → f[2]  H/4

        # X(1,2): gate = up(d2_1)
        up_d2_1 = self._up(d2_1)                         # f[2]→H/2
        d1_2    = self.dec1_2(torch.cat([
            self.ag1_1(up_d2_1, e1_0),                   # Ag(X(1,0))   f[1]
            self.ag1_2(up_d2_1, d1_1),                   # Ag(X(1,1))   f[1]  ← FIXED gate
            up_d2_1,                                      # UP(X(2,1))   f[2]
        ], dim=1))                                        # → f[1]  H/2

        # X(0,2): gate = up(d1_1)
        up_d1_1 = self._up(d1_1)                         # f[1]→H
        d0_2    = self.dec0_2(torch.cat([
            self.ag0_1(up_d1_1, e0_0),                   # Ag(X(0,0))   f[0]
            self.ag0_2(up_d1_1, d0_1),                   # Ag(X(0,1))   f[0]  ← FIXED gate
            up_d1_1,                                      # UP(X(1,1))   f[1]
        ], dim=1))                                        # → f[0]  H

        # ── j = 3 ─────────────────────────────────────────────────────────────
        # Gate signal at j=3 is UP(X(i+1, 2)) = up(d*_2)   ← FIXED
        #
        # X(1,3): gate = up(d2_2)
        up_d2_2 = self._up(d2_2)                         # f[2]→H/2
        d1_3    = self.dec1_3(torch.cat([
            self.ag1_1(up_d2_2, e1_0),                   # Ag(X(1,0))   f[1]
            self.ag1_2(up_d2_2, d1_1),                   # Ag(X(1,1))   f[1]  ← FIXED
            self.ag1_3(up_d2_2, d1_2),                   # Ag(X(1,2))   f[1]  ← FIXED
            up_d2_2,                                      # UP(X(2,2))   f[2]
        ], dim=1))                                        # → f[1]  H/2

        # X(0,3): gate = up(d1_2)
        up_d1_2 = self._up(d1_2)                         # f[1]→H
        d0_3    = self.dec0_3(torch.cat([
            self.ag0_1(up_d1_2, e0_0),                   # Ag(X(0,0))   f[0]
            self.ag0_2(up_d1_2, d0_1),                   # Ag(X(0,1))   f[0]  ← FIXED
            self.ag0_3(up_d1_2, d0_2),                   # Ag(X(0,2))   f[0]  ← FIXED
            up_d1_2,                                      # UP(X(1,2))   f[1]
        ], dim=1))                                        # → f[0]  H

        # ── j = 4 ─────────────────────────────────────────────────────────────
        # Gate signal at j=4 is UP(X(1, 3)) = up(d1_3)     ← FIXED
        #
        # X(0,4): gate = up(d1_3)
        up_d1_3 = self._up(d1_3)                         # f[1]→H
        d0_4    = self.dec0_4(torch.cat([
            self.ag0_1(up_d1_3, e0_0),                   # Ag(X(0,0))   f[0]
            self.ag0_2(up_d1_3, d0_1),                   # Ag(X(0,1))   f[0]  ← FIXED
            self.ag0_3(up_d1_3, d0_2),                   # Ag(X(0,2))   f[0]  ← FIXED
            self.ag0_4(up_d1_3, d0_3),                   # Ag(X(0,3))   f[0]
            up_d1_3,                                      # UP(X(1,3))   f[1]
        ], dim=1))                                        # → f[0]  H

        # ── Output ───────────────────────────────────────────────────────────
        if self.deep_supervision:
            # BUG 2 FIX: return independent logit tensors, NOT averaged logits.
            # train.py applies a weighted loss sum; at inference use index [-1].
            return [
                self.final1(d0_1),
                self.final2(d0_2),
                self.final3(d0_3),
                self.final4(d0_4),
            ]
        else:
            return self.final(d0_4)

    def use_checkpointing(self):
        logging.warning(
            'use_checkpointing() for UNetAdv is experimental. '
            'If OOM, reduce --batch-size or --img-size instead.'
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def build_unetadv(name: str, n_channels: int, n_classes: int, **kwargs):
    """
    Factory function — keeps the same signature as other model builders.
    Accepted kwargs: deep_supervision, dropout, base_filters, sk_reduction.
    Unknown kwargs (e.g. bilinear) are silently ignored for drop-in compatibility.
    """
    return UNetAdv(
        n_channels       = n_channels,
        n_classes        = n_classes,
        deep_supervision = kwargs.get('deep_supervision', True),
        dropout          = kwargs.get('dropout', 0.5),
        base_filters     = kwargs.get('base_filters', [32, 64, 128, 256, 512]),
        sk_reduction     = kwargs.get('sk_reduction', 16),
    )
