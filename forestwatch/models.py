"""Siamese U-Net for bitemporal change detection (Caye Daudt et al. 2018 + Ronneberger et al. 2015).

mode="siamese": ONE shared-weight encoder processes BEFORE and AFTER; at every level the absolute
                feature difference |f_before - f_after| is passed to the U-Net decoder (the report's design).
mode="diff"   : "UNet-diff" ablation (Isaienkov et al. 2021): the encoder sees the pixel difference
                image (after - before) directly, i.e. a plain U-Net on the difference.
"""
import torch
import torch.nn as nn


def block(cin: int, cout: int) -> nn.Sequential:
    """Two 3x3 conv + BatchNorm + ReLU layers (the standard U-Net block)."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SiameseUNet(nn.Module):
    def __init__(self, in_ch: int = 4, base: int = 32, levels: int = 3, mode: str = "siamese"):
        super().__init__()
        assert mode in ("siamese", "diff") and 2 <= levels <= 5
        self.in_ch, self.base, self.levels, self.mode = in_ch, base, levels, mode
        w = [base * 2 ** i for i in range(levels)]                     # channel width per level
        self.enc = nn.ModuleList([block(in_ch if i == 0 else w[i - 1], w[i]) for i in range(levels)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[i + 1], w[i], 2, stride=2) for i in range(levels - 1)])
        self.dec = nn.ModuleList([block(2 * w[i], w[i]) for i in range(levels - 1)])
        self.head = nn.Conv2d(w[0], 1, 1)

    @property
    def divisor(self) -> int:
        """Input height/width must be a multiple of this."""
        return 2 ** (self.levels - 1)

    def encode(self, x):
        """Shared encoder: returns the feature map of every level."""
        feats = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else self.pool(x))
            feats.append(x)
        return feats

    def forward(self, before, after):
        if self.mode == "siamese":
            fa, fb = self.encode(before), self.encode(after)          # same weights for both dates
            d = [torch.abs(p - q) for p, q in zip(fa, fb)]           # feature difference = change
        else:
            d = self.encode(after - before)
        x = d[-1]
        for i in reversed(range(self.levels - 1)):
            x = self.dec[i](torch.cat([self.up[i](x), d[i]], dim=1))
        return self.head(x)                                          # logits, (B,1,H,W)


class BCEDiceLoss(nn.Module):
    """Weighted BCE (cleared pixels are rare) + soft Dice, as in Isaienkov et al. and Aashika et al."""

    def __init__(self, pos_weight: float = 5.0, dice_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pw", torch.tensor([pos_weight]))
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pw)
        p = torch.sigmoid(logits)
        inter = (p * target).sum()
        dice = 1 - (2 * inter + 1) / (p.sum() + target.sum() + 1)
        return bce + self.dice_weight * dice
