# G:/VETNet_pilot/models/backbone/blocks.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn

from models.backbone.mdta_strategy import MDTA
from models.backbone.gdfn_volterra import GDFN
from models.backbone.volterra_layer import VolterraLayer2D

class VETBlock(nn.Module):
    """
    VET Block = LayerNorm + MDTA + Volterra + LayerNorm + GDFN
    Phase1에서는 Strategy 없이 순수한 Restormer+Volterra 블록.
    """
    def __init__(self, dim, num_heads=8, volterra_rank=4, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.dim = dim

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDTA(dim=dim, num_heads=num_heads, bias=bias)
        self.volterra = VolterraLayer2D(in_channels=dim, out_channels=dim,
                                        kernel_size=3, rank=volterra_rank, bias=bias)
        self.norm2 = nn.LayerNorm(dim)
        self.gdfn = GDFN(dim=dim, expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape

        # ---- Global branch: MDTA + Volterra ----
        x_ln = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x_ln = self.norm1(x_ln)
        x_ln = x_ln.reshape(b, h, w, c).permute(0, 3, 1, 2)

        g = self.attn(x_ln)
        g = self.volterra(g)
        x = x + g

        # ---- Local branch: GDFN ----
        x_ln2 = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x_ln2 = self.norm2(x_ln2)
        x_ln2 = x_ln2.reshape(b, h, w, c).permute(0, 3, 1, 2)

        x = x + self.gdfn(x_ln2)
        return x


# ----------------- 테스트 코드 ----------------- #
if __name__ == "__main__":
    x = torch.randn(1, 48, 64, 64)
    block = VETBlock(dim=48, num_heads=8, volterra_rank=4)
    y = block(x)
    print("=== VETBlock Test ===")
    print(f"Input  Shape: {x.shape}")
    print(f"Output Shape: {y.shape}")
