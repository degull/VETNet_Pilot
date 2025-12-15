""" # G:/VETNet_pilot/models/backbone/blocks.py
# # phase -1 (vetnet backbone)
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn

from models.backbone.mdta_strategy import MDTA
from models.backbone.gdfn_volterra import GDFN
from models.backbone.volterra_layer import VolterraLayer2D

class VETBlock(nn.Module):
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
 """

# phase -2 (control bridge)

# G:/VETNet_pilot/models/backbone/blocks.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn

from models.backbone.mdta_strategy import MDTAWithStrategy
from models.backbone.gdfn_volterra import GDFN
from models.backbone.volterra_layer import VolterraLayer2D


class VETBlock(nn.Module):
    """
    Phase-2 대응 VET Block

    - Strategy Token은 MDTAWithStrategy로 전달만 함
    - attention 조향 논리는 mdta_strategy.py에만 존재
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        volterra_rank=4,
        ffn_expansion_factor=2.66,
        bias=False,
    ):
        super().__init__()
        self.dim = dim

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDTAWithStrategy(dim=dim, num_heads=num_heads, bias=bias)

        self.volterra = VolterraLayer2D(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            rank=volterra_rank,
            bias=bias,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.gdfn = GDFN(
            dim=dim,
            expansion_factor=ffn_expansion_factor,
            bias=bias,
        )

    def forward(self, x, strategy_tokens=None):
        """
        x: (B, C, H, W)
        strategy_tokens: (B, K, C) or None
        """
        B, C, H, W = x.shape
        N = H * W

        # ---------------- Attention branch ----------------
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
        x_flat = self.norm1(x_flat)

        # 🔥 핵심: Strategy Token 전달
        attn_out = self.attn(x_flat, S_tokens=strategy_tokens)

        attn_out = attn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        attn_out = self.volterra(attn_out)

        x = x + attn_out

        # ---------------- FFN branch ----------------
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
        x_flat = self.norm2(x_flat)
        x_ffn = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x = x + self.gdfn(x_ffn)
        return x


# =========================================================
# Self-test
# =========================================================
if __name__ == "__main__":
    print("\n[blocks.py] Self-test 시작")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, H, W = 2, 64, 32, 32
    K = 4

    x = torch.randn(B, C, H, W, device=device)
    S = torch.randn(B, K, C, device=device)

    blk = VETBlock(dim=C, num_heads=4).to(device)
    blk.eval()

    with torch.no_grad():
        y_noS = blk(x, strategy_tokens=None)
        y_S   = blk(x, strategy_tokens=S)

    diff = (y_noS - y_S).abs().mean().item()

    print(f"Input shape : {x.shape}")
    print(f"Output shape: {y_S.shape}")
    print(f"mean(|withS - noS|) = {diff:.6f}")
    print("→ 0이 아니면 Strategy Token이 실제로 영향을 줌")

    print("[blocks.py] Self-test 완료\n")
