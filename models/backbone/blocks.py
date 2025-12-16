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
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

# ------------------------------------------------------------
# ROOT 경로 세팅
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../models/backbone
ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))      # .../VETNet_pilot
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[blocks] import OK")

from models.backbone.mdta_strategy import MDTA, MDTAWithStrategy
from models.backbone.gdfn_volterra import GDFN
from models.backbone.volterra_layer import VolterraLayer2D


class VETBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        volterra_rank: int = 4,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim

        # LN은 Restormer 스타일로 (B,H,W,C) 토큰에 적용
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # ✅ Phase-1 attention (ckpt 호환용)
        self.attn_conv = MDTA(dim=dim, num_heads=num_heads, bias=bias)

        # ✅ Phase-2 attention (strategy token steering)
        # - v2(OOM-safe) 구현이 mdta_strategy.py에 들어가 있어야 함
        self.attn_tok = MDTAWithStrategy(dim=dim, num_heads=num_heads, bias=bias)

        self.volterra = VolterraLayer2D(
            in_channels=dim, out_channels=dim,
            kernel_size=3, rank=volterra_rank, bias=bias
        )
        self.gdfn = GDFN(dim=dim, expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x: torch.Tensor, strategy_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:

        b, c, h, w = x.shape
        assert c == self.dim, f"[VETBlock] channel mismatch: x.C={c} vs dim={self.dim}"

        # -------------------------
        # Global branch: (LN -> Attention -> Volterra) + residual
        # -------------------------
        # LN 준비: (B,C,H,W) -> (B,HW,C) -> LN -> 다시 (B,C,H,W)
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x_flat = self.norm1(x_flat)

        if strategy_tokens is None:
            # ✅ Phase-1: conv-MDTA
            x_ln = x_flat.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            g = self.attn_conv(x_ln)
        else:
            # ✅ Phase-2: token-MDTAWithStrategy (steering)
            # x_flat: (B, HW, C) / strategy_tokens: (B,K,C)
            g_flat = self.attn_tok(x_flat, S_tokens=strategy_tokens)  # (B, HW, C)
            g = g_flat.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        g = self.volterra(g)
        x = x + g

        # -------------------------
        # Local branch: (LN -> GDFN) + residual
        # -------------------------
        x2_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x2_flat = self.norm2(x2_flat)
        x2 = x2_flat.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        x = x + self.gdfn(x2)
        return x


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print("\n[blocks.py] Self-test 시작")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[blocks.py] Device =", device)

    B, C, H, W = 2, 64, 32, 32
    K = 4
    x = torch.randn(B, C, H, W, device=device)
    S = torch.randn(B, K, C, device=device)

    blk = VETBlock(dim=C, num_heads=8, volterra_rank=4).to(device)
    blk.eval()

    with torch.no_grad():
        y_no = blk(x, strategy_tokens=None)
        y_yes = blk(x, strategy_tokens=S)

    diff = (y_yes - y_no).abs().mean().item()
    print("Input shape :", x.shape)
    print("Output shape:", y_yes.shape)
    print(f"mean(|withS - noS|) = {diff:.6f}")
    if diff > 0:
        print("✅ Strategy Token이 실제로 attention 경로를 바꿉니다.")
    else:
        print("❌ diff=0 → 아직 strategy가 반영되지 않습니다.")
    print("[blocks.py] Self-test 완료\n")
