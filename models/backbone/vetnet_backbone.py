# G:/VETNet_pilot/models/backbone/vetnet_backbone.py
# phase -1 (vetnet backbone)
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.blocks import VETBlock

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(in_channels, in_channels * 2,
                              kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x1 conv + PixelShuffle(2) => x2 업샘플
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class EncoderStage(nn.Module):
    def __init__(self, dim, depth, num_heads, volterra_rank, ffn_expansion_factor, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            VETBlock(dim=dim,
                     num_heads=num_heads,
                     volterra_rank=volterra_rank,
                     ffn_expansion_factor=ffn_expansion_factor,
                     bias=bias)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, dim, depth, num_heads, volterra_rank, ffn_expansion_factor, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            VETBlock(dim=dim,
                     num_heads=num_heads,
                     volterra_rank=volterra_rank,
                     ffn_expansion_factor=ffn_expansion_factor,
                     bias=bias)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class VETNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        heads=(1, 2, 4, 8),
        volterra_rank=2,
        ffn_expansion_factor=2.66,
        bias=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim

        # Shallow feature extraction
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)

        # ----------------- Encoder -----------------
        self.encoder1 = EncoderStage(dim=dim,
                                     depth=num_blocks[0],
                                     num_heads=heads[0],
                                     volterra_rank=volterra_rank,
                                     ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias)
        self.down1 = Downsample(dim)

        self.encoder2 = EncoderStage(dim=dim * 2,
                                     depth=num_blocks[1],
                                     num_heads=heads[1],
                                     volterra_rank=volterra_rank,
                                     ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias)
        self.down2 = Downsample(dim * 2)

        self.encoder3 = EncoderStage(dim=dim * 4,
                                     depth=num_blocks[2],
                                     num_heads=heads[2],
                                     volterra_rank=volterra_rank,
                                     ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias)
        self.down3 = Downsample(dim * 4)

        # ----------------- Bottleneck -----------------
        self.latent = EncoderStage(dim=dim * 8,
                                   depth=num_blocks[3],
                                   num_heads=heads[3],
                                   volterra_rank=volterra_rank,
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias)

        # ----------------- Decoder -----------------
        self.up3 = Upsample(dim * 8, dim * 4)
        self.decoder3 = DecoderStage(dim=dim * 4,
                                     depth=num_blocks[2],
                                     num_heads=heads[2],
                                     volterra_rank=volterra_rank,
                                     ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias)

        self.up2 = Upsample(dim * 4, dim * 2)
        self.decoder2 = DecoderStage(dim=dim * 2,
                                     depth=num_blocks[1],
                                     num_heads=heads[1],
                                     volterra_rank=volterra_rank,
                                     ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias)

        self.up1 = Upsample(dim * 2, dim)
        self.decoder1 = DecoderStage(dim=dim,
                                     depth=num_blocks[0],
                                     num_heads=heads[0],
                                     volterra_rank=volterra_rank,
                                     ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias)

        # ----------------- Refinement & Output -----------------
        self.refinement = EncoderStage(dim=dim,
                                       depth=num_blocks[0],
                                       num_heads=heads[0],
                                       volterra_rank=volterra_rank,
                                       ffn_expansion_factor=ffn_expansion_factor,
                                       bias=bias)
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _pad_and_add(up_tensor, skip_tensor):
        if up_tensor.shape[-2:] != skip_tensor.shape[-2:]:
            up_tensor = F.interpolate(up_tensor, size=skip_tensor.shape[-2:],
                                      mode="bilinear", align_corners=False)
        return up_tensor + skip_tensor

    def forward(self, x):
        x_embed = self.patch_embed(x)

        # Encoder
        e1 = self.encoder1(x_embed)          # (B, dim, H, W)
        e2 = self.encoder2(self.down1(e1))   # (B, 2dim, H/2, W/2)
        e3 = self.encoder3(self.down2(e2))   # (B, 4dim, H/4, W/4)
        b  = self.latent(self.down3(e3))     # (B, 8dim, H/8, W/8)

        # Decoder
        d3 = self._pad_and_add(self.up3(b), e3)
        d3 = self.decoder3(d3)

        d2 = self._pad_and_add(self.up2(d3), e2)
        d2 = self.decoder2(d2)

        d1 = self._pad_and_add(self.up1(d2), e1)
        d1 = self.decoder1(d1)

        # Refinement + Residual
        r = self.refinement(d1)
        out = self.output(r + x_embed)
        return out


# ----------------- 테스트 코드 ----------------- #
if __name__ == "__main__":
    print("=== VETNetBackbone Phase1 Test ===")
    B, C, H, W = 1, 3, 256, 256
    x = torch.randn(B, C, H, W)
    model = VETNetBackbone(in_channels=3, out_channels=3, dim=48)
    y = model(x)
    print(f"Input  Shape: {x.shape}")
    print(f"Output Shape: {y.shape}")
    assert y.shape == x.shape, "입력과 출력 해상도가 일치해야 합니다!"
    print(">> VETNetBackbone Phase1: OK (입력/출력 shape 일치)")


# phase -2 (control bridge)
# G:/VETNet_pilot/models/backbone/vetnet_backbone.py
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.blocks import VETBlock


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(in_channels, in_channels * 2, 3, 2, 1)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class EncoderStage(nn.Module):
    def __init__(self, dim, depth, num_heads, volterra_rank, ffn_expansion_factor, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            VETBlock(
                dim=dim,
                num_heads=num_heads,
                volterra_rank=volterra_rank,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
            )
            for _ in range(depth)
        ])

    def forward(self, x, strategy_tokens=None):
        for blk in self.blocks:
            x = blk(x, strategy_tokens=strategy_tokens)
        return x


class DecoderStage(EncoderStage):
    pass


class VETNetBackbone(nn.Module):


    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        dim=64,
        num_blocks=(4, 6, 6, 8),
        heads=(1, 2, 4, 8),
        volterra_rank=4,
        ffn_expansion_factor=2.66,
        bias=False,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, dim, 3, 1, 1)

        # Encoder
        self.encoder1 = EncoderStage(dim,     num_blocks[0], heads[0], volterra_rank, ffn_expansion_factor, bias)
        self.down1    = Downsample(dim)

        self.encoder2 = EncoderStage(dim * 2, num_blocks[1], heads[1], volterra_rank, ffn_expansion_factor, bias)
        self.down2    = Downsample(dim * 2)

        self.encoder3 = EncoderStage(dim * 4, num_blocks[2], heads[2], volterra_rank, ffn_expansion_factor, bias)
        self.down3    = Downsample(dim * 4)

        self.latent   = EncoderStage(dim * 8, num_blocks[3], heads[3], volterra_rank, ffn_expansion_factor, bias)

        # Decoder
        self.up3 = Upsample(dim * 8, dim * 4)
        self.decoder3 = DecoderStage(dim * 4, num_blocks[2], heads[2], volterra_rank, ffn_expansion_factor, bias)

        self.up2 = Upsample(dim * 4, dim * 2)
        self.decoder2 = DecoderStage(dim * 2, num_blocks[1], heads[1], volterra_rank, ffn_expansion_factor, bias)

        self.up1 = Upsample(dim * 2, dim)
        self.decoder1 = DecoderStage(dim, num_blocks[0], heads[0], volterra_rank, ffn_expansion_factor, bias)

        self.refinement = EncoderStage(dim, num_blocks[0], heads[0], volterra_rank, ffn_expansion_factor, bias)
        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1)

    def _add(self, x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return x + skip

    def forward(self, x, strategy_tokens=None):

        strategy_tokens = strategy_tokens or {}

        x = self.patch_embed(x)

        e1 = self.encoder1(x, strategy_tokens.get("stage1"))
        e2 = self.encoder2(self.down1(e1), strategy_tokens.get("stage2"))
        e3 = self.encoder3(self.down2(e2), strategy_tokens.get("stage3"))
        b  = self.latent(self.down3(e3), strategy_tokens.get("stage4"))

        d3 = self.decoder3(self._add(self.up3(b), e3), strategy_tokens.get("stage3"))
        d2 = self.decoder2(self._add(self.up2(d3), e2), strategy_tokens.get("stage2"))
        d1 = self.decoder1(self._add(self.up1(d2), e1), strategy_tokens.get("stage1"))

        r = self.refinement(d1, strategy_tokens.get("stage1"))
        return self.output(r + x)


# =========================================================
# Self-test
# =========================================================
if __name__ == "__main__":
    print("\n[vetnet_backbone.py] Phase-2 Self-test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VETNetBackbone().to(device).eval()

    B, H, W = 2, 128, 128
    x = torch.randn(B, 3, H, W, device=device)

    strategy_tokens = {
        "stage1": torch.randn(B, 4, 64, device=device),
        "stage2": torch.randn(B, 4, 128, device=device),
        "stage3": torch.randn(B, 4, 256, device=device),
        "stage4": torch.randn(B, 4, 512, device=device),
    }

    with torch.no_grad():
        y = model(x, strategy_tokens=strategy_tokens)

    print("Input :", x.shape)
    print("Output:", y.shape)
    print("[vetnet_backbone.py] Phase-2 Self-test 완료\n")


""" 

# phase 2
import os, sys
from typing import Optional, Dict, TYPE_CHECKING

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.blocks import VETBlock

# ============================================================
# TYPE CHECKING ONLY (🔥 핵심 수정 포인트)
# ============================================================
if TYPE_CHECKING:
    from models.bridge.strategy_router import StrategyRouter


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(in_channels, in_channels * 2,
                              kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class EncoderStage(nn.Module):
    def __init__(self, dim, depth, num_heads, volterra_rank, ffn_expansion_factor, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            VETBlock(
                dim=dim,
                num_heads=num_heads,
                volterra_rank=volterra_rank,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias
            )
            for _ in range(depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        strategy_tokens: Optional[torch.Tensor] = None,
        router: Optional["StrategyRouter"] = None,
        stage_name: Optional[str] = None,
    ) -> torch.Tensor:
        for blk in self.blocks:
            routed = router(x, strategy_tokens, stage_name) if router is not None else strategy_tokens
            x = blk(x, strategy_tokens=routed)
        return x


class DecoderStage(nn.Module):
    def __init__(self, dim, depth, num_heads, volterra_rank, ffn_expansion_factor, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            VETBlock(
                dim=dim,
                num_heads=num_heads,
                volterra_rank=volterra_rank,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias
            )
            for _ in range(depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        strategy_tokens: Optional[torch.Tensor] = None,
        router: Optional["StrategyRouter"] = None,
        stage_name: Optional[str] = None,
    ) -> torch.Tensor:
        for blk in self.blocks:
            routed = router(x, strategy_tokens, stage_name) if router is not None else strategy_tokens
            x = blk(x, strategy_tokens=routed)
        return x


class VETNetBackbone(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        dim=64,
        num_blocks=(4, 6, 6, 8),
        heads=(1, 2, 4, 8),
        volterra_rank=4,
        ffn_expansion_factor=2.66,
        bias=False,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, dim, 3, 1, 1)

        self.encoder1 = EncoderStage(dim, num_blocks[0], heads[0], volterra_rank, ffn_expansion_factor, bias)
        self.down1 = Downsample(dim)

        self.encoder2 = EncoderStage(dim*2, num_blocks[1], heads[1], volterra_rank, ffn_expansion_factor, bias)
        self.down2 = Downsample(dim*2)

        self.encoder3 = EncoderStage(dim*4, num_blocks[2], heads[2], volterra_rank, ffn_expansion_factor, bias)
        self.down3 = Downsample(dim*4)

        self.latent = EncoderStage(dim*8, num_blocks[3], heads[3], volterra_rank, ffn_expansion_factor, bias)

        self.up3 = Upsample(dim*8, dim*4)
        self.decoder3 = DecoderStage(dim*4, num_blocks[2], heads[2], volterra_rank, ffn_expansion_factor, bias)

        self.up2 = Upsample(dim*4, dim*2)
        self.decoder2 = DecoderStage(dim*2, num_blocks[1], heads[1], volterra_rank, ffn_expansion_factor, bias)

        self.up1 = Upsample(dim*2, dim)
        self.decoder1 = DecoderStage(dim, num_blocks[0], heads[0], volterra_rank, ffn_expansion_factor, bias)

        self.refinement = EncoderStage(dim, num_blocks[0], heads[0], volterra_rank, ffn_expansion_factor, bias)
        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1)

    @staticmethod
    def _pad_and_add(up, skip):
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return up + skip

    def forward(
        self,
        x: torch.Tensor,
        strategy_tokens: Optional[Dict[str, torch.Tensor]] = None,
        router: Optional["StrategyRouter"] = None,
    ) -> torch.Tensor:

        if strategy_tokens is None:
            strategy_tokens = {}

        x0 = self.patch_embed(x)

        e1 = self.encoder1(x0, strategy_tokens.get("stage1"), router, "stage1")
        e2 = self.encoder2(self.down1(e1), strategy_tokens.get("stage2"), router, "stage2")
        e3 = self.encoder3(self.down2(e2), strategy_tokens.get("stage3"), router, "stage3")
        b  = self.latent(self.down3(e3), strategy_tokens.get("stage4"), router, "stage4")

        d3 = self.decoder3(self._pad_and_add(self.up3(b), e3))
        d2 = self.decoder2(self._pad_and_add(self.up2(d3), e2))
        d1 = self.decoder1(self._pad_and_add(self.up1(d2), e1))

        r = self.refinement(d1)
        return self.output(r + x0)


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print("[vetnet_backbone] import & forward OK")
