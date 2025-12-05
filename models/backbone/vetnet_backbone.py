# G:/VETNet_pilot/models/backbone/vetnet_backbone.py
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
    """
    Phase1용 VETNet Backbone (Restormer + Volterra 기반 U-Net)
    - 입력: (B, 3, H, W)
    - 출력: (B, 3, H, W)
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        heads=(1, 2, 4, 8),
        volterra_rank=4,
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
        """
        업샘플된 텐서와 skip 연결 텐서의 spatial size가 다를 경우
        bilinear interpolate로 맞추고 더해줌.
        """
        if up_tensor.shape[-2:] != skip_tensor.shape[-2:]:
            up_tensor = F.interpolate(up_tensor, size=skip_tensor.shape[-2:],
                                      mode="bilinear", align_corners=False)
        return up_tensor + skip_tensor

    def forward(self, x):
        """
        x: (B, 3, H, W)
        """
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
