# E:/VETNet_Pilot/models/backbone/vetnet_backbone.py
# phase-1 (vetnet backbone) + phase-2 gating support
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
    VETNetBackbone
    - Phase-1: standard restoration backbone
    - Phase-2: supports stage-wise scalar gates (g_stage)

    Stage-wise gating definition (stable residual scaling):
        y = x + g * (F(x) - x)   where g in [0,1] (or [g_min,1])

    We apply gates to 8 "macro stages":
      0: encoder1 output
      1: encoder2 output
      2: encoder3 output
      3: latent (bottleneck) output
      4: decoder3 output
      5: decoder2 output
      6: decoder1 output
      7: refinement output

    Note: This adds NO learnable parameters to backbone.
          Phase-1 checkpoints remain strictly loadable.
    """

    NUM_MACRO_STAGES = 8

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

    @staticmethod
    def _apply_gate(x_in: torch.Tensor, x_out: torch.Tensor, g: torch.Tensor):
        """
        Stable residual scaling:
            y = x_in + g * (x_out - x_in)
        """
        return x_in + g * (x_out - x_in)

    def _normalize_gates(self, g_stage, batch_size, device, dtype):
        """
        Accepts:
          - None
          - Tensor [B, S]
          - Tensor [S]
          - list/tuple length S
        Returns:
          - Tensor [B, S] on correct device/dtype
        """
        if g_stage is None:
            return None

        if isinstance(g_stage, (list, tuple)):
            g_stage = torch.tensor(g_stage, device=device, dtype=dtype)

        if not torch.is_tensor(g_stage):
            raise TypeError(f"g_stage must be Tensor/list/tuple/None, got {type(g_stage)}")

        if g_stage.dim() == 1:
            # [S] -> [B,S]
            g_stage = g_stage.view(1, -1).repeat(batch_size, 1)
        elif g_stage.dim() == 2:
            # [B,S]
            pass
        else:
            raise ValueError(f"g_stage must be [S] or [B,S], got shape={tuple(g_stage.shape)}")

        if g_stage.size(1) != self.NUM_MACRO_STAGES:
            raise ValueError(
                f"Expected g_stage with S={self.NUM_MACRO_STAGES}, "
                f"but got {g_stage.size(1)}. "
                f"(Use GateController(num_stages={self.NUM_MACRO_STAGES}))"
            )

        return g_stage.to(device=device, dtype=dtype)

    def forward(self, x, g_stage=None):
        """
        Args:
            x: input image [B,3,H,W]
            g_stage: optional stage-wise gates
                     shape [B,8] or [8], gates in [0,1] or [g_min,1]

        Returns:
            out: restored image [B,3,H,W]
        """
        B = x.size(0)
        device = x.device
        dtype = x.dtype
        g_stage = self._normalize_gates(g_stage, B, device, dtype)

        x_embed = self.patch_embed(x)

        # ---------------- Encoder ----------------
        # stage 0: encoder1
        x_in = x_embed
        e1 = self.encoder1(x_in)
        if g_stage is not None:
            g0 = g_stage[:, 0].view(B, 1, 1, 1)
            e1 = self._apply_gate(x_in, e1, g0)

        # stage 1: encoder2
        x_in = self.down1(e1)
        e2 = self.encoder2(x_in)
        if g_stage is not None:
            g1 = g_stage[:, 1].view(B, 1, 1, 1)
            e2 = self._apply_gate(x_in, e2, g1)

        # stage 2: encoder3
        x_in = self.down2(e2)
        e3 = self.encoder3(x_in)
        if g_stage is not None:
            g2 = g_stage[:, 2].view(B, 1, 1, 1)
            e3 = self._apply_gate(x_in, e3, g2)

        # stage 3: latent
        x_in = self.down3(e3)
        b = self.latent(x_in)
        if g_stage is not None:
            g3 = g_stage[:, 3].view(B, 1, 1, 1)
            b = self._apply_gate(x_in, b, g3)

        # ---------------- Decoder ----------------
        # stage 4: decoder3
        d3_in = self._pad_and_add(self.up3(b), e3)
        d3 = self.decoder3(d3_in)
        if g_stage is not None:
            g4 = g_stage[:, 4].view(B, 1, 1, 1)
            d3 = self._apply_gate(d3_in, d3, g4)

        # stage 5: decoder2
        d2_in = self._pad_and_add(self.up2(d3), e2)
        d2 = self.decoder2(d2_in)
        if g_stage is not None:
            g5 = g_stage[:, 5].view(B, 1, 1, 1)
            d2 = self._apply_gate(d2_in, d2, g5)

        # stage 6: decoder1
        d1_in = self._pad_and_add(self.up1(d2), e1)
        d1 = self.decoder1(d1_in)
        if g_stage is not None:
            g6 = g_stage[:, 6].view(B, 1, 1, 1)
            d1 = self._apply_gate(d1_in, d1, g6)

        # ---------------- Refinement ----------------
        # stage 7: refinement
        r_in = d1
        r = self.refinement(r_in)
        if g_stage is not None:
            g7 = g_stage[:, 7].view(B, 1, 1, 1)
            r = self._apply_gate(r_in, r, g7)

        # Output (keep original residual design)
        out = self.output(r + x_embed)
        return out


# ----------------- 테스트 코드 ----------------- #
if __name__ == "__main__":
    print("=== VETNetBackbone Phase1/2 Test ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, C, H, W = 2, 3, 256, 256
    x = torch.randn(B, C, H, W, device=device)

    model = VETNetBackbone(in_channels=3, out_channels=3, dim=48).to(device)
    model.eval()

    with torch.no_grad():
        # Phase1 behavior (no gates)
        y0 = model(x)
        # Phase2 behavior (with gates)
        g = torch.full((B, model.NUM_MACRO_STAGES), 0.9, device=device)
        y1 = model(x, g_stage=g)

    print(f"Input  Shape: {x.shape}")
    print(f"Output Shape (no gate) : {y0.shape}")
    print(f"Output Shape (with gate): {y1.shape}")
    assert y0.shape == x.shape, "입력과 출력 해상도가 일치해야 합니다!"
    assert y1.shape == x.shape, "입력과 출력 해상도가 일치해야 합니다!"
    print(">> OK: Phase1/2 forward shapes match")

    # Check that gates actually affect output (should differ slightly)
    diff = (y0 - y1).abs().mean().item()
    print(f"Mean |y(no_gate) - y(with_gate)|: {diff:.8f}")
    print(">> If diff is ~0, gates may not be applied correctly.")
