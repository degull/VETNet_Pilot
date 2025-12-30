# E:\VETNet_Pilot\models\pilot\phase3_policy.py
# VLM(vision+language)로부터 g_stage(8개 게이트) 를 예측하는 정책 모듈
# E:\VETNet_Pilot\models\pilot\phase3_policy.py
# ------------------------------------------------------------
# Phase-3 Policy (VLM Controller)
# - Vision encoder (CLIP) -> z_v
# - Text encoder (LLM backbone as encoder) -> z_t
# - Gate head: [z_v || z_t] -> g_stage (8 scalars)
#
# Notes:
# - This file is "research module" (not toy test).
# - It does NOT restore images. It only outputs gates + optional strategy logits.
# ------------------------------------------------------------

import torch
import torch.nn as nn


def _count_params(m: nn.Module):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Phase3GatePolicy(nn.Module):
    """
    Phase3GatePolicy
    ----------------
    Inputs:
      - z_v: vision embedding [B, Dv]
      - z_t: text embedding   [B, Dt]

    Outputs:
      - g_stage: [B, S] in [g_min, 1]
      - strategy_logits: optional [B, K] (if num_strategies is given)
    """
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        num_stages: int = 8,
        g_min: float = 0.1,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        num_strategies: int = 0,
        init_gate_bias: float = 2.0,  # sigmoid(2)=0.88 then scaled => near-identity start
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.num_stages = num_stages
        self.g_min = g_min
        self.num_strategies = num_strategies

        fused_dim = vision_dim + text_dim

        self.gate_head = MLP(fused_dim, hidden_dim, num_stages, dropout=dropout)

        # Optional discrete strategy head (classification)
        self.strategy_head = None
        if num_strategies and num_strategies > 0:
            self.strategy_head = MLP(fused_dim, hidden_dim, num_strategies, dropout=dropout)

        # Safer init: start near pass-through
        nn.init.zeros_(self.gate_head.net[-1].weight)
        nn.init.constant_(self.gate_head.net[-1].bias, init_gate_bias)

    def forward(self, z_v: torch.Tensor, z_t: torch.Tensor):
        assert z_v.ndim == 2 and z_t.ndim == 2
        assert z_v.size(0) == z_t.size(0)

        h = torch.cat([z_v, z_t], dim=-1)  # [B, Dv+Dt]
        g_raw = self.gate_head(h)          # [B, S]
        g = self.g_min + (1.0 - self.g_min) * torch.sigmoid(g_raw)

        out = {"g_stage": g}

        if self.strategy_head is not None:
            out["strategy_logits"] = self.strategy_head(h)

        return out


if __name__ == "__main__":
    print("=" * 72)
    print("[Phase3GatePolicy] Debug & Sanity Check")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 2
    Dv = 768
    Dt = 1024
    S = 8

    model = Phase3GatePolicy(vision_dim=Dv, text_dim=Dt, num_stages=S, g_min=0.1).to(device)
    total, trainable = _count_params(model)
    print(f"Device: {device}")
    print(f"Params: total={total:,} trainable={trainable:,}")

    z_v = torch.randn(B, Dv, device=device)
    z_t = torch.randn(B, Dt, device=device)
    out = model(z_v, z_t)

    g = out["g_stage"]
    print("g_stage shape:", tuple(g.shape))
    print("g stats:", float(g.min()), float(g.max()), float(g.mean()))

    loss = g.mean()
    loss.backward()
    gn = 0.0
    for p in model.parameters():
        if p.grad is not None:
            gn += p.grad.norm().item()
    print("grad_norm:", gn)
    print("[OK] Phase3GatePolicy sanity passed.")
    print("=" * 72)
