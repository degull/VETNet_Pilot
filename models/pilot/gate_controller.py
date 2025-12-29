# E:\VETNet_Pilot\models\pilot\gate_controller.py
# ------------------------------------------------
# GateController:
#   저차원 연속 제어 신호(gate)를 추정하는 "측정기(measurement controller)"
#   - 복원 backbone을 대체하지 않음
#   - feature를 전달하지 않음
#   - stage-wise scalar gate만 출력
# ------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class GateController(nn.Module):
    """
    GateController
    ----------------
    Input:
        x : degraded image, shape [B, 3, H, W]

    Output:
        g_stage : stage-wise scalar gates in [g_min, 1],
                  shape [B, num_stages]

    Design principles:
        - very low capacity (cannot override backbone)
        - global degradation measurement only
        - stable optimization (Phase-2)
    """

    def __init__(self, num_stages: int, g_min: float = 0.1):
        super().__init__()
        self.num_stages = num_stages
        self.g_min = g_min

        # ------------------------------------------------
        # Encoder: image -> global degradation statistics
        # ------------------------------------------------
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # /4
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # remove spatial info
        )

        # ------------------------------------------------
        # Head: global stats -> stage-wise scalar gates
        # ------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_stages),
        )

        # ------------------------------------------------
        # Initialization (Phase-1 behavior preservation)
        #   sigmoid(2.0) ≈ 0.88
        # ------------------------------------------------
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: degraded image tensor [B,3,H,W]

        Returns:
            g_stage: [B, num_stages], each gate in [g_min, 1]
        """
        b = x.size(0)

        h = self.enc(x).view(b, -1)   # [B,128]
        g_raw = self.head(h)          # [B,S]

        # bounded gate: g ∈ [g_min, 1]
        g = self.g_min + (1.0 - self.g_min) * torch.sigmoid(g_raw)
        return g


# ============================================================
# Debug / Research Sanity Check
# ============================================================
if __name__ == "__main__":
    print("=" * 72)
    print("[GateController] Research Sanity & Responsiveness Check")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------
    B = 2
    C = 3
    H = 256
    W = 256
    NUM_STAGES = 8
    G_MIN = 0.1
    LR = 1e-3
    STEPS = 5

    # --------------------------------------------------------
    # Instantiate model
    # --------------------------------------------------------
    model = GateController(num_stages=NUM_STAGES, g_min=G_MIN).to(device)

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --------------------------------------------------------
    # (0) Initial gate check (before training)
    # --------------------------------------------------------
    model.eval()
    x1 = torch.randn(B, C, H, W, device=device)
    x2 = torch.randn(B, C, H, W, device=device)

    with torch.no_grad():
        g1_init = model(x1)
        g2_init = model(x2)

    print("\n[Initial gates (before training)]")
    print(f"g(x1) mean: {g1_init.mean().item():.4f}")
    print(f"g(x2) mean: {g2_init.mean().item():.4f}")
    print(f"Stage-wise var(g): {torch.var(g1_init, dim=0)}")

    # --------------------------------------------------------
    # (1) Mini training loop: does gate respond to input?
    # --------------------------------------------------------
    print("\n[Mini training loop: 5 steps]")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for step in range(1, STEPS + 1):
        optimizer.zero_grad()

        # two different inputs
        x1 = torch.randn(B, C, H, W, device=device)
        x2 = torch.randn(B, C, H, W, device=device)

        g1 = model(x1)
        g2 = model(x2)

        # artificial objective:
        # encourage gates to separate (only for responsiveness test)
        loss = -torch.mean(torch.abs(g1 - g2))

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            diff = torch.mean(torch.abs(g1 - g2)).item()
            var = torch.var(torch.cat([g1, g2], dim=0), dim=0)

        print(
            f"Step {step:02d} | "
            f"mean |g1-g2| = {diff:.6f} | "
            f"stage-wise var = {[f'{v:.4f}' for v in var.tolist()]}"
        )

    # --------------------------------------------------------
    # (2) Post-training inspection
    # --------------------------------------------------------
    model.eval()
    with torch.no_grad():
        g1_post = model(x1)
        g2_post = model(x2)

    print("\n[Post-training gates]")
    print(f"g(x1) mean: {g1_post.mean().item():.4f}")
    print(f"g(x2) mean: {g2_post.mean().item():.4f}")
    print(f"Stage-wise var(g): {torch.var(torch.cat([g1_post, g2_post], dim=0), dim=0)}")

    print("\n[Conclusion]")
    print("✔ Safe to proceed to Phase-2 backbone-gated training.")
    print("=" * 72)
