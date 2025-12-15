# G:/VETNet_pilot/models/bridge/control_projection.py
"""
ControlProjection
-----------------
StrategyHead 에서 나온 글로벌 strategy vector z (B, D_z)를
Backbone 각 Stage에서 사용할 strategy tokens로 투영(projection)하는 모듈.

입력:
    - z: (B, D_z)  (예: D_z = 256)

설정 (Config):
    - strategy_dim: D_z (z의 차원)
    - stage_dims:   (C1, C2, C3, C4, ...)  각 Stage의 채널 차원
    - num_tokens:   K (Stage마다 K개의 토큰 생성)

출력:
    - tokens_by_stage: List[Tensor]
        * len = num_stages
        * tokens_by_stage[i] shape = (B, K, C_i)
    - tokens_dict: Dict[str, Tensor]
        * "stage1", "stage2", ... 키로 접근 가능
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import torch
import torch.nn as nn


# -------------------------------------------------------------------------
# PATH 설정 (VETNet_pilot 루트 등록)
# -------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../models/bridge
ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))           # .../VETNet_pilot

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print(f"[control_projection] ROOT = {ROOT}")


# -------------------------------------------------------------------------
# Config Dataclass
# -------------------------------------------------------------------------
@dataclass
class ControlProjectionConfig:
    """
    ControlProjection 설정값.

    strategy_dim : Strategy vector z의 차원 (D_z)
    stage_dims   : 각 Backbone Stage의 채널 차원 튜플
                   예) (64, 128, 256, 512)
    num_tokens   : Stage별 생성할 Strategy Tokens 개수 (K)
    """
    strategy_dim: int = 256
    stage_dims: Tuple[int, ...] = (64, 128, 256, 512)
    num_tokens: int = 4


# -------------------------------------------------------------------------
# ControlProjection 본체
# -------------------------------------------------------------------------
class ControlProjection(nn.Module):
    """
    z (B, D_z)  →  [ (B, K, C1), (B, K, C2), ... ] 로 변환.

    사용 예시:
        cfg = ControlProjectionConfig(strategy_dim=256,
                                      stage_dims=(64, 128, 256, 512),
                                      num_tokens=4)
        proj = ControlProjection(cfg)
        out = proj(z)   # z: (B, 256)

        tokens_by_stage = out["tokens_by_stage"]
        tokens_stage1 = out["tokens_dict"]["stage1"]   # (B, 4, 64)
    """

    def __init__(self, cfg: ControlProjectionConfig):
        super().__init__()
        self.cfg = cfg

        self.num_stages = len(cfg.stage_dims)
        self.num_tokens = cfg.num_tokens
        self.strategy_dim = cfg.strategy_dim

        # Stage별 Linear layer:
        #   D_z → (K * C_i)
        layers: List[nn.Linear] = []
        for i, c in enumerate(cfg.stage_dims):
            layer = nn.Linear(self.strategy_dim, self.num_tokens * c)
            layers.append(layer)

        self.stage_projs = nn.ModuleList(layers)

        print(
            f"[ControlProjection] Init with strategy_dim={self.strategy_dim}, "
            f"stage_dims={cfg.stage_dims}, num_tokens={self.num_tokens}"
        )

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> Dict[str, Any]:
        """
        z: (B, D_z)

        반환:
            {
                "tokens_by_stage": List[Tensor],  # len = num_stages
                "tokens_dict": { "stage1": ..., "stage2": ... }
            }
        """
        assert z.dim() == 2, f"[ControlProjection] z must be (B, D_z), got {z.shape}"
        b, d = z.shape
        assert d == self.strategy_dim, (
            f"[ControlProjection] z dim mismatch: expected {self.strategy_dim}, got {d}"
        )

        tokens_by_stage: List[torch.Tensor] = []
        tokens_dict: Dict[str, torch.Tensor] = {}

        for i, (proj, c) in enumerate(zip(self.stage_projs, self.cfg.stage_dims)):
            # (B, D_z) → (B, K * C_i)
            t_flat = proj(z)
            # (B, K, C_i) 로 reshape
            t = t_flat.view(b, self.num_tokens, c)

            tokens_by_stage.append(t)
            stage_key = f"stage{i+1}"
            tokens_dict[stage_key] = t

        return {
            "tokens_by_stage": tokens_by_stage,
            "tokens_dict": tokens_dict,
        }


# -------------------------------------------------------------------------
# Self-test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    """
    간단 Self-test:

    1) 더미 strategy vector z 생성 (B=2, D_z=256)
    2) ControlProjection 통과
    3) Stage별 tokens shape 출력해서 구조 확인
    """

    print("\n[control_projection] Self-test 시작")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[control_projection] Device =", device)

    # 1) Config 생성
    cfg = ControlProjectionConfig(
        strategy_dim=256,
        stage_dims=(64, 128, 256, 512),
        num_tokens=4,
    )

    # 2) 모듈 생성
    proj = ControlProjection(cfg).to(device)
    proj.eval()

    # 3) 더미 z 생성
    batch_size = 2
    z = torch.randn(batch_size, cfg.strategy_dim, device=device)

    # 4) Forward
    with torch.no_grad():
        out = proj(z)

    tokens_by_stage = out["tokens_by_stage"]
    tokens_dict = out["tokens_dict"]

    print("\n[control_projection] tokens_by_stage len:", len(tokens_by_stage))
    for i, t in enumerate(tokens_by_stage):
        print(f"  - Stage {i+1}: tokens shape = {t.shape}")  # (B, K, C_i)

    print("\n[control_projection] tokens_dict keys:", list(tokens_dict.keys()))
    for k, v in tokens_dict.items():
        print(f"  * {k}: {tuple(v.shape)}")

    print("\n[control_projection] Self-test 완료.\n")
