# G:/VETNet_pilot/models/bridge/strategy_router.py
"""
strategy_router.py

Phase 2에서 Strategy Tokens를 각 Stage의 MDTA에 주입하는 헬퍼 모듈.

역할:
- Stage별로 전달된 feature (B, N, C)에 대해,
  해당 Stage용 strategy_tokens (B, K, C)를 concat 해서
  (B, N+K, C) 형태의 입력을 만들어준다.

사용 예시 (MDTA 블록 내부에서):

    # x: (B, N, C)  -- flatten된 이미지 토큰
    # tokens_dict: {"stage1": (B, K, C1), "stage2": (B, K, C2), ...}

    x_in = router.concat_tokens(
        x,
        stage_name="stage1",
        strategy_tokens=tokens_dict  # dict 또는 direct tensor 모두 지원
    )

    # 이후 MDTA:
    y = self.attn(x_in, x_in, x_in)
    y_img = y[:, :N, :]   # 앞쪽 N개만 다시 이미지 토큰으로 사용
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Union, Any

import torch
import torch.nn as nn

# -------------------------------------------------------------------------
# PATH 설정 (VETNet_pilot 루트 추가)
# -------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../models/bridge
ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))              # .../VETNet_pilot

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print(f"[strategy_router] ROOT = {ROOT}")


# -------------------------------------------------------------------------
# Config Dataclass
# -------------------------------------------------------------------------
@dataclass
class StrategyRouterConfig:
    """
    StrategyRouter 설정값 모음.

    - stage_names: backbone에서 사용하는 Stage 식별자
    - use_strategy: False면 전체 Stage에서 Strategy Token 주입 비활성화
    - enabled_stages: 특정 Stage에서만 Strategy Token을 사용하고 싶을 때 지정
    """
    stage_names: tuple = ("stage1", "stage2", "stage3", "stage4")
    use_strategy: bool = True
    enabled_stages: Optional[tuple] = None  # 예: ("stage1", "stage3")


class StrategyRouter(nn.Module):
    """
    Strategy Tokens를 Stage별로 MDTA에 주입하는 모듈.

    지원 입력 형식:
      - strategy_tokens: Dict[str, Tensor] (추천)
          {"stage1": (B, K, C1), "stage2": (B, K, C2), ...}
      - 혹은 Tensor (B, K, C) 1개 (Stage 관계없이 동일 토큰 사용 시)

    concat 결과:
      - x: (B, N, C)
      - tokens: (B, K, C)
      - 반환: (B, N+K, C)
    """

    def __init__(self, cfg: StrategyRouterConfig):
        super().__init__()
        self.cfg = cfg

        # enabled_stages가 None이면 stage_names 전체를 사용
        if cfg.enabled_stages is None:
            self.enabled_stages = set(cfg.stage_names)
        else:
            self.enabled_stages = set(cfg.enabled_stages)

        print(
            "[StrategyRouter] init | use_strategy =", cfg.use_strategy,
            "| enabled_stages =", self.enabled_stages
        )

    # ------------------------------------------------------------------
    def is_stage_enabled(self, stage_name: str) -> bool:
        """
        해당 stage_name에 대해 Strategy Token을 사용할지 여부.
        """
        if not self.cfg.use_strategy:
            return False
        return stage_name in self.enabled_stages

    # ------------------------------------------------------------------
    def _get_tokens_for_stage(
        self,
        stage_name: str,
        strategy_tokens: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """
        stage_name에 해당하는 Strategy Tokens를 꺼내온다.
        - dict인 경우: strategy_tokens[stage_name]
        - tensor인 경우: 그대로 반환
        - 없거나 비활성화면 None
        """
        if not self.is_stage_enabled(stage_name):
            return None

        # dict 형식: {"stage1": (B,K,C1), ...}
        if isinstance(strategy_tokens, dict):
            if stage_name not in strategy_tokens:
                # 해당 Stage 토큰이 없으면 사용하지 않음
                return None
            return strategy_tokens[stage_name]

        # tensor 형식: (B,K,C)
        if isinstance(strategy_tokens, torch.Tensor):
            return strategy_tokens

        # 지원하지 않는 타입
        return None

    # ------------------------------------------------------------------
    def concat_tokens(
        self,
        x: torch.Tensor,
        stage_name: str,
        strategy_tokens: Union[torch.Tensor, Dict[str, torch.Tensor], None],
    ) -> torch.Tensor:
        """
        x: (B, N, C)  - flatten된 이미지 feature
        stage_name: "stage1", "stage2" 등
        strategy_tokens:
            - dict: {"stage1": (B, K, C1), ...}
            - tensor: (B, K, C)
            - None: 적용 안 함

        반환:
            - X_in: (B, N+K, C)  (Strategy Token concat 포함)
            - 또는 strategy_tokens가 없으면 x 그대로 반환
        """
        if strategy_tokens is None or (not self.cfg.use_strategy):
            return x

        s = self._get_tokens_for_stage(stage_name, strategy_tokens)
        if s is None:
            return x

        # shape check
        if s.dim() != 3 or x.dim() != 3:
            raise ValueError(
                f"[StrategyRouter] x, s 모두 (B,N,C)/(B,K,C) 이어야 합니다. "
                f"Got: x.shape={x.shape}, s.shape={s.shape}"
            )

        b1, n, c1 = x.shape
        b2, k, c2 = s.shape

        if b1 != b2:
            raise ValueError(
                f"[StrategyRouter] 배치 크기가 다릅니다: x({b1}) vs s({b2})"
            )
        if c1 != c2:
            raise ValueError(
                f"[StrategyRouter] 채널 차원 C가 다릅니다: x({c1}) vs s({c2}). "
                "control_projection에서 stage별 token_dim을 backbone dim과 맞추어야 합니다."
            )

        # concat: (B, N+K, C)
        x_in = torch.cat([x, s], dim=1)
        return x_in


# -------------------------------------------------------------------------
# Self-test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    """
    간단한 self-test:

    1) 더미 feature x: (B=2, N=1024, C=64)
    2) 더미 strategy_tokens:
         - dict 방식: {"stage1": (2,4,64), "stage2": (2,4,128) ...}
    3) stage1/2 에 대해 concat 결과 shape 검사
    """

    print("\n[strategy_router] Self-test 시작")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[strategy_router] Device =", device)

    # ---------------------------
    # 1) Router 생성
    # ---------------------------
    cfg = StrategyRouterConfig(
        stage_names=("stage1", "stage2", "stage3", "stage4"),
        use_strategy=True,
        enabled_stages=("stage1", "stage3"),  # 예시: 1,3 stage에서만 사용
    )
    router = StrategyRouter(cfg).to(device)

    # ---------------------------
    # 2) 더미 feature / tokens 생성
    # ---------------------------
    B = 2
    N = 1024
    C1 = 64
    K = 4

    # stage1용 feature
    x_stage1 = torch.randn(B, N, C1, device=device)

    # stage1, stage3용 tokens dict (여기선 둘 다 C=64로 맞춤)
    tokens_dict = {
        "stage1": torch.randn(B, K, C1, device=device),
        "stage3": torch.randn(B, K, C1, device=device),
    }

    # ---------------------------
    # 3) stage1 concat 테스트
    # ---------------------------
    x_in_1 = router.concat_tokens(
        x_stage1,
        stage_name="stage1",
        strategy_tokens=tokens_dict,
    )
    print("[Self-test] stage1 concat: x_in_1.shape =", x_in_1.shape)
    # 기대: (B, N+K, C1) = (2, 1028, 64)

    # ---------------------------
    # 4) stage2 concat 테스트 (enabled_stages에 없으므로 그냥 통과)
    # ---------------------------
    x_stage2 = torch.randn(B, N, C1, device=device)
    x_in_2 = router.concat_tokens(
        x_stage2,
        stage_name="stage2",
        strategy_tokens=tokens_dict,
    )
    print("[Self-test] stage2 concat (disabled): x_in_2.shape =", x_in_2.shape)
    # 기대: (B, N, C1) = (2, 1024, 64)

    # ---------------------------
    # 5) direct tensor 방식 테스트
    # ---------------------------
    tokens_direct = torch.randn(B, K, C1, device=device)
    x_in_direct = router.concat_tokens(
        x_stage1,
        stage_name="stage1",
        strategy_tokens=tokens_direct,
    )
    print("[Self-test] direct tensor concat: x_in_direct.shape =", x_in_direct.shape)

    print("\n[strategy_router] Self-test 완료.\n")
