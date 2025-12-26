# StrategyRouter:
# - backbone stage별로 strategy token을 "쓸지/말지" 결정해주는 라우터
# - backbone 내부에서 호출되어 stage별로 토큰을 None 처리하거나 그대로 통과시킴
#
# ✅ backbone 통합을 위해 핵심 동작은 "tokens or None" 반환이어야 함.
#   (concat은 MDTAWithStrategy가 담당)

# E:\VETNet_Pilot\models\bridge\strategy_router.py
import os
import sys
from dataclasses import dataclass
from typing import Optional, Set, Dict, Union

import torch

# ------------------------------------------------------------
# ROOT 경로 세팅
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print(f"[strategy_router] ROOT = {ROOT}")


@dataclass
class StrategyRouterConfig:
    """
    use_strategy: 전체적으로 strategy를 사용할지
    enabled_stages: 사용할 stage 이름 집합 {"stage1","stage2","stage3","stage4", ...}
    detach_tokens: True면 backbone으로 들어가기 전에 토큰을 detach
    """
    use_strategy: bool = True
    enabled_stages: Optional[Set[str]] = None
    detach_tokens: bool = False


class StrategyRouter:
    """
    backbone 내부에서 호출:
        routed = router(x, strategy_tokens, stage_name)

    strategy_tokens 형태:
      - Dict[str, Tensor]: {"stage1": (B,K,C1), ...}
      - Tensor: (B,K,C)  -> 그대로 통과 (fallback)
    """

    def __init__(self, cfg: StrategyRouterConfig):
        self.cfg = cfg
        if self.cfg.enabled_stages is None:
            self.cfg.enabled_stages = {"stage1", "stage2", "stage3", "stage4"}
        print(
            f"[StrategyRouter] init | use_strategy={cfg.use_strategy} | "
            f"enabled_stages={self.cfg.enabled_stages}"
        )

    def __call__(
        self,
        x: torch.Tensor,
        strategy_tokens: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]],
        stage_name: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """
        x: backbone feature (B,C,H,W) or (B,N,C) – 사용 안 함 (인터페이스 유지용)
        strategy_tokens:
            - dict[str, Tensor] or
            - Tensor
        stage_name: 현재 backbone stage 이름
        """

        # ---- global off ----
        if not self.cfg.use_strategy:
            return None
        if strategy_tokens is None:
            return None

        # ---- stage filtering ----
        if stage_name is not None:
            if stage_name not in self.cfg.enabled_stages:
                return None

        # ---- select token ----
        if isinstance(strategy_tokens, dict):
            if stage_name is None:
                # stage_name이 없으면 아무 것도 안 주는 게 안전
                return None
            if stage_name not in strategy_tokens:
                return None
            out = strategy_tokens[stage_name]
        else:
            # Tensor fallback
            out = strategy_tokens

        # ---- detach if needed ----
        if self.cfg.detach_tokens:
            out = out.detach()

        return out


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print("\n[strategy_router] Self-test 시작")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[strategy_router] Device =", device)

    B = 2
    K = 4

    tokens = {
        "stage1": torch.randn(B, K, 64, device=device),
        "stage2": torch.randn(B, K, 128, device=device),
        "stage3": torch.randn(B, K, 256, device=device),
        "stage4": torch.randn(B, K, 512, device=device),
    }

    x = torch.randn(B, 64, 32, 32, device=device)

    # stage1, stage3만 enable
    cfg = StrategyRouterConfig(
        use_strategy=True,
        enabled_stages={"stage1", "stage3"},
        detach_tokens=False,
    )
    router = StrategyRouter(cfg)

    t1 = router(x, tokens, "stage1")
    t2 = router(x, tokens, "stage2")  # disabled
    t3 = router(x, tokens, "stage3")
    t4 = router(x, tokens, None)      # no stage_name

    print("[Self-test] stage1 routed:", None if t1 is None else tuple(t1.shape))
    print("[Self-test] stage2 routed:", t2)
    print("[Self-test] stage3 routed:", None if t3 is None else tuple(t3.shape))
    print("[Self-test] stage=None routed:", t4)

    assert t1 is not None and t1.shape == (B, K, 64)
    assert t2 is None
    assert t3 is not None and t3.shape == (B, K, 256)
    assert t4 is None

    print("[strategy_router] Self-test 완료\n")
