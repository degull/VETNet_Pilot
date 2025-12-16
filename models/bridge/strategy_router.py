# G:/VETNet_pilot/models/bridge/strategy_router.py
# StrategyRouter:
# - backbone stage별로 strategy token을 "쓸지/말지" 결정해주는 라우터
# - backbone 내부에서 호출되어 stage별로 토큰을 None 처리하거나 그대로 통과시킴
#
# ✅ backbone 통합을 위해 핵심 동작은 "tokens or None" 반환이어야 함.
#   (concat은 MDTAWithStrategy가 담당)

import os
import sys
from dataclasses import dataclass
from typing import Optional, Set

import torch

# ------------------------------------------------------------
# ROOT 경로 세팅
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../models/bridge
ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))       # .../VETNet_pilot
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print(f"[strategy_router] ROOT = {ROOT}")


@dataclass
class StrategyRouterConfig:
    """
    use_strategy: 전체적으로 strategy를 사용할지
    enabled_stages: 사용할 stage 이름 집합 {"stage1","stage2","stage3","stage4","dec3","dec2","dec1","refine"...}
    detach_tokens: True면 backbone으로 들어가기 전에 토큰을 detach (보통 False 권장)
                  - Phase2 학습에서 gradient가 projection/LLM으로 흘러야 하므로 False
    """
    use_strategy: bool = True
    enabled_stages: Optional[Set[str]] = None
    detach_tokens: bool = False


class StrategyRouter:
    """
    backbone 내부에서 호출:
        routed = router(x, strategy_tokens, stage_name)

    반환 규칙:
      - use_strategy=False -> None
      - stage_name이 enabled_stages에 없으면 -> None
      - 그렇지 않으면 -> strategy_tokens (B,K,C) 그대로
    """

    def __init__(self, cfg: StrategyRouterConfig):
        self.cfg = cfg
        if self.cfg.enabled_stages is None:
            self.cfg.enabled_stages = {"stage1", "stage2", "stage3", "stage4"}
        print(f"[StrategyRouter] init | use_strategy = {cfg.use_strategy} | enabled_stages = {self.cfg.enabled_stages}")

    def __call__(
        self,
        x: torch.Tensor,
        strategy_tokens: Optional[torch.Tensor],
        stage_name: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """
        x: backbone feature tensor (B,C,H,W) or (B,N,C) 등 (여기서는 검사만, 사용은 안 함)
        strategy_tokens: (B,K,C) or None
        stage_name: "stage1"..."stage4" 등
        """
        if not self.cfg.use_strategy:
            return None
        if strategy_tokens is None:
            return None

        if stage_name is None:
            # stage_name이 없으면 기본적으로 통과
            out = strategy_tokens
        else:
            if stage_name not in self.cfg.enabled_stages:
                return None
            out = strategy_tokens

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

    B, C, H, W = 2, 64, 32, 32
    K = 4
    x = torch.randn(B, C, H, W, device=device)
    tok = torch.randn(B, K, C, device=device)

    # stage1, stage3만 enable
    cfg = StrategyRouterConfig(
        use_strategy=True,
        enabled_stages={"stage1", "stage3"},
        detach_tokens=False,
    )
    router = StrategyRouter(cfg)

    t1 = router(x, tok, "stage1")
    t2 = router(x, tok, "stage2")  # disabled
    t3 = router(x, tok, "stage3")

    print("[Self-test] stage1 routed:", None if t1 is None else tuple(t1.shape))
    print("[Self-test] stage2 routed:", t2)
    print("[Self-test] stage3 routed:", None if t3 is None else tuple(t3.shape))

    assert t1 is not None and t1.shape == tok.shape
    assert t2 is None
    assert t3 is not None and t3.shape == tok.shape

    print("[strategy_router] Self-test 완료\n")
