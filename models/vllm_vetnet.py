# G:/VETNet_pilot/models/vllm_vetnet.py
"""
VLLM-VETNet (Phase-2 Wrapper)

- Path A (VLLM Pilot): StrategyHead (CLIP (+ optional LLM))
    -> strategy_vector Z (B, Dz)
    -> strategy_texts (optional, for XAI)

- Control Bridge:
    -> ControlProjection: Z -> stage-wise Strategy Tokens
    -> StrategyRouter: (optional) stage on/off gating + concat helper

- Path B (VETNet Backbone):
    -> 실제 복원 수행
    -> (중요) Backbone이 stage-wise strategy token 주입을 지원해야 함.
       이 wrapper는 다양한 kwarg 이름으로 주입을 시도하고,
       Backbone이 아직 지원하지 않으면 경고 후 "그냥 복원만" 수행합니다.

Forward 인터페이스:
    restored, strategy_texts = model(img_hr, img_lr=None, dataset_tag="Rain100H", generate_text=False)

Self-test:
    python G:/VETNet_pilot/models/vllm_vetnet.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# PATH 설정 (VETNet_pilot 루트 추가)
# -------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../models
ROOT = os.path.dirname(CURRENT_DIR)                        # .../VETNet_pilot
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
print(f"[vllm_vetnet] ROOT = {ROOT}")


# -------------------------------------------------------------------------
# Imports (프로젝트 내부 모듈)
# -------------------------------------------------------------------------
from models.backbone.vetnet_backbone import VETNetBackbone
from models.pilot.strategy_head import StrategyHead, StrategyHeadConfig
from models.bridge.control_projection import ControlProjection, ControlProjectionConfig
from models.bridge.strategy_router import StrategyRouter, StrategyRouterConfig


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
@dataclass
class VLLMVETNetConfig:
    # ----- Backbone -----
    in_channels: int = 3
    out_channels: int = 3
    dim: int = 64
    num_blocks: Tuple[int, int, int, int] = (4, 6, 6, 8)
    heads: Tuple[int, int, int, int] = (1, 2, 4, 8)
    volterra_rank: int = 4
    ffn_expansion_factor: float = 2.66
    bias: bool = False

    # Backbone stage dims (VETNet stage channels)
    # 보통 Restormer 계열: (dim, 2dim, 4dim, 8dim)
    # 네 phase1이 dim=64면 stage_dims=(64,128,256,512)
    stage_dims: Tuple[int, int, int, int] = (64, 128, 256, 512)

    # ----- Strategy / Tokens -----
    strategy_dim: int = 256          # Z dim
    num_tokens: int = 4              # K
    enabled_stages: Tuple[str, ...] = ("stage1", "stage2", "stage3", "stage4")
    use_strategy: bool = True

    # ----- Pilot (CLIP/LLM) -----
    enable_llm: bool = False         # Phase2 학습 때 True 가능(무거움)
    clip_model_name: str = "openai/clip-vit-large-patch14"
    clip_image_size: int = 224

    # ----- Convenience -----
    lr_image_size: int = 336         # img_lr 없으면 내부에서 resize할 크기


# -------------------------------------------------------------------------
# Wrapper Model
# -------------------------------------------------------------------------
class VLLMVETNet(nn.Module):
    """
    Path A + Bridge + Path B 통합 wrapper.

    - forward()에서 strategy를 만들고,
      backbone에 stage-wise tokens를 주입하려고 시도한다.
    """

    def __init__(self, cfg: VLLMVETNetConfig):
        super().__init__()
        self.cfg = cfg

        # ----------------------------
        # Path B: Backbone
        # ----------------------------
        self.backbone = VETNetBackbone(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            dim=cfg.dim,
            num_blocks=cfg.num_blocks,
            heads=cfg.heads,
            volterra_rank=cfg.volterra_rank,
            ffn_expansion_factor=cfg.ffn_expansion_factor,
            bias=cfg.bias,
        )

        # ----------------------------
        # Path A: StrategyHead (CLIP + optional LLM)
        # ----------------------------
        sh_cfg = StrategyHeadConfig(
            clip_model_name=cfg.clip_model_name,
            clip_image_size=cfg.clip_image_size,
            strategy_dim=cfg.strategy_dim,
            num_tokens=cfg.num_tokens,
            token_dim=cfg.stage_dims[0],   # head가 tokens까지 뽑아주는 경우 stage1 dim으로 맞춤
            enable_llm=cfg.enable_llm,
            llm_config=None,
        )
        self.strategy_head = StrategyHead(sh_cfg)

        # ----------------------------
        # Bridge: Z -> stage tokens
        # ----------------------------
        cp_cfg = ControlProjectionConfig(
            strategy_dim=cfg.strategy_dim,
            stage_dims=cfg.stage_dims,
            num_tokens=cfg.num_tokens,
        )
        self.control_projection = ControlProjection(cp_cfg)

        # ----------------------------
        # Bridge: stage gating + concat helper
        # ----------------------------
        sr_cfg = StrategyRouterConfig(
            use_strategy=cfg.use_strategy,
            enabled_stages=set(cfg.enabled_stages),
        )
        self.strategy_router = StrategyRouter(sr_cfg)

    # ------------------------------------------------------------------
    def load_phase1_backbone(self, ckpt_path: str, strict: bool = False) -> None:
        """
        Phase1에서 학습된 backbone ckpt를 로드.
        ckpt가 dict면 state_dict 키를 자동 탐지.
        """
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[VLLMVETNet] ckpt not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "model" in ckpt:
                state = ckpt["model"]
            else:
                # 이미 state_dict일 수도
                state = ckpt
        else:
            state = ckpt

        missing, unexpected = self.backbone.load_state_dict(state, strict=strict)
        print(f"[VLLMVETNet] Loaded phase1 backbone: {ckpt_path}")
        print(f"  - strict={strict}")
        print(f"  - missing keys: {len(missing)}")
        print(f"  - unexpected keys: {len(unexpected)}")

    # ------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        print("[VLLMVETNet] Backbone frozen (requires_grad=False).")

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_lr(img_hr: torch.Tensor, lr_size: int) -> torch.Tensor:
        """img_lr가 없으면 img_hr로부터 lr_size로 축소"""
        return F.interpolate(img_hr, size=(lr_size, lr_size), mode="bicubic", align_corners=False)

    # ------------------------------------------------------------------
    def _try_backbone_forward(
        self,
        img_hr: torch.Tensor,
        tokens_by_stage: List[torch.Tensor],
        tokens_dict: Dict[str, torch.Tensor],
        dataset_tag: Optional[str],
    ) -> torch.Tensor:
        """
        Backbone이 strategy token 주입을 지원하는지 다양한 kwarg로 시도.
        지원 안 하면 그냥 backbone(img_hr).
        """
        # 1) 가장 흔한 케이스들부터 시도
        candidates: List[Dict[str, Any]] = [
            {"strategy_tokens_by_stage": tokens_by_stage},
            {"tokens_by_stage": tokens_by_stage},
            {"strategy_tokens": tokens_dict},
            {"tokens_dict": tokens_dict},
            {"strategy_dict": tokens_dict},
            {"strategy_tokens_stage": tokens_dict},
        ]

        if dataset_tag is not None:
            # dataset_tag도 받는 backbone이 있을 수 있어 같이 시도
            extra = []
            for kw in candidates:
                kk = dict(kw)
                kk["dataset_tag"] = dataset_tag
                extra.append(kk)
            candidates = extra + candidates

        for kw in candidates:
            try:
                out = self.backbone(img_hr, **kw)  # type: ignore
                return out
            except TypeError:
                continue

        # 2) 여전히 실패하면 fallback
        print(
            "[VLLMVETNet WARNING] Backbone forward()가 strategy token 주입 kwarg를 받지 않습니다.\n"
            "  → 현재는 'strategy 없이 backbone만' 실행됩니다.\n"
            "  → 해결: models/backbone 쪽(예: mdta_strategy.py / blocks.py / vetnet_backbone.py)에서\n"
            "           stage-wise tokens를 받아 concat 후 MDTA를 수행하도록 forward 인터페이스를 맞춰주세요."
        )
        return self.backbone(img_hr)

    # ------------------------------------------------------------------
    def forward(
        self,
        img_hr: torch.Tensor,
        img_lr: Optional[torch.Tensor] = None,
        dataset_tag: Optional[str] = None,
        generate_text: bool = False,
        extra_text: Optional[str] = None,
    ):
        """
        img_hr: (B,3,H,W) 복원 대상
        img_lr: (B,3,h,w) strategy용 입력 (없으면 내부 resize)
        dataset_tag: "Rain100H", "CSD", ...
        generate_text: True면 LLM으로 strategy_text 생성 (느림)
        """
        if img_lr is None:
            img_lr = self._ensure_lr(img_hr, self.cfg.lr_image_size)

        # ----------------------------
        # Path A: strategy 생성
        # ----------------------------
        pilot_out = self.strategy_head(
            img=img_lr,
            dataset_tag=dataset_tag,
            extra_text=extra_text,
            generate_text=generate_text,
        )
        z: torch.Tensor = pilot_out["strategy_vector"]          # (B, Dz)
        strategy_texts: Optional[List[str]] = pilot_out["strategy_texts"]

        # ----------------------------
        # Bridge: Z -> stage-wise tokens
        # ----------------------------
        proj_out = self.control_projection(z)
        tokens_by_stage: List[torch.Tensor] = proj_out["tokens_by_stage"]     # [ (B,K,Cs1), ...]
        tokens_dict: Dict[str, torch.Tensor] = proj_out["tokens_dict"]        # {"stage1":...}

        # ----------------------------
        # Path B: backbone 복원 (token 주입 시도)
        # ----------------------------
        restored = self._try_backbone_forward(
            img_hr=img_hr,
            tokens_by_stage=tokens_by_stage,
            tokens_dict=tokens_dict,
            dataset_tag=dataset_tag,
        )

        return restored, strategy_texts


# -------------------------------------------------------------------------
# Self-test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Self-test 목표:
    1) StrategyHead -> Z 생성이 되는지
    2) ControlProjection -> stage별 tokens shape이 맞는지
    3) Backbone에 주입 시도 -> TypeError 없이 돌아가면 "주입 인터페이스 OK"
       (만약 Backbone이 아직 주입을 안 받으면 경고 출력 + backbone만 실행)
    """
    print("\n[vllm_vetnet] Self-test 시작")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[vllm_vetnet] Device =", device)

    # config (LLM은 꺼두는 게 테스트가 빠름)
    cfg = VLLMVETNetConfig(
        dim=64,
        stage_dims=(64, 128, 256, 512),
        strategy_dim=256,
        num_tokens=4,
        enable_llm=False,
        use_strategy=True,
        enabled_stages=("stage1", "stage2", "stage3", "stage4"),
    )

    model = VLLMVETNet(cfg).to(device)
    model.eval()

    # dummy input
    img_hr = torch.rand(2, 3, 256, 256, device=device)  # 복원 대상
    img_lr = F.interpolate(img_hr, size=(336, 336), mode="bicubic", align_corners=False)

    with torch.no_grad():
        restored, texts = model(
            img_hr=img_hr,
            img_lr=img_lr,
            dataset_tag="Rain100H",
            generate_text=False,
        )

    print("[vllm_vetnet] restored shape:", tuple(restored.shape))
    print("[vllm_vetnet] strategy_texts:", texts)

    # stage token shapes도 한번 더 확인
    with torch.no_grad():
        pilot_out = model.strategy_head(img_lr, dataset_tag="Rain100H", generate_text=False)
        z = pilot_out["strategy_vector"]
        proj_out = model.control_projection(z)
        td = proj_out["tokens_dict"]
        print("[vllm_vetnet] tokens_dict shapes:")
        for k, v in td.items():
            print(f"  - {k}: {tuple(v.shape)}")

    print("\n[vllm_vetnet] Self-test 완료\n")
