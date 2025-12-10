# G:/VETNet_pilot/models/pilot/strategy_head.py
"""
StrategyHead: CLIP Vision Encoder + (Optional) LLM Strategy Generator

- 입력:  이미지 텐서 (B, 3, H, W), 0~1 범위 가정
- 출력:
    - strategy_tokens: (B, K, C_token)
    - strategy_vector: (B, D_z)
    - strategy_texts : List[str] 또는 None

Phase 2에서:
    - strategy_tokens → MDTA에 concat (X; S) 형태로 주입
    - strategy_vector → 필요 시 FiLM / 추가 컨트롤에 사용
    - strategy_texts → XAI / 로그 / 분석용
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# PATH 설정 (VETNet_pilot 루트 추가)
# -------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../models/pilot
ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))              # .../VETNet_pilot

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print(f"[strategy_head] ROOT = {ROOT}")

# -------------------------------------------------------------------------
# 외부 모듈 import 시도
# -------------------------------------------------------------------------
try:
    from transformers import CLIPVisionModel
    HAS_TRANSFORMERS = True
except Exception as e:
    HAS_TRANSFORMERS = False
    print("[strategy_head WARNING] transformers/CLIPVisionModel 로드 실패:", repr(e))
    print("  → pip install transformers accelerate 필요")

# LLM 로더
try:
    from models.pilot.llm_loader import LLMConfig, load_llm
    HAS_LLM_LOADER = True
except Exception as e:
    HAS_LLM_LOADER = False
    LLMConfig = None  # type: ignore
    print("[strategy_head WARNING] llm_loader import 실패:", repr(e))

# (선택) tokenizer_utils 사용
try:
    from models.pilot.tokenizer_utils import build_strategy_prompt
    HAS_TOKENIZER_UTILS = True
except Exception as e:
    HAS_TOKENIZER_UTILS = False
    print("[strategy_head WARNING] tokenizer_utils import 실패:", repr(e))


# -------------------------------------------------------------------------
# Config Dataclass
# -------------------------------------------------------------------------
@dataclass
class StrategyHeadConfig:
    """
    StrategyHead 설정값 모음.
    """
    # CLIP Vision 모델
    clip_model_name: str = "openai/clip-vit-large-patch14"
    clip_image_size: int = 224     # CLIP 입력 해상도

    # Strategy Vector / Tokens 차원
    strategy_dim: int = 256        # Z 차원
    num_tokens: int = 4            # K
    token_dim: int = 64            # C_token (Backbone Stage1 dim과 맞추면 좋음)

    # LLM 사용 여부
    enable_llm: bool = False       # 기본은 끔 (자원 절약)
    # 👉 타입 충돌 방지를 위해 Any 사용 (Pylance 경고 제거 목적)
    llm_config: Optional[Any] = None

    # 프롬프트 관련
    default_dataset_tag: str = "Generic"
    language: str = "en"


# -------------------------------------------------------------------------
# StrategyHead
# -------------------------------------------------------------------------
class StrategyHead(nn.Module):
    """
    CLIP Vision Encoder + (Optional) LLM을 이용해
    Strategy Vector / Tokens / Text를 생성하는 모듈.

    forward(
        img,             # (B,3,H,W), 0~1
        dataset_tag,     # "Rain100H", "CSD", ...
        extra_text,      # optional prompt context
        generate_text    # True면 LLM으로 strategy_text 생성 (느림)
    ) -> dict
    """

    def __init__(self, cfg: StrategyHeadConfig):
        super().__init__()
        self.cfg = cfg

        # ------------------------------
        # 1) CLIP Vision Encoder
        # ------------------------------
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "[StrategyHead] transformers/CLIPVisionModel 이 필요합니다. "
                "pip install transformers accelerate"
            )

        print(f"[StrategyHead] Loading CLIP Vision Model: {cfg.clip_model_name}")
        self.clip = CLIPVisionModel.from_pretrained(cfg.clip_model_name)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False     # Phase2 에서는 CLIP freeze 권장

        # CLIP feature dimension
        vision_dim = self.clip.config.hidden_size

        # ------------------------------
        # 2) Strategy Vector / Tokens Projection
        # ------------------------------
        self.proj_z = nn.Linear(vision_dim, cfg.strategy_dim)
        self.proj_tokens = nn.Linear(cfg.strategy_dim, cfg.num_tokens * cfg.token_dim)

        # ------------------------------
        # 3) LLM (Optional)
        # ------------------------------
        self.llm = None
        if cfg.enable_llm:
            if not HAS_LLM_LOADER:
                raise ImportError(
                    "[StrategyHead] enable_llm=True 이지만 llm_loader 를 가져올 수 없습니다."
                )
            # llm_config 타입을 Any로 둔 상태라 여기선 그냥 그대로 사용
            llm_cfg = cfg.llm_config if cfg.llm_config is not None else LLMConfig()
            print(f"[StrategyHead] Loading LLM: {llm_cfg.base_model_name}")
            self.llm = load_llm(llm_cfg)
        else:
            print("[StrategyHead] LLM 비활성화 (enable_llm=False). "
                  "strategy_text는 None으로 반환됩니다.")

        # CLIP 전처리용 mean/std (openai/clip-vit-large-patch14 기준)
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

    # ------------------------------------------------------------------
    def _preprocess_for_clip(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B,3,H,W), 0~1 범위 가정.
        CLIP 입력 사이즈로 resize + CLIP mean/std 정규화.
        """
        b, c, h, w = img.shape
        if (h, w) != (self.cfg.clip_image_size, self.cfg.clip_image_size):
            img = F.interpolate(
                img,
                size=(self.cfg.clip_image_size, self.cfg.clip_image_size),
                mode="bicubic",
                align_corners=False,
            )

        img = img.clamp(0.0, 1.0)
        img = (img - self.clip_mean) / self.clip_std
        return img

    # ------------------------------------------------------------------
    def _build_prompt(self, dataset_tag: Optional[str], extra_text: Optional[str]) -> str:
        """
        tokenizer_utils.build_strategy_prompt가 있으면 사용하고,
        없으면 내부 default 프롬프트를 사용.
        """
        tag = dataset_tag if dataset_tag is not None else self.cfg.default_dataset_tag

        # tokenizer_utils가 있으면 우선 사용
        if HAS_TOKENIZER_UTILS:
            return build_strategy_prompt(dataset_tag=tag, extra_text=extra_text)

        # Fallback 프롬프트
        base = (
            "You are an expert in blind image restoration for rain, snow, haze, "
            "raindrop and illumination degradations. "
        )
        task = f"The current dataset is {tag}. "
        instr = (
            "Describe a restoration strategy that maximizes PSNR and SSIM, "
            "while preserving important edges, textures, and removing artifacts."
        )
        if extra_text is not None:
            return base + task + extra_text + " " + instr
        else:
            return base + task + instr

    # ------------------------------------------------------------------
    def forward(
        self,
        img: torch.Tensor,
        dataset_tag: Optional[str] = None,
        extra_text: Optional[str] = None,
        generate_text: bool = False,
    ) -> Dict[str, Any]:
        """
        img: (B,3,H,W), 0~1
        generate_text=True 이면 LLM을 사용해서 strategy_text도 생성 (느려질 수 있음)

        반환:
            {
                "strategy_tokens": (B, K, C_token),
                "strategy_vector": (B, D_z),
                "strategy_texts": List[str] 또는 None
            }
        """
        device = img.device
        b = img.size(0)

        # 1) CLIP Vision Encoding
        x = self._preprocess_for_clip(img)
        with torch.no_grad():
            vision_out = self.clip(x, output_hidden_states=False)
            # pooled_output: (B, D_v)
            v = vision_out.pooler_output

        # 2) Strategy Vector & Tokens (CLIP 기반)
        z = self.proj_z(v)                     # (B, D_z)
        tokens_flat = self.proj_tokens(z)      # (B, K*C_token)
        tokens = tokens_flat.view(
            b,
            self.cfg.num_tokens,
            self.cfg.token_dim,
        )                                      # (B, K, C_token)

        # 3) Optional: LLM으로 Strategy Text 생성
        strategy_texts: Optional[List[str]] = None
        if generate_text:
            if self.llm is None:
                raise RuntimeError(
                    "[StrategyHead] generate_text=True 이지만 enable_llm=False 입니다. "
                    "config.enable_llm=True 로 LLM을 활성화해주세요."
                )

            strategy_texts = []
            for i in range(b):
                prompt = self._build_prompt(dataset_tag, extra_text)
                txt, _hidden = self.llm.generate_with_hidden(prompt, device=device)
                strategy_texts.append(txt)

        return {
            "strategy_tokens": tokens,
            "strategy_vector": z,
            "strategy_texts": strategy_texts,
        }


# -------------------------------------------------------------------------
# Self-test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    """
    간단한 self-test:

    - 더미 이미지 (B=2, 3x256x256)를 생성
    - StrategyHead(enable_llm=False) 로 통과
    - strategy_tokens / vector shape 출력

    LLM은 기본 비활성화라, 무거운 모델 다운로드 없이 CLIP + Projection만 테스트된다.
    """

    print("\n[strategy_head] Self-test 시작")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[strategy_head] Device =", device)

    # 더미 이미지 생성 (0~1)
    dummy_img = torch.rand(2, 3, 256, 256, device=device)

    # Config: LLM은 일단 끔
    cfg = StrategyHeadConfig(
        clip_model_name="openai/clip-vit-large-patch14",
        clip_image_size=224,
        strategy_dim=256,
        num_tokens=4,
        token_dim=64,
        enable_llm=False,
    )

    # 모델 생성
    head = StrategyHead(cfg).to(device)
    head.eval()

    with torch.no_grad():
        out = head(dummy_img, dataset_tag="Rain100H", extra_text=None, generate_text=False)

    tokens = out["strategy_tokens"]
    z = out["strategy_vector"]
    texts = out["strategy_texts"]

    print("[strategy_head] strategy_tokens shape:", tokens.shape)  # (B, K, C_token)
    print("[strategy_head] strategy_vector shape:", z.shape)       # (B, D_z)
    print("[strategy_head] strategy_texts:", texts)

    print("\n[strategy_head] Self-test 완료.\n")
