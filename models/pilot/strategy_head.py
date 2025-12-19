""" # G:/VETNet_pilot/models/pilot/strategy_head.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# PATH 설정
# -------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print(f"[strategy_head] ROOT = {ROOT}")

# -------------------------------------------------------------------------
# External imports
# -------------------------------------------------------------------------
from transformers import CLIPVisionModel

try:
    from models.pilot.llm_loader import LLMConfig, load_llm
    HAS_LLM_LOADER = True
except Exception:
    HAS_LLM_LOADER = False
    LLMConfig = None  # type: ignore

try:
    from models.pilot.tokenizer_utils import build_strategy_prompt
    HAS_TOKENIZER_UTILS = True
except Exception:
    HAS_TOKENIZER_UTILS = False


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
@dataclass
class StrategyHeadConfig:
    clip_model_name: str = "openai/clip-vit-large-patch14"
    clip_image_size: int = 224

    strategy_dim: int = 256
    num_tokens: int = 4
    token_dim: int = 64

    enable_llm: bool = False
    llm_config: Optional[Any] = None

    default_dataset_tag: str = "Generic"
    language: str = "en"


# -------------------------------------------------------------------------
# StrategyHead
# -------------------------------------------------------------------------
class StrategyHead(nn.Module):
    def __init__(self, cfg: StrategyHeadConfig):
        super().__init__()
        self.cfg = cfg

        # ---- CLIP Vision Encoder (Frozen) ----
        print(f"[StrategyHead] Loading CLIP Vision Model: {cfg.clip_model_name}")
        self.clip = CLIPVisionModel.from_pretrained(cfg.clip_model_name)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        vision_dim = self.clip.config.hidden_size

        # ---- Trainable projections ----
        self.proj_z = nn.Linear(vision_dim, cfg.strategy_dim)
        self.proj_tokens = nn.Linear(cfg.strategy_dim, cfg.num_tokens * cfg.token_dim)

        # ---- Optional LLM ----
        self.llm = None
        if cfg.enable_llm:
            if not HAS_LLM_LOADER:
                raise ImportError("enable_llm=True but llm_loader not found")
            llm_cfg = cfg.llm_config if cfg.llm_config is not None else LLMConfig()
            self.llm = load_llm(llm_cfg)
        else:
            print("[StrategyHead] LLM 비활성화 (enable_llm=False). strategy_text는 None으로 반환됩니다.")

        # ---- CLIP normalization ----
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

    def _preprocess_for_clip(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[-2:] != (self.cfg.clip_image_size, self.cfg.clip_image_size):
            img = F.interpolate(
                img,
                size=(self.cfg.clip_image_size, self.cfg.clip_image_size),
                mode="bicubic",
                align_corners=False,
            )
        img = img.clamp(0, 1)
        return (img - self.clip_mean) / self.clip_std

    def forward(
        self,
        img: torch.Tensor,
        dataset_tag: Optional[str] = None,
        extra_text: Optional[str] = None,
        generate_text: bool = False,
    ) -> Dict[str, Any]:

        B = img.size(0)

        # ---- CLIP (no_grad) ----
        x = self._preprocess_for_clip(img)
        with torch.no_grad():
            v = self.clip(x).pooler_output  # (B, D_v)

        # ---- Trainable projections (GRAD ON) ----
        z = self.proj_z(v)                              # (B, D_z)
        tokens = self.proj_tokens(z).view(
            B, self.cfg.num_tokens, self.cfg.token_dim
        )

        strategy_texts = None
        if generate_text:
            if self.llm is None:
                raise RuntimeError("generate_text=True but enable_llm=False")
            strategy_texts = []
            for _ in range(B):
                prompt = "Describe a restoration strategy."
                txt, _ = self.llm.generate_with_hidden(prompt, device=img.device)
                strategy_texts.append(txt)

        return {
            "strategy_tokens": tokens,
            "strategy_vector": z,
            "strategy_texts": strategy_texts,
        }


if __name__ == "__main__":
    print("\n[strategy_head] Self-test 시작")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.rand(2, 3, 256, 256, device=device)

    cfg = StrategyHeadConfig(enable_llm=False)
    model = StrategyHead(cfg).to(device).eval()

    out = model(dummy)
    print("tokens:", out["strategy_tokens"].shape)
    print("z:", out["strategy_vector"].shape)
    print("texts:", out["strategy_texts"])
    print("[strategy_head] Self-test 완료")
 """


 # ver2
 # G:\VETNet_pilot\models\pilot\strategy_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class StrategyHead(nn.Module):
    """
    Takes:
      - LLM hidden states (B, T, D_lm) or a pooled vector (B, D_lm)
    Outputs:
      - strategy tokens S: (B, K, C)

    This is the "z -> S" adapter g_psi(z).
    """

    def __init__(
        self,
        lm_dim: int,
        strategy_dim: int = 512,   # z dim
        K: int = 3,               # number of strategy tokens
        C: int = 256,             # channel dim expected by VETNet control bridge
        dropout: float = 0.1
    ):
        super().__init__()
        self.lm_dim = lm_dim
        self.strategy_dim = strategy_dim
        self.K = K
        self.C = C

        self.pool = "mean"  # stable default
        self.z_proj = nn.Sequential(
            nn.LayerNorm(lm_dim),
            nn.Linear(lm_dim, strategy_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(strategy_dim, strategy_dim),
        )

        self.s_proj = nn.Sequential(
            nn.LayerNorm(strategy_dim),
            nn.Linear(strategy_dim, K * C),
        )

    def forward(self, lm_hidden: torch.Tensor):
        """
        lm_hidden:
          - (B, T, D) or (B, D)
        returns:
          S: (B, K, C)
          z: (B, strategy_dim)
        """
        if lm_hidden.dim() == 3:
            # (B,T,D) -> pooled (B,D)
            if self.pool == "mean":
                pooled = lm_hidden.mean(dim=1)
            else:
                pooled = lm_hidden[:, 0]  # fallback: first token
        else:
            pooled = lm_hidden

        z = self.z_proj(pooled)  # (B, strategy_dim)
        s_flat = self.s_proj(z)  # (B, K*C)
        S = s_flat.view(-1, self.K, self.C)  # (B,K,C)
        return S, z
