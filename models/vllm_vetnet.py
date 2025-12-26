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

""" from __future__ import annotations

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

    #Path A + Bridge + Path B 통합 wrapper.
    #- forward()에서 strategy를 만들고,
    #  backbone에 stage-wise tokens를 주입하려고 시도한다.


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
        #Phase1에서 학습된 backbone ckpt를 로드.
        #ckpt가 dict면 state_dict 키를 자동 탐지.
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
        return F.interpolate(img_hr, size=(lr_size, lr_size), mode="bicubic", align_corners=False)

    # ------------------------------------------------------------------
    def _try_backbone_forward(
        self,
        img_hr: torch.Tensor,
        tokens_by_stage: List[torch.Tensor],
        tokens_dict: Dict[str, torch.Tensor],
        dataset_tag: Optional[str],
    ) -> torch.Tensor:

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

    #Self-test 목표:
    #1) StrategyHead -> Z 생성이 되는지
    #2) ControlProjection -> stage별 tokens shape이 맞는지
    #3) Backbone에 주입 시도 -> TypeError 없이 돌아가면 "주입 인터페이스 OK"
    #   (만약 Backbone이 아직 주입을 안 받으면 경고 출력 + backbone만 실행)

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
 """


# ============================================================
# Path A (StrategyHead) + Control Bridge + Path B (VETNetBackbone)
# ============================================================

# Path A(StrategyHead) + ControlBridge(ControlProjection/StrategyRouter) + Path B(VETNetBackbone)
# - Self-test: Phase-1 ckpt 로드 + strategy ON/OFF 비교

# E:\VETNet_Pilot\models\vllm_vetnet.py

# ============================================================
# VLLM-VETNet (Phase2/3 pilot wrapper)
# - backbone: VETNetBackbone
# - strategy_head: produces strategy vector (and optional texts)
# - control_projection: maps strategy vector -> stage tokens
# - strategy_router: routes tokens into backbone stages
#
# Goals of this file:
# 1) Run in multiple project states (StrategyHeadConfig may/may not exist)
# 2) Accept different StrategyHead.forward signatures (with/without dataset_tag)
# 3) Accept different output formats from StrategyHead and ControlProjection
# 4) Provide a self-test that prints:
#    [A] mean(|ON-OFF|) > 0
#    [B] random tokens diff > 0
#    [C] control tokens diff > 0
# ============================================================

# E:\VETNet_Pilot\models\vllm_vetnet.py
# E:\VETNet_Pilot\models\vllm_vetnet.py
import os
import sys
import inspect
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# ROOT 경로 세팅
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print(f"[vllm_vetnet] ROOT = {ROOT}")

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from models.backbone.vetnet_backbone import VETNetBackbone
from models.bridge.control_projection import ControlProjection, ControlProjectionConfig
from models.bridge.strategy_router import StrategyRouter, StrategyRouterConfig

# ============================================================
# StrategyHead import (robust)
# ============================================================
try:
    from models.pilot.strategy_head import StrategyHead, StrategyHeadConfig  # type: ignore
    print("[vllm_vetnet] StrategyHeadConfig import OK")
except Exception as e:
    from models.pilot.strategy_head import StrategyHead  # type: ignore
    print("[vllm_vetnet] StrategyHeadConfig missing -> fallback")
    print("  - import error:", repr(e))

    @dataclass
    class StrategyHeadConfig:
        # Phase-2에서는 StrategyHead를 "latent generator"로만 사용하므로
        # 최소한 StrategyHead.__init__이 요구하는 인자(lm_dim 등)를 채워준다.
        lm_dim: int = 256
        strategy_dim: int = 256
        num_tokens: int = 4
        enable_llm: bool = False


# ============================================================
# Config
# ============================================================
@dataclass
class VLLMVETNetConfig:
    # backbone
    backbone_in_channels: int = 3
    backbone_out_channels: int = 3
    backbone_dim: int = 64
    backbone_num_blocks: Tuple[int, int, int, int] = (4, 6, 6, 8)
    backbone_heads: Tuple[int, int, int, int] = (1, 2, 4, 8)
    backbone_volterra_rank: int = 4
    backbone_ffn_expansion_factor: float = 2.66
    backbone_bias: bool = False

    # strategy/control
    strategy_dim: int = 256
    num_tokens: int = 4
    stage_dims: Tuple[int, int, int, int] = (64, 128, 256, 512)
    enabled_stages: Tuple[str, ...] = ("stage1", "stage2", "stage3", "stage4")
    enable_llm: bool = False

    # ------------------------------------------------
    # (1) 토큰 영향 게이트 (learnable)
    #   - 초기에는 거의 0(OFF와 유사)로 시작 → 점진적으로 토큰 영향 학습
    # ------------------------------------------------
    token_gate_init: float = -4.0   # sigmoid(-4) ≈ 0.018

    # ------------------------------------------------
    # (2) backward 에러 방지 + grad 경로 강제 연결
    # backbone이 freeze되어 있고 backbone이 tokens를 무시해도
    # loss.requires_grad=False가 되지 않도록 restored에 아주 작은 제어 스칼라를 더한다.
    # ------------------------------------------------
    grad_hook_eps: float = 1e-4


# ============================================================
# Helper: checkpoint state_dict extraction
# ============================================================
def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format")


# ============================================================
# StrategyHead builder (FIXED)
# ============================================================
def _build_strategy_head(cfg: VLLMVETNetConfig) -> nn.Module:
    """
    Phase-2에서는 StrategyHead를 'latent generator'로만 사용.
    StrategyHead.__init__이 cfg-based인지 kwargs-based인지 모두 대응.
    lm_dim이 필요한 구현이 있으니 fallback에서도 반드시 제공.
    """
    sh_cfg = StrategyHeadConfig(
        lm_dim=cfg.strategy_dim,          # 중요: StrategyHead가 요구하는 lm_dim 채움
        strategy_dim=cfg.strategy_dim,
        num_tokens=cfg.num_tokens,
        enable_llm=cfg.enable_llm,
    )

    sig = inspect.signature(StrategyHead.__init__)
    arg_names = [p.name for p in sig.parameters.values() if p.name != "self"]

    # __init__(self, cfg) 형태
    if len(arg_names) == 1:
        try:
            return StrategyHead(sh_cfg)
        except Exception as e:
            print("[vllm_vetnet] StrategyHead(cfg) failed -> try kwargs. err:", repr(e))

    # kwargs 형태
    kwargs = asdict(sh_cfg) if hasattr(sh_cfg, "__dataclass_fields__") else sh_cfg.__dict__
    kwargs = {k: v for k, v in kwargs.items() if k in arg_names}

    # 일부 구현이 lm_dim을 "required positional"로 잡아두는 경우 방지
    if "lm_dim" in arg_names and "lm_dim" not in kwargs:
        kwargs["lm_dim"] = cfg.strategy_dim

    return StrategyHead(**kwargs)


# ============================================================
# StrategyHead call (Phase-2 ONLY)
# ============================================================
def _call_strategy_head_phase2(sh: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    StrategyHead 출력 포맷이 다양할 수 있으므로 robust 파싱.
    """
    out = sh(x)

    if isinstance(out, dict):
        for k in ("strategy_vector", "z", "latent", "embedding", "feat", "features"):
            if k in out and isinstance(out[k], torch.Tensor):
                return out[k]
        tensor_keys = [k for k, v in out.items() if isinstance(v, torch.Tensor)]
        if len(tensor_keys) == 1:
            return out[tensor_keys[0]]

    if isinstance(out, (list, tuple)) and len(out) >= 1 and isinstance(out[0], torch.Tensor):
        return out[0]

    if isinstance(out, torch.Tensor):
        return out

    raise RuntimeError(f"Invalid StrategyHead output type: {type(out)}")


# ============================================================
# ControlProjection output normalization (dict로 강제)
# ============================================================
def _unwrap_tokens_container(tokens_any: Any) -> Any:
    if not isinstance(tokens_any, dict):
        return tokens_any
    for k in ("tokens_dict", "stage_tokens", "tokens", "controls", "out"):
        if k in tokens_any and isinstance(tokens_any[k], (dict, list, tuple, torch.Tensor)):
            return tokens_any[k]
    return tokens_any


def _coerce_stage_token(x: Any, stage_name: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for t in x:
            if isinstance(t, torch.Tensor):
                return t
    raise RuntimeError(f"[vllm_vetnet] stage={stage_name} token type not supported: {type(x)}")


def _coerce_tokens_to_dict(tokens_any: Any, num_tokens: int, stage_dims: Tuple[int, int, int, int]) -> Dict[str, torch.Tensor]:
    if tokens_any is None:
        return {}

    tokens_any = _unwrap_tokens_container(tokens_any)

    if isinstance(tokens_any, dict):
        out: Dict[str, torch.Tensor] = {}
        for k, v in tokens_any.items():
            out[str(k)] = _coerce_stage_token(v, str(k))
        return out

    if isinstance(tokens_any, (list, tuple)):
        if len(tokens_any) != 4:
            raise RuntimeError(f"[vllm_vetnet] control_projection returned list/tuple len={len(tokens_any)} != 4")
        return {
            "stage1": _coerce_stage_token(tokens_any[0], "stage1"),
            "stage2": _coerce_stage_token(tokens_any[1], "stage2"),
            "stage3": _coerce_stage_token(tokens_any[2], "stage3"),
            "stage4": _coerce_stage_token(tokens_any[3], "stage4"),
        }

    if isinstance(tokens_any, torch.Tensor):
        B, K, C = tokens_any.shape
        out2: Dict[str, torch.Tensor] = {}
        for i, sd in enumerate(stage_dims):
            name = f"stage{i+1}"
            if C == sd:
                out2[name] = tokens_any
            elif C > sd:
                out2[name] = tokens_any[:, :, :sd]
            else:
                pad = sd - C
                out2[name] = torch.cat(
                    [tokens_any, torch.zeros(B, K, pad, device=tokens_any.device, dtype=tokens_any.dtype)],
                    dim=-1,
                )
        return out2

    raise RuntimeError(f"[vllm_vetnet] Unsupported tokens type from control_projection: {type(tokens_any)}")


def _tokens_scalar(tokens_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    if not tokens_dict:
        raise RuntimeError("[vllm_vetnet] tokens_dict is empty")
    s = None
    for t in tokens_dict.values():
        v = t.abs().mean()
        s = v if s is None else (s + v)
    return s


def _apply_token_gate(tokens_dict: Dict[str, torch.Tensor], gate: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    gate: scalar tensor (0~1). 각 stage token에 동일 스케일을 곱한다.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in tokens_dict.items():
        out[k] = v * gate
    return out


# ============================================================
# Main Model
# ============================================================
class VLLMVETNet(nn.Module):
    """
    Phase-2 Pilot Model
    - backbone은 phase1 로드 후 freeze
    - StrategyHead + ControlProjection + token_gate만 학습
    - token_gate로 초반 토큰 영향이 거의 0이 되게 해서 ON이 망가지는 걸 방지
    - backbone이 tokens를 무시해도 backward가 죽지 않도록 grad_hook 유지
    """

    def __init__(self, cfg: VLLMVETNetConfig):
        super().__init__()
        self.cfg = cfg

        # debug caches
        self._last_z: Optional[torch.Tensor] = None
        self._last_tokens_dict: Optional[Dict[str, torch.Tensor]] = None
        self._last_gate: Optional[torch.Tensor] = None

        self.backbone = VETNetBackbone(
            in_channels=cfg.backbone_in_channels,
            out_channels=cfg.backbone_out_channels,
            dim=cfg.backbone_dim,
            num_blocks=cfg.backbone_num_blocks,
            heads=cfg.backbone_heads,
            volterra_rank=cfg.backbone_volterra_rank,
            ffn_expansion_factor=cfg.backbone_ffn_expansion_factor,
            bias=cfg.backbone_bias,
        )

        self.strategy_head = _build_strategy_head(cfg)

        self.control_projection = ControlProjection(
            ControlProjectionConfig(
                strategy_dim=cfg.strategy_dim,
                stage_dims=cfg.stage_dims,
                num_tokens=cfg.num_tokens,
            )
        )

        self.strategy_router = StrategyRouter(
            StrategyRouterConfig(
                use_strategy=True,
                enabled_stages=set(cfg.enabled_stages),
                detach_tokens=False,
            )
        )

        # ------------------------------------------------
        # ✅ Learnable token gate (초반 안정화 핵심)
        # sigmoid(token_gate_logit) 이 실제 gate 값.
        # ------------------------------------------------
        self.token_gate_logit = nn.Parameter(torch.tensor(float(cfg.token_gate_init)))

    def load_phase1_backbone(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = _extract_state_dict(ckpt)
        self.backbone.load_state_dict(sd, strict=False)
        print("[VLLMVETNet] Phase-1 backbone loaded")

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def forward(
        self,
        img_hr: torch.Tensor,
        img_lr: Optional[torch.Tensor] = None,
        dataset_tag: str = "Generic",
        use_strategy: bool = True,
    ):
        if img_lr is None:
            img_lr = img_hr

        # reset caches
        self._last_z = None
        self._last_tokens_dict = None
        self._last_gate = None

        if use_strategy:
            pooled = img_lr.mean(dim=(2, 3))  # (B,3)

            if pooled.shape[1] != self.cfg.strategy_dim:
                rep = (self.cfg.strategy_dim + pooled.shape[1] - 1) // pooled.shape[1]
                pooled = pooled.repeat(1, rep)[:, : self.cfg.strategy_dim]  # (B, D)

            z = _call_strategy_head_phase2(self.strategy_head, pooled)

            if isinstance(z, torch.Tensor) and z.dim() == 3:
                z = z.mean(dim=1)
            elif isinstance(z, torch.Tensor) and z.dim() == 1:
                z = z.unsqueeze(0)
            elif not isinstance(z, torch.Tensor) or z.dim() != 2:
                raise RuntimeError(
                    f"[vllm_vetnet] StrategyHead z must be (B,D) or (B,T,D), got {type(z)} shape={getattr(z,'shape',None)}"
                )

            self._last_z = z

            tokens_any = self.control_projection(z)
            tokens_dict = _coerce_tokens_to_dict(tokens_any, self.cfg.num_tokens, self.cfg.stage_dims)

            # ✅ apply learnable gate (초반 ON이 OFF에 가깝게 시작)
            gate_raw = torch.sigmoid(self.token_gate_logit)

            if self.training and hasattr(self, "current_epoch") and self.current_epoch <= 10:
                gate = gate_raw.clamp(max=0.2)
            else:
                gate = gate_raw

            tokens_dict = _apply_token_gate(tokens_dict, gate)
            self._last_gate = gate.detach()

            self._last_tokens_dict = tokens_dict

            restored = self.backbone(
                img_hr,
                strategy_tokens=tokens_dict,
                router=self.strategy_router,
            )

            # ✅ grad_hook 유지 (backbone이 tokens를 무시해도 backward가 죽지 않게)
            B = img_hr.shape[0]
            ctrl_s = _tokens_scalar(tokens_dict)  # requires_grad=True
            restored = restored + (self.cfg.grad_hook_eps * ctrl_s).view(B, 1, 1, 1)

        else:
            restored = self.backbone(
                img_hr,
                strategy_tokens=None,
                router=None,
            )

        return restored, None
