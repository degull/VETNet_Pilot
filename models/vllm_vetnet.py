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


# G:/VETNet_pilot/models/vllm_vetnet.py
# ============================================================
# Path A (StrategyHead) + Control Bridge + Path B (VETNetBackbone)
# ============================================================
# G:/VETNet_pilot/models/vllm_vetnet.py
# Path A(StrategyHead) + ControlBridge(ControlProjection/StrategyRouter) + Path B(VETNetBackbone)
# - Self-test: Phase-1 ckpt 로드 + strategy ON/OFF 비교
import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# ROOT 경로 세팅
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../models
ROOT = os.path.dirname(CURRENT_DIR)                        # .../VETNet_pilot
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print(f"[vllm_vetnet] ROOT = {ROOT}")

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from models.backbone.vetnet_backbone import VETNetBackbone
from models.pilot.strategy_head import StrategyHead, StrategyHeadConfig
from models.bridge.control_projection import ControlProjection, ControlProjectionConfig
from models.bridge.strategy_router import StrategyRouter, StrategyRouterConfig


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

    # strategy head
    clip_model_name: str = "openai/clip-vit-large-patch14"
    clip_image_size: int = 224
    strategy_dim: int = 256
    num_tokens: int = 4

    # stage dims
    stage_dims: Tuple[int, int, int, int] = (64, 128, 256, 512)

    # LLM
    enable_llm: bool = False

    # router stages
    enabled_stages: Tuple[str, ...] = ("stage1", "stage2", "stage3", "stage4")

    # strategy input image
    strategy_use_hr_as_input: bool = True


# ============================================================
# Model
# ============================================================
class VLLMVETNet(nn.Module):
    """
    forward(img_hr, img_lr=None, dataset_tag="Rain100H", use_strategy=True, generate_text=False)
      -> restored, strategy_texts
    """

    def __init__(self, cfg: VLLMVETNetConfig):
        super().__init__()
        self.cfg = cfg

        # debug cache
        self._last_strategy_z: Optional[torch.Tensor] = None
        self._last_tokens_dict: Optional[Dict[str, torch.Tensor]] = None

        # Backbone
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

        # StrategyHead
        sh_cfg = StrategyHeadConfig(
            clip_model_name=cfg.clip_model_name,
            clip_image_size=cfg.clip_image_size,
            strategy_dim=cfg.strategy_dim,
            num_tokens=cfg.num_tokens,
            token_dim=cfg.stage_dims[0],
            enable_llm=cfg.enable_llm,
        )
        self.strategy_head = StrategyHead(sh_cfg)

        # ControlProjection
        cp_cfg = ControlProjectionConfig(
            strategy_dim=cfg.strategy_dim,
            stage_dims=cfg.stage_dims,
            num_tokens=cfg.num_tokens,
        )
        self.control_projection = ControlProjection(cp_cfg)

        # StrategyRouter (detach 금지)
        rt_cfg = StrategyRouterConfig(
            use_strategy=True,
            enabled_stages=set(cfg.enabled_stages),
            detach_tokens=False,
        )
        self.strategy_router = StrategyRouter(rt_cfg)

    # --------------------------------------------------------
    @staticmethod
    def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            return ckpt["state_dict"]
        if isinstance(ckpt, dict):
            return ckpt
        raise ValueError("[VLLMVETNet] Unsupported checkpoint format")

    @staticmethod
    def _remap_phase1_key(k: str) -> str:
        return k.replace(".attn.", ".attn_conv.")

    def load_phase1_backbone(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd_raw = self._extract_state_dict(ckpt)

        sd = {self._remap_phase1_key(k): v for k, v in sd_raw.items()}
        model_sd = self.backbone.state_dict()

        load_sd = {}
        loaded, skipped = [], []

        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                load_sd[k] = v
                loaded.append(k)
            else:
                skipped.append(k)

        self.backbone.load_state_dict(load_sd, strict=False)

        print("[VLLMVETNet] Phase-1 backbone loaded")
        print("  - loaded:", len(loaded))
        print("  - skipped:", len(skipped))

    # --------------------------------------------------------
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    # --------------------------------------------------------
    def _unwrap_tokens_container(self, tokens_any: Any) -> Any:
        """
        control_projection이 종종 다음처럼 래핑해서 반환할 수 있음:
          {"tokens_dict": {...}} 또는 {"stage_tokens": {...}} 등

        이 함수는 그런 1단 래핑을 자동으로 벗긴다.
        """
        if not isinstance(tokens_any, dict):
            return tokens_any

        # 흔히 나오는 래핑 키들
        unwrap_keys = ("tokens_dict", "stage_tokens", "tokens", "controls", "out")
        for k in unwrap_keys:
            if k in tokens_any and isinstance(tokens_any[k], (dict, list, tuple)):
                return tokens_any[k]

        return tokens_any

    # --------------------------------------------------------
    def _coerce_stage_token(self, x: Any, stage_name: str) -> torch.Tensor:
        """
        stage token을 반드시 torch.Tensor로 정규화한다.

        케이스:
        - Tensor -> OK
        - [Tensor] 또는 (Tensor, ...) -> 첫 Tensor 사용
        """
        if isinstance(x, torch.Tensor):
            return x

        if isinstance(x, (list, tuple)):
            for t in x:
                if isinstance(t, torch.Tensor):
                    return t
            raise RuntimeError(f"[VLLMVETNet] stage={stage_name} token is list/tuple but contains no Tensor")

        raise RuntimeError(f"[VLLMVETNet] stage={stage_name} token is not Tensor/list/tuple. type={type(x)}")

    def _coerce_tokens_to_dict(self, tokens_any: Any) -> Dict[str, torch.Tensor]:
        """
        control_projection 출력이 dict/list/tuple 어느 형태든,
        최종적으로는 {'stage1': Tensor, ...} 형태로 강제한다.
        """
        if tokens_any is None:
            return {}

        # ✅ 1) 래핑 해제(핵심)
        tokens_any = self._unwrap_tokens_container(tokens_any)

        # 2) dict인 경우: value까지 Tensor로 보장
        if isinstance(tokens_any, dict):
            # 이미 stage1~4를 담고있는 dict여야 함
            out: Dict[str, torch.Tensor] = {}
            for k, v in tokens_any.items():
                out[str(k)] = self._coerce_stage_token(v, str(k))
            return out

        # 3) list/tuple인 경우: stage1~4로 매핑 후 value Tensor 보장
        if isinstance(tokens_any, (list, tuple)):
            if len(tokens_any) != 4:
                raise RuntimeError(f"[VLLMVETNet] control_projection returned list/tuple but len != 4: len={len(tokens_any)}")
            mapped = {
                "stage1": tokens_any[0],
                "stage2": tokens_any[1],
                "stage3": tokens_any[2],
                "stage4": tokens_any[3],
            }
            out2: Dict[str, torch.Tensor] = {}
            for k, v in mapped.items():
                out2[k] = self._coerce_stage_token(v, k)
            return out2

        raise RuntimeError(f"[VLLMVETNet] Unsupported tokens type from control_projection: {type(tokens_any)}")

    # --------------------------------------------------------
    def forward(
        self,
        img_hr: torch.Tensor,
        img_lr: Optional[torch.Tensor] = None,
        dataset_tag: str = "Generic",
        use_strategy: bool = True,
        generate_text: bool = False,
        extra_text: Optional[str] = None,
    ):
        self._last_strategy_z = None
        self._last_tokens_dict = None

        if img_lr is None:
            if self.cfg.strategy_use_hr_as_input:
                img_lr = img_hr
            else:
                img_lr = F.interpolate(img_hr, size=(336, 336), mode="bicubic", align_corners=False)

        strategy_texts = None

        if use_strategy:
            out = self.strategy_head(
                img_lr,
                dataset_tag=dataset_tag,
                extra_text=extra_text,
                generate_text=generate_text,
            )

            z = out["strategy_vector"]  # ✅ detach 금지
            strategy_texts = out["strategy_texts"]

            self._last_strategy_z = z

            tokens_any = self.control_projection(z)     # ✅ detach 금지
            tokens_dict = self._coerce_tokens_to_dict(tokens_any)

            self._last_tokens_dict = tokens_dict

            restored = self.backbone(
                img_hr,
                strategy_tokens=tokens_dict,
                router=self.strategy_router,
            )
        else:
            restored = self.backbone(
                img_hr,
                strategy_tokens=None,
                router=None,
            )

        return restored, strategy_texts


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print("\n[vllm_vetnet] Self-test 시작")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    cfg = VLLMVETNetConfig()
    model = VLLMVETNet(cfg).to(device)

    ckpt_path = r"G:\VETNet_pilot\checkpoints\phase1_backbone\epoch_089_L0.0085_P38.07_S0.9662.pth"
    if os.path.isfile(ckpt_path):
        model.load_phase1_backbone(ckpt_path)
    else:
        print("[WARNING] ckpt not found:", ckpt_path)

    model.freeze_backbone()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Trainable] {trainable/1e6:.3f}M / {total/1e6:.3f}M")

    B, H, W = 2, 256, 256
    img = torch.rand(B, 3, H, W, device=device)

    model.eval()
    with torch.no_grad():
        # (A) Model-level OFF vs ON
        out_off, _ = model(img_hr=img, img_lr=None, dataset_tag="Rain100H", use_strategy=False)
        out_on, _  = model(img_hr=img, img_lr=None, dataset_tag="Rain100H", use_strategy=True)
        diffA = (out_on - out_off).abs().mean().item()
        print("[A] mean(|ON-OFF|):", diffA)

        # (B) Backbone OFF vs Backbone + RANDOM TOKENS
        rand_tokens = {
            "stage1": torch.randn(B, 4, 64, device=device),
            "stage2": torch.randn(B, 4, 128, device=device),
            "stage3": torch.randn(B, 4, 256, device=device),
            "stage4": torch.randn(B, 4, 512, device=device),
        }
        y0 = model.backbone(img, strategy_tokens=None, router=None)
        yR = model.backbone(img, strategy_tokens=rand_tokens, router=model.strategy_router)
        diffB = (yR - y0).abs().mean().item()
        print("[B] random tokens diff:", diffB)

        # (C) CONTROL TOKENS
        out = model.strategy_head(img, dataset_tag="Rain100H", generate_text=False)
        z = out["strategy_vector"]

        ctrl_any = model.control_projection(z)
        ctrl_tokens = model._coerce_tokens_to_dict(ctrl_any)

        print("[C] ctrl token keys:", list(ctrl_tokens.keys()))
        for k, v in ctrl_tokens.items():
            print(f"  - {k}: type={type(v)} shape={tuple(v.shape)}")

        t_norm = {k: float(v.abs().mean().item()) for k, v in ctrl_tokens.items()}
        print("[C] control token norm:", t_norm)

        yC = model.backbone(img, strategy_tokens=ctrl_tokens, router=model.strategy_router)
        diffC = (yC - y0).abs().mean().item()
        print("[C] control tokens diff:", diffC)

    print("\n[vllm_vetnet] Self-test 완료\n")
