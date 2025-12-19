# G:/VETNet_pilot/trainers/train_phase2_pilot.py
# Phase-2 training: freeze backbone, train (LoRA/strategy_head + control_projection (+ optional vision_adapter))
# - No torch.no_grad(): backbone frozen이어도 그래프는 유지되어 projection/LoRA로 grad가 흘러야 함
# - Includes list->dict safety for control tokens
# - Includes checkpointing + logging + simple validation
""" 
import os
import sys
import json
import time
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ------------------------------------------------------------
# ROOT 경로 세팅
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))     # .../trainers
ROOT = os.path.dirname(CURRENT_DIR)                          # .../VETNet_pilot
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
print(f"[train_phase2_pilot] ROOT = {ROOT}")

# ------------------------------------------------------------
# Imports (project)
# ------------------------------------------------------------
from models.vllm_vetnet import VLLMVETNet, VLLMVETNetConfig

# SSIM (optional)
try:
    from pytorch_msssim import ssim as msssim_ssim
    _HAS_MSSSIM = True
    print("[Loss] Using pytorch_msssim.ssim")
except Exception:
    _HAS_MSSSIM = False
    print("[Loss] pytorch_msssim not found -> SSIM term disabled (L1 only)")

# ============================================================
# Utils
# ============================================================
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_makedirs(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _to_01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def _psnr(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mse = (pred - gt).pow(2).mean(dim=(1, 2, 3)).clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


# ============================================================
# Dataset
# ============================================================
class PreloadCacheDataset(Dataset):

    def __init__(self, pairs: List[Tuple[str, str]], patch_size: int = 256):
        self.pairs = pairs
        self.patch_size = patch_size
        from torchvision.io import read_image
        self._read_image = read_image

    def __len__(self):
        return len(self.pairs)

    def _load(self, p: str) -> torch.Tensor:
        x = self._read_image(p).float() / 255.0
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        return x[:3]

    def __getitem__(self, idx):
        inp_path, gt_path = self.pairs[idx]
        inp = self._load(inp_path)
        gt = self._load(gt_path)

        _, H, W = inp.shape
        ps = self.patch_size
        if H > ps and W > ps:
            top = random.randint(0, H - ps)
            left = random.randint(0, W - ps)
            inp = inp[:, top:top+ps, left:left+ps]
            gt = gt[:, top:top+ps, left:left+ps]

        return {
            "inp": inp,
            "gt": gt,
            "inp_path": inp_path,
            "gt_path": gt_path,
        }


def build_pairs_from_preload_cache(root_dir: str) -> List[Tuple[str, str]]:
    pairs = []
    for d in sorted(os.listdir(root_dir)):
        dpath = os.path.join(root_dir, d)
        if not os.path.isdir(dpath):
            continue

        files = os.listdir(dpath)
        ins = sorted([f for f in files if f.endswith("_in.png")])
        for f in ins:
            base = f.replace("_in.png", "")
            inp = os.path.join(dpath, f)
            gt = os.path.join(dpath, base + "_gt.png")
            if os.path.isfile(gt):
                pairs.append((inp, gt))

        print(f"[DATA] {d}: {len(ins)} pairs")

    return pairs


# ============================================================
# Phase-2 Config
# ============================================================
@dataclass
class Phase2Config:
    seed: int = 1234
    device: str = "cuda"
    use_amp: bool = True

    phase1_ckpt: str = r"G:\VETNet_pilot\checkpoints\phase1_backbone\epoch_089_L0.0085_P38.07_S0.9662.pth"
    out_dir: str = r"G:\VETNet_pilot\checkpoints\phase2_pilot"

    preload_cache_root: str = r"G:\VETNet_pilot\preload_cache"
    pairs_cache_json: str = r"G:\VETNet_pilot\data_cache\pairs_cache_phase2.json"

    batch_size: int = 1
    num_workers: int = 4
    patch_size: int = 256
    epochs: int = 10
    lr: float = 1e-4
    lambda_ssim: float = 0.0

    backbone_dim: int = 64
    backbone_num_blocks: Tuple[int, int, int, int] = (4, 6, 6, 8)
    backbone_heads: Tuple[int, int, int, int] = (1, 2, 4, 8)
    backbone_volterra_rank: int = 4

    strategy_dim: int = 256
    num_tokens: int = 4
    stage_dims: Tuple[int, int, int, int] = (64, 128, 256, 512)

    enable_llm: bool = False
    enabled_stages: Tuple[str, ...] = ("stage1", "stage2", "stage3", "stage4")
    dataset_tag: str = "MultiPreload"


# ============================================================
# Loss
# ============================================================
class Phase2Loss(nn.Module):
    def __init__(self, lambda_ssim: float):
        super().__init__()
        self.lambda_ssim = lambda_ssim

    def forward(self, pred, gt):
        pred = _to_01(pred)
        gt = _to_01(gt)
        l1 = (pred - gt).abs().mean()
        if self.lambda_ssim > 0 and _HAS_MSSSIM:
            s = msssim_ssim(pred, gt, data_range=1.0, size_average=True)
            return l1 + self.lambda_ssim * (1 - s)
        return l1


# ============================================================
# Train
# ============================================================
def train_phase2(cfg: Phase2Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    _safe_makedirs(cfg.out_dir)
    _safe_makedirs(os.path.dirname(cfg.pairs_cache_json))

    if os.path.isfile(cfg.pairs_cache_json):
        with open(cfg.pairs_cache_json, "r") as f:
            pairs = [(d["inp"], d["gt"]) for d in json.load(f)]
        print(f"[CACHE] Loaded pairs = {len(pairs)}")
    else:
        pairs = build_pairs_from_preload_cache(cfg.preload_cache_root)
        with open(cfg.pairs_cache_json, "w") as f:
            json.dump([{"inp": a, "gt": b} for a, b in pairs], f, indent=2)
        print(f"[CACHE] Saved pairs = {len(pairs)}")

    if len(pairs) == 0:
        raise RuntimeError("No training pairs found")

    dataset = PreloadCacheDataset(pairs, patch_size=cfg.patch_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size,
                        shuffle=True, num_workers=cfg.num_workers,
                        pin_memory=True, drop_last=True)

    mcfg = VLLMVETNetConfig(
        backbone_dim=cfg.backbone_dim,
        backbone_num_blocks=cfg.backbone_num_blocks,
        backbone_heads=cfg.backbone_heads,
        backbone_volterra_rank=cfg.backbone_volterra_rank,
        stage_dims=cfg.stage_dims,
        strategy_dim=cfg.strategy_dim,
        num_tokens=cfg.num_tokens,
        enable_llm=cfg.enable_llm,
        enabled_stages=cfg.enabled_stages,
    )
    model = VLLMVETNet(mcfg).to(device)
    model.load_phase1_backbone(cfg.phase1_ckpt)
    model.freeze_backbone()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[Trainable Params] {sum(p.numel() for p in trainable)/1e6:.3f} M")

    opt = torch.optim.AdamW(trainable, lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
    criterion = Phase2Loss(cfg.lambda_ssim)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"[Epoch {epoch}/{cfg.epochs}]")
        for batch in pbar:
            inp = batch["inp"].to(device)
            gt = batch["gt"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                pred, _ = model(img_hr=inp, use_strategy=True, dataset_tag=cfg.dataset_tag)

                # -----------------------------
                # 🔎 DEBUG: 그래프 연결 체크
                # -----------------------------
                z = getattr(model, "_last_strategy_z", None)
                z_req = None if z is None else bool(z.requires_grad)
                print(
                    "[DEBUG]",
                    "pred.requires_grad =", bool(pred.requires_grad),
                    "| z.requires_grad =", z_req
                )

                if not pred.requires_grad:
                    raise RuntimeError("pred.requires_grad=False → no_grad/detach 존재")

                loss = criterion(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        ckpt_path = os.path.join(cfg.out_dir, f"phase2_epoch_{epoch:03d}.pth")
        torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_path)
        print("[Saved]", ckpt_path)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    cfg = Phase2Config()
    train_phase2(cfg)
 """

""" import os
import sys
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

# ------------------------------------------------------------
# ROOT 경로 세팅
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
print(f"[train_phase2_pilot] ROOT = {ROOT}")

# ------------------------------------------------------------
# Imports (project)
# ------------------------------------------------------------
from models.vllm_vetnet import VLLMVETNet, VLLMVETNetConfig

# ------------------------------------------------------------
# SSIM (metric & optional loss)
# ------------------------------------------------------------
try:
    from pytorch_msssim import ssim as msssim_ssim
    _HAS_MSSSIM = True
    print("[Loss/Metric] Using pytorch_msssim.ssim")
except Exception:
    _HAS_MSSSIM = False
    print("[Loss/Metric] pytorch_msssim not found -> SSIM disabled")

# ============================================================
# Utils
# ============================================================
def set_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)

def _to_01(x: torch.Tensor):
    return x.clamp(0.0, 1.0)

@torch.no_grad()
def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    # ✅ metrics는 FP32 강제
    pred = _to_01(pred).float()
    gt = _to_01(gt).float()
    mse = (pred - gt).pow(2).mean(dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
    return psnr.mean().item()

@torch.no_grad()
def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    if not _HAS_MSSSIM:
        return float("nan")
    # ✅ SSIM은 FP32 강제 (AMP half 방지)
    pred = _to_01(pred).float()
    gt = _to_01(gt).float()
    s = msssim_ssim(pred, gt, data_range=1.0, size_average=True)
    return float(s.item())

def _isnan(x: float) -> bool:
    return x != x

# ============================================================
# EMA Helper
# ============================================================
class EMA:
    def __init__(self, beta: float = 0.95):
        self.beta = beta
        self.v: Dict[str, float] = {}

    def update(self, key: str, value: float):
        if _isnan(value):
            return
        if key not in self.v:
            self.v[key] = value
        else:
            self.v[key] = self.beta * self.v[key] + (1.0 - self.beta) * value

    def get(self, key: str, default: float = float("nan")) -> float:
        return self.v.get(key, default)

# ============================================================
# Dataset
# ============================================================
class PreloadCacheDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], patch_size: int = 256):
        self.pairs = pairs
        self.patch_size = patch_size
        from torchvision.io import read_image
        self._read_image = read_image

    def __len__(self):
        return len(self.pairs)

    def _load(self, p: str):
        x = self._read_image(p).float() / 255.0
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        return x[:3]

    def __getitem__(self, idx):
        inp_path, gt_path = self.pairs[idx]
        inp = self._load(inp_path)
        gt = self._load(gt_path)

        _, H, W = inp.shape
        ps = self.patch_size
        if H > ps and W > ps:
            top = random.randint(0, H - ps)
            left = random.randint(0, W - ps)
            inp = inp[:, top:top+ps, left:left+ps]
            gt = gt[:, top:top+ps, left:left+ps]

        return {
            "inp": inp,
            "gt": gt,
            "inp_path": inp_path,
            "gt_path": gt_path,
        }

# ============================================================
# Config
# ============================================================
@dataclass
class Phase2Config:
    seed: int = 1234
    device: str = "cuda"
    use_amp: bool = True

    phase1_ckpt: str = r"G:\VETNet_pilot\checkpoints\phase1_backbone\epoch_060_L0.0116_P35.76_S0.9594.pth"
    out_dir: str = r"G:\VETNet_pilot\checkpoints\phase2_pilot"

    preload_cache_root: str = r"G:\VETNet_pilot\preload_cache"
    pairs_cache_json: str = r"G:\VETNet_pilot\data_cache\pairs_cache_phase2.json"

    preview_dir: str = r"G:\VETNet_pilot\results\phase2_pilot\iter_preview"
    preview_interval: int = 100

    # ✅ metrics 계산/갱신 주기
    metric_interval: int = 20

    # ✅ alpha warm-up: 0 -> alpha_max 까지 선형 증가
    alpha_min: float = 0.3
    alpha_max: float = 0.7  # ⭐ 0.3~0.7 구간에 오래 머무르게
    warmup_iters = 1

    # ✅ strategy 폭주 방지: token 노름 클램프
    token_norm_clip: float = 5.0

    # ✅ modulation 폭주 방지: (모델 내부에서 사용될 수 있도록 전달 시도)
    modulation_scale_clip: float = 2.0

    batch_size: int = 1
    num_workers: int = 4
    patch_size: int = 256
    epochs: int = 120
    lr: float = 2e-5

    backbone_dim: int = 64
    backbone_num_blocks: Tuple[int, int, int, int] = (4, 6, 6, 8)
    backbone_heads: Tuple[int, int, int, int] = (1, 2, 4, 8)
    backbone_volterra_rank: int = 4

    strategy_dim: int = 256
    num_tokens: int = 4
    stage_dims: Tuple[int, int, int, int] = (64, 128, 256, 512)

    enable_llm: bool = False
    enabled_stages: Tuple[str, ...] = ("stage1", "stage2", "stage3", "stage4")
    dataset_tag: str = "MultiPreload"

# ============================================================
# Strategy clamp (best-effort)
# ============================================================
def try_enable_strategy_clamp(model: nn.Module, token_norm_clip: float, modulation_scale_clip: float):
    candidates = []

    # 흔한 네이밍들
    for attr in ["strategy_router", "router", "control_projection", "projection", "strategy_head"]:
        if hasattr(model, attr):
            candidates.append(getattr(model, attr))

    # 모델 자신도 포함
    candidates.append(model)

    for m in candidates:
        # token norm clip
        for key in ["token_norm_clip", "strategy_token_norm_clip", "s_norm_clip", "clip_token_norm"]:
            if hasattr(m, key):
                try:
                    setattr(m, key, float(token_norm_clip))
                    print(f"[Clamp] set {m.__class__.__name__}.{key} = {token_norm_clip}")
                    break
                except Exception:
                    pass

        # modulation scale clip
        for key in ["modulation_scale_clip", "mod_scale_clip", "gamma_beta_clip", "clip_modulation_scale"]:
            if hasattr(m, key):
                try:
                    setattr(m, key, float(modulation_scale_clip))
                    print(f"[Clamp] set {m.__class__.__name__}.{key} = {modulation_scale_clip}")
                    break
                except Exception:
                    pass

# ============================================================
# Train
# ============================================================
def train_phase2(cfg: Phase2Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    _safe_makedirs(cfg.out_dir)
    _safe_makedirs(cfg.preview_dir)
    _safe_makedirs(os.path.dirname(cfg.pairs_cache_json))

    if not os.path.isfile(cfg.pairs_cache_json):
        raise FileNotFoundError(f"pairs_cache_json not found: {cfg.pairs_cache_json}")

    with open(cfg.pairs_cache_json) as f:
        pairs = [(d["inp"], d["gt"]) for d in json.load(f)]
    print(f"[CACHE] Loaded pairs = {len(pairs)}")

    dataset = PreloadCacheDataset(pairs, cfg.patch_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    model = VLLMVETNet(
        VLLMVETNetConfig(
            backbone_dim=cfg.backbone_dim,
            backbone_num_blocks=cfg.backbone_num_blocks,
            backbone_heads=cfg.backbone_heads,
            backbone_volterra_rank=cfg.backbone_volterra_rank,
            stage_dims=cfg.stage_dims,
            strategy_dim=cfg.strategy_dim,
            num_tokens=cfg.num_tokens,
            enable_llm=cfg.enable_llm,
            enabled_stages=cfg.enabled_stages,
        )
    ).to(device)

    model.load_phase1_backbone(cfg.phase1_ckpt)
    model.freeze_backbone()

    # ✅ clamp 옵션 시도(모델이 지원하면 적용)
    try_enable_strategy_clamp(model, cfg.token_norm_clip, cfg.modulation_scale_clip)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[Trainable Params] {sum(p.numel() for p in trainable)/1e6:.3f} M")

    opt = torch.optim.AdamW(trainable, lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    ema = EMA(beta=0.95)

    # 마지막 측정값(출력용)
    last = {
        "PSNR_ON": float("nan"),
        "SSIM_ON": float("nan"),
        "PSNR_OFF": float("nan"),
        "SSIM_OFF": float("nan"),
        "dPSNR": float("nan"),
        "dSSIM": float("nan"),
        "alpha": 0.0
    }

    global_iter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"[Epoch {epoch}/{cfg.epochs}]")

        for batch in pbar:
            global_iter += 1

            # ✅ 동일 batch, 동일 crop 보장: dataset에서 crop 완료된 텐서가 넘어옴
            inp = batch["inp"].to(device, non_blocking=True)
            gt  = batch["gt"].to(device, non_blocking=True)

            # ✅ alpha 스케줄: 0->alpha_max 선형, 이후 alpha_min~alpha_max 구간 유지
            ramp = min(1.0, global_iter / max(1, cfg.warmup_iters))
            alpha = cfg.alpha_min + (cfg.alpha_max - cfg.alpha_min) * ramp
            alpha = float(max(cfg.alpha_min, min(cfg.alpha_max, alpha)))

            opt.zero_grad(set_to_none=True)

            # ----------------------------
            # OFF (no grad) - baseline
            # ----------------------------
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
                    pred_off_ng, _ = model(
                        img_hr=inp,
                        use_strategy=False,
                        dataset_tag=cfg.dataset_tag,
                    )

            # ----------------------------
            # ON (grad) + BLEND
            # ----------------------------
            with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
                pred_on, _ = model(
                    img_hr=inp,
                    use_strategy=True,
                    dataset_tag=cfg.dataset_tag,
                )

                # ✅ 안정화: ON이 OFF를 망가뜨리지 않게 blend
                pred_blend = pred_off_ng + alpha * (pred_on - pred_off_ng)

                # ✅ loss는 학습 출력(pred_blend) 기준
                loss = (pred_blend - gt).abs().mean()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # ----------------------------
            # METRICS (주기적으로만 계산)
            # ----------------------------
            if global_iter % cfg.metric_interval == 0:
                model.eval()
                with torch.no_grad():
                    # ✅ 같은 batch로 ON/OFF 각각 재추론 (동일 inp)
                    with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
                        pred_on_m, _ = model(img_hr=inp, use_strategy=True, dataset_tag=cfg.dataset_tag)
                        pred_off_m, _ = model(img_hr=inp, use_strategy=False, dataset_tag=cfg.dataset_tag)

                    psnr_on  = compute_psnr(pred_on_m, gt)
                    psnr_off = compute_psnr(pred_off_m, gt)
                    ssim_on  = compute_ssim(pred_on_m, gt)
                    ssim_off = compute_ssim(pred_off_m, gt)

                    dpsnr = psnr_on - psnr_off
                    dssim = ssim_on - ssim_off

                # EMA 업데이트
                ema.update("PSNR_ON", psnr_on)
                ema.update("PSNR_OFF", psnr_off)
                ema.update("SSIM_ON", ssim_on)
                ema.update("SSIM_OFF", ssim_off)
                ema.update("dPSNR", dpsnr)
                ema.update("dSSIM", dssim)

                last.update({
                    "PSNR_ON": psnr_on,
                    "PSNR_OFF": psnr_off,
                    "SSIM_ON": ssim_on,
                    "SSIM_OFF": ssim_off,
                    "dPSNR": dpsnr,
                    "dSSIM": dssim,
                    "alpha": alpha
                })

                model.train()

            # ✅ 항상 출력: EMA를 기본으로 보여주고(안정적), last는 디버깅용으로 값 유지
            psnr_on_ema  = ema.get("PSNR_ON")
            psnr_off_ema = ema.get("PSNR_OFF")
            ssim_on_ema  = ema.get("SSIM_ON")
            ssim_off_ema = ema.get("SSIM_OFF")
            dpsnr_ema    = ema.get("dPSNR")
            dssim_ema    = ema.get("dSSIM")

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "PSNR_ON(EMA)":  f"{psnr_on_ema:.2f}"  if not _isnan(psnr_on_ema)  else "NA",
                "SSIM_ON(EMA)":  f"{ssim_on_ema:.4f}"  if not _isnan(ssim_on_ema)  else "NA",
                "PSNR_OFF(EMA)": f"{psnr_off_ema:.2f}" if not _isnan(psnr_off_ema) else "NA",
                "SSIM_OFF(EMA)": f"{ssim_off_ema:.4f}" if not _isnan(ssim_off_ema) else "NA",
                "dPSNR(EMA)":    f"{dpsnr_ema:.2f}"    if not _isnan(dpsnr_ema)    else "NA",
                "dSSIM(EMA)":    f"{dssim_ema:.4f}"    if not _isnan(dssim_ema)    else "NA",
                "alpha": f"{alpha:.3f}",
            })

            # ----------------------------
            # PREVIEW (blend 저장: 실제 학습 출력과 동일)
            # ----------------------------
            if global_iter % cfg.preview_interval == 0:
                with torch.no_grad():
                    model.eval()
                    with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
                        pred_off_v, _ = model(img_hr=inp, use_strategy=False, dataset_tag=cfg.dataset_tag)
                        pred_on_v, _ = model(img_hr=inp, use_strategy=True, dataset_tag=cfg.dataset_tag)
                        pred_blend_v = pred_off_v + alpha * (pred_on_v - pred_off_v)

                    name = os.path.basename(batch["inp_path"][0]).replace("_in.png", "")
                    out_img = torch.cat([inp, pred_blend_v, gt], dim=3)
                    save_path = os.path.join(
                        cfg.preview_dir,
                        f"ep{epoch:02d}_it{global_iter:06d}_{name}_a{alpha:.3f}.png"
                    )
                    save_image(out_img, save_path)
                    print(f"[Preview] saved -> {save_path}")
                    model.train()

        ckpt_path = os.path.join(cfg.out_dir, f"phase2_epoch_{epoch:03d}.pth")
        torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_path)
        print(f"[Saved] {ckpt_path}")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    cfg = Phase2Config()
    train_phase2(cfg)
 """

# 설정변경
import os
import sys
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

# ------------------------------------------------------------
# ROOT 경로 세팅
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
print(f"[train_phase2_pilot] ROOT = {ROOT}")

# ------------------------------------------------------------
# Imports (project)
# ------------------------------------------------------------
from models.vllm_vetnet import VLLMVETNet, VLLMVETNetConfig

# ------------------------------------------------------------
# SSIM (metric & optional loss)
# ------------------------------------------------------------
try:
    from pytorch_msssim import ssim as msssim_ssim
    _HAS_MSSSIM = True
    print("[Loss/Metric] Using pytorch_msssim.ssim")
except Exception:
    _HAS_MSSSIM = False
    print("[Loss/Metric] pytorch_msssim not found -> SSIM disabled")

# ============================================================
# Utils
# ============================================================
def set_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)

def _to_01(x: torch.Tensor):
    return x.clamp(0.0, 1.0)

@torch.no_grad()
def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = _to_01(pred).float()
    gt = _to_01(gt).float()
    mse = (pred - gt).pow(2).mean(dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
    return psnr.mean().item()

@torch.no_grad()
def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    if not _HAS_MSSSIM:
        return float("nan")
    pred = _to_01(pred).float()
    gt = _to_01(gt).float()
    s = msssim_ssim(pred, gt, data_range=1.0, size_average=True)
    return float(s.item())

def _isnan(x: float) -> bool:
    return x != x

# ============================================================
# EMA Helper
# ============================================================
class EMA:
    def __init__(self, beta: float = 0.95):
        self.beta = beta
        self.v: Dict[str, float] = {}

    def update(self, key: str, value: float):
        if _isnan(value):
            return
        if key not in self.v:
            self.v[key] = value
        else:
            self.v[key] = self.beta * self.v[key] + (1.0 - self.beta) * value

    def get(self, key: str, default: float = float("nan")) -> float:
        return self.v.get(key, default)

# ============================================================
# Dataset
# ============================================================
class PreloadCacheDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], patch_size: int = 256):
        self.pairs = pairs
        self.patch_size = patch_size
        from torchvision.io import read_image
        self._read_image = read_image

    def __len__(self):
        return len(self.pairs)

    def _load(self, p: str):
        x = self._read_image(p).float() / 255.0
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        return x[:3]

    def __getitem__(self, idx):
        inp_path, gt_path = self.pairs[idx]
        inp = self._load(inp_path)
        gt = self._load(gt_path)

        _, H, W = inp.shape
        ps = self.patch_size
        if H > ps and W > ps:
            top = random.randint(0, H - ps)
            left = random.randint(0, W - ps)
            inp = inp[:, top:top+ps, left:left+ps]
            gt = gt[:, top:top+ps, left:left+ps]

        return {
            "inp": inp,
            "gt": gt,
            "inp_path": inp_path,
            "gt_path": gt_path,
        }

# ============================================================
# Config
# ============================================================
@dataclass
class Phase2Config:
    seed: int = 1234
    device: str = "cuda"
    use_amp: bool = True

    phase1_ckpt: str = r"G:\VETNet_pilot\checkpoints\phase1_backbone\epoch_060_L0.0116_P35.76_S0.9594.pth"
    out_dir: str = r"G:\VETNet_pilot\checkpoints\phase2_pilot"

    preload_cache_root: str = r"G:\VETNet_pilot\preload_cache"
    pairs_cache_json: str = r"G:\VETNet_pilot\data_cache\pairs_cache_phase2.json"

    preview_dir: str = r"G:\VETNet_pilot\results\phase2_pilot\iter_preview"
    preview_interval: int = 100

    metric_interval: int = 20

    alpha_min: float = 0.3
    alpha_max: float = 0.7
    warmup_iters = 1

    token_norm_clip: float = 5.0
    modulation_scale_clip: float = 2.0

    batch_size: int = 1
    num_workers: int = 4
    patch_size: int = 256
    epochs: int = 120
    lr: float = 2e-5

    backbone_dim: int = 64
    backbone_num_blocks: Tuple[int, int, int, int] = (4, 6, 6, 8)
    backbone_heads: Tuple[int, int, int, int] = (1, 2, 4, 8)
    backbone_volterra_rank: int = 4

    strategy_dim: int = 256
    num_tokens: int = 4
    stage_dims: Tuple[int, int, int, int] = (64, 128, 256, 512)

    enable_llm: bool = False
    enabled_stages: Tuple[str, ...] = ("stage1", "stage2", "stage3", "stage4")
    dataset_tag: str = "MultiPreload"

# ============================================================
# Train
# ============================================================
def train_phase2(cfg: Phase2Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    with open(cfg.pairs_cache_json) as f:
        pairs = [(d["inp"], d["gt"]) for d in json.load(f)]
    print(f"[CACHE] Loaded pairs = {len(pairs)}")

    dataset = PreloadCacheDataset(pairs, cfg.patch_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    model = VLLMVETNet(
        VLLMVETNetConfig(
            backbone_dim=cfg.backbone_dim,
            backbone_num_blocks=cfg.backbone_num_blocks,
            backbone_heads=cfg.backbone_heads,
            backbone_volterra_rank=cfg.backbone_volterra_rank,
            stage_dims=cfg.stage_dims,
            strategy_dim=cfg.strategy_dim,
            num_tokens=cfg.num_tokens,
            enable_llm=cfg.enable_llm,
            enabled_stages=cfg.enabled_stages,
        )
    ).to(device)

    model.load_phase1_backbone(cfg.phase1_ckpt)
    model.freeze_backbone()

    # ===== resume (epoch 포함) =====
    ckpt_resume = r"G:\VETNet_pilot\checkpoints\phase2_pilot\phase2_epoch_011.pth"
    ckpt = torch.load(ckpt_resume, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    start_epoch = ckpt["epoch"] + 1
    print(f"[Resume] Loaded checkpoint: {ckpt_resume}")
    print(f"[Resume] Start epoch = {start_epoch}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    ema = EMA(beta=0.95)
    global_iter = 0

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"[Epoch {epoch}/{cfg.epochs}]")

        for batch in pbar:
            global_iter += 1
            inp = batch["inp"].to(device)
            gt = batch["gt"].to(device)

            alpha = cfg.alpha_max

            opt.zero_grad(set_to_none=True)

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=cfg.use_amp):
                pred_off, _ = model(inp, use_strategy=False, dataset_tag=cfg.dataset_tag)

            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                pred_on, _ = model(inp, use_strategy=True, dataset_tag=cfg.dataset_tag)
                pred = pred_off + alpha * (pred_on - pred_off)
                loss = (pred - gt).abs().mean()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if global_iter % cfg.metric_interval == 0:
                with torch.no_grad():
                    psnr_on = compute_psnr(pred_on, gt)
                    psnr_off = compute_psnr(pred_off, gt)
                    ssim_on = compute_ssim(pred_on, gt)
                    ssim_off = compute_ssim(pred_off, gt)
                    ema.update("PSNR_ON", psnr_on)
                    ema.update("PSNR_OFF", psnr_off)
                    ema.update("SSIM_ON", ssim_on)
                    ema.update("SSIM_OFF", ssim_off)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "PSNR_ON(EMA)": f"{ema.get('PSNR_ON'):.2f}",
                "SSIM_ON(EMA)": f"{ema.get('SSIM_ON'):.4f}",
                "PSNR_OFF(EMA)": f"{ema.get('PSNR_OFF'):.2f}",
                "SSIM_OFF(EMA)": f"{ema.get('SSIM_OFF'):.4f}",
                "alpha": f"{alpha:.3f}",
            })

        ckpt_path = os.path.join(cfg.out_dir, f"phase2_epoch_{epoch:03d}.pth")
        torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_path)
        print(f"[Saved] {ckpt_path}")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    cfg = Phase2Config()
    train_phase2(cfg)
