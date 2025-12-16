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

# 100iteration당 저장
# G:/VETNet_pilot/trainers/train_phase2_pilot.py

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

# SSIM
try:
    from pytorch_msssim import ssim as msssim_ssim
    _HAS_MSSSIM = True
    print("[Loss] Using pytorch_msssim.ssim")
except Exception:
    _HAS_MSSSIM = False
    print("[Loss] pytorch_msssim not found -> SSIM disabled")

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

def build_pairs_from_preload_cache(root_dir: str):
    pairs = []
    for d in sorted(os.listdir(root_dir)):
        dpath = os.path.join(root_dir, d)
        if not os.path.isdir(dpath):
            continue
        ins = [f for f in os.listdir(dpath) if f.endswith("_in.png")]
        for f in sorted(ins):
            base = f.replace("_in.png", "")
            inp = os.path.join(dpath, f)
            gt = os.path.join(dpath, base + "_gt.png")
            if os.path.isfile(gt):
                pairs.append((inp, gt))
        print(f"[DATA] {d}: {len(ins)} pairs")
    return pairs

# ============================================================
# Config
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

    # 🔥 preview 저장 경로
    preview_dir: str = r"G:\VETNet_pilot\results\phase2_pilot\iter_preview"
    preview_interval: int = 100

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
    device = torch.device(cfg.device)
    print("[Device]", device)

    _safe_makedirs(cfg.out_dir)
    _safe_makedirs(cfg.preview_dir)

    if os.path.isfile(cfg.pairs_cache_json):
        with open(cfg.pairs_cache_json) as f:
            pairs = [(d["inp"], d["gt"]) for d in json.load(f)]
        print(f"[CACHE] Loaded pairs = {len(pairs)}")
    else:
        pairs = build_pairs_from_preload_cache(cfg.preload_cache_root)
        with open(cfg.pairs_cache_json, "w") as f:
            json.dump([{"inp": a, "gt": b} for a, b in pairs], f, indent=2)

    dataset = PreloadCacheDataset(pairs, cfg.patch_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True)

    model = VLLMVETNet(VLLMVETNetConfig(
        backbone_dim=cfg.backbone_dim,
        backbone_num_blocks=cfg.backbone_num_blocks,
        backbone_heads=cfg.backbone_heads,
        backbone_volterra_rank=cfg.backbone_volterra_rank,
        stage_dims=cfg.stage_dims,
        strategy_dim=cfg.strategy_dim,
        num_tokens=cfg.num_tokens,
        enable_llm=cfg.enable_llm,
        enabled_stages=cfg.enabled_stages,
    )).to(device)

    model.load_phase1_backbone(cfg.phase1_ckpt)
    model.freeze_backbone()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
    criterion = Phase2Loss(cfg.lambda_ssim)

    global_iter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"[Epoch {epoch}/{cfg.epochs}]")
        for batch in pbar:
            global_iter += 1
            inp = batch["inp"].to(device)
            gt = batch["gt"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                pred, _ = model(img_hr=inp, use_strategy=True, dataset_tag=cfg.dataset_tag)
                loss = criterion(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # 🔥 100 iter마다 preview 저장
            if global_iter % cfg.preview_interval == 0:
                name = os.path.basename(batch["inp_path"][0]).replace("_in.png", "")
                out_img = torch.cat([inp, pred, gt], dim=3)
                save_path = os.path.join(
                    cfg.preview_dir,
                    f"ep{epoch:02d}_it{global_iter:06d}_{name}.png"
                )
                save_image(out_img, save_path)
                print(f"[Preview] saved -> {save_path}")

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        torch.save(
            {"epoch": epoch, "model": model.state_dict()},
            os.path.join(cfg.out_dir, f"phase2_epoch_{epoch:03d}.pth")
        )

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    cfg = Phase2Config()
    train_phase2(cfg)
