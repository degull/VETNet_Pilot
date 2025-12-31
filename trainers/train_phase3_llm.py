# E:\VETNet_Pilot\trainers\train_phase3_llm_clean.py
# ------------------------------------------------------------
# Phase-3 CLEAN Trainer (Cached CLIP-Vision ONLY) + Method-1 XAI
#
# ✅ ABSOLUTE RULE:
#   - Phase-3 NEVER runs CLIP Vision (no import, no load, no forward)
#   - Vision embedding z_v is loaded ONLY from *_clip.pt
#
# Model:
#   - Backbone: VETNetBackbone (frozen, Phase-1 ckpt)
#   - Optional Teacher: Phase-2 GateController (frozen)
#   - Policy: Phase3GatePolicy (trainable) maps (z_v, z_t) -> g_stage
#
# Loss:
#   L = L1(pred, gt) + w_cons * MSE(g_pred, g_teacher)  (optional)
#
# Text:
#   Prefer xai_llm -> xai_blip -> fallback prompt
#   (Text is NOT fixed: it is per-sample, read from cache meta if present, else fallback)
#
# XAI (Method-1):
#   - strategy_id inferred deterministically from g_stage pattern
#   - strategy_name + explanation template printed/logged
#   - preview overlays include:
#       (1) input text
#       (2) strategy_id/name + explanation
#       (3) g_stage vector (rounded)
#
# Cache folder format expected:
#   E:/VETNet_Pilot/preload_cache/<DATASET_NAME>/
#       000354_in.png
#       000354_gt.png
#       000354_clip.pt   # cached CLIP vision embedding tensor [Dv] or [1,Dv]
#
# Optional meta file (if you have it):
#   E:/VETNet_Pilot/preload_cache/<DATASET_NAME>/meta.json
#     - can map index/file -> {"xai_llm": "...", "xai_blip": "..."} etc.
#
# ------------------------------------------------------------

import os
import sys
import time
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# ---- AMP (new API to avoid FutureWarning) ----
from torch.amp import autocast, GradScaler

# ---- project root wiring ----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- project imports ----
from models.backbone.vetnet_backbone import VETNetBackbone
from models.pilot.gate_controller import GateController
from models.pilot.phase3_policy import Phase3GatePolicy

# ---- metrics (optional) ----
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    USE_SKIMAGE = True
except Exception:
    USE_SKIMAGE = False


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    # Cache root
    cache_root: str = "E:/VETNet_Pilot/preload_cache"

    # Crop size
    crop_size: int = 192

    # Speed (subset)
    subset_size: int = 3000
    subset_seed: int = 123

    # Phase-1 backbone ckpt
    phase1_ckpt: str = r"E:\VETNet_Pilot\checkpoints\phase1_backbone\epoch_021_L0.0204_P31.45_S0.9371.pth"

    # Phase-2 teacher ckpt (optional)
    use_phase2_teacher: bool = True
    phase2_ckpt: str = r"E:\VETNet_Pilot\checkpoints\phase2_gating\epoch_002_L0.0236_P30.42_S0.9293.pth"

    # Output
    save_root: str = "E:/VETNet_Pilot/checkpoints/phase3_vlm_clean"
    results_root: str = "E:/VETNet_Pilot/results/phase3_vlm_clean"

    # Train
    epochs: int = 5
    batch_size: int = 4
    num_workers: int = 0
    lr: float = 2e-4
    weight_decay: float = 1e-4
    use_amp: bool = True

    # Gates
    num_stages: int = VETNetBackbone.NUM_MACRO_STAGES  # expected 8
    g_min: float = 0.1

    # Consistency to teacher
    w_cons: float = 0.1  # set 0.0 to disable teacher consistency

    # Preview / logging
    preview_every: int = 100          # iterations
    print_xai_every: int = 200        # iterations
    max_preview_text_chars: int = 420 # allow more text

    # Text encoder (CLIP text only)
    # NOTE: CLIP vision is NEVER used here.
    clip_text_model_name: str = "openai/clip-vit-large-patch14"

    # Method-1 thresholds
    xai_thresholds: Dict[str, float] = None

    def __post_init__(self):
        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(self.results_root, exist_ok=True)
        if self.xai_thresholds is None:
            self.xai_thresholds = {
                "low": 0.93,
                "high": 0.985,
                "near_one": 0.997,
            }


cfg = Config()


# ============================================================
# Helpers
# ============================================================
def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def safe_load_backbone(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        sd = ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[CKPT] Backbone loaded:", ckpt_path)
    print("[CKPT] Missing:", len(missing), "Unexpected:", len(unexpected))


def safe_load_phase2_controller(controller: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "controller" in ckpt:
        sd = ckpt["controller"]
    else:
        sd = ckpt
    missing, unexpected = controller.load_state_dict(sd, strict=False)
    print("[CKPT] Phase-2 controller loaded:", ckpt_path)
    print("[CKPT] Missing:", len(missing), "Unexpected:", len(unexpected))


def tensor_to_img_uint8(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255.0).astype("uint8")


def compute_psnr_ssim(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float]:
    if not USE_SKIMAGE:
        return 0.0, 0.0
    p = tensor_to_img_uint8(pred[0])
    g = tensor_to_img_uint8(gt[0])
    psnr = peak_signal_noise_ratio(g, p, data_range=255)
    ssim = structural_similarity(g, p, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


def gate_stats(g: torch.Tensor) -> Dict:
    return {
        "g_mean": float(g.mean().item()),
        "g_min": float(g.min().item()),
        "g_max": float(g.max().item()),
        "g_var_stage": g.var(dim=0, unbiased=False).detach().cpu().tolist(),
        "g_mean_stage": g.mean(dim=0).detach().cpu().tolist(),
    }


# ============================================================
# Dataset (direct scan of preload_cache)
# ============================================================
class Phase3CachedClipVisionDataset(Dataset):
    """
    Scans:
      cache_root/<dataset_name>/*_in.png
    and expects matching:
      *_gt.png, *_clip.pt

    Also optionally loads:
      cache_root/<dataset_name>/meta.json
    where meta.json can store xai text for each id.
    """

    def __init__(self, cache_root: str, crop_size: int = 192):
        self.cache_root = cache_root
        self.crop_size = crop_size

        self.items: List[Dict] = []
        self.meta_by_dataset: Dict[str, Dict] = {}

        self._scan()

        if len(self.items) == 0:
            raise RuntimeError(f"[Phase3Dataset] No items found under: {cache_root}")

        # quick sanity: check one clip tensor shape
        z = self._load_clip_tensor(self.items[0]["clip_path"])
        if z.ndim == 2 and z.size(0) == 1:
            z = z[0]
        if z.ndim != 1:
            raise RuntimeError(
                f"[Phase3Dataset] clip tensor must be [Dv] (or [1,Dv]). Got shape: {tuple(z.shape)} "
                f"from {self.items[0]['clip_path']}"
            )
        self.vision_dim = int(z.numel())

        print(f"[Phase3Dataset] total items: {len(self.items)}")
        print(f"[Phase3Dataset] cached vision dim: {self.vision_dim}")

    def _scan(self):
        # Each subfolder is a dataset name
        if not os.path.isdir(self.cache_root):
            raise RuntimeError(f"[Phase3Dataset] cache_root not found: {self.cache_root}")

        dataset_names = [d for d in os.listdir(self.cache_root) if os.path.isdir(os.path.join(self.cache_root, d))]
        dataset_names.sort()

        for dname in dataset_names:
            ddir = os.path.join(self.cache_root, dname)

            # load meta.json if exists
            meta_path = os.path.join(ddir, "meta.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        self.meta_by_dataset[dname] = json.load(f)
                    print(f"[Phase3Dataset] meta loaded: {meta_path}")
                except Exception as e:
                    print(f"[Phase3Dataset] meta load failed: {meta_path} ({e})")
                    self.meta_by_dataset[dname] = {}

            # find *_in.png
            for fn in os.listdir(ddir):
                if not fn.endswith("_in.png"):
                    continue
                base = fn[:-7]  # remove "_in.png"
                in_path = os.path.join(ddir, f"{base}_in.png")
                gt_path = os.path.join(ddir, f"{base}_gt.png")
                clip_path = os.path.join(ddir, f"{base}_clip.pt")

                if not os.path.isfile(gt_path):
                    continue
                if not os.path.isfile(clip_path):
                    continue

                self.items.append({
                    "dataset": dname,
                    "id": base,
                    "in_path": in_path,
                    "gt_path": gt_path,
                    "clip_path": clip_path,
                })

    def __len__(self):
        return len(self.items)

    def _load_image_rgb(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def _random_crop_pair(self, inp: Image.Image, gt: Image.Image) -> Tuple[Image.Image, Image.Image]:
        w, h = inp.size
        cs = self.crop_size

        if w < cs or h < cs:
            # pad to crop size
            pad_w = max(0, cs - w)
            pad_h = max(0, cs - h)
            inp = Image.fromarray(
                np.pad(np.array(inp), ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            )
            gt = Image.fromarray(
                np.pad(np.array(gt), ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            )
            w, h = inp.size

        x = random.randint(0, w - cs)
        y = random.randint(0, h - cs)
        inp_c = inp.crop((x, y, x + cs, y + cs))
        gt_c = gt.crop((x, y, x + cs, y + cs))
        return inp_c, gt_c

    def _to_tensor01(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
        return t

    def _load_clip_tensor(self, clip_path: str) -> torch.Tensor:
        z = torch.load(clip_path, map_location="cpu")
        if isinstance(z, dict) and "feat" in z:
            z = z["feat"]
        if not torch.is_tensor(z):
            z = torch.tensor(z, dtype=torch.float32)
        z = z.float()
        if z.ndim == 2 and z.size(0) == 1:
            z = z[0]
        return z.contiguous()

    def _get_xai_text(self, dataset: str, base_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns (xai_llm, xai_blip) if available.
        meta.json can be any format; we try common patterns.
        """
        meta = self.meta_by_dataset.get(dataset, {})
        if not isinstance(meta, dict):
            return None, None

        # Common pattern A: meta["000354"] = {...}
        if base_id in meta and isinstance(meta[base_id], dict):
            d = meta[base_id]
            return d.get("xai_llm", None), d.get("xai_blip", None)

        # Common pattern B: meta uses file names as keys
        k1 = f"{base_id}_in.png"
        if k1 in meta and isinstance(meta[k1], dict):
            d = meta[k1]
            return d.get("xai_llm", None), d.get("xai_blip", None)

        return None, None

    def __getitem__(self, idx: int) -> Dict:
        it = self.items[idx]
        dataset = it["dataset"]
        base_id = it["id"]

        inp_img = self._load_image_rgb(it["in_path"])
        gt_img  = self._load_image_rgb(it["gt_path"])
        inp_img, gt_img = self._random_crop_pair(inp_img, gt_img)

        inp = self._to_tensor01(inp_img)
        gt  = self._to_tensor01(gt_img)

        # ✅ cached CLIP vision embedding ONLY
        z_v = self._load_clip_tensor(it["clip_path"])  # [Dv]

        xai_llm, xai_blip = self._get_xai_text(dataset, base_id)

        sample = {
            "input": inp,                 # [3,H,W] in [0,1]
            "gt": gt,                     # [3,H,W] in [0,1]
            "clip_vision_feat": z_v,      # [Dv]
            "dataset": dataset,
            "id": base_id,
        }
        if xai_llm is not None:
            sample["xai_llm"] = str(xai_llm)
        if xai_blip is not None:
            sample["xai_blip"] = str(xai_blip)
        return sample
    
def build_subset_dataset(full_dataset: Dataset, subset_size: int, seed: int) -> Dataset:
    n = len(full_dataset)
    if subset_size is None or subset_size <= 0 or subset_size >= n:
        print(f"[Phase3] Using FULL dataset (n={n})")
        return full_dataset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:subset_size].tolist()
    print(f"[Phase3] Using SUBSET: {subset_size}/{n} (seed={seed})")
    return Subset(full_dataset, idx)


def get_text_from_batch(batch: Dict) -> List[str]:
    """
    Prefer xai_llm -> xai_blip -> fallback prompt.
    Handles both collated strings and missing keys.
    """
    B = int(batch["input"].size(0))

    def _to_list(v):
        # DataLoader collate: if dataset returns str, it becomes list[str]
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        if isinstance(v, str):
            return [v] * B
        return None

    if "xai_llm" in batch:
        out = _to_list(batch["xai_llm"])
        if out is not None and len(out) == B:
            return out

    if "xai_blip" in batch:
        out = _to_list(batch["xai_blip"])
        if out is not None and len(out) == B:
            return out

    # fallback with dataset name if available
    ds = batch.get("dataset", None)
    if isinstance(ds, (list, tuple)) and len(ds) == B:
        return [f"Restore a degraded image from dataset {str(d)}. Describe artifacts factually." for d in ds]
    if isinstance(ds, str):
        return [f"Restore a degraded image from dataset {ds}. Describe artifacts factually."] * B

    return ["Restore the degraded image. Describe visible artifacts and restoration intent."] * B


# ============================================================
# Method-1 XAI (Strategy ID + Template Explanation)
# ============================================================
def _mean_stage(g_stage: torch.Tensor) -> torch.Tensor:
    return g_stage.detach().mean(dim=0)


def infer_strategy_id_from_gates(g_stage: torch.Tensor, thresholds: Dict[str, float]) -> Tuple[int, str]:
    """
    Rule-based strategy inference from predicted gates.
    Robust for S != 8 (uses group slicing with bounds).
    """
    g = _mean_stage(g_stage)
    S = int(g.numel())

    def gi(i, default=1.0):
        if 0 <= i < S:
            return float(g[i].item())
        return float(default)

    enc = [gi(i) for i in range(min(3, S))]
    mid = [gi(i) for i in range(3, min(5, S))]
    dec = [gi(i) for i in range(5, S)]

    enc_m = float(sum(enc) / max(1, len(enc)))
    mid_m = float(sum(mid) / max(1, len(mid)))
    dec_m = float(sum(dec) / max(1, len(dec)))

    low = float(thresholds.get("low", 0.93))
    high = float(thresholds.get("high", 0.985))
    near_one = float(thresholds.get("near_one", 0.997))

    if (enc_m > near_one) and (mid_m > near_one) and (dec_m > near_one):
        return 0, "Identity / Mild Degradation"
    if (enc_m < low) and (mid_m >= low) and (dec_m >= low):
        return 1, "Encoder Suppression (Low-level Artifacts)"
    if (mid_m < low) and (dec_m >= low):
        return 2, "Bottleneck Suppression (Structured Artifacts)"
    if (dec_m > high) and (mid_m >= low):
        return 3, "Decoder/Refine Emphasis (Blur-like)"
    if (enc_m < low) and (mid_m < low) and (dec_m < low):
        return 4, "Global Suppression (Severe Degradation)"
    return 5, "Mixed / Uncertain Strategy"


def strategy_template_explanation(strategy_id: int, strategy_name: str) -> str:
    table = {
        0: "Gates remain near 1 across stages, indicating mild degradation; the backbone operates close to its default restoration behavior.",
        1: "Early-stage gates are reduced, prioritizing suppression of low-level corruption (e.g., noise-like artifacts) while preserving higher-level reconstruction.",
        2: "Mid/bottleneck gates are reduced, suppressing structured artifacts (e.g., streak-like or veil-like degradations) to stabilize global context reconstruction.",
        3: "Later-stage gates are relatively higher, emphasizing decoder/refinement to recover fine details commonly degraded by blur-like distortions.",
        4: "Gates are reduced broadly, indicating severe degradation; the controller dampens multiple stages to avoid amplifying corrupted features before reconstruction.",
        5: "Stage-wise gates show a mixed pattern; the controller applies a blended strategy to balance artifact suppression and detail reconstruction.",
    }
    base = table.get(strategy_id, table[5])
    return f"{base} (Strategy: {strategy_name}, ID={strategy_id})"


def format_gate_vector(g_stage: torch.Tensor, decimals: int = 3) -> str:
    g = _mean_stage(g_stage)
    vals = [round(float(v.item()), decimals) for v in g]
    return "[" + ", ".join([f"{v:.{decimals}f}" for v in vals]) + "]"


def build_xai_text_block(user_text: str, sid: int, sname: str, explain: str, g_stage: torch.Tensor) -> str:
    user_text = (user_text or "").strip().replace("\n", " ")
    gvec = format_gate_vector(g_stage, decimals=3)
    block = (
        f"Input text: {user_text}\n"
        f"Strategy ID: {sid} | {sname}\n"
        f"Explain: {explain}\n"
        f"g_stage(mean): {gvec}"
    )
    return block


def draw_text_overlay(img_uint8: np.ndarray, text: str, max_chars: int) -> np.ndarray:
    img = Image.fromarray(img_uint8)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    text = (text or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    # multi-line wrap (simple)
    lines = text.split("\n")
    wrapped: List[str] = []
    for ln in lines:
        ln = ln.strip()
        if len(ln) <= 90:
            wrapped.append(ln)
        else:
            for i in range(0, len(ln), 90):
                wrapped.append(ln[i:i+90])

    pad = 6
    line_h = 18
    max_w = 0
    for ln in wrapped:
        bbox = draw.textbbox((0, 0), ln, font=font)
        max_w = max(max_w, bbox[2] - bbox[0])

    box_w = max_w + 2 * pad
    box_h = len(wrapped) * line_h + 2 * pad

    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0))
    y = pad
    for ln in wrapped:
        draw.text((pad, y), ln, fill=(255, 255, 255), font=font)
        y += line_h

    return np.array(img)


def save_triplet_with_text(inp: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, text: str, path: str, max_chars: int):
    inp_u = tensor_to_img_uint8(inp)
    pred_u = tensor_to_img_uint8(pred)
    gt_u = tensor_to_img_uint8(gt)

    inp_u = draw_text_overlay(inp_u, text, max_chars=max_chars)

    H, W, _ = inp_u.shape
    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)
    canvas[:, 0:W] = inp_u
    canvas[:, W:2*W] = pred_u
    canvas[:, 2*W:3*W] = gt_u

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(canvas).save(path)


# ============================================================
# CLIP Text Encoder ONLY (Vision is cached)
# ============================================================
def load_clip_text_encoder(model_name: str, device: str):
    """
    ✅ Only text model is used online.
    ❌ No CLIP vision import here.
    """
    try:
        from transformers import CLIPProcessor, CLIPTextModel
    except Exception as e:
        raise ImportError(
            "transformers is required for Phase-3 CLIP text encoder.\n"
            "Please install: pip install transformers"
        ) from e

    processor = CLIPProcessor.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name).to(device)
    text_model.eval()
    freeze_module(text_model)
    return processor, text_model


@torch.no_grad()
def clip_text_embed(processor, text_model, texts: List[str], device: str) -> torch.Tensor:
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    out = text_model(input_ids=input_ids, attention_mask=attention_mask)
    z_t = out.pooler_output  # [B, Dt]
    return z_t


# ============================================================
# Train
# ============================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Phase3] device:", device)

    # 1) Dataset (direct scan)
    full_ds = Phase3CachedClipVisionDataset(cfg.cache_root, crop_size=cfg.crop_size)
    ds = build_subset_dataset(full_ds, cfg.subset_size, cfg.subset_seed)

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print("[Phase3] Total train samples =", len(ds))
    print("[Phase3] Steps per epoch =", len(dl))

    # 2) Backbone (frozen)
    backbone = VETNetBackbone(
        in_channels=3, out_channels=3,
        dim=64,
        num_blocks=(4, 6, 6, 8),
        heads=(1, 2, 4, 8),
        volterra_rank=4,
        ffn_expansion_factor=2.66,
        bias=False,
    ).to(device)

    safe_load_backbone(backbone, cfg.phase1_ckpt)
    backbone.eval()
    freeze_module(backbone)
    print("[Phase3] backbone frozen:", all(not p.requires_grad for p in backbone.parameters()))

    # 3) Teacher (optional)
    teacher = None
    if cfg.use_phase2_teacher and cfg.w_cons > 0:
        teacher = GateController(
        num_stages=cfg.num_stages,
        g_min=cfg.g_min,
        hidden_dim=256   # ⭐ Phase-2와 반드시 동일
    ).to(device)

        safe_load_phase2_controller(teacher, cfg.phase2_ckpt)
        teacher.eval()
        freeze_module(teacher)
        print("[Phase3] teacher enabled: True")
    else:
        print("[Phase3] teacher enabled: False")

    # 4) Text encoder only
    processor, clip_text = load_clip_text_encoder(cfg.clip_text_model_name, device)
    text_dim = int(clip_text.config.hidden_size)

    # 5) Policy (trainable)
    vision_dim = int(full_ds.vision_dim)  # from cached clip vectors
    policy = Phase3GatePolicy(
        vision_dim=vision_dim,
        text_dim=text_dim,
        num_stages=cfg.num_stages,
        g_min=cfg.g_min,
        hidden_dim=512,
        dropout=0.0,
        num_strategies=0,
        init_gate_bias=2.0,
    ).to(device)

    policy.train()
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[Phase3] cached vision_dim={vision_dim}, text_dim={text_dim}")
    print("[Phase3] policy trainable params:", trainable_params / 1e6, "M")

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler("cuda", enabled=cfg.use_amp)

    log_path = os.path.join(cfg.results_root, "phase3_log.txt")
    print("[Phase3] log file:", log_path)

    global_iter = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        policy.train()

        loss_sum = 0.0
        rec_sum = 0.0
        cons_sum = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        cnt = 0

        g_stats_accum = None

        pbar = tqdm(dl, ncols=120, desc=f"Epoch {epoch:03d}/{cfg.epochs}")
        for batch in pbar:
            global_iter += 1

            inp = batch["input"].to(device, non_blocking=True)   # [B,3,H,W]
            gt  = batch["gt"].to(device, non_blocking=True)

            # ✅ cached CLIP vision features ONLY
            # batch["clip_vision_feat"] expected shape: [B, Dv] after collate
            z_v = batch["clip_vision_feat"].to(device, non_blocking=True).float()
            if z_v.ndim == 1:
                z_v = z_v.unsqueeze(0)  # safety

            # per-sample text
            texts = get_text_from_batch(batch)

            opt.zero_grad(set_to_none=True)

            # text embedding (frozen, no grad)
            with torch.no_grad():
                z_t = clip_text_embed(processor, clip_text, texts, device)

                g_teacher = None
                if teacher is not None:
                    g_teacher = teacher(inp)  # [B,S]

            with autocast("cuda", enabled=cfg.use_amp):
                out = policy(z_v, z_t)
                g_pred = out["g_stage"]  # [B,S]

                pred = backbone(inp, g_stage=g_pred)
                rec_loss = F.l1_loss(pred, gt)

                cons_loss = torch.tensor(0.0, device=device)
                if (g_teacher is not None) and (cfg.w_cons > 0):
                    cons_loss = F.mse_loss(g_pred, g_teacher)

                loss = rec_loss + cfg.w_cons * cons_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                pred_c = pred.clamp(0, 1)
                gt_c   = gt.clamp(0, 1)
                ps, ss = compute_psnr_ssim(pred_c, gt_c)

                loss_sum += float(loss.item())
                rec_sum  += float(rec_loss.item())
                cons_sum += float(cons_loss.item()) if isinstance(cons_loss, torch.Tensor) else float(cons_loss)
                psnr_sum += float(ps)
                ssim_sum += float(ss)
                cnt += 1

                st = gate_stats(g_pred)
                if g_stats_accum is None:
                    g_stats_accum = st
                else:
                    g_stats_accum["g_mean"] += st["g_mean"]
                    g_stats_accum["g_min"] = min(g_stats_accum["g_min"], st["g_min"])
                    g_stats_accum["g_max"] = max(g_stats_accum["g_max"], st["g_max"])
                    g_stats_accum["g_var_stage"] = [
                        a + b for a, b in zip(g_stats_accum["g_var_stage"], st["g_var_stage"])
                    ]
                    g_stats_accum["g_mean_stage"] = [
                        a + b for a, b in zip(g_stats_accum["g_mean_stage"], st["g_mean_stage"])
                    ]

                pbar.set_postfix({
                    "L": f"{loss_sum/cnt:.4f}",
                    "Rec": f"{rec_sum/cnt:.4f}",
                    "Cons": f"{cons_sum/cnt:.4f}",
                    "P": f"{psnr_sum/cnt:.2f}",
                    "S": f"{ssim_sum/cnt:.3f}",
                    "g": f"{st['g_mean']:.3f}",
                })

            # --------------------------
            # Method-1 XAI prints
            # --------------------------
            if (cfg.print_xai_every > 0) and (global_iter % cfg.print_xai_every == 0):
                sid, sname = infer_strategy_id_from_gates(g_pred, thresholds=cfg.xai_thresholds)
                explain = strategy_template_explanation(sid, sname)
                gvec = format_gate_vector(g_pred, decimals=3)
                t0_txt = texts[0] if isinstance(texts, list) and len(texts) > 0 else str(texts)

                print("\n" + "-" * 72)
                print(f"[XAI @ iter {global_iter:07d}] Strategy ID: {sid} | {sname}")
                print("Explanation:", explain)
                print("g_stage(mean):", gvec)
                print("Input text(0):", (t0_txt[:200] + "...") if len(t0_txt) > 200 else t0_txt)
                print("-" * 72 + "\n")

            # --------------------------
            # Preview save
            # --------------------------
            if (cfg.preview_every > 0) and (global_iter % cfg.preview_every == 0):
                prev_dir = os.path.join(cfg.results_root, "iter_preview")
                os.makedirs(prev_dir, exist_ok=True)
                path = os.path.join(prev_dir, f"iter_{global_iter:07d}.png")

                sid, sname = infer_strategy_id_from_gates(g_pred, thresholds=cfg.xai_thresholds)
                explain = strategy_template_explanation(sid, sname)

                xai_block = build_xai_text_block(
                    user_text=texts[0] if isinstance(texts, list) and len(texts) > 0 else str(texts),
                    sid=sid,
                    sname=sname,
                    explain=explain,
                    g_stage=g_pred
                )

                save_triplet_with_text(
                    inp=inp[0].detach().cpu(),
                    pred=pred_c[0].detach().cpu(),
                    gt=gt_c[0].detach().cpu(),
                    text=xai_block,
                    path=path,
                    max_chars=cfg.max_preview_text_chars
                )

        # epoch averages
        epoch_loss = loss_sum / max(1, cnt)
        epoch_rec  = rec_sum  / max(1, cnt)
        epoch_cons = cons_sum / max(1, cnt)
        epoch_psnr = psnr_sum / max(1, cnt)
        epoch_ssim = ssim_sum / max(1, cnt)

        # finalize gate stats
        steps = max(1, cnt)
        if g_stats_accum is None:
            g_stats_accum = {
                "g_mean": 0.0, "g_min": 0.0, "g_max": 0.0,
                "g_var_stage": [0.0] * cfg.num_stages,
                "g_mean_stage": [0.0] * cfg.num_stages,
            }
        else:
            g_stats_accum["g_mean"] /= steps
            g_stats_accum["g_var_stage"] = [v / steps for v in g_stats_accum["g_var_stage"]]
            g_stats_accum["g_mean_stage"] = [v / steps for v in g_stats_accum["g_mean_stage"]]

        # epoch-level representative XAI from mean gates
        g_mean_stage_tensor = torch.tensor([g_stats_accum["g_mean_stage"]], device=device, dtype=torch.float32)
        sid_ep, sname_ep = infer_strategy_id_from_gates(g_mean_stage_tensor, thresholds=cfg.xai_thresholds)
        explain_ep = strategy_template_explanation(sid_ep, sname_ep)

        ckpt_path = os.path.join(
            cfg.save_root,
            f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.pth"
        )
        torch.save({
            "epoch": epoch,
            "policy": policy.state_dict(),
            "config": cfg.__dict__,
            "vision_dim": vision_dim,
            "text_dim": text_dim,
        }, ckpt_path)

        epoch_msg = (
            f"[Epoch {epoch:03d}] "
            f"loss={epoch_loss:.6f} rec={epoch_rec:.6f} cons={epoch_cons:.6f} "
            f"psnr={epoch_psnr:.3f} ssim={epoch_ssim:.6f} | "
            f"g_mean={g_stats_accum['g_mean']:.4f} g_min={g_stats_accum['g_min']:.4f} g_max={g_stats_accum['g_max']:.4f} | "
            f"g_mean_stage={[round(v,4) for v in g_stats_accum['g_mean_stage']]} | "
            f"g_var_stage={[round(v,6) for v in g_stats_accum['g_var_stage']]} | "
            f"XAI(strategy_id={sid_ep}, name='{sname_ep}') | "
            f"time={time.time()-t0:.1f}s | saved={ckpt_path}"
        )
        print("\n" + epoch_msg)
        print("[XAI][Epoch Summary]", explain_ep)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(epoch_msg + "\n")
            f.write("[XAI][Epoch Summary] " + explain_ep + "\n")

    print("\n[Phase3] Training completed.")


if __name__ == "__main__":
    train()
