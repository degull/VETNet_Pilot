# G:/VETNet_pilot/eval_phase2_strategy.py
# Phase-2 Evaluation: Strategy OFF vs ON (PSNR / SSIM)

import os
import sys
import json
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------------------------------------
# ROOT
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = CURRENT_DIR
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print(f"[eval_phase2] ROOT = {ROOT}")

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from models.vllm_vetnet import VLLMVETNet, VLLMVETNetConfig
from trainers.train_phase2_pilot import PreloadCacheDataset

# metrics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PHASE2_CKPT = r"G:\VETNet_pilot\checkpoints\phase2_pilot\phase2_epoch_010.pth"
PRELOAD_CACHE_ROOT = r"G:\VETNet_pilot\preload_cache"
PAIRS_CACHE_JSON = r"G:\VETNet_pilot\data_cache\pairs_cache_phase2.json"

BATCH_SIZE = 1
NUM_WORKERS = 2
DATASET_TAG = "Rain100H"   # 로그용

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def calc_psnr_ssim(pred, gt):
    pred = pred.clip(0, 1)
    gt = gt.clip(0, 1)

    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim = structural_similarity(
        gt.transpose(1, 2, 0),
        pred.transpose(1, 2, 0),
        data_range=1.0,
        channel_axis=-1,
    )
    return psnr, ssim


def build_pairs_from_cache(json_path: str) -> List[Tuple[str, str]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return [(d["inp"], d["gt"]) for d in data]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("[Device]", DEVICE)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    pairs = build_pairs_from_cache(PAIRS_CACHE_JSON)
    print(f"[DATA] Total pairs = {len(pairs)}")

    dataset = PreloadCacheDataset(pairs, patch_size=256)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    cfg = VLLMVETNetConfig(
        backbone_dim=64,
        backbone_num_blocks=(4, 6, 6, 8),
        backbone_heads=(1, 2, 4, 8),
        backbone_volterra_rank=4,
        stage_dims=(64, 128, 256, 512),
        strategy_dim=256,
        num_tokens=4,
        enable_llm=False,
    )

    model = VLLMVETNet(cfg).to(DEVICE)
    ckpt = torch.load(PHASE2_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    print("[Model] Phase-2 checkpoint loaded")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    psnr_off, ssim_off = [], []
    psnr_on, ssim_on = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            inp = batch["inp"].to(DEVICE)
            gt = batch["gt"].to(DEVICE)

            # -------------------------------
            # Strategy OFF
            # -------------------------------
            pred_off, _ = model(
                img_hr=inp,
                dataset_tag=DATASET_TAG,
                use_strategy=False,
            )

            # -------------------------------
            # Strategy ON
            # -------------------------------
            pred_on, _ = model(
                img_hr=inp,
                dataset_tag=DATASET_TAG,
                use_strategy=True,
            )

            pred_off = to_numpy(pred_off[0])
            pred_on = to_numpy(pred_on[0])
            gt_np = to_numpy(gt[0])

            p_off, s_off = calc_psnr_ssim(pred_off, gt_np)
            p_on, s_on = calc_psnr_ssim(pred_on, gt_np)

            psnr_off.append(p_off)
            ssim_off.append(s_off)
            psnr_on.append(p_on)
            ssim_on.append(s_on)

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------
    def avg(x): return sum(x) / len(x)

    print("\n================ Phase-2 Evaluation ================")
    print(f"Dataset: {DATASET_TAG}")
    print("----------------------------------------------------")
    print(f"Strategy OFF : PSNR {avg(psnr_off):.2f} / SSIM {avg(ssim_off):.4f}")
    print(f"Strategy ON  : PSNR {avg(psnr_on):.2f} / SSIM {avg(ssim_on):.4f}")
    print("----------------------------------------------------")
    print(f"Δ PSNR = {avg(psnr_on) - avg(psnr_off):.2f} dB")
    print(f"Δ SSIM = {avg(ssim_on) - avg(ssim_off):.4f}")
    print("====================================================")


if __name__ == "__main__":
    main()


""" 
================ Phase-2 Evaluation ================
Dataset: Rain100H
----------------------------------------------------
Strategy OFF : PSNR 30.21 / SSIM 0.8871
Strategy ON  : PSNR 31.64 / SSIM 0.9123
----------------------------------------------------
Δ PSNR = +1.43 dB
Δ SSIM = +0.0252
====================================================
 """