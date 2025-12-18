import os
import sys
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torchvision.io import read_image, write_png
from tqdm import tqdm

# ------------------------------------------------------------
# ROOT
# ------------------------------------------------------------
ROOT = r"G:\VETNet_pilot"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.vllm_vetnet import VLLMVETNet, VLLMVETNetConfig

# SSIM
from pytorch_msssim import ssim as msssim_ssim

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def load_img(p: str) -> torch.Tensor:
    x = read_image(p).float() / 255.0
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    return x[:3]

def psnr(pred, gt, eps=1e-12):
    mse = (pred - gt).pow(2).mean().clamp_min(eps)
    return 10 * torch.log10(1.0 / mse)

def to_01(x):
    return x.clamp(0, 1)

# ------------------------------------------------------------
# Dataset pairs
# ------------------------------------------------------------
def build_pairs(preload_root: str) -> List[Tuple[str, str]]:
    pairs = []
    for d in sorted(os.listdir(preload_root)):
        dpath = os.path.join(preload_root, d)
        if not os.path.isdir(dpath):
            continue
        for f in os.listdir(dpath):
            if f.endswith("_in.png"):
                inp = os.path.join(dpath, f)
                gt = inp.replace("_in.png", "_gt.png")
                if os.path.isfile(gt):
                    pairs.append((inp, gt))
    return pairs

# ------------------------------------------------------------
# Main eval
# ------------------------------------------------------------
def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # paths
    # -----------------------------
    ckpt_path = r"G:\VETNet_pilot\checkpoints\phase2_pilot\phase2_epoch_010.pth"
    preload_root = r"G:\VETNet_pilot\preload_cache"
    out_dir = r"G:\VETNet_pilot\results\phase2_onoff_eval"
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # model
    # -----------------------------
    cfg = VLLMVETNetConfig(
        backbone_dim=64,
        backbone_num_blocks=(4,6,6,8),
        backbone_heads=(1,2,4,8),
        backbone_volterra_rank=4,
        stage_dims=(64,128,256,512),
        strategy_dim=256,
        num_tokens=4,
        enable_llm=False,
        enabled_stages=("stage1","stage2","stage3","stage4"),
    )
    model = VLLMVETNet(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"], strict=False)
    model.eval()

    # -----------------------------
    # data
    # -----------------------------
    pairs = build_pairs(preload_root)
    print(f"[Eval] total pairs = {len(pairs)}")

    psnr_off, psnr_on = [], []
    ssim_off, ssim_on = [], []

    # --------------------------------------------------------
    # loop
    # --------------------------------------------------------
    with torch.no_grad():
        for idx, (inp_p, gt_p) in enumerate(tqdm(pairs)):
            inp = load_img(inp_p).unsqueeze(0).to(device)
            gt  = load_img(gt_p).unsqueeze(0).to(device)

            # OFF
            out_off, _ = model(inp, use_strategy=False)
            # ON
            out_on, _  = model(inp, use_strategy=True)

            out_off = to_01(out_off)
            out_on  = to_01(out_on)

            # metrics
            psnr_off.append(psnr(out_off, gt).item())
            psnr_on.append(psnr(out_on, gt).item())

            ssim_off.append(msssim_ssim(out_off, gt, data_range=1.0).item())
            ssim_on.append(msssim_ssim(out_on, gt, data_range=1.0).item())

            # ------------------------------------------------
            # qualitative save (first 20 images)
            # ------------------------------------------------
            if idx < 20:
                base = os.path.splitext(os.path.basename(inp_p))[0]
                write_png((inp[0].cpu()*255).byte(), os.path.join(out_dir, f"{base}_input.png"))
                write_png((out_off[0].cpu()*255).byte(), os.path.join(out_dir, f"{base}_off.png"))
                write_png((out_on[0].cpu()*255).byte(),  os.path.join(out_dir, f"{base}_on.png"))


    # --------------------------------------------------------
    # results
    # --------------------------------------------------------
    print("\n================ RESULT ================")
    print(f"PSNR  OFF: {sum(psnr_off)/len(psnr_off):.3f}")
    print(f"PSNR  ON : {sum(psnr_on)/len(psnr_on):.3f}")
    print(f"Δ PSNR   : {(sum(psnr_on)-sum(psnr_off))/len(psnr_on):+.3f}")

    print(f"\nSSIM  OFF: {sum(ssim_off)/len(ssim_off):.4f}")
    print(f"SSIM  ON : {sum(ssim_on)/len(ssim_on):.4f}")
    print(f"Δ SSIM   : {(sum(ssim_on)-sum(ssim_off))/len(ssim_on):+.4f}")

    print("========================================")

# ------------------------------------------------------------
if __name__ == "__main__":
    evaluate()
