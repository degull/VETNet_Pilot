# ============================================================
# save_xai_restoration_figure.py
# ============================================================

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# -------------------------------
# Project imports
# -------------------------------
from datasets.multitask_dataset_cache import MultiTaskDatasetCache
from models.backbone.vetnet_backbone import VETNetBackbone
from models.pilot.phase3_policy import Phase3GatePolicy
from models.pilot.gate_controller import GateController

from transformers import CLIPProcessor, CLIPTextModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Config
# ============================================================

CACHE_ROOT = "E:/VETNet_Pilot/preload_cache"

PHASE1_CKPT = r"E:\VETNet_Pilot\checkpoints\phase1_backbone\epoch_021_L0.0204_P31.45_S0.9371.pth"
PHASE3_CKPT = r"E:\VETNet_Pilot\checkpoints\phase3_vlm_clean\epoch_004_L0.0274_P29.59_S0.9315.pth"

CLIP_TEXT_MODEL = "openai/clip-vit-large-patch14"

SAVE_PATH = "E:/VETNet_Pilot/results/xai_restoration_figure.png"

# ============================================================
# Utils
# ============================================================

def load_ckpt_flexible(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "policy" in ckpt:
            return ckpt["policy"]
    return ckpt


def tensor_to_uint8(img_t):
    img = img_t.detach().cpu().clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    return (img * 255.0).astype(np.uint8)


def compute_psnr_ssim(pred, gt):
    p = tensor_to_uint8(pred)
    g = tensor_to_uint8(gt)
    psnr = peak_signal_noise_ratio(g, p, data_range=255)
    ssim = structural_similarity(g, p, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


# ============================================================
# XAI logic (same as Phase-3)
# ============================================================

def gate_to_xai_text(g_stage: torch.Tensor):
    """
    g_stage: [1, S]
    """
    g = g_stage.mean(dim=0)
    g = g / g.sum()   # normalize

    enc = g[:3].mean().item()
    mid = g[3:5].mean().item()
    dec = g[5:].mean().item()

    text = (
        f"Early-stage modulation ({enc:.2f}) suppresses low-level artifacts. "
        f"Mid-stage modulation ({mid:.2f}) addresses structured degradations. "
        f"Late-stage modulation ({dec:.2f}) refines fine details during reconstruction."
    )

    strategy = "Adaptive Stage-wise Modulation (VLM-driven)"

    return strategy, text



# ============================================================
# Figure drawing
# ============================================================

def save_restoration_xai_figure(
    degraded,
    restored,
    gt,
    psnr,
    ssim,
    strategy,
    xai_text,
    g_stage,
    save_path,
):
    deg_u = tensor_to_uint8(degraded)
    res_u = tensor_to_uint8(restored)
    gt_u  = tensor_to_uint8(gt)

    H, W, _ = deg_u.shape
    TEXT_H = 150

    canvas = np.zeros((H + TEXT_H, W * 3, 3), dtype=np.uint8)
    canvas[:H, 0:W]     = deg_u
    canvas[:H, W:2*W]   = res_u
    canvas[:H, 2*W:3*W] = gt_u

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 22)
        font_body  = ImageFont.truetype("arial.ttf", 18)
    except:
        font_title = ImageFont.load_default()
        font_body = font_title

    # titles
    draw.text((W//2-50, 10), "Degraded", fill=(255,255,255), font=font_title)
    draw.text((W+W//2-50, 10), "Restored", fill=(255,255,255), font=font_title)
    draw.text((2*W+W//2-70, 10), "Ground Truth", fill=(255,255,255), font=font_title)

    # text box
    draw.rectangle([0, H, W*3, H+TEXT_H], fill=(0,0,0))

    lines = [
        f"PSNR: {psnr:.2f} dB   |   SSIM: {ssim:.4f}",
        f"Strategy: {strategy}",
        f'XAI: "{xai_text}"',
        f"g_stage: {[round(float(v),3) for v in g_stage]}"
    ]

    y = H + 12
    for i, line in enumerate(lines):
        draw.text((12, y), line, fill=(255,255,255),
                  font=font_title if i==0 else font_body)
        y += 30

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
    print(f"[Saved] {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("[INFO] Loading dataset...")
    ds = MultiTaskDatasetCache(CACHE_ROOT)
    sample = ds[0]

    degraded = sample["input"].to(DEVICE)
    gt = sample["gt"].to(DEVICE)

    # load cached vision
    ds_name = sample["dataset"]
    clip_path = os.path.join(CACHE_ROOT, ds_name, "000354_clip.pt")
    z_v = torch.load(clip_path).unsqueeze(0).to(DEVICE)

    # text encoder
    processor = CLIPProcessor.from_pretrained(CLIP_TEXT_MODEL)
    text_model = CLIPTextModel.from_pretrained(CLIP_TEXT_MODEL).to(DEVICE)
    text_model.eval()

    text = ["Restore the degraded image."]
    inputs = processor(text=text, return_tensors="pt", padding=True)
    z_t = text_model(
        input_ids=inputs["input_ids"].to(DEVICE),
        attention_mask=inputs["attention_mask"].to(DEVICE),
    ).pooler_output

    # backbone
    backbone = VETNetBackbone(
        in_channels=3, out_channels=3,
        dim=64,
        num_blocks=(4,6,6,8),
        heads=(1,2,4,8),
        volterra_rank=4,
        ffn_expansion_factor=2.66,
        bias=False,
    ).to(DEVICE)
    backbone.load_state_dict(load_ckpt_flexible(PHASE1_CKPT), strict=False)
    backbone.eval()

    # policy
    policy = Phase3GatePolicy(
        vision_dim=z_v.size(-1),
        text_dim=z_t.size(-1),
        num_stages=8,
        g_min=0.1,
        hidden_dim=512,
    ).to(DEVICE)
    policy.load_state_dict(load_ckpt_flexible(PHASE3_CKPT), strict=False)
    policy.eval()

    with torch.no_grad():
        g_stage = policy(z_v, z_t)["g_stage"]
        restored = backbone(degraded.unsqueeze(0), g_stage=g_stage)[0]

    psnr, ssim = compute_psnr_ssim(restored, gt)
    strategy, xai_text = gate_to_xai_text(g_stage)

    save_restoration_xai_figure(
        degraded,
        restored,
        gt,
        psnr,
        ssim,
        strategy,
        xai_text,
        g_stage[0].cpu().tolist(),
        SAVE_PATH
    )


if __name__ == "__main__":
    main()
