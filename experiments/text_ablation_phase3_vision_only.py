# ============================================================
# Vision-only Ablation for Phase-3
# - Fixed image
# - Fixed cached CLIP vision feature z_v
# - Text embedding z_t is FORCED to ZERO
# ============================================================

import os, sys
import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
print("[ROOT]", ROOT)

from datasets.multitask_dataset_cache import MultiTaskDatasetCache
from models.pilot.phase3_policy import Phase3GatePolicy
from transformers import CLIPProcessor, CLIPTextModel

# ----------------------------
# Config
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_ROOT = "E:/VETNet_Pilot/preload_cache"
PHASE3_CKPT = r"E:\VETNet_Pilot\checkpoints\phase3_vlm_clean\epoch_004_L0.0274_P29.59_S0.9315.pth"

CLIP_MODEL = "openai/clip-vit-large-patch14"

# Texts are DIFFERENT, but will be IGNORED
TEXTS = [
    "Restore the degraded image.",
    "Remove snow streaks and improve visibility.",
    "Remove haze and restore global contrast.",
    "Recover sharp details lost due to blur.",
]

# ----------------------------
# Load dataset
# ----------------------------
print("[INFO] Loading dataset...")
ds = MultiTaskDatasetCache(CACHE_ROOT)
print("[INFO] Total samples =", len(ds))

# Pick ONE fixed sample
sample = ds[0]
dataset_name = sample["dataset"]

# Load cached vision feature directly
clip_path = os.path.join(
    CACHE_ROOT, dataset_name, "000354_clip.pt"
)
print("[INFO] Loading cached vision feature:")
print(" ", clip_path)

z_v = torch.load(clip_path).unsqueeze(0).to(DEVICE)  # [1, Dv]
vision_dim = z_v.shape[-1]
print("[INFO] vision_dim =", vision_dim)

# ----------------------------
# Load CLIP text encoder (only to get dim)
# ----------------------------
print("[INFO] Loading CLIP text encoder...")
processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
text_model = CLIPTextModel.from_pretrained(CLIP_MODEL).to(DEVICE)
text_model.eval()
for p in text_model.parameters():
    p.requires_grad = False

with torch.no_grad():
    dummy = processor(text=["dummy"], return_tensors="pt", padding=True)
    out = text_model(
        input_ids=dummy["input_ids"].to(DEVICE),
        attention_mask=dummy["attention_mask"].to(DEVICE),
    )
    text_dim = out.pooler_output.shape[-1]

print("[INFO] text_dim =", text_dim)

# ----------------------------
# Load Phase-3 policy
# ----------------------------
print("[INFO] Loading Phase-3 policy checkpoint...")
policy = Phase3GatePolicy(
    vision_dim=vision_dim,
    text_dim=text_dim,
    num_stages=8,
    g_min=0.1,
    hidden_dim=512,
).to(DEVICE)

ckpt = torch.load(PHASE3_CKPT, map_location="cpu")
policy.load_state_dict(ckpt["policy"])
policy.eval()

print("[INFO] Phase-3 policy loaded successfully")

# ============================================================
# Vision-only Ablation
# ============================================================
print("\n" + "=" * 72)
print("VISION-ONLY ABLATION (z_t = 0, text ignored)")
print("=" * 72)

g_list = []

with torch.no_grad():
    for text in TEXTS:
        # 🚫 TEXT REMOVED HERE
        z_t = torch.zeros((1, text_dim), device=DEVICE)

        out = policy(z_v, z_t)
        g = out["g_stage"].squeeze(0).cpu().numpy()
        g_list.append(g)

        print("\n[TEXT IGNORED]")
        print(" ", text)
        print("[g_stage]")
        print(" ", np.round(g, 4))

# ============================================================
# Pairwise Δg
# ============================================================
print("\n" + "=" * 72)
print("PAIRWISE Δg (L2 distance) — EXPECT ≈ 0")
print("=" * 72)

def l2(a, b):
    return float(np.linalg.norm(a - b))

for i in range(len(g_list)):
    for j in range(i + 1, len(g_list)):
        d = l2(g_list[i], g_list[j])
        print(f"\nΔg || TEXT[{i}] vs TEXT[{j}] = {d:.6f}")

print("\n[Done] Vision-only ablation completed.")
