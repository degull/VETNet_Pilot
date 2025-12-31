# E:/VETNet_Pilot/experiments/text_ablation_phase3.py
# ============================================================
# Phase-3 Text Ablation Experiment
#
# Goal:
#   Fix visual embedding z_v (cached CLIP vision feature)
#   Change ONLY text input
#   Observe changes in predicted g_stage
#
# This answers:
#   "Is the Phase-3 controller a true VLM, or vision-only?"
# ============================================================

import os
import sys
import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[ROOT]", ROOT)

# -----------------------------
# Imports from project
# -----------------------------
from models.pilot.phase3_policy import Phase3GatePolicy
from models.backbone.vetnet_backbone import VETNetBackbone

# -----------------------------
# CLIP Text Encoder (ONLY text)
# -----------------------------
from transformers import CLIPProcessor, CLIPTextModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Paths
# -----------------------------
PHASE3_CKPT = r"E:\VETNet_Pilot\checkpoints\phase3_vlm_clean\epoch_004_L0.0274_P29.59_S0.9315.pth"

CACHE_ROOT = r"E:\VETNet_Pilot\preload_cache"
DATASET_NAME = "CSD"        # change if needed
SAMPLE_ID = "000354"        # any valid sample id

CLIP_PT_PATH = os.path.join(
    CACHE_ROOT, DATASET_NAME, f"{SAMPLE_ID}_clip.pt"
)

# -----------------------------
# Load cached vision embedding
# -----------------------------
print("[INFO] Loading cached vision feature:")
print(" ", CLIP_PT_PATH)

z_v = torch.load(CLIP_PT_PATH)  # [768]
assert z_v.ndim == 1, "Expected shape [Dv]"
z_v = z_v.unsqueeze(0).to(DEVICE)  # [1, 768]

vision_dim = z_v.shape[-1]
print("[INFO] vision_dim =", vision_dim)

# -----------------------------
# Load CLIP text encoder
# -----------------------------
print("[INFO] Loading CLIP text encoder...")

clip_model_name = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(clip_model_name)
text_model = CLIPTextModel.from_pretrained(clip_model_name).to(DEVICE)
text_model.eval()

text_dim = text_model.config.hidden_size
print("[INFO] text_dim =", text_dim)

# -----------------------------
# Load Phase-3 Policy
# -----------------------------
print("[INFO] Loading Phase-3 policy checkpoint...")
ckpt = torch.load(PHASE3_CKPT, map_location="cpu")

policy = Phase3GatePolicy(
    vision_dim=vision_dim,
    text_dim=text_dim,
    num_stages=VETNetBackbone.NUM_MACRO_STAGES,
    g_min=0.1,
    hidden_dim=512,
).to(DEVICE)

policy.load_state_dict(ckpt["policy"])
policy.eval()

print("[INFO] Phase-3 policy loaded successfully")

# -----------------------------
# Text prompts for ablation
# -----------------------------
TEXTS = [
    "Restore the degraded image.",

    "Remove snow streaks and improve visibility.",

    "Remove haze and restore global contrast.",

    "Recover sharp details lost due to blur."
]

# -----------------------------
# Run ablation
# -----------------------------
print("\n" + "=" * 72)
print("TEXT ABLATION RESULTS (fixed vision, varying text)")
print("=" * 72)

g_results = {}

with torch.no_grad():
    for text in TEXTS:
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        z_t = text_model(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=inputs["attention_mask"].to(DEVICE),
        ).pooler_output  # [1, Dt]

        out = policy(z_v, z_t)
        g = out["g_stage"]  # [1, S]

        g_np = g.squeeze(0).cpu().numpy()
        g_results[text] = g_np

        print("\n[TEXT]")
        print(" ", text)
        print("[g_stage]")
        print(" ", np.round(g_np, 4))

# -----------------------------
# Pairwise difference analysis
# -----------------------------
print("\n" + "=" * 72)
print("PAIRWISE Δg (L2 distance)")
print("=" * 72)

keys = list(g_results.keys())

for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        g1 = g_results[keys[i]]
        g2 = g_results[keys[j]]
        delta = np.linalg.norm(g1 - g2)

        print(f"\nΔg || '{keys[i][:30]}...' vs '{keys[j][:30]}...'")
        print(" ", f"{delta:.6f}")

print("\n[Done] Text ablation experiment completed.")
