# preprocess_clip_vision.py
# ---------------------------------------------
# Precompute CLIP vision features for Phase-3
# Run ONCE before Phase-3 training
# ---------------------------------------------

import os
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

from transformers import CLIPProcessor, CLIPVisionModel
from datasets.multitask_dataset_cache import MultiTaskDatasetCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_ROOT = "E:/VETNet_Pilot/preload_cache"
OUT_KEY = "clip_vision_feat"
CLIP_NAME = "openai/clip-vit-large-patch14"

processor = CLIPProcessor.from_pretrained(CLIP_NAME)
vision = CLIPVisionModel.from_pretrained(CLIP_NAME).to(DEVICE)
vision.eval()

ds = MultiTaskDatasetCache(CACHE_ROOT, size=192)

print("[CLIP PREPROCESS] total samples:", len(ds))

for i in tqdm(range(len(ds))):
    sample = ds[i]
    img = sample["input"]          # [3,H,W] float [0,1]

    img_np = (img.clamp(0,1)*255).byte().permute(1,2,0).numpy()
    inputs = processor(images=[img_np], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():
        z_v = vision(pixel_values=pixel_values).pooler_output[0].cpu()

    # 🔥 저장 위치 (dataset cache 내부)
    torch.save(z_v, os.path.join(ds.cache_dir, f"{i:06d}_{OUT_KEY}.pt"))

print("[CLIP PREPROCESS] DONE")
