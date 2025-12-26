import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.pilot.vision_adapter import CLIPVisionAdapter

# ==============================
# Config
# ==============================
CACHE_ROOT = "E:/VETNet_Pilot/preload_cache"
FOLDERS = ["CSD", "DayRainDrop", "NightRainDrop", "rain100H", "RESIDE-6K"]
CLIP_NAME = "openai/clip-vit-base-patch32"  # ✅ 빠르고 충분
DEVICE = "cuda"

# ==============================
def main():
    clip = CLIPVisionAdapter(CLIP_NAME, device=DEVICE)
    clip.eval()

    for folder in FOLDERS:
        fdir = os.path.join(CACHE_ROOT, folder)
        in_list = sorted(glob.glob(os.path.join(fdir, "*_in.png")))

        print(f"[{folder}] {len(in_list)} images")

        for p in tqdm(in_list):
            out_p = p.replace("_in.png", "_clip.pt")
            if os.path.exists(out_p):
                continue

            img = Image.open(p).convert("RGB")
            x = np.array(img).astype(np.float32) / 255.0
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                feat, _ = clip(x)   # (1, D)

            torch.save(feat.squeeze(0).cpu(), out_p)

    print("[DONE] CLIP cache generated.")

if __name__ == "__main__":
    main()
