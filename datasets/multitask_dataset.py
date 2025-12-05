# G:\VETNet_pilot\datasets\multitask_dataset.py
import os, sys, time

# -------------------------------------------------------------------------
# PATH 설정
# -------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../datasets
ROOT = os.path.dirname(CURRENT_DIR)                        # .../VETNet_pilot

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[DEBUG] Using ROOT:", ROOT)

import random
from torch.utils.data import Dataset

from datasets.utils_dataset import load_image, build_all_pairs
from datasets.transforms import TrainTransform


# -------------------------------------------------------------------------
# MultiTaskDataset — FAST VERSION (Memory Preload + Progress Bar)
# -------------------------------------------------------------------------
class MultiTaskDataset(Dataset):

    def __init__(self, data_root, transform=None, preload=True):
        super().__init__()
        self.data_root = data_root
        self.transform = transform if transform is not None else TrainTransform(size=256)
        self.preload = preload

        # 1) 전체 pair 목록 생성
        self.all_pairs = build_all_pairs(self.data_root)

        # 2) flatten
        self.samples = []
        for name, pairs in self.all_pairs.items():
            for inp, gt in pairs:
                self.samples.append((inp, gt, name))

        total_samples = len(self.samples)
        print(f"[MultiTaskDataset] Total Samples = {total_samples}")

        # -------------------------------------------------------------------------
        # 3) 프리로드: RAM에 이미지 전체 로딩 (+ 진행률 % 출력)
        # -------------------------------------------------------------------------
        if self.preload:
            print("[MultiTaskDataset] Preloading all images into RAM... (once only)")

            self.memory_in = [None] * total_samples
            self.memory_gt = [None] * total_samples

            start_time = time.time()

            for i, (inp_path, gt_path, _) in enumerate(self.samples):

                # 실제 이미지 로딩
                self.memory_in[i] = load_image(inp_path)
                self.memory_gt[i] = load_image(gt_path)

                # -------------------------------
                # 진행률 출력 (% + ETA)
                # -------------------------------
                if (i + 1) % 100 == 0 or (i + 1) == total_samples:
                    progress = (i + 1) / total_samples
                    elapsed = time.time() - start_time
                    eta = elapsed / progress - elapsed

                    print(
                        f"\r[Preload] {i+1}/{total_samples} "
                        f"({progress*100:5.1f}%) | "
                        f"Elapsed: {elapsed:6.1f}s | ETA: {eta:6.1f}s",
                        end=""
                    )

            print("\n[MultiTaskDataset] Preload finished. (All images stored in memory)")

    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # -------------------------------------------------------------------------
    def __getitem__(self, idx):

        # RAM에서 즉시 로딩 → 매우 빠름
        if self.preload:
            img_in = self.memory_in[idx]
            img_gt = self.memory_gt[idx]
        else:
            inp_path, gt_path, _ = self.samples[idx]
            img_in = load_image(inp_path)
            img_gt = load_image(gt_path)

        img_in, img_gt = self.transform(img_in, img_gt)

        return {
            "input": img_in,
            "gt": img_gt,
            "dataset": self.samples[idx][2],
        }


# -------------------------------------------------------------------------
# Standalone Test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    root = "G:/VETNet_pilot/data"
    ds = MultiTaskDataset(root, preload=True)
    print("\nLen =", len(ds))

    sample = ds[100]
    print(sample["input"].shape, sample["gt"].shape, sample["dataset"])
