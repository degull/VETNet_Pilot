import os, sys

# 현재 파일 = VETNet_pilot/datasets/multitask_dataset.py
# 상위 폴더 = VETNet_pilot/datasets
# 상위 상위 = VETNet_pilot  ← 여기까지만 올라가야 한다

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))              # .../datasets
ROOT = os.path.dirname(CURRENT_DIR)                                   # .../VETNet_pilot

# 중복 방지 후 sys.path에 추가
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[DEBUG] Using ROOT:", ROOT)

import random
from torch.utils.data import Dataset

from datasets.utils_dataset import load_image, build_all_pairs
from datasets.transforms import TrainTransform

class MultiTaskDataset(Dataset):
    """
    Phase1 Backbone Training Dataset
    - 5 datasets merged into one
    - __getitem__ randomly selects one sample across all datasets
    """

    def __init__(self, data_root, transform=None):
        super().__init__()
        self.data_root = data_root
        self.transform = transform if transform is not None else TrainTransform(size=256)

        # Load dataset file pairs
        self.all_pairs = build_all_pairs(self.data_root)

        # Flatten into single list: (input_path, gt_path, dataset_name)
        self.samples = []
        for name, pairs in self.all_pairs.items():
            for (inp, gt) in pairs:
                self.samples.append((inp, gt, name))

        print(f"[MultiTaskDataset] Total Samples = {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp_path, gt_path, name = self.samples[idx]

        img_in = load_image(inp_path)
        img_gt = load_image(gt_path)

        img_in, img_gt = self.transform(img_in, img_gt)

        return {
            "input": img_in,
            "gt": img_gt,
            "dataset": name
        }


if __name__ == "__main__":
    root = "G:/VETNet_pilot/data"
    ds = MultiTaskDataset(root)
    print("Len =", len(ds))
    sample = ds[100]
    print(sample["input"].shape, sample["gt"].shape, sample["dataset"])
