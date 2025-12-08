# G:/VETNet_pilot/datasets/utils_dataset.py
import os
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_img(path):
    return os.path.splitext(path)[-1].lower() in IMG_EXT


# -----------------------------------------------------------
# 로딩: PIL 이미지 반환
# -----------------------------------------------------------
def load_image(path):
    """경로에서 이미지를 RGB로 로딩"""
    return Image.open(path).convert("RGB")


# -----------------------------------------------------------
# 전체 dataset pair 생성기
# (이전 create_cache.py 와 동일한 pair 규칙 반영)
# -----------------------------------------------------------
def build_all_pairs(root):
    """
    data_root 내부의 모든 데이터셋에서 (input, gt) 페어를 전부 찾는다.
    반환 형태:
        {
            "DayRainDrop": [(in_path, gt_path), ...],
            "NightRainDrop": [...],
            ...
        }
    """
    all_pairs = {}

    # ============ DayRainDrop ============ 
    ds = "DayRainDrop"
    day_root = os.path.join(root, ds)
    drop_root = os.path.join(day_root, "Drop")
    clear_root = os.path.join(day_root, "Clear")
    pairs = []
    if os.path.isdir(drop_root) and os.path.isdir(clear_root):
        seq_dirs = sorted(os.listdir(drop_root))
        for seq in seq_dirs:
            drop_seq = os.path.join(drop_root, seq)
            clear_seq = os.path.join(clear_root, seq)
            if not (os.path.isdir(drop_seq) and os.path.isdir(clear_seq)):
                continue

            # 첫 프레임만 사용
            fname = "00001.png"
            inp = os.path.join(drop_seq, fname)
            gt  = os.path.join(clear_seq, fname)
            if os.path.isfile(inp) and os.path.isfile(gt):
                pairs.append((inp, gt))

    all_pairs[ds] = pairs


    # ============ NightRainDrop ============
    ds = "NightRainDrop"
    nr_root = os.path.join(root, ds)
    drop_root = os.path.join(nr_root, "Drop")
    clear_root = os.path.join(nr_root, "Clear")
    pairs = []
    if os.path.isdir(drop_root) and os.path.isdir(clear_root):
        seq_dirs = sorted(os.listdir(drop_root))
        for seq in seq_dirs:
            drop_seq = os.path.join(drop_root, seq)
            clear_seq = os.path.join(clear_root, seq)
            if not (os.path.isdir(drop_seq) and os.path.isdir(clear_seq)):
                continue

            # 첫 프레임만 사용
            fname = "00001.png"
            inp = os.path.join(drop_seq, fname)
            gt  = os.path.join(clear_seq, fname)
            if os.path.isfile(inp) and os.path.isfile(gt):
                pairs.append((inp, gt))

    all_pairs[ds] = pairs


    # ============ CSD (Train + Test) ============
    ds = "CSD"
    csd_root = os.path.join(root, ds)
    pairs = []
    for split in ["Train", "Test"]:
        snow_dir = os.path.join(csd_root, split, "Snow")
        gt_dir   = os.path.join(csd_root, split, "Gt")
        if not (os.path.isdir(snow_dir) and os.path.isdir(gt_dir)):
            continue
        for f in sorted(os.listdir(snow_dir)):
            inp = os.path.join(snow_dir, f)
            gt  = os.path.join(gt_dir, f)
            if os.path.isfile(inp) and os.path.isfile(gt) and is_img(inp):
                pairs.append((inp, gt))

    all_pairs[ds] = pairs


    # ============ RESIDE-6K (train/test 공통) ============
    ds = "RESIDE-6K"
    reside_root = os.path.join(root, ds)
    pairs = []
    for split in ["train", "test"]:
        hazy_dir = os.path.join(reside_root, split, "hazy")
        gt_dir   = os.path.join(reside_root, split, "GT")
        if not (os.path.isdir(hazy_dir) and os.path.isdir(gt_dir)):
            continue
        for f in sorted(os.listdir(hazy_dir)):
            inp = os.path.join(hazy_dir, f)
            gt  = os.path.join(gt_dir, f)
            if os.path.isfile(inp) and os.path.isfile(gt) and is_img(inp):
                pairs.append((inp, gt))

    all_pairs[ds] = pairs


    # ============ rain100H ============
    for ds in ["rain100H", "Rain100H"]:
        base = os.path.join(root, ds)
        train_root = os.path.join(base, "train")
        if not os.path.isdir(train_root):
            continue

        # rain / no-rain 탐색
        subdirs = os.listdir(train_root)
        inp_dir = gt_dir = None
        for d in subdirs:
            dd = d.lower()
            if "rain" in dd:
                inp_dir = os.path.join(train_root, d)
            if "gt" in dd or "norain" in dd:
                gt_dir = os.path.join(train_root, d)

        pairs = []
        if inp_dir and gt_dir:
            for f in sorted(os.listdir(inp_dir)):
                inp = os.path.join(inp_dir, f)
                gt  = os.path.join(gt_dir, f)
                if os.path.isfile(inp) and os.path.isfile(gt) and is_img(inp):
                    pairs.append((inp, gt))

            all_pairs[ds] = pairs
            break

    return all_pairs
