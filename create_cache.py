# G:/VETNet_pilot/create_cache.py
import os
from tqdm import tqdm
from PIL import Image

# ------------------------------------------------------------------
# 경로 설정
# ------------------------------------------------------------------
ROOT = os.path.abspath(os.path.dirname(__file__))       # G:/VETNet_pilot
DATA_ROOT = os.path.join(ROOT, "data")                  # G:/VETNet_pilot/data
CACHE_ROOT = os.path.join(ROOT, "preload_cache")        # G:/VETNet_pilot/preload_cache

os.makedirs(CACHE_ROOT, exist_ok=True)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image_file(path):
    return os.path.splitext(path)[-1].lower() in IMG_EXTS


# ------------------------------------------------------------------
# 1) DayRainDrop (Drop/xxx.png ↔ Clear/xxx.png)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 1) DayRainDrop (Drop/00001/00001.png ↔ Clear/00001/00001.png, 첫 프레임만)
# ------------------------------------------------------------------
def collect_dayraindrop_pairs():
    """
    DayRainDrop 구조:

    DayRainDrop/
       Clear/00001/00001.png, 00002.png ...
       Drop/00001/00001.png, 00002.png ...

    → 각 seq 폴더에서 '00001.png' 한 장만 pair 로 사용
    """

    pairs = []
    root = os.path.join(DATA_ROOT, "DayRainDrop")
    drop_root = os.path.join(root, "Drop")
    clear_root = os.path.join(root, "Clear")

    if not (os.path.isdir(drop_root) and os.path.isdir(clear_root)):
        print("[DayRainDrop] directory missing")
        return pairs

    print(f"[DayRainDrop Debug] root={root}")

    seq_dirs = sorted(os.listdir(drop_root))
    count = 0

    for seq in seq_dirs:
        drop_seq_dir = os.path.join(drop_root, seq)
        clear_seq_dir = os.path.join(clear_root, seq)

        if not (os.path.isdir(drop_seq_dir) and os.path.isdir(clear_seq_dir)):
            continue

        frame_name = "00001.png"  # ★ 첫 프레임만 사용
        in_path = os.path.join(drop_seq_dir, frame_name)
        gt_path = os.path.join(clear_seq_dir, frame_name)

        if not (os.path.isfile(in_path) and os.path.isfile(gt_path)):
            continue

        pairs.append(("DayRainDrop", in_path, gt_path))
        count += 1

    print(f"[DayRainDrop] Found {count} pairs.")
    return pairs

# ------------------------------------------------------------------
# 2) NightRainDrop (Drop/00001/00001.png ↔ Clear/00001/00001.png, 첫 프레임만)
# ------------------------------------------------------------------
def collect_nightraindrop_pairs():
    pairs = []
    root = os.path.join(DATA_ROOT, "NightRainDrop")
    drop_root = os.path.join(root, "Drop")
    clear_root = os.path.join(root, "Clear")
    if not (os.path.isdir(drop_root) and os.path.isdir(clear_root)):
        return pairs

    print(f"[NightRainDrop Debug] root={root}")
    seq_dirs = sorted(os.listdir(drop_root))
    count = 0
    for seq in seq_dirs:
        drop_seq_dir = os.path.join(drop_root, seq)
        clear_seq_dir = os.path.join(clear_root, seq)
        if not (os.path.isdir(drop_seq_dir) and os.path.isdir(clear_seq_dir)):
            continue

        frame_name = "00001.png"
        in_path = os.path.join(drop_seq_dir, frame_name)
        gt_path = os.path.join(clear_seq_dir, frame_name)
        if not (os.path.isfile(in_path) and os.path.isfile(gt_path)):
            continue

        pairs.append(("NightRainDrop", in_path, gt_path))
        count += 1
    print(f"[NightRainDrop] Found {count} pairs.")
    return pairs


# ------------------------------------------------------------------
# 3) CSD (Train/Test 공통: Snow/*.tif ↔ Gt/*.tif)
# ------------------------------------------------------------------
def collect_csd_pairs():
    pairs = []
    root = os.path.join(DATA_ROOT, "CSD")
    for split in ["Train", "Test"]:
        snow_dir = os.path.join(root, split, "Snow")
        gt_dir = os.path.join(root, split, "Gt")
        if not (os.path.isdir(snow_dir) and os.path.isdir(gt_dir)):
            continue

        snow_files = sorted(os.listdir(snow_dir))
        count_split = 0
        for f in snow_files:
            in_path = os.path.join(snow_dir, f)
            gt_path = os.path.join(gt_dir, f)
            if not is_image_file(in_path):
                continue
            if not os.path.isfile(gt_path):
                continue
            pairs.append(("CSD", in_path, gt_path))
            count_split += 1
        print(f"[CSD-{split}] Found {count_split} pairs.")
    print(f"[CSD] Total pairs = {len(pairs)}")
    return pairs


# ------------------------------------------------------------------
# 4) RESIDE-6K (train/test: hazy/*.jpg ↔ GT/*.jpg)
# ------------------------------------------------------------------
def collect_reside_pairs():
    pairs = []
    root = os.path.join(DATA_ROOT, "RESIDE-6K")
    if not os.path.isdir(root):
        return pairs

    for split in ["train", "test"]:
        hazy_dir = os.path.join(root, split, "hazy")
        gt_dir = os.path.join(root, split, "GT")
        if not (os.path.isdir(hazy_dir) and os.path.isdir(gt_dir)):
            continue

        hazy_files = sorted(os.listdir(hazy_dir))
        count_split = 0
        for f in hazy_files:
            in_path = os.path.join(hazy_dir, f)
            gt_path = os.path.join(gt_dir, f)
            if not is_image_file(in_path):
                continue
            if not os.path.isfile(gt_path):
                continue
            pairs.append(("RESIDE-6K", in_path, gt_path))
            count_split += 1
        print(f"[RESIDE-6K-{split}] Found {count_split} pairs.")

    print(f"[RESIDE-6K] Total pairs = {len(pairs)}")
    return pairs


# ------------------------------------------------------------------
# 5) Rain100H (가능하면 자동 탐색, 실패하면 스킵)
#    - 예상: data/rain100H/train/rain/*.png ↔ train/norain/*.png
# ------------------------------------------------------------------
def collect_rain100H_pairs():
    pairs = []
    for ds_name in ["rain100H", "Rain100H"]:
        base = os.path.join(DATA_ROOT, ds_name)
        train_root = os.path.join(base, "train")
        if not os.path.isdir(train_root):
            continue

        # 후보 디렉토리 탐색
        subdirs = [d for d in os.listdir(train_root)
                   if os.path.isdir(os.path.join(train_root, d))]
        if len(subdirs) < 2:
            continue

        in_dir = gt_dir = None
        for d in subdirs:
            lower = d.lower()
            if "gt" in lower or "norain" in lower:
                gt_dir = os.path.join(train_root, d)
            elif "rain" in lower or "input" in lower:
                in_dir = os.path.join(train_root, d)

        if not (in_dir and gt_dir and
                os.path.isdir(in_dir) and os.path.isdir(gt_dir)):
            continue

        in_files = sorted(os.listdir(in_dir))
        count = 0
        for f in in_files:
            in_path = os.path.join(in_dir, f)
            gt_path = os.path.join(gt_dir, f)
            if not is_image_file(in_path):
                continue
            if not os.path.isfile(gt_path):
                continue
            pairs.append((ds_name, in_path, gt_path))
            count += 1
        print(f"[{ds_name}] Found {count} pairs.")
        break

    return pairs


# ------------------------------------------------------------------
# 메인: PNG 캐시 생성
# ------------------------------------------------------------------
if __name__ == "__main__":
    all_pairs = []

    all_pairs += collect_dayraindrop_pairs()
    all_pairs += collect_nightraindrop_pairs()
    all_pairs += collect_csd_pairs()
    all_pairs += collect_reside_pairs()
    all_pairs += collect_rain100H_pairs()

    print(f"[CacheBuilder] Total pairs = {len(all_pairs)}")

    idx_global = 0
    for ds_name, in_path, gt_path in tqdm(all_pairs, desc="Caching", ncols=100):
        ds_dir = os.path.join(CACHE_ROOT, ds_name)
        os.makedirs(ds_dir, exist_ok=True)

        idx_str = f"{idx_global:06d}"
        cache_in = os.path.join(ds_dir, f"{idx_str}_in.png")
        cache_gt = os.path.join(ds_dir, f"{idx_str}_gt.png")

        # 이미 있으면 스킵 (재실행 가능)
        if os.path.isfile(cache_in) and os.path.isfile(cache_gt):
            idx_global += 1
            continue

        try:
            img_in = Image.open(in_path).convert("RGB").resize((256, 256), Image.BICUBIC)
            img_gt = Image.open(gt_path).convert("RGB").resize((256, 256), Image.BICUBIC)

            img_in.save(cache_in)
            img_gt.save(cache_gt)
        except Exception as e:
            print(f"[ERROR] Skip pair ({in_path}, {gt_path}) because: {e}")
            continue

        idx_global += 1

    print(f"[CacheBuilder] Done. PNG cache saved under: {CACHE_ROOT}")
