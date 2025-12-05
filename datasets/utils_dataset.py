# G:\VETNet_pilot\datasets\utils_dataset.py
import os
import glob
from PIL import Image
import tifffile
import numpy as np

# ===========================
#     GLOBAL SETTINGS
# ===========================

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ===========================
#     HELPER FUNCTIONS
# ===========================

def load_image(path):
    """Load an image supporting JPG/PNG/TIF."""
    ext = os.path.splitext(path)[1].lower()

    if ext in [".tif", ".tiff"]:
        img = tifffile.imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return Image.fromarray(img.astype(np.uint8))
    else:
        return Image.open(path).convert("RGB")


def _list_images_in_dir(folder):
    """현재 폴더 바로 아래의 이미지 파일만 리스트."""
    if not os.path.isdir(folder):
        return []
    files = []
    for f in os.listdir(folder):
        fpath = os.path.join(folder, f)
        if os.path.isfile(fpath) and os.path.splitext(f)[1].lower() in IMG_EXTS:
            files.append(fpath)
    return sorted(files)


def _list_images_recursive(folder):
    """하위 폴더까지 재귀적으로 모든 이미지 파일 리스트."""
    if not os.path.isdir(folder):
        return []
    files = []
    for root, _, fnames in os.walk(folder):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                files.append(os.path.join(root, f))
    return sorted(files)


def list_pairs_by_filename(folder_input, folder_gt):
    """
    Match pairs by identical filenames in two folders.
    Example: rain100H, RESIDE-6K
    """
    inputs = _list_images_in_dir(folder_input)
    gts    = _list_images_in_dir(folder_gt)

    input_dict = {os.path.basename(p): p for p in inputs}
    gt_dict    = {os.path.basename(p): p for p in gts}

    pairs = []
    for fname, ip in input_dict.items():
        if fname in gt_dict:
            pairs.append((ip, gt_dict[fname]))

    return pairs


# ===========================
#   RAIN DROP DATASET PAIRS
# ===========================

def list_pairs_rain_drop(root):
    """
    DayRainDrop / NightRainDrop pair builder.

    기대 구조:
      root/Clear/00001/*.png (or jpg, tif...)
      root/Drop/00001/*.png

    폴더 구조가 조금 달라도:
      - Clear/하위폴더/어디든 이미지
      - Drop/하위폴더/어디든 이미지
    ⇒ 같은 폴더 이름(예: 00001) 안에서 파일명으로 매칭.
    """
    clear_root = os.path.join(root, "Clear")
    drop_root  = os.path.join(root, "Drop")

    if not os.path.isdir(clear_root) or not os.path.isdir(drop_root):
        print(f"[WARN] RainDrop root missing Clear/Drop: {root}")
        return []

    # Clear 쪽 폴더 기준으로 스캔 (00001, 00002, ...)
    clear_folders = [
        d for d in sorted(os.listdir(clear_root))
        if os.path.isdir(os.path.join(clear_root, d))
    ]

    pairs = []

    for folder in clear_folders:
        cdir = os.path.join(clear_root, folder)
        ddir = os.path.join(drop_root, folder)

        if not os.path.isdir(ddir):
            # Drop 쪽에 동일 이름 폴더 없으면 스킵
            continue

        # 각 폴더 하위의 모든 이미지 재귀적으로 수집
        cfiles = _list_images_recursive(cdir)
        dfiles = _list_images_recursive(ddir)

        if len(cfiles) == 0 or len(dfiles) == 0:
            continue

        # 파일명을 기준으로 매칭
        c_dict = {os.path.basename(f): f for f in cfiles}
        d_dict = {os.path.basename(f): f for f in dfiles}

        common_names = sorted(list(set(c_dict.keys()) & set(d_dict.keys())))
        for fname in common_names:
            inp = d_dict[fname]   # Drop: input
            gt  = c_dict[fname]   # Clear: gt
            pairs.append((inp, gt))

    # 디버그용 출력 (원하면 주석 처리 가능)
    print(f"[RainDrop Debug] root={root}")
    print(f"  - Clear root: {clear_root}")
    print(f"  - Drop  root: {drop_root}")
    print(f"  - Matched pairs: {len(pairs)}")

    return pairs


# ===========================
#       CSD DATASET
# ===========================

def list_pairs_csd(root):
    """
    CSD dataset:
      root/Train/Snow/*.tif
      root/Train/Gt/*.tif
    """
    snow_dir = os.path.join(root, "Snow")
    gt_dir   = os.path.join(root, "Gt")

    snow_files = sorted(glob.glob(os.path.join(snow_dir, "*.tif")))
    gt_files   = sorted(glob.glob(os.path.join(gt_dir, "*.tif")))

    snow_dict = {os.path.basename(f): f for f in snow_files}
    gt_dict   = {os.path.basename(f): f for f in gt_files}

    pairs = []
    for fname, ip in snow_dict.items():
        if fname in gt_dict:
            pairs.append((ip, gt_dict[fname]))

    return pairs


# ===========================
#    MAIN BUILD FUNCTION
# ===========================

def build_all_pairs(data_root):
    """
    Build all dataset pairs for:
      - CSD (Train only)
      - Rain100H (Train only)
      - RESIDE-6K (Train only)
      - DayRainDrop
      - NightRainDrop
    """
    all_pairs = {}

    # --- CSD ---
    csd_root = os.path.join(data_root, "CSD", "Train")
    csd_pairs = list_pairs_csd(csd_root)
    all_pairs["CSD"] = csd_pairs

    # --- rain100H ---
    r100_root = os.path.join(data_root, "rain100H", "train")
    r100_pairs = list_pairs_by_filename(
        folder_input=os.path.join(r100_root, "rain"),
        folder_gt=os.path.join(r100_root, "norain")
    )
    all_pairs["rain100H"] = r100_pairs

    # --- RESIDE-6K ---
    reside_root = os.path.join(data_root, "RESIDE-6K", "train")
    reside_pairs = list_pairs_by_filename(
        folder_input=os.path.join(reside_root, "hazy"),
        folder_gt=os.path.join(reside_root, "GT")
    )
    all_pairs["RESIDE-6K"] = reside_pairs

    # --- DayRainDrop ---
    day_root = os.path.join(data_root, "DayRainDrop")
    day_pairs = list_pairs_rain_drop(day_root)
    all_pairs["DayRainDrop"] = day_pairs

    # --- NightRainDrop ---
    night_root = os.path.join(data_root, "NightRainDrop")
    night_pairs = list_pairs_rain_drop(night_root)
    all_pairs["NightRainDrop"] = night_pairs

    return all_pairs


# ===========================
#          TEST
# ===========================

if __name__ == "__main__":
    root = "G:/VETNet_pilot/data"
    pairs = build_all_pairs(root)
    for name, p in pairs.items():
        print(f"[{name}] Pairs = {len(p)}")
