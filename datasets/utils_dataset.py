# datasets/utils_dataset.py

import os
from glob import glob

def build_all_pairs(data_root):
    """
    모든 데이터셋 pair를 (input_path, gt_path, tag) 형태로 반환
    + 디버그 안전장치 포함
    """
    pairs = []

    def add_pairs(input_dir, gt_dir, tag):
        input_files = sorted(glob(os.path.join(input_dir, "*")))
        gt_files    = sorted(glob(os.path.join(gt_dir, "*")))

        if len(input_files) != len(gt_files):
            print(f"[WARN] Mismatch count in {tag}: input={len(input_files)}, gt={len(gt_files)}")

        for inp, gt in zip(input_files, gt_files):

            # 파일 이름 검증
            if not os.path.isfile(inp):
                print(f"[ERROR] Not a file: {inp}")
                continue

            if not os.path.isfile(gt):
                print(f"[ERROR] Not a file: {gt}")
                continue

            # 디버그 로그 추가
            if len(inp) < 10:
                print(f"[DEBUG WARNING] Strange input path: {inp}")

            pairs.append((inp.replace("\\", "/"), gt.replace("\\", "/"), tag))

    # ---------------------------------------------------------
    # 1) DayRainDrop
    # ---------------------------------------------------------
    dr = os.path.join(data_root, "DayRainDrop")
    add_pairs(os.path.join(dr, "Drop"),  os.path.join(dr, "Clear"), "DayRainDrop")

    # ---------------------------------------------------------
    # 2) NightRainDrop
    # ---------------------------------------------------------
    nr = os.path.join(data_root, "NightRainDrop")
    add_pairs(os.path.join(nr, "Drop"),  os.path.join(nr, "Clear"), "NightRainDrop")

    # ---------------------------------------------------------
    # 3) Snow / CSD
    # ---------------------------------------------------------
    csd = os.path.join(data_root, "CSD", "Train")
    add_pairs(os.path.join(csd, "Snow"), os.path.join(csd, "Gt"), "CSD")

    # ---------------------------------------------------------
    # 4) Rain100H
    # ---------------------------------------------------------
    r100h = os.path.join(data_root, "rain100H", "train")
    add_pairs(os.path.join(r100h, "rain"), os.path.join(r100h, "norain"), "rain100H")

    # ---------------------------------------------------------
    # 5) RESIDE-6K (예: haze 제거)
    # ---------------------------------------------------------
    res = os.path.join(data_root, "RESIDE-6K", "train")
    add_pairs(os.path.join(res, "hazy"), os.path.join(res, "clear"), "RESIDE-6K")

    # ---------------------------------------------------------
    print(f"[build_all_pairs] Final pairs = {len(pairs)}")

    # pair 예시 출력
    for i in range(5):
        print("[PAIR]", pairs[i])

    return pairs
