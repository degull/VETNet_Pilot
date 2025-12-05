# G:/VETNet_pilot/train_phase1_backbone.py
# 코드1
""" import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[DEBUG] Using ROOT:", ROOT)

import time
from datetime import timedelta

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

from datasets.multitask_dataset import MultiTaskDataset
from models.backbone.vetnet_backbone import VETNetBackbone

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    USE_SKIMAGE = True
except:
    USE_SKIMAGE = False

from torch.amp import autocast, GradScaler


# ============================================================
# Config
# ============================================================
class Config:
    data_root = "G:/VETNet_pilot/data"
    save_root = "G:/VETNet_pilot/checkpoints/phase1_backbone"
    results_root = "G:/VETNet_pilot/results/phase1_backbone"

    epochs = 100
    batch_size = 4
    num_workers = 4

    lr = 3e-4
    weight_decay = 1e-8

    in_channels = 3
    out_channels = 3
    dim = 48
    num_blocks = (4, 6, 6, 8)
    heads = (1, 2, 4, 8)
    volterra_rank = 4
    ffn_expansion_factor = 2.66
    bias = False

    metric_images_per_batch = 2
    print_freq = 100
    use_amp = True


cfg = Config()


# ============================================================
# Utils
# ============================================================
def seed_everything(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_numpy_img(t):
    t = t.detach().cpu().clamp(0, 1)
    t = t.permute(1, 2, 0).numpy()
    return (t * 255).astype("uint8")


def compute_batch_psnr_ssim(pred, gt, max_samples=2):
    if not USE_SKIMAGE:
        return 0.0, 0.0

    n = min(pred.size(0), max_samples)
    ps, ss = 0, 0

    for i in range(n):
        p = pred[i].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        g = gt[i].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

        ps += peak_signal_noise_ratio(g, p, data_range=1.0)
        ss += structural_similarity(g, p, channel_axis=2, data_range=1.0)

    return ps / n, ss / n


def format_time(sec):
    return str(timedelta(seconds=int(sec)))


# ============================================================
# 이미지 3개를 가로로 붙여 저장하는 함수
# ============================================================
def save_triplet_image(img_in, img_pred, img_gt, save_path):
    img_in = tensor_to_numpy_img(img_in)
    img_pred = tensor_to_numpy_img(img_pred)
    img_gt = tensor_to_numpy_img(img_gt)

    H, W, _ = img_in.shape
    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)

    canvas[:, 0:W] = img_in
    canvas[:, W:2 * W] = img_pred
    canvas[:, 2 * W:3 * W] = img_gt

    Image.fromarray(canvas).save(save_path)


# ============================================================
# Training Loop
# ============================================================
def train_phase1():
    seed_everything(42)

    os.makedirs(cfg.save_root, exist_ok=True)
    os.makedirs(cfg.results_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # Dataset
    train_dataset = MultiTaskDataset(cfg.data_root)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    num_train = len(train_dataset)
    print("[Data] Total Train Samples =", num_train)

    # Model
    model = VETNetBackbone(
        in_channels=cfg.in_channels, out_channels=cfg.out_channels,
        dim=cfg.dim, num_blocks=cfg.num_blocks, heads=cfg.heads,
        volterra_rank=cfg.volterra_rank,
        ffn_expansion_factor=cfg.ffn_expansion_factor, bias=cfg.bias
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Params: {num_params/1e6:.3f} M")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=(device.type == "cuda" and cfg.use_amp))

    best_ssim = -1
    hist_loss, hist_psnr, hist_ssim = [], [], []
    global_start = time.time()

    # ============================================================
    # Epoch Loop
    # ============================================================
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_start = time.time()

        running_loss = 0
        running_psnr = 0
        running_ssim = 0
        metric_cnt = 0

        pbar = tqdm(enumerate(train_loader, start=1),
                    total=len(train_loader),
                    desc=f"Epoch [{epoch}/{cfg.epochs}]",
                    ncols=120)

        first_batch_input = None
        first_batch_pred = None
        first_batch_gt = None

        for it, batch in pbar:
            inp = batch["input"].to(device)
            gt = batch["gt"].to(device)

            if first_batch_input is None:
                first_batch_input = inp[0].detach()
                first_batch_gt = gt[0].detach()

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=cfg.use_amp):
                pred = model(inp)
                loss = F.l1_loss(pred, gt)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            pred_c = pred.clamp(0, 1)
            gt_c = gt.clamp(0, 1)

            if first_batch_pred is None:
                first_batch_pred = pred_c[0].detach()

            b_psnr, b_ssim = compute_batch_psnr_ssim(pred_c, gt_c)
            running_psnr += b_psnr
            running_ssim += b_ssim
            metric_cnt += 1

            running_loss += loss.item() * inp.size(0)

            cur_loss = running_loss / (it * cfg.batch_size)
            cur_psnr = running_psnr / metric_cnt
            cur_ssim = running_ssim / metric_cnt

            pbar.set_postfix({
                "L": f"{cur_loss:.4f}",
                "P": f"{cur_psnr:.2f}",
                "S": f"{cur_ssim:.3f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.1e}",
            })

        # ---------------- Epoch finished ----------------
        epoch_loss = running_loss / num_train
        epoch_psnr = running_psnr / metric_cnt
        epoch_ssim = running_ssim / metric_cnt

        hist_loss.append(epoch_loss)
        hist_psnr.append(epoch_psnr)
        hist_ssim.append(epoch_ssim)

        scheduler.step()

        # ============================================================
        # 결과 이미지 저장 (원본|복원|GT) 1장
        # ============================================================
        img_name = (
            f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.png"
        )
        save_path = os.path.join(cfg.results_root, img_name)

        save_triplet_image(first_batch_input, first_batch_pred, first_batch_gt, save_path)
        print(f"[Saved Image] {save_path}")

        # ============================================================
        # Checkpoint 저장
        # ============================================================
        ckpt_name = (
            f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.pth"
        )
        ckpt_path = os.path.join(cfg.save_root, ckpt_name)

        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss": epoch_loss,
            "psnr": epoch_psnr,
            "ssim": epoch_ssim,
        }, ckpt_path)

        print(f"[Checkpoint] Saved: {ckpt_path}")

        if epoch_ssim > best_ssim:
            best_ssim = epoch_ssim
            best_path = os.path.join(cfg.save_root, "best_phase1_backbone.pth")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": epoch_loss,
                "psnr": epoch_psnr,
                "ssim": epoch_ssim,
            }, best_path)
            print(f"[Best] SSIM Updated → {best_path}")

    print("\n[Training Finished]")
    print("Best SSIM:", best_ssim)


if __name__ == "__main__":
    train_phase1()
 """

# G:/VETNet_pilot/train_phase1_backbone.py
# 시간 최적화
# G:/VETNet_pilot/trainers/train_phase1_backbone.py
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[DEBUG] Using ROOT:", ROOT)

import time
from datetime import timedelta

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

from datasets.multitask_dataset import MultiTaskDataset
from models.backbone.vetnet_backbone import VETNetBackbone

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    USE_SKIMAGE = True
except Exception:
    USE_SKIMAGE = False

from torch.amp import autocast, GradScaler


# ============================================================
# Config
# ============================================================
class Config:
    data_root = "G:/VETNet_pilot/data"
    save_root = "G:/VETNet_pilot/checkpoints/phase1_backbone"
    results_root = "G:/VETNet_pilot/results/phase1_backbone"

    epochs = 100
    batch_size = 4
    num_workers = 4   # DataLoader workers (Windows 최적값 근처)

    lr = 3e-4
    weight_decay = 1e-8

    in_channels = 3
    out_channels = 3
    dim = 48
    num_blocks = (4, 6, 6, 8)
    heads = (1, 2, 4, 8)
    volterra_rank = 4
    ffn_expansion_factor = 2.66
    bias = False

    metric_images_per_batch = 2
    print_freq = 100
    use_amp = True   # AMP 항상 사용 (FP16)


cfg = Config()


# ============================================================
# Utils
# ============================================================
def seed_everything(seed=42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_numpy_img(t: torch.Tensor):
    t = t.detach().cpu().clamp(0, 1)
    if t.ndim == 3:
        t = t.permute(1, 2, 0)
    return (t.numpy() * 255).astype("uint8")


def compute_batch_psnr_ssim(pred, gt, max_samples=2):
    if not USE_SKIMAGE:
        return 0.0, 0.0

    n = min(pred.size(0), max_samples)
    ps, ss = 0.0, 0.0

    for i in range(n):
        p = pred[i].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        g = gt[i].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

        ps += peak_signal_noise_ratio(g, p, data_range=1.0)
        ss += structural_similarity(g, p, channel_axis=2, data_range=1.0)

    return ps / n, ss / n


def format_time(sec):
    return str(timedelta(seconds=int(sec)))


# ============================================================
# 이미지 3개를 가로로 붙여 저장하는 함수
# ============================================================
def save_triplet_image(img_in, img_pred, img_gt, save_path):
    """
    img_in, img_pred, img_gt: (C, H, W) torch tensors
    → 하나의 가로 이미지로 붙여 저장
    """
    img_in = tensor_to_numpy_img(img_in)
    img_pred = tensor_to_numpy_img(img_pred)
    img_gt = tensor_to_numpy_img(img_gt)

    H, W, _ = img_in.shape
    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)

    canvas[:, 0:W] = img_in
    canvas[:, W:2 * W] = img_pred
    canvas[:, 2 * W:3 * W] = img_gt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(canvas).save(save_path)


# ============================================================
# Training Loop (RAM preload 최적화 버전)
# ============================================================
def train_phase1():
    seed_everything(42)

    os.makedirs(cfg.save_root, exist_ok=True)
    os.makedirs(cfg.results_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # ---------------- Dataset (RAM Preload ON) ----------------
    # MultiTaskDataset 쪽에서 이미:
    #  - build_all_pairs → 파일 리스트 구성
    #  - preload=True → 모든 이미지를 RAM에 로딩 (한 번만)
    train_dataset = MultiTaskDataset(
        cfg.data_root,
        preload=True,       # ★ RAM preload 활성화
    )

    # DataLoader: 이제 디스크 I/O는 없음 → num_workers 적당히만
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    num_train = len(train_dataset)
    print("[Data] Total Train Samples =", num_train)

    # ---------------- Model ----------------
    model = VETNetBackbone(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        dim=cfg.dim,
        num_blocks=cfg.num_blocks,
        heads=cfg.heads,
        volterra_rank=cfg.volterra_rank,
        ffn_expansion_factor=cfg.ffn_expansion_factor,
        bias=cfg.bias,
    ).to(device)

    # Windows + eager에서는 torch.compile이 느려서 비활성화
    if device.type == "cuda":
        print("[torch.compile] Disabled on Windows for performance.")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Params: {num_params/1e6:.3f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
    )

    # AMP: FP16 + loss scaling
    scaler = GradScaler(
        enabled=(device.type == "cuda" and cfg.use_amp),
        init_scale=2.0 ** 16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    )

    best_ssim = -1.0
    hist_loss, hist_psnr, hist_ssim = [], [], []
    global_start = time.time()

    # ============================================================
    # Epoch Loop
    # ============================================================
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_start = time.time()

        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        metric_cnt = 0

        pbar = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"Epoch [{epoch}/{cfg.epochs}]",
            ncols=120,
        )

        first_batch_input = None
        first_batch_pred = None
        first_batch_gt = None

        for it, batch in pbar:
            inp = batch["input"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            if first_batch_input is None:
                first_batch_input = inp[0].detach()
                first_batch_gt = gt[0].detach()

            optimizer.zero_grad(set_to_none=True)

            # AMP FP16
            with autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else None,
                enabled=(device.type == "cuda" and cfg.use_amp),
            ):
                pred = model(inp)
                loss = F.l1_loss(pred, gt)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            pred_c = pred.clamp(0, 1)
            gt_c = gt.clamp(0, 1)

            if first_batch_pred is None:
                first_batch_pred = pred_c[0].detach()

            # PSNR / SSIM (배치 단위)
            b_psnr, b_ssim = compute_batch_psnr_ssim(
                pred_c,
                gt_c,
                max_samples=cfg.metric_images_per_batch,
            )
            running_psnr += b_psnr
            running_ssim += b_ssim
            metric_cnt += 1

            running_loss += loss.item() * inp.size(0)

            cur_loss = running_loss / (it * cfg.batch_size)
            cur_psnr = running_psnr / max(metric_cnt, 1)
            cur_ssim = running_ssim / max(metric_cnt, 1)

            pbar.set_postfix({
                "L": f"{cur_loss:.4f}",
                "P": f"{cur_psnr:.2f}",
                "S": f"{cur_ssim:.3f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.1e}",
            })

        # ---------------- Epoch finished ----------------
        epoch_time = time.time() - epoch_start
        total_time = time.time() - global_start

        epoch_loss = running_loss / num_train
        epoch_psnr = running_psnr / max(metric_cnt, 1)
        epoch_ssim = running_ssim / max(metric_cnt, 1)

        hist_loss.append(epoch_loss)
        hist_psnr.append(epoch_psnr)
        hist_ssim.append(epoch_ssim)

        scheduler.step()

        print("\n" + "=" * 80)
        print(f"Epoch [{epoch}/{cfg.epochs}] Finished")
        print(f"  - Loss : {epoch_loss:.6f}")
        print(f"  - PSNR : {epoch_psnr:.3f}")
        print(f"  - SSIM : {epoch_ssim:.4f}")
        print(f"  - Epoch Time    : {format_time(epoch_time)}")
        print(f"  - Total Elapsed : {format_time(total_time)}")
        print("=" * 80 + "\n")

        # ====================================================
        # 결과 이미지 저장 (원본|복원|GT)
        # ====================================================
        img_name = (
            f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.png"
        )
        img_path = os.path.join(cfg.results_root, img_name)
        save_triplet_image(first_batch_input, first_batch_pred, first_batch_gt, img_path)
        print(f"[Saved Image] {img_path}")

        # ====================================================
        # Checkpoint 저장
        # ====================================================
        ckpt_name = (
            f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.pth"
        )
        ckpt_path = os.path.join(cfg.save_root, ckpt_name)

        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss": epoch_loss,
            "psnr": epoch_psnr,
            "ssim": epoch_ssim,
        }, ckpt_path)
        print(f"[Checkpoint] Saved: {ckpt_path}")

        # Best model (by SSIM)
        if epoch_ssim > best_ssim:
            best_ssim = epoch_ssim
            best_path = os.path.join(cfg.save_root, "best_phase1_backbone.pth")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": epoch_loss,
                "psnr": epoch_psnr,
                "ssim": epoch_ssim,
            }, best_path)
            print(f"[Best] SSIM Updated → {best_path}")

    print("\n[Training Finished]")
    print("Best SSIM:", best_ssim)


if __name__ == "__main__":
    train_phase1()
