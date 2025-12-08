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


# PNG 읽어오기
# G:/VETNet_pilot/trainers/train_phase1_backbone.py
import os, sys, time, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[DEBUG] Using ROOT:", ROOT)

from datasets.multitask_dataset_cache import MultiTaskDatasetCache
from models.backbone.vetnet_backbone import VETNetBackbone
from torch.amp import autocast, GradScaler

# skimage (PSNR/SSIM용)
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    USE_SKIMAGE = True
except:
    USE_SKIMAGE = False

# MS-SSIM (구조 보존용)
try:
    from pytorch_msssim import ms_ssim
    USE_MSSSIM = True
    print("[Loss] Using MS-SSIM from pytorch_msssim.")
except:
    USE_MSSSIM = False
    print("[Loss WARNING] pytorch_msssim not found. "
          "MS-SSIM term will be disabled. Install with: pip install pytorch-msssim")


# ============================================================
# Config
# ============================================================
class Config:
    cache_root = "G:/VETNet_pilot/preload_cache"

    save_root = "G:/VETNet_pilot/checkpoints/phase1_backbone"
    results_root = "G:/VETNet_pilot/results/phase1_backbone"
    iter_preview_root = "G:/VETNet_pilot/results/phase1_backbone/iter_preview"   # [NEW]

    epochs = 100
    batch_size = 2
    num_workers = 0
    lr = 3e-4

    in_channels = 3
    out_channels = 3
    dim = 64
    num_blocks = (4, 6, 6, 8)
    heads = (1, 2, 4, 8)
    volterra_rank = 4
    ffn_expansion_factor = 2.66
    bias = False

    metric_images_per_batch = 2
    use_amp = True

    # epoch 내 iteration preview 확률
    iter_preview_prob = 0.0015     # 평균 1~2장 저장됨 (9577 iter 기준)
    iter_preview_count = 1         # 1 iteration 당 1장만 저장


cfg = Config()


# ============================================================
# Charbonnier Loss
# ============================================================
class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt):
        diff = pred - gt
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


# ============================================================
# Helper
# ============================================================
def tensor_to_img(t):
    t = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255).astype("uint8")


def save_triplet(input, pred, gt, path):
    inp = tensor_to_img(input)
    pr = tensor_to_img(pred)
    gt_img = tensor_to_img(gt)

    H, W, _ = inp.shape
    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)
    canvas[:, 0:W] = inp
    canvas[:, W:2*W] = pr
    canvas[:, 2*W:3*W] = gt_img

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(canvas).save(path)


def compute_psnr_ssim(pred, gt):
    if not USE_SKIMAGE:
        return 0, 0
    p = tensor_to_img(pred[0])
    g = tensor_to_img(gt[0])
    psnr = peak_signal_noise_ratio(g, p, data_range=255)
    ssim = structural_similarity(g, p, channel_axis=2)
    return psnr, ssim


# ============================================================
# Training Loop
# ============================================================
def train_phase1():

    os.makedirs(cfg.save_root, exist_ok=True)
    os.makedirs(cfg.results_root, exist_ok=True)
    os.makedirs(cfg.iter_preview_root, exist_ok=True)   # [NEW]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    dataset = MultiTaskDatasetCache(cfg.cache_root, size=256)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print("[Data] Total cached samples =", len(dataset))

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

    print("[Model Params]", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.use_amp)

    charb_loss = CharbonnierLoss().to(device)

    best_ssim = -1

    # ============================================================
    for epoch in range(1, cfg.epochs + 1):

        model.train()
        loss_sum = psnr_sum = ssim_sum = 0.0
        cnt = 0

        pbar = tqdm(loader, ncols=120, desc=f"Epoch {epoch}")

        for it, batch in enumerate(pbar):
            inp = batch["input"].to(device)
            gt = batch["gt"].to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward
            with autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
                pred = model(inp)
                pred_c = pred.clamp(0, 1)

                l1 = F.l1_loss(pred_c, gt)
                lc = charb_loss(pred_c, gt)

                if USE_MSSSIM:
                    lssim = 1.0 - ms_ssim(pred_c, gt, data_range=1.0)
                    loss = 0.4 * lc + 0.4 * l1 + 0.2 * lssim
                else:
                    loss = 0.5 * lc + 0.5 * l1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            ps, ss = compute_psnr_ssim(pred_c, gt)
            loss_sum += loss.item()
            psnr_sum += ps
            ssim_sum += ss
            cnt += 1

            pbar.set_postfix({
                "L": f"{loss_sum/cnt:.4f}",
                "P": f"{psnr_sum/cnt:.2f}",
                "S": f"{ssim_sum/cnt:.3f}",
            })

            # ============================================================
            # 🔵 [NEW] ITERATION 중 랜덤 PREVIEW 저장
            # ============================================================
            if np.random.rand() < cfg.iter_preview_prob:
                save_path = os.path.join(
                    cfg.iter_preview_root,
                    f"epoch{epoch:03d}_iter{it:05d}.png"
                )
                save_triplet(
                    inp[0].detach().cpu(),
                    pred_c[0].detach().cpu(),
                    gt[0].detach().cpu(),
                    save_path
                )
                # 속도 영향 최소화를 위해 1장만 저장
                # (iter_preview_count 옵션 활용 가능)

        # ============================================================
        # Epoch 종료 후 스케줄러 업데이트
        # ============================================================
        epoch_loss = loss_sum / cnt
        epoch_psnr = psnr_sum / cnt
        epoch_ssim = ssim_sum / cnt

        scheduler.step()

        print(f"\n[Epoch {epoch}] Loss={epoch_loss:.4f}  PSNR={epoch_psnr:.2f}  SSIM={epoch_ssim:.4f}")

        # ---------------- Checkpoint 저장 ----------------
        ckpt = os.path.join(
            cfg.save_root,
            f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            ckpt,
        )

        # 베스트 모델 저장
        if epoch_ssim > best_ssim:
            best_ssim = epoch_ssim
            torch.save(model.state_dict(), os.path.join(cfg.save_root, "best_phase1_backbone.pth"))
            print("[BEST] Updated best SSIM model")

    print("\nTraining Finished.")
    print("Best SSIM:", best_ssim)


if __name__ == "__main__":
    train_phase1()
