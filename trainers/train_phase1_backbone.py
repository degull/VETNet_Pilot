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

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    USE_SKIMAGE = True
except:
    USE_SKIMAGE = False


# ============================================================
class Config:
    cache_root = "G:/VETNet_pilot/preload_cache"

    save_root = "G:/VETNet_pilot/checkpoints/phase1_backbone"
    results_root = "G:/VETNet_pilot/results/phase1_backbone"

    epochs = 100
    batch_size = 2
    num_workers = 0     # PNG 캐시에서는 0이 가장 빠름
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

    # 🔵 새로 추가: 미리보기로 저장할 이미지 수
    preview_count = 3


cfg = Config()


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


# 🔵 랜덤 미리보기 저장 기능
def save_preview_images(inputs, preds, gts, epoch, save_dir, count=3):
    os.makedirs(save_dir, exist_ok=True)

    total = inputs.size(0)
    count = min(count, total)

    # 랜덤 선택
    idxs = np.random.choice(total, count, replace=False)

    for i, idx in enumerate(idxs):
        path = os.path.join(save_dir, f"epoch_{epoch:03d}_preview_{i:02d}.png")
        save_triplet(inputs[idx], preds[idx], gts[idx], path)


def compute_psnr_ssim(pred, gt):
    if not USE_SKIMAGE:
        return 0, 0
    p = tensor_to_img(pred[0])
    g = tensor_to_img(gt[0])
    psnr = peak_signal_noise_ratio(g, p, data_range=255)
    ssim = structural_similarity(g, p, channel_axis=2)
    return psnr, ssim


# ============================================================
def train_phase1():

    os.makedirs(cfg.save_root, exist_ok=True)
    os.makedirs(cfg.results_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    # ============================================================
    dataset = MultiTaskDatasetCache(cfg.cache_root, size=256)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    print("[Data] Total cached samples =", len(dataset))

    # ------------------------------------------------------------
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
    scaler = GradScaler()

    best_ssim = -1

    # ============================================================
    for epoch in range(1, cfg.epochs + 1):

        model.train()
        loss_sum = 0
        psnr_sum = 0
        ssim_sum = 0
        cnt = 0

        pbar = tqdm(loader, ncols=120, desc=f"Epoch {epoch}")

        # 🔵 미리보기 저장용 임시 버퍼
        preview_inp = None
        preview_gt = None
        preview_pred = None

        for batch in pbar:
            inp = batch["input"].to(device)
            gt = batch["gt"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
                pred = model(inp)
                loss = F.l1_loss(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred_c = pred.clamp(0, 1)

            # 🔵 미리보기용 첫 배치를 저장
            if preview_inp is None:
                preview_inp = inp.detach().cpu()
                preview_gt = gt.detach().cpu()
                preview_pred = pred_c.detach().cpu()

            # 평가
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

        epoch_loss = loss_sum / cnt
        epoch_psnr = psnr_sum / cnt
        epoch_ssim = ssim_sum / cnt

        scheduler.step()

        print(f"\n[Epoch {epoch}] Loss={epoch_loss:.4f}  PSNR={epoch_psnr:.2f}  SSIM={epoch_ssim:.4f}")

        # ======================================================
        # 🔵 랜덤 Preview 이미지 저장
        # ======================================================
        save_preview_images(preview_inp, preview_pred, preview_gt,
                            epoch, cfg.results_root, count=cfg.preview_count)

        # ======================================================
        # 원래 epoch 이미지 저장 (첫 1장)
        img_path = os.path.join(
            cfg.results_root,
            f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.png",
        )
        save_triplet(preview_inp[0], preview_pred[0], preview_gt[0], img_path)

        # ======================================================
        # checkpoint 저장
        ckpt_path = os.path.join(
            cfg.save_root,
            f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.pth",
        )
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            ckpt_path,
        )

        if epoch_ssim > best_ssim:
            best_ssim = epoch_ssim
            torch.save(model.state_dict(), os.path.join(cfg.save_root, "best_phase1_backbone.pth"))
            print("[BEST] Updated best SSIM model")

    print("\nTraining Finished.")
    print("Best SSIM:", best_ssim)


if __name__ == "__main__":
    train_phase1()
    