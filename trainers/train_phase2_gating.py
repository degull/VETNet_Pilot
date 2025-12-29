# E:\VETNet_Pilot\trainers\train_phase2_gating.py
# ------------------------------------------------------------
# Phase-2 (Gating Controller Training)
# - Load Phase-1 backbone checkpoint
# - Freeze backbone completely
# - Train GateController only
# - Gates are applied to backbone macro stages (8 stages)
# - Dataset creation is IDENTICAL to Phase-1:
#     dataset = MultiTaskDatasetCache(cache_root, size=256)
#     batch keys: "input", "gt"
# ------------------------------------------------------------

import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# AMP: Phase-1 used torch.cuda.amp; keep consistent & stable
from torch.cuda.amp import autocast, GradScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.multitask_dataset_cache import MultiTaskDatasetCache
from models.backbone.vetnet_backbone import VETNetBackbone
from models.pilot.gate_controller import GateController

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    USE_SKIMAGE = True
except:
    USE_SKIMAGE = False


# ---------------------------
# Config
# ---------------------------
class Config:
    # Phase-1과 동일하게 사용 (분포 고정)
    cache_root = "E:/VETNet_Pilot/preload_cache"
    crop_size = 192  # Phase-1: size=256

    # Phase-1 checkpoint
    phase1_ckpt = r"E:\VETNet_Pilot\checkpoints\phase1_backbone\epoch_021_L0.0204_P31.45_S0.9371.pth"

    # Save / Log
    save_root = "E:/VETNet_Pilot/checkpoints/phase2_gating"
    log_root  = "E:/VETNet_Pilot/results/phase2_gating"
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    # Train
    epochs = 30
    batch_size = 8
    num_workers = 0
    lr = 2e-4
    weight_decay = 1e-4

    use_amp = True

    # IMPORTANT: must match backbone macro stages (=8)
    num_stages = VETNetBackbone.NUM_MACRO_STAGES
    g_min = 0.1


cfg = Config()


# ---------------------------
# Utilities
# ---------------------------
def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def safe_load_state_dict(model: nn.Module, ckpt_path: str):
    """
    Robust checkpoint loader supporting:
      - {'model': state_dict, ...}
      - {'state_dict': state_dict, ...}
      - raw state_dict
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            # maybe dict itself is state_dict
            sd = ckpt
    else:
        sd = ckpt

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[CKPT] Loaded:", ckpt_path)
    print("[CKPT] Missing keys   :", len(missing))
    print("[CKPT] Unexpected keys:", len(unexpected))
    if len(missing) > 0:
        print("  e.g. missing:", missing[:10])
    if len(unexpected) > 0:
        print("  e.g. unexpected:", unexpected[:10])


def tensor_to_img_uint8(t):
    t = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255.0).astype("uint8")


def compute_psnr_ssim(pred, gt):
    """
    Phase-1과 동일한 방식: 배치 첫 장 기준
    """
    if not USE_SKIMAGE:
        return 0.0, 0.0
    p = tensor_to_img_uint8(pred[0])
    g = tensor_to_img_uint8(gt[0])
    psnr = peak_signal_noise_ratio(g, p, data_range=255)
    ssim = structural_similarity(g, p, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


def gate_stats(g):
    """
    g: [B,S]
    """
    return {
        "g_mean": float(g.mean().item()),
        "g_min": float(g.min().item()),
        "g_max": float(g.max().item()),
        "g_var_stage": g.var(dim=0, unbiased=False).detach().cpu().tolist(),
        "g_mean_stage": g.mean(dim=0).detach().cpu().tolist(),
    }


def save_controller_ckpt(controller, epoch, epoch_loss, epoch_psnr, epoch_ssim, save_root):
    ckpt_path = os.path.join(
        save_root,
        f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.pth",
    )
    torch.save(
        {
            "epoch": epoch,
            "controller": controller.state_dict(),
            "config": vars(cfg),
        },
        ckpt_path,
    )
    return ckpt_path


# ---------------------------
# Train
# ---------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Phase2] device:", device)

    # ============================================================
    # 1) Dataset / Loader  (Phase-1과 동일!)
    # ============================================================
    dataset = MultiTaskDatasetCache(cfg.cache_root, size=cfg.crop_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print("[Phase2] Total cached samples =", len(dataset))

    # ============================================================
    # 2) Backbone (Frozen) + Load Phase-1 ckpt
    # ============================================================
    backbone = VETNetBackbone(
        in_channels=3,
        out_channels=3,
        dim=64,                      # ⭐ Phase-1과 동일
        num_blocks=(4, 6, 6, 8),
        heads=(1, 2, 4, 8),
        volterra_rank=4,
        ffn_expansion_factor=2.66,
        bias=False,
    ).to(device)
    safe_load_state_dict(backbone, cfg.phase1_ckpt)
    backbone.eval()
    freeze_module(backbone)
    print("[Phase2] backbone frozen:", all(not p.requires_grad for p in backbone.parameters()))

    # ============================================================
    # 3) Controller (Trainable)
    # ============================================================
    controller = GateController(num_stages=cfg.num_stages, g_min=cfg.g_min).to(device)
    controller.train()
    print("[Phase2] controller trainable params:", sum(p.numel() for p in controller.parameters()) / 1e6, "M")

    opt = torch.optim.AdamW(controller.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.use_amp)

    log_path = os.path.join(cfg.log_root, "phase2_log.txt")
    print("[Phase2] log file:", log_path)

    # ============================================================
    # 4) Training Loop
    # ============================================================
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        controller.train()

        loss_sum = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        cnt = 0

        g_stats_accum = None

        pbar = tqdm(loader, ncols=120, desc=f"Epoch {epoch:03d}/{cfg.epochs}")

        for batch in pbar:
            # Phase-1과 동일한 키 사용
            inp = batch["input"].to(device, non_blocking=True)
            gt  = batch["gt"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.use_amp):
                g_stage = controller(inp)                   # [B,8]
                pred = backbone(inp, g_stage=g_stage)       # gated forward
                loss = F.l1_loss(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # Metrics (Phase-1과 동일: 첫 장 기준)
            with torch.no_grad():
                pred_c = pred.clamp(0, 1)
                gt_c = gt.clamp(0, 1)

                ps, ss = compute_psnr_ssim(pred_c, gt_c)

                loss_sum += float(loss.item())
                psnr_sum += float(ps)
                ssim_sum += float(ss)
                cnt += 1

                st = gate_stats(g_stage)
                if g_stats_accum is None:
                    g_stats_accum = st
                else:
                    g_stats_accum["g_mean"] += st["g_mean"]
                    g_stats_accum["g_min"] = min(g_stats_accum["g_min"], st["g_min"])
                    g_stats_accum["g_max"] = max(g_stats_accum["g_max"], st["g_max"])
                    g_stats_accum["g_var_stage"] = [
                        a + b for a, b in zip(g_stats_accum["g_var_stage"], st["g_var_stage"])
                    ]
                    g_stats_accum["g_mean_stage"] = [
                        a + b for a, b in zip(g_stats_accum["g_mean_stage"], st["g_mean_stage"])
                    ]

                pbar.set_postfix({
                    "L": f"{loss_sum/cnt:.4f}",
                    "P": f"{psnr_sum/cnt:.2f}",
                    "S": f"{ssim_sum/cnt:.3f}",
                    "g": f"{st['g_mean']:.3f}",
                })

        # epoch averages
        epoch_loss = loss_sum / max(1, cnt)
        epoch_psnr = psnr_sum / max(1, cnt)
        epoch_ssim = ssim_sum / max(1, cnt)

        # finalize gate stats (average over steps)
        steps = max(1, cnt)
        if g_stats_accum is None:
            g_stats_accum = {
                "g_mean": 0.0, "g_min": 0.0, "g_max": 0.0,
                "g_var_stage": [0.0] * cfg.num_stages,
                "g_mean_stage": [0.0] * cfg.num_stages,
            }
        else:
            g_stats_accum["g_mean"] /= steps
            g_stats_accum["g_var_stage"] = [v / steps for v in g_stats_accum["g_var_stage"]]
            g_stats_accum["g_mean_stage"] = [v / steps for v in g_stats_accum["g_mean_stage"]]

        # save
        ckpt_path = save_controller_ckpt(controller, epoch, epoch_loss, epoch_psnr, epoch_ssim, cfg.save_root)

        # log
        epoch_msg = (
            f"[Epoch {epoch:03d}] "
            f"loss={epoch_loss:.6f} psnr={epoch_psnr:.3f} ssim={epoch_ssim:.6f} | "
            f"g_mean={g_stats_accum['g_mean']:.4f} g_min={g_stats_accum['g_min']:.4f} g_max={g_stats_accum['g_max']:.4f} | "
            f"g_mean_stage={[round(v,4) for v in g_stats_accum['g_mean_stage']]} | "
            f"g_var_stage={[round(v,6) for v in g_stats_accum['g_var_stage']]} | "
            f"time={time.time()-t0:.1f}s | saved={ckpt_path}"
        )
        print("\n" + epoch_msg)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(epoch_msg + "\n")

    print("\n[Phase2] Training completed.")


if __name__ == "__main__":
    train()
