# E:\VETNet_Pilot\trainers\train_phase2_gating.py
# ------------------------------------------------------------
# Phase-2 (Gating Controller Training) - FIXED for "real controller"
# - Load Phase-1 backbone checkpoint
# - Freeze backbone completely
# - Train controller only (sample-wise gates)
# - Gates are applied to backbone macro stages (S=8)
# - Dataset: MultiTaskDatasetCache(cache_root, size=crop_size)
# - SPEED: Use random Subset (e.g., 3000)
#
# IMPORTANT:
#   If Phase-2 collapses to g≈1, add "budget + diversity" regularizers.
#   This script includes:
#     L_total = L_rec
#             + λ_budget * (mean(g)-g_target)^2
#             - λ_div    * mean(var_batch(g_stage))
#             + λ_ent    * entropy_loss(optional)
# ------------------------------------------------------------

import os, sys, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# AMP
from torch.cuda.amp import autocast, GradScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.multitask_dataset_cache import MultiTaskDatasetCache
from models.backbone.vetnet_backbone import VETNetBackbone

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    USE_SKIMAGE = True
except:
    USE_SKIMAGE = False


# ============================================================
# Config
# ============================================================
class Config:
    cache_root = "E:/VETNet_Pilot/preload_cache"
    crop_size  = 192

    phase1_ckpt = r"E:\VETNet_Pilot\checkpoints\phase1_backbone\epoch_021_L0.0204_P31.45_S0.9371.pth"

    save_root = "E:/VETNet_Pilot/checkpoints/phase2_gating"
    log_root  = "E:/VETNet_Pilot/results/phase2_gating"
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    epochs = 50
    batch_size = 4
    num_workers = 0
    lr = 2e-4
    weight_decay = 1e-4
    use_amp = True

    subset_size = 3000
    subset_seed = 123

    # gate
    num_stages = VETNetBackbone.NUM_MACRO_STAGES  # usually 8
    g_min = 0.10

    # -------- Regularization to avoid collapse (IMPORTANT) --------
    # gate mean target: lower => forces controller to "choose"
    g_target = 0.80

    # budget: keep average g near g_target (prevents g≈1 collapse)
    lambda_budget = 0.30

    # diversity: maximize batch-wise variance across samples (encourage sample-wise control)
    lambda_div = 0.50

    # entropy: optional (small). encourage non-degenerate stage distribution per sample
    lambda_ent = 0.02

    # stage smoothness: optional (penalize sharp zigzag across stages)
    lambda_smooth = 0.00

    # logging / saving
    save_every_epoch = True


cfg = Config()


# ============================================================
# Utils
# ============================================================
def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def safe_load_state_dict(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
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
    if not USE_SKIMAGE:
        return 0.0, 0.0
    p = tensor_to_img_uint8(pred[0])
    g = tensor_to_img_uint8(gt[0])
    psnr = peak_signal_noise_ratio(g, p, data_range=255)
    ssim = structural_similarity(g, p, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


def gate_stats(g):
    # g: [B,S]
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


def build_subset_dataset(full_dataset, subset_size: int, seed: int):
    n = len(full_dataset)
    if subset_size is None or subset_size <= 0 or subset_size >= n:
        print(f"[Phase2] Using FULL dataset (n={n})")
        return full_dataset
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(n, generator=g)[:subset_size].tolist()
    print(f"[Phase2] Using SUBSET: {subset_size}/{n} (seed={seed})")
    return Subset(full_dataset, indices)


# ============================================================
# Controller (sample-wise, prevents collapse)
# ============================================================
class Phase2GateController(nn.Module):
    """
    A simple image-conditioned controller that outputs per-sample g_stage.
    Output range: [g_min, 1].
    """
    def __init__(self, num_stages: int, g_min: float):
        super().__init__()
        self.num_stages = num_stages
        self.g_min = float(g_min)

        # lightweight encoder -> global avg pool -> MLP
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, num_stages),
        )

        # initialize bias so initial gates are not all ~1
        # sigmoid(0)=0.5 -> g≈ g_min + (1-g_min)*0.5
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        # x: [B,3,H,W]
        h = self.enc(x)                       # [B,256,h,w]
        h = h.mean(dim=(2, 3))                # GAP -> [B,256]
        raw = self.head(h)                    # [B,S]
        g01 = torch.sigmoid(raw)              # [0,1]
        g = self.g_min + (1.0 - self.g_min) * g01
        return g


# ============================================================
# Regularizers
# ============================================================
def budget_loss(g, target_mean: float):
    # keep mean gate around target, prevents collapse to 1
    return (g.mean() - target_mean) ** 2


def diversity_loss(g):
    # maximize batch variance => we minimize negative variance
    # g: [B,S]
    if g.size(0) < 2:
        return g.sum() * 0.0
    var = g.var(dim=0, unbiased=False)  # [S]
    return -var.mean()


def entropy_loss(g, eps=1e-8):
    """
    Treat stage weights as a distribution per sample (normalize across stages),
    and encourage non-degenerate distribution (small weight).
    """
    p = g / (g.sum(dim=1, keepdim=True) + eps)  # [B,S]
    ent = -(p * (p + eps).log()).sum(dim=1).mean()  # mean entropy
    # We *maximize* entropy -> minimize negative entropy
    return -ent


def smoothness_loss(g):
    # encourage neighboring stages to be similar (optional)
    if g.size(1) <= 1:
        return g.sum() * 0.0
    return (g[:, 1:] - g[:, :-1]).abs().mean()


# ============================================================
# Train
# ============================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Phase2] device:", device)

    # 1) Dataset / Loader
    full_dataset = MultiTaskDatasetCache(cfg.cache_root, size=cfg.crop_size)
    print("[CACHE DATASET] Total cached pairs =", len(full_dataset))

    dataset = build_subset_dataset(full_dataset, cfg.subset_size, cfg.subset_seed)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print("[Phase2] Total train samples =", len(dataset))
    print("[Phase2] Steps per epoch =", len(loader))

    # 2) Backbone (Frozen)
    backbone = VETNetBackbone(
        in_channels=3,
        out_channels=3,
        dim=64,
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

    # 3) Controller (Trainable)
    controller = Phase2GateController(num_stages=cfg.num_stages, g_min=cfg.g_min).to(device)
    controller.train()
    print("[Phase2] controller trainable params:", sum(p.numel() for p in controller.parameters()) / 1e6, "M")

    opt = torch.optim.AdamW(controller.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.use_amp)

    log_path = os.path.join(cfg.log_root, "phase2_log.txt")
    print("[Phase2] log file:", log_path)

    # 4) Training Loop
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        controller.train()

        loss_sum = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        cnt = 0

        g_stats_accum = None

        pbar = tqdm(loader, ncols=140, desc=f"Epoch {epoch:03d}/{cfg.epochs}")
        for batch in pbar:
            inp = batch["input"].to(device, non_blocking=True)
            gt  = batch["gt"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.use_amp):
                g_stage = controller(inp)                      # [B,S] sample-wise
                pred = backbone(inp, g_stage=g_stage)          # gated forward (backbone frozen)

                # reconstruction objective (what Phase-2 should optimize)
                L_rec = F.l1_loss(pred, gt)

                # regularizers (prevent trivial g≈1)
                L_budget = budget_loss(g_stage, cfg.g_target) * cfg.lambda_budget
                L_div    = diversity_loss(g_stage) * cfg.lambda_div
                L_ent    = entropy_loss(g_stage) * cfg.lambda_ent
                L_smooth = smoothness_loss(g_stage) * cfg.lambda_smooth

                loss = L_rec + L_budget + L_div + L_ent + L_smooth

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # Metrics
            with torch.no_grad():
                pred_c = pred.clamp(0, 1)
                gt_c   = gt.clamp(0, 1)
                ps, ss = compute_psnr_ssim(pred_c, gt_c)

                loss_sum += float(L_rec.item())  # log rec loss as "main"
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
                    "Lrec": f"{loss_sum/cnt:.4f}",
                    "P": f"{psnr_sum/cnt:.2f}",
                    "S": f"{ssim_sum/cnt:.3f}",
                    "g": f"{st['g_mean']:.3f}",
                    "V": f"{float(torch.tensor(st['g_var_stage']).mean()):.4f}",
                })

        # epoch averages
        epoch_rec  = loss_sum / max(1, cnt)
        epoch_psnr = psnr_sum / max(1, cnt)
        epoch_ssim = ssim_sum / max(1, cnt)

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

        # save ckpt
        ckpt_path = save_controller_ckpt(controller, epoch, epoch_rec, epoch_psnr, epoch_ssim, cfg.save_root)

        # log
        epoch_msg = (
            f"[Epoch {epoch:03d}] "
            f"loss={epoch_rec:.6f} psnr={epoch_psnr:.3f} ssim={epoch_ssim:.6f} | "
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
