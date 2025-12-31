# E:\VETNet_Pilot\trainers\train_phase2_gating.py
# ------------------------------------------------------------
# Phase-2 (MoE Gating Controller Training) + LOW-RES PROXY LOSS
# - Load Phase-1 backbone checkpoint
# - Freeze backbone completely
# - Train MoE-GateController only (K experts + routing + load balancing)
# - Gates are applied to backbone macro stages (8 stages)
# - Dataset creation is IDENTICAL to Phase-1:
#     dataset = MultiTaskDatasetCache(cache_root, size=256)
#     batch keys: "input", "gt"
# - Speed-up:
#     inp/gt are downsampled for proxy reconstruction loss
# ------------------------------------------------------------

import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# AMP (new API to avoid warnings)
from torch.amp import autocast, GradScaler

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


# ---------------------------
# Config
# ---------------------------
class Config:
    # Phase-1과 동일하게 사용 (분포 고정)
    cache_root = "E:/VETNet_Pilot/preload_cache"
    crop_size = 256  # Phase-1: size=256

    # Phase-1 checkpoint
    phase1_ckpt = r"E:\VETNet_Pilot\checkpoints\phase1_backbone\epoch_021_L0.0204_P31.45_S0.9371.pth"

    # Save / Log
    save_root = "E:/VETNet_Pilot/checkpoints/phase2_gating_1231_moe_proxy"
    log_root  = "E:/VETNet_Pilot/results/phase2_gating_1231_moe_proxy"
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    # Train
    epochs = 100
    batch_size = 12     # gate-only라 키워도 됨 (원하면 4로 되돌려도 OK)
    num_workers = 4
    lr = 2e-4
    weight_decay = 1e-4
    use_amp = True

    # IMPORTANT: must match backbone macro stages (=8)
    num_stages = VETNetBackbone.NUM_MACRO_STAGES

    # Gate range
    g_min = 0.1

    # MoE gate
    num_experts = 2
    router_temp = 1.0
    lb_weight = 0.01
    ent_weight = 0.0   # 기본 0 추천

    # Proxy (speed)
    down_scale = 0.5   # 핵심: 0.5면 2~3x 빨라짐

    # Metrics (speed)
    metric_every = 50  # PSNR/SSIM은 너무 자주하면 느려짐 (50~200 추천)

    # Stats / debug
    print_router_every = 1


cfg = Config()


# ---------------------------
# Utilities
# ---------------------------
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
    # pred/gt: [B,3,H,W] in [0,1]
    if not USE_SKIMAGE:
        return 0.0, 0.0
    p = tensor_to_img_uint8(pred[0])
    g = tensor_to_img_uint8(gt[0])
    psnr = peak_signal_noise_ratio(g, p, data_range=255)
    ssim = structural_similarity(g, p, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


def save_controller_ckpt(controller, epoch, epoch_loss, epoch_psnr, epoch_ssim, save_root, extra=None):
    ckpt_path = os.path.join(
        save_root,
        f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.pth",
    )
    payload = {
        "epoch": epoch,
        "controller": controller.state_dict(),
        "config": vars(cfg),
    }
    if extra is not None and isinstance(extra, dict):
        payload.update(extra)

    torch.save(payload, ckpt_path)
    return ckpt_path


# ---------------------------
# MoE Gate Controller
# ---------------------------
class MoEGateController(nn.Module):
    """
    Produces per-image stage gates via K experts + routing.
    - experts: K heads produce gate vectors (B, K, S)
    - router : produces routing probs (B, K)
    - output : weighted gate g_stage = sum_k p_k * g_k  -> (B, S)
    """

    def __init__(self, num_stages: int, g_min: float = 0.1, num_experts: int = 2, router_temp: float = 1.0):
        super().__init__()
        self.num_stages = int(num_stages)
        self.g_min = float(g_min)
        self.num_experts = int(num_experts)
        self.router_temp = float(router_temp)

        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(inplace=True),   # 128
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),  # 64
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True), # 32
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )

        self.expert_heads = nn.ModuleList([nn.Linear(256, self.num_stages) for _ in range(self.num_experts)])
        self.router = nn.Linear(256, self.num_experts)

        # init: near-open gates
        for head in self.expert_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, 2.0)

        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)  # uniform at start

    def _squash_gate(self, x):
        g01 = torch.sigmoid(x)
        return self.g_min + (1.0 - self.g_min) * g01

    def forward(self, inp):
        b = inp.shape[0]
        z = self.enc(inp).view(b, 128)
        h = self.fc(z)  # [B,256]

        gates = []
        for k in range(self.num_experts):
            gates.append(self._squash_gate(self.expert_heads[k](h)))  # [B,S]
        g_experts = torch.stack(gates, dim=1)  # [B,K,S]

        logits = self.router(h)  # [B,K]
        probs = F.softmax(logits / max(self.router_temp, 1e-6), dim=1)  # [B,K]

        g_stage = torch.sum(probs.unsqueeze(-1) * g_experts, dim=1)  # [B,S]
        return g_stage, probs, g_experts


def load_balance_loss(probs: torch.Tensor):
    # probs: [B,K]
    p = probs.mean(dim=0)  # [K]
    k = probs.shape[1]
    target = torch.full_like(p, 1.0 / float(k))
    return F.mse_loss(p, target)


def routing_entropy_loss(probs: torch.Tensor):
    eps = 1e-8
    ent = -(probs * (probs + eps).log()).sum(dim=1).mean()
    return -ent  # minimize => encourage higher entropy


def gate_stats(g_stage: torch.Tensor):
    return {
        "g_mean": float(g_stage.mean().item()),
        "g_min": float(g_stage.min().item()),
        "g_max": float(g_stage.max().item()),
        "g_var_stage": g_stage.var(dim=0, unbiased=False).detach().cpu().tolist(),
        "g_mean_stage": g_stage.mean(dim=0).detach().cpu().tolist(),
    }


def routing_stats(probs: torch.Tensor):
    p_mean = probs.mean(dim=0).detach().cpu().tolist()
    p_max = probs.max(dim=1).values.mean().item()
    return {
        "route_mean": p_mean,
        "route_peak_mean": float(p_max),
    }


# ---------------------------
# Train
# ---------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Phase2-MoE-Proxy] device:", device)

    # 1) Dataset / Loader
    dataset = MultiTaskDatasetCache(cfg.cache_root, size=cfg.crop_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print("[Phase2-MoE-Proxy] Total cached samples =", len(dataset))

    # 2) Backbone (Frozen) + Load Phase-1 ckpt
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
    print("[Phase2-MoE-Proxy] backbone frozen:", all(not p.requires_grad for p in backbone.parameters()))

    # 3) MoE Controller
    controller = MoEGateController(
        num_stages=cfg.num_stages,
        g_min=cfg.g_min,
        num_experts=cfg.num_experts,
        router_temp=cfg.router_temp,
    ).to(device)
    controller.train()
    print("[Phase2-MoE-Proxy] controller trainable params:", sum(p.numel() for p in controller.parameters()) / 1e6, "M")

    opt = torch.optim.AdamW(controller.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler("cuda", enabled=cfg.use_amp)

    log_path = os.path.join(cfg.log_root, "phase2_moe_proxy_log.txt")
    print("[Phase2-MoE-Proxy] log file:", log_path)

    # Running metrics (so P/S always show in pbar)
    psnr_meas_sum = 0.0
    ssim_meas_sum = 0.0
    meas_cnt = 0

    # 4) Loop
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        controller.train()

        loss_sum = 0.0
        rec_sum  = 0.0
        lb_sum   = 0.0
        ent_sum  = 0.0
        cnt = 0

        g_stats_accum = None
        r_stats_accum = None

        pbar = tqdm(loader, ncols=140, desc=f"Epoch {epoch:03d}/{cfg.epochs}")

        for it, batch in enumerate(pbar):
            inp = batch["input"].to(device, non_blocking=True)
            gt  = batch["gt"].to(device, non_blocking=True)

            # ✅ LOW-RES PROXY (speed)
            if cfg.down_scale != 1.0:
                inp_ds = F.interpolate(inp, scale_factor=cfg.down_scale, mode="bilinear", align_corners=False)
                gt_ds  = F.interpolate(gt,  scale_factor=cfg.down_scale, mode="bilinear", align_corners=False)
            else:
                inp_ds, gt_ds = inp, gt

            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=cfg.use_amp):
                g_stage, probs, _gexp = controller(inp)         # gate는 원본 inp로 예측
                pred = backbone(inp_ds, g_stage=g_stage)        # 복원은 다운샘플로 proxy
                rec_loss = F.l1_loss(pred, gt_ds)

                lb = load_balance_loss(probs)
                ent = routing_entropy_loss(probs) if cfg.ent_weight > 0 else (rec_loss * 0.0)

                loss = rec_loss + cfg.lb_weight * lb + cfg.ent_weight * ent

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # stats
            loss_sum += float(loss.item())
            rec_sum  += float(rec_loss.item())
            lb_sum   += float(lb.item())
            ent_sum  += float(ent.item()) if cfg.ent_weight > 0 else 0.0
            cnt += 1

            # gate/routing current
            with torch.no_grad():
                st = gate_stats(g_stage)
                rt = routing_stats(probs)

                # sparse PSNR/SSIM (but keep showing averages in pbar)
                if (it % max(1, cfg.metric_every)) == 0:
                    pred_c = pred.clamp(0, 1)
                    gt_c = gt_ds.clamp(0, 1)
                    ps, ss = compute_psnr_ssim(pred_c, gt_c)
                    psnr_meas_sum += float(ps)
                    ssim_meas_sum += float(ss)
                    meas_cnt += 1

                psnr_avg_show = psnr_meas_sum / max(1, meas_cnt)
                ssim_avg_show = ssim_meas_sum / max(1, meas_cnt)

                # accumulate epoch-level gate stats (optional)
                if g_stats_accum is None:
                    g_stats_accum = st
                else:
                    g_stats_accum["g_mean"] += st["g_mean"]
                    g_stats_accum["g_min"] = min(g_stats_accum["g_min"], st["g_min"])
                    g_stats_accum["g_max"] = max(g_stats_accum["g_max"], st["g_max"])
                    g_stats_accum["g_var_stage"] = [a + b for a, b in zip(g_stats_accum["g_var_stage"], st["g_var_stage"])]
                    g_stats_accum["g_mean_stage"] = [a + b for a, b in zip(g_stats_accum["g_mean_stage"], st["g_mean_stage"])]

                if r_stats_accum is None:
                    r_stats_accum = rt
                else:
                    r_stats_accum["route_mean"] = [a + b for a, b in zip(r_stats_accum["route_mean"], rt["route_mean"])]
                    r_stats_accum["route_peak_mean"] += rt["route_peak_mean"]

                # ✅ 너가 원한 포맷 그대로 다시 출력
                pbar.set_postfix({
                    "L":   f"{loss_sum/cnt:.4f}",
                    "Rec": f"{rec_sum/cnt:.4f}",
                    "LB":  f"{lb_sum/cnt:.4f}",
                    "P":   f"{psnr_avg_show:.2f}",
                    "S":   f"{ssim_avg_show:.3f}",
                    "g":   f"{st['g_mean']:.3f}",
                    "rp":  f"{rt['route_peak_mean']:.2f}",
                })

        # epoch averages
        steps = max(1, cnt)
        epoch_loss = loss_sum / steps
        epoch_rec  = rec_sum / steps
        epoch_lb   = lb_sum / steps
        epoch_ent  = ent_sum / steps

        # epoch psnr/ssim shown as measured-average (sparse)
        epoch_psnr = psnr_meas_sum / max(1, meas_cnt)
        epoch_ssim = ssim_meas_sum / max(1, meas_cnt)

        # finalize routing mean for saving/log
        if r_stats_accum is None:
            route_mean = [0.0] * cfg.num_experts
            route_peak = 0.0
        else:
            route_mean = [v / steps for v in r_stats_accum["route_mean"]]
            route_peak = float(r_stats_accum["route_peak_mean"] / steps)

        ckpt_path = save_controller_ckpt(
            controller, epoch, epoch_loss, epoch_psnr, epoch_ssim, cfg.save_root,
            extra={
                "epoch_rec_loss": epoch_rec,
                "epoch_lb_loss": epoch_lb,
                "epoch_ent_loss": epoch_ent,
                "routing_mean": route_mean,
                "routing_peak_mean": route_peak,
                "proxy_down_scale": cfg.down_scale,
                "metric_every": cfg.metric_every,
            }
        )

        epoch_msg = (
            f"[Epoch {epoch:03d}] "
            f"loss={epoch_loss:.6f} rec={epoch_rec:.6f} lb={epoch_lb:.6f} "
            f"psnr={epoch_psnr:.3f} ssim={epoch_ssim:.6f} | "
            f"route_mean={[round(v,4) for v in route_mean]} route_peak_mean={route_peak:.4f} | "
            f"proxy_scale={cfg.down_scale} metric_every={cfg.metric_every} | "
            f"time={time.time()-t0:.1f}s | saved={ckpt_path}"
        )
        print("\n" + epoch_msg)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(epoch_msg + "\n")

        if (epoch % max(1, cfg.print_router_every)) == 0:
            print(f"[Phase2-MoE-Proxy] routing usage (mean probs): {route_mean}")

    print("\n[Phase2-MoE-Proxy] Training completed.")


if __name__ == "__main__":
    train()
