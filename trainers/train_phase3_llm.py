""" # G:\VETNet_pilot\trainers\train_phase3_llm.py
import os
import sys
import time
import glob
import random
import numpy as np
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[train_phase3_llm] ROOT =", ROOT)

# ========= imports from your project =========
from models.backbone.vetnet_backbone import VETNetBackbone  # must exist in your project

from models.pilot.vision_adapter import CLIPVisionAdapter
from models.pilot.llm_loader import load_llm_with_lora
from models.pilot.strategy_head import StrategyHead
from models.pilot.tokenizer_utils import (
    tokenize_prompt,
    ensure_pad_token,
    build_strategy_prompt,
    build_xai_prompt,
)

try:
    from skimage.metrics import structural_similarity as ssim_fn
    SKIMAGE_OK = True
except:
    SKIMAGE_OK = False


# ============================================================
# Dataset: preload_cache/* with *_in.png and *_gt.png
# ============================================================
class PreloadCachePairedDataset(Dataset):
    def __init__(self, root_dir, folders, crop_size=256, training=True):
        self.root_dir = root_dir
        self.folders = folders
        self.crop_size = crop_size
        self.training = training

        self.pairs = []
        for folder in folders:
            fdir = os.path.join(root_dir, folder)
            if not os.path.isdir(fdir):
                print(f"[Dataset] WARNING: folder not found: {fdir}")
                continue

            in_list = sorted(glob.glob(os.path.join(fdir, "*_in.png")))
            for in_path in in_list:
                gt_path = in_path.replace("_in.png", "_gt.png")
                if os.path.exists(gt_path):
                    self.pairs.append((in_path, gt_path))

        print(f"[Dataset] Found {len(self.pairs)} pairs from {folders}")
        if len(self.pairs) == 0:
            raise RuntimeError(f"No pairs found under {root_dir} for {folders}")

    def __len__(self):
        return len(self.pairs)

    def _load_img(self, path):
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t

    def _random_crop(self, inp, gt):
        _, H, W = inp.shape
        cs = self.crop_size
        if H < cs or W < cs:
            pad_h = max(0, cs - H)
            pad_w = max(0, cs - W)
            inp = F.pad(inp, (0, pad_w, 0, pad_h), mode="reflect")
            gt = F.pad(gt, (0, pad_w, 0, pad_h), mode="reflect")
            _, H, W = inp.shape

        top = random.randint(0, H - cs)
        left = random.randint(0, W - cs)
        inp_c = inp[:, top:top + cs, left:left + cs]
        gt_c = gt[:, top:top + cs, left:left + cs]
        return inp_c, gt_c

    def _augment(self, inp, gt):
        if random.random() < 0.5:
            inp = torch.flip(inp, dims=[2])
            gt = torch.flip(gt, dims=[2])
        if random.random() < 0.5:
            inp = torch.flip(inp, dims=[1])
            gt = torch.flip(gt, dims=[1])
        k = random.randint(0, 3)
        if k > 0:
            inp = torch.rot90(inp, k, dims=[1, 2])
            gt = torch.rot90(gt, k, dims=[1, 2])
        return inp, gt

    def __getitem__(self, idx):
        in_path, gt_path = self.pairs[idx]
        inp = self._load_img(in_path)
        gt = self._load_img(gt_path)

        if self.training:
            inp, gt = self._random_crop(inp, gt)
            inp, gt = self._augment(inp, gt)

        return {"inp": inp, "gt": gt, "in_path": in_path, "gt_path": gt_path}


# ============================================================
# Metrics
# ============================================================
def calc_psnr(pred, gt, eps=1e-8):
    mse = torch.mean((pred - gt) ** 2, dim=[1, 2, 3]) + eps
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr.mean().item()


def calc_ssim(pred, gt):
    if not SKIMAGE_OK:
        l1 = torch.mean(torch.abs(pred - gt)).item()
        return max(0.0, 1.0 - l1)

    # ✅ bfloat16 -> float (numpy unsupported for bfloat16)
    pred_np = pred.detach().float().clamp(0, 1).cpu().numpy()
    gt_np = gt.detach().float().clamp(0, 1).cpu().numpy()

    B = pred_np.shape[0]
    vals = []
    for i in range(B):
        p = np.transpose(pred_np[i], (1, 2, 0))
        g = np.transpose(gt_np[i], (1, 2, 0))
        vals.append(ssim_fn(p, g, channel_axis=2, data_range=1.0))
    return float(np.mean(vals))


# ============================================================
# ✅ Visualization utils: save (inp|pred|gt) + XAI text every N iters
# ============================================================
def _tensor_to_pil(x_chw: torch.Tensor) -> Image.Image:
    x = x_chw.detach().float().clamp(0, 1).cpu()
    x = (x * 255.0 + 0.5).to(torch.uint8)
    x = x.permute(1, 2, 0).numpy()  # HWC
    return Image.fromarray(x, mode="RGB")


def _draw_label(draw: ImageDraw.ImageDraw, xy, text, font):
    # subtle outline for readability
    x, y = xy
    for ox, oy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
        draw.text((x+ox, y+oy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))


def save_triplet_with_xai(
    save_path: str,
    inp_img: Image.Image,
    pred_img: Image.Image,
    gt_img: Image.Image,
    xai_text: str,
    meta_text: str = "",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # fonts (default is safest)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    # layout
    pad = 12
    gap = 10
    label_h = 18
    img_w, img_h = inp_img.size

    # xai area
    xai_text = (xai_text or "").strip()
    if len(xai_text) == 0:
        xai_text = "(XAI: empty)"

    # wrap text
    wrap_width = 110  # tuned for 3x256 layout; safe for default font
    wrapped = textwrap.fill(xai_text, width=wrap_width)

    # estimate xai height
    lines = wrapped.count("\n") + 1
    line_h = 14
    xai_h = pad + lines * line_h + pad
    if meta_text:
        xai_h += line_h + 6

    out_w = pad + img_w + gap + img_w + gap + img_w + pad
    out_h = pad + label_h + 4 + img_h + pad + xai_h + pad

    canvas = Image.new("RGB", (out_w, out_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    # paste images
    x0 = pad
    y0 = pad + label_h + 4
    canvas.paste(inp_img, (x0, y0))
    canvas.paste(pred_img, (x0 + img_w + gap, y0))
    canvas.paste(gt_img, (x0 + (img_w + gap) * 2, y0))

    # labels
    _draw_label(draw, (x0, pad), "INPUT", font)
    _draw_label(draw, (x0 + img_w + gap, pad), "RESTORED", font)
    _draw_label(draw, (x0 + (img_w + gap) * 2, pad), "GT", font)

    # xai box
    box_y = y0 + img_h + pad
    box_x = pad
    box_w = out_w - 2 * pad
    box_h = out_h - box_y - pad
    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(245, 245, 245))

    tx = box_x + pad
    ty = box_y + pad

    if meta_text:
        draw.text((tx, ty), meta_text, font=font, fill=(0, 0, 0))
        ty += line_h + 6

    draw.text((tx, ty), wrapped, font=font, fill=(0, 0, 0))

    canvas.save(save_path)


# ============================================================
# Phase-3 Controller (Dual-head: strategy path + XAI text path)
# ============================================================
class Phase3Controller(nn.Module):


    def __init__(
        self,
        vetnet: nn.Module,
        clip_adapter: nn.Module,
        llm_model: nn.Module,
        llm_tokenizer,
        strategy_head: nn.Module,
        strategy_prompt: str,
        xai_prompt: str,
        device: torch.device,
        prompt_max_len: int = 128,
        prefix_tokens: int = 8,
    ):
        super().__init__()
        self.vetnet = vetnet
        self.clip_adapter = clip_adapter
        self.llm = llm_model
        self.tok = llm_tokenizer
        self.strategy_head = strategy_head

        self.strategy_prompt = strategy_prompt
        self.xai_prompt = xai_prompt

        self.device = device
        self.prompt_max_len = prompt_max_len
        self.prefix_tokens = prefix_tokens

        self.lm_dim = self.llm.config.hidden_size
        self.clip_to_prefix = None

        ensure_pad_token(self.tok)

        # ============================================================
        # ✅ stage_proj 정의 (없어서 AttributeError 났던 부분)
        # ============================================================
        base_C = 256
        self.stage_proj = nn.ModuleDict({
            "stage1": nn.Linear(base_C, 64, bias=True),
            "stage2": nn.Linear(base_C, 128, bias=True),
            "stage3": nn.Linear(base_C, 256, bias=True),
        }).to(self.device)

    def _build_clip_to_prefix(self, clip_dim: int):
        if self.clip_to_prefix is None:
            self.clip_to_prefix = nn.Sequential(
                nn.LayerNorm(clip_dim),
                nn.Linear(clip_dim, self.prefix_tokens * self.lm_dim),
            ).to(self.device)

    def _make_prefix(self, inp: torch.Tensor):
        B = inp.size(0)
        with torch.no_grad():
            v_pool, _ = self.clip_adapter(inp)  # (B, D_clip)
        self._build_clip_to_prefix(v_pool.shape[-1])
        prefix = self.clip_to_prefix(v_pool).view(B, self.prefix_tokens, self.lm_dim)
        return prefix

    def _prepare_inputs_embeds(self, prefix_embeds: torch.Tensor, prompt: str):
        B = prefix_embeds.size(0)

        tok = tokenize_prompt(self.tok, prompt, device=self.device, max_length=self.prompt_max_len)
        input_ids = tok["input_ids"]
        attn_mask = tok["attention_mask"]

        prompt_embeds = self.llm.get_input_embeddings()(input_ids)  # (1,T,D)
        # 🔑 batch size 맞추기
        if prompt_embeds.size(0) == 1 and prefix_embeds.size(0) > 1:
            prompt_embeds = prompt_embeds.expand(prefix_embeds.size(0), -1, -1)

        inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

        # 🔑 attention mask도 batch size 맞추기
        if attn_mask.size(0) == 1 and B > 1:
            attn_mask = attn_mask.expand(B, -1)

        prefix_mask = torch.ones((B, self.prefix_tokens), device=self.device, dtype=attn_mask.dtype)
        attn_mask2 = torch.cat([prefix_mask, attn_mask], dim=1)

        return inputs_embeds, attn_mask2

    def forward(self, inp: torch.Tensor):
        prefix = self._make_prefix(inp)  # (B,P,lm_dim)
        inputs_embeds, attn_mask2 = self._prepare_inputs_embeds(prefix, self.strategy_prompt)

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask2,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = out.hidden_states[-1]  # (B,P+T,lm_dim)

        # pooled over prefix positions -> stable continuous regressor behavior
        prefix_hidden = last_hidden[:, :self.prefix_tokens, :]
        pooled = prefix_hidden.mean(dim=1)  # (B,lm_dim)

        S_raw, z = self.strategy_head(pooled)  # (B,K,256)

        # ✅ VETNet stage별 channel 요구에 맞춰 projection 후 dict로 전달
        strategy_tokens = {
            "stage1": self.stage_proj["stage1"](S_raw),  # (B,K,64)
            "stage2": self.stage_proj["stage2"](S_raw),  # (B,K,128)
            "stage3": self.stage_proj["stage3"](S_raw),  # (B,K,256)
        }

        pred = self.vetnet(inp, strategy_tokens=strategy_tokens)
        return pred, S_raw, z

    @torch.no_grad()
    def generate_xai(self, inp: torch.Tensor, max_new_tokens: int = 96, do_sample: bool = False):
        self.llm.eval()

        prefix = self._make_prefix(inp)
        inputs_embeds, attn_mask2 = self._prepare_inputs_embeds(prefix, self.xai_prompt)

        gen_out = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask2,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
        )

        # ✅ "입력 길이"만큼 자르고 생성된 부분만 디코딩
        seq = gen_out.sequences  # (B, seq_len_total)
        input_lens = attn_mask2.sum(dim=1).to(torch.long)  # (B,)

        texts = []
        for b in range(seq.size(0)):
            gen_ids = seq[b, input_lens[b]:]  # 생성된 토큰만
            texts.append(self.tok.decode(gen_ids, skip_special_tokens=True).strip())
        return texts


# ============================================================
# Config
# ============================================================
class Config:
    # Data
    cache_root = "E:/VETNet_pilot/preload_cache"
    folders = ["CSD", "DayRainDrop", "NightRainDrop", "rain100H", "RESIDE-6K"]
    crop_size = 256

    # Training
    epochs = 20
    batch_size = 2
    num_workers = 0
    lr = 1e-4
    weight_decay = 0.0

    # Models
    clip_name = "openai/clip-vit-large-patch14"
    llm_name = "microsoft/Phi-3-mini-4k-instruct"
    llm_4bit = True

    # Strategy token spec
    K = 3
    C = 256
    strategy_dim = 512
    prefix_tokens = 8

    # Prompt hints
    task_hint = "rain removal / raindrop / haze removal; preserve structure and texture."

    # XAI logging control
    enable_xai = True
    xai_every_epochs = 1
    xai_max_new_tokens = 96
    xai_do_sample = False

    # Save / log
    save_root = "E:/VETNet_pilot/checkpoints/phase3_llm"
    results_root = "E:/VETNet_pilot/results/phase3_llm"
    log_every = 20

    # ✅ iteration마다 결과 저장
    save_vis_every_iters = 100  # <==== 요청: iteration 100마다 저장


# ============================================================
# Training
# ============================================================
def train():
    cfg = Config()
    os.makedirs(cfg.save_root, exist_ok=True)
    os.makedirs(cfg.results_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # Dataset
    ds = PreloadCachePairedDataset(
        root_dir=cfg.cache_root,
        folders=cfg.folders,
        crop_size=cfg.crop_size,
        training=True,
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    # VETNet backbone (freeze)
    vetnet = VETNetBackbone().to(device)
    vetnet.eval()
    for p in vetnet.parameters():
        p.requires_grad = False
    print("[Backbone] Loaded & Frozen: VETNetBackbone")

    # CLIP adapter (freeze)
    clip_adapter = CLIPVisionAdapter(model_name=cfg.clip_name, device=str(device), use_fp16=False)
    print("[Vision] CLIP loaded & frozen:", cfg.clip_name)

    # LLM + LoRA (trainable)
    llm, tok = load_llm_with_lora(
        model_name=cfg.llm_name,
        device=str(device),
        load_in_4bit=cfg.llm_4bit,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    tok = ensure_pad_token(tok)
    lm_dim = llm.config.hidden_size
    print("[LLM] hidden_size =", lm_dim)

    # Strategy head (trainable)
    strategy_head = StrategyHead(
        lm_dim=lm_dim,
        strategy_dim=cfg.strategy_dim,
        K=cfg.K,
        C=cfg.C,
        dropout=0.1,
    ).to(device)
    print(f"[StrategyHead] K={cfg.K}, C={cfg.C}, strategy_dim={cfg.strategy_dim}")

    strategy_prompt = build_strategy_prompt(cfg.task_hint)
    xai_prompt = build_xai_prompt(cfg.task_hint)
    print("[Prompt:Strategy]", strategy_prompt)
    print("[Prompt:XAI     ]", xai_prompt)

    # Controller
    model = Phase3Controller(
        vetnet=vetnet,
        clip_adapter=clip_adapter,
        llm_model=llm,
        llm_tokenizer=tok,
        strategy_head=strategy_head,
        strategy_prompt=strategy_prompt,
        xai_prompt=xai_prompt,
        device=device,
        prompt_max_len=128,
        prefix_tokens=cfg.prefix_tokens,
    ).to(device)

    # Train params
    train_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in train_params)
    print("[Trainable] total params =", f"{total_trainable:,}")

    opt = torch.optim.AdamW(train_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    l1 = nn.L1Loss()

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        avg_loss = 0.0
        avg_psnr = 0.0
        avg_ssim = 0.0

        pbar = tqdm(dl, desc=f"Epoch {epoch:03d}/{cfg.epochs}")
        did_xai_this_epoch = False

        for it, batch in enumerate(pbar, 1):
            inp = batch["inp"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                pred, S, z = model(inp)
                pred = pred.clamp(0, 1)
                loss = l1(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                psnr = calc_psnr(pred, gt)
                ssim = calc_ssim(pred, gt)

            avg_loss += loss.item()
            avg_psnr += psnr
            avg_ssim += ssim
            global_step += 1

            # progress log
            if it % cfg.log_every == 0 or it == 1:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "psnr": f"{psnr:.2f}",
                    "ssim": f"{ssim:.4f}",
                    "S_mean": f"{S.detach().mean().item():.4f}",
                    "z_mean": f"{z.detach().mean().item():.4f}",
                })

            # ============================================================
            # ✅ iteration 100마다 (원본|복원|gt) + XAI 텍스트를 한 장으로 저장
            # ============================================================
            if (global_step % cfg.save_vis_every_iters) == 0:
                model.eval()
                try:
                    # 1 sample only (속도/메모리)
                    inp1 = inp[:1]
                    gt1 = gt[:1]
                    pred1 = pred[:1]

                    xai_text = ""
                    if cfg.enable_xai:
                        xai_texts = model.generate_xai(
                            inp1,
                            max_new_tokens=cfg.xai_max_new_tokens,
                            do_sample=cfg.xai_do_sample,
                        )
                        xai_text = xai_texts[0] if len(xai_texts) > 0 else ""

                    # make PIL images
                    inp_img = _tensor_to_pil(inp1[0])
                    pred_img = _tensor_to_pil(pred1[0])
                    gt_img = _tensor_to_pil(gt1[0])

                    meta = f"epoch={epoch:03d}  iter={it:05d}  global_step={global_step:07d}  loss={loss.item():.4f}  psnr={psnr:.2f}  ssim={ssim:.4f}"
                    out_name = f"e{epoch:03d}_it{it:05d}_gs{global_step:07d}.png"
                    out_path = os.path.join(cfg.results_root, out_name)

                    save_triplet_with_xai(
                        save_path=out_path,
                        inp_img=inp_img,
                        pred_img=pred_img,
                        gt_img=gt_img,
                        xai_text=xai_text,
                        meta_text=meta,
                    )
                    print(f"\n[Saved:VIS] {out_path}\n")
                except Exception as e:
                    print("\n[VIS] save failed:", repr(e), "\n")
                model.train()

            # XAI generation (optional, no-grad) - do once per epoch on first batch
            if cfg.enable_xai and (not did_xai_this_epoch) and (epoch % cfg.xai_every_epochs == 0):
                model.eval()
                try:
                    xai_texts = model.generate_xai(
                        inp[:1],
                        max_new_tokens=cfg.xai_max_new_tokens,
                        do_sample=cfg.xai_do_sample,
                    )
                    print("\n[XAI] sample explanation:")
                    print(xai_texts[0] if len(xai_texts) > 0 else "(empty)")
                    print("-" * 60)
                except Exception as e:
                    print("\n[XAI] generation failed:", repr(e))
                model.train()
                did_xai_this_epoch = True

        n = len(dl)
        avg_loss /= n
        avg_psnr /= n
        avg_ssim /= n
        dt = time.time() - t0

        print(f"\n[Epoch {epoch:03d}] time={dt:.1f}s  loss={avg_loss:.4f}  psnr={avg_psnr:.2f}  ssim={avg_ssim:.4f}")

        ckpt = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
        }
        ckpt_name = f"phase3_epoch_{epoch:03d}_L{avg_loss:.4f}_P{avg_psnr:.2f}_S{avg_ssim:.4f}.pth"
        ckpt_path = os.path.join(cfg.save_root, ckpt_name)
        torch.save(ckpt, ckpt_path)
        print("[Saved]", ckpt_path)

    print("\n[Done] Phase-3 training complete.")


if __name__ == "__main__":
    # 디버깅용 print 포함
    print("============================================================")
    print("[MAIN] train_phase3_llm.py starting...")
    print("[MAIN] Python:", sys.version)
    print("[MAIN] Torch:", torch.__version__)
    print("[MAIN] CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[MAIN] GPU:", torch.cuda.get_device_name(0))
    print("[MAIN] ROOT:", ROOT)
    print("[MAIN] Expected data root:", Config.cache_root)
    for f in Config.folders:
        print("   -", os.path.join(Config.cache_root, f))
    print("[MAIN] XAI enabled:", Config.enable_xai)
    print("[MAIN] XAI every epochs:", Config.xai_every_epochs)
    print("[MAIN] Results root:", Config.results_root)
    print("[MAIN] Save vis every iters:", Config.save_vis_every_iters)
    print("============================================================")

    train() """

# ============================================================
# Phase-3 Training (CLIP Preload Cache Version)
# - Fresh start (resume OFF)
# - Windows-safe DataLoader (num_workers=0 => persistent_workers OFF)
# ============================================================
# ============================================================
# Phase-3 Training (CLIP cache + LLM Core w/ prompt & prefix)
# Architecture (matches your diagram):
#   Image
#    → CLIP (cached vector v)
#    → LLM (prefix=v + prompt tokens)
#        ├─ generate() → XAI text
#        └─ forward hidden → strategy_head → Z (and S)
#    → projection → strategy tokens S
#    → VETNet
#
# Notes:
# - Uses cached CLIP vectors (_clip.pt) generated by preprocess_clip_cache.py
# - Uses an MLP "clip_to_prefix" to convert v → prefix token embeddings
# - Uses LLM forward() to get hidden for strategy control (no xai_head hack)
# - Uses LLM generate() to produce real XAI sentence (decoded text)
# - Saves [INPUT|RESTORED|GT] + XAI text every N iterations
# - Resume OFF by default (fresh start)
# ============================================================
# ============================================================
# Phase-3 Training (CLIP cache + LLM Core w/ prompt & prefix)
# Architecture:
#   Image → CLIP(cache)
#        → LLM(prefix+prompt)
#            ├─ generate() → XAI text
#            └─ hidden → strategy_head → Z,S
#        → projection → strategy tokens
#        → VETNet
# ============================================================
import os
import sys
import glob
import time
import textwrap
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------
# Path (🔥 최우선)
# ------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[train_phase3_llm] ROOT =", ROOT)

# 🔥 이제부터 models import 가능
import models.pilot.strategy_head as sh
print("[DEBUG] StrategyHead loaded from:", sh.__file__)

from transformers import AutoTokenizer, AutoModelForCausalLM
from models.backbone.vetnet_backbone import VETNetBackbone
from models.pilot.strategy_head import StrategyHead


# ============================================================
# Utils
# ============================================================
DEFAULT_XAI = (
    "The image was restored by removing rain streaks while "
    "preserving structural edges and natural textures."
)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().clamp(0, 1).cpu()
    x = (x * 255.0 + 0.5).to(torch.uint8)
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x, mode="RGB")


def save_triplet_with_xai(inp, pred, gt, xai_text, save_path, meta=""):
    inp_img = tensor_to_pil(inp)
    pred_img = tensor_to_pil(pred)
    gt_img = tensor_to_pil(gt)

    w, h = inp_img.size
    pad = 12
    gap = 10
    label_h = 18

    font = ImageFont.load_default()

    xai_text = (xai_text or "").strip()
    if len(xai_text) == 0:
        xai_text = DEFAULT_XAI  # ✅ 반드시 출력 보장

    wrapped = textwrap.fill(xai_text, width=110)
    line_h = 14
    xai_h = pad + (wrapped.count("\n") + 1) * line_h + pad

    out_w = pad + w * 3 + gap * 2 + pad
    out_h = pad + label_h + h + pad + xai_h + pad

    canvas = Image.new("RGB", (out_w, out_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    y0 = pad + label_h
    canvas.paste(inp_img, (pad, y0))
    canvas.paste(pred_img, (pad + w + gap, y0))
    canvas.paste(gt_img, (pad + 2 * (w + gap), y0))

    draw.text((pad, pad), "INPUT", fill=(255, 255, 255), font=font)
    draw.text((pad + w + gap, pad), "RESTORED", fill=(255, 255, 255), font=font)
    draw.text((pad + 2 * (w + gap), pad), "GT", fill=(255, 255, 255), font=font)

    y1 = y0 + h + pad
    draw.rectangle([pad, y1, out_w - pad, out_h - pad], fill=(245, 245, 245))
    draw.text((pad + 6, y1 + 6), f"XAI: {wrapped}", fill=(0, 0, 0), font=font)

    if meta:
        draw.text((pad + 6, y1 - 18), meta, fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)


# ============================================================
# Dataset (CLIP cache)
# ============================================================
class PreloadCacheDataset(Dataset):
    def __init__(self, root, folders):
        self.items = []
        for f in folders:
            fdir = os.path.join(root, f)
            if not os.path.isdir(fdir):
                print("[Dataset] WARNING folder missing:", fdir)
                continue

            for p in glob.glob(os.path.join(fdir, "*_in.png")):
                gt = p.replace("_in.png", "_gt.png")
                clip = p.replace("_in.png", "_clip.pt")
                if os.path.exists(gt) and os.path.exists(clip):
                    self.items.append((p, gt, clip))

        print(f"[Dataset] Total pairs: {len(self.items)}")
        if len(self.items) == 0:
            raise RuntimeError("No pairs found. Check preload_cache and *_clip.pt existence.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        in_p, gt_p, c_p = self.items[i]
        inp = torch.from_numpy(np.array(Image.open(in_p).convert("RGB"))).permute(2, 0, 1).float() / 255.0
        gt = torch.from_numpy(np.array(Image.open(gt_p).convert("RGB"))).permute(2, 0, 1).float() / 255.0
        clip = torch.load(c_p, map_location="cpu").float().view(-1)  # (D,)
        return {"inp": inp, "gt": gt, "clip": clip}


# ============================================================
# Phase-3 Model
# (Image → CLIP(cache) → LLM(prefix+prompt)
#   ├ generate() → XAI text
#   └ hidden → StrategyHead → (S,z) → proj → tokens → VETNet)
# ============================================================
class Phase3Model(nn.Module):
    def __init__(self, vetnet, llm, tokenizer, strategy_head, clip_dim, prefix_tokens=8):
        super().__init__()
        self.vetnet = vetnet
        self.llm = llm
        self.tokenizer = tokenizer
        self.strategy_head = strategy_head
        self.prefix_tokens = int(prefix_tokens)
        self.lm_dim = int(llm.config.hidden_size)

        self.clip_to_prefix = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, self.prefix_tokens * self.lm_dim),
        )

        self.stage_proj = nn.ModuleDict({
            "stage1": nn.Linear(256, 64),
            "stage2": nn.Linear(256, 128),
            "stage3": nn.Linear(256, 256),
        })

    def _make_prefix(self, clip_feat: torch.Tensor) -> torch.Tensor:
        # clip_feat: (B, D_clip)
        x = self.clip_to_prefix(clip_feat)
        x = x.view(clip_feat.size(0), self.prefix_tokens, self.lm_dim)
        return x.to(dtype=self.llm.dtype)  # ✅ FIX ②

    def _prompt_embeds(self, device: torch.device) -> torch.Tensor:
        prompt = (
            "You are a restoration pilot. "
            "Describe in ONE sentence how the image was restored."
        )
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        emb = self.llm.get_input_embeddings()(ids)
        return emb.to(dtype=self.llm.dtype)  # ✅ FIX ③

    def forward(self, inp: torch.Tensor, clip: torch.Tensor):
        # clip: (B,D) or (D,)
        if clip.ndim == 1:
            clip = clip.unsqueeze(0)

        prefix = self._make_prefix(clip)  # (B,P,D_lm)
        prompt = self._prompt_embeds(prefix.device).expand(prefix.size(0), -1, -1)  # (B,T,D_lm)
        embeds = torch.cat([prefix, prompt], dim=1)  # (B,P+T,D_lm)

        out = self.llm(
            inputs_embeds=embeds,
            output_hidden_states=True,
            use_cache=False
        )

        hidden = out.hidden_states[-1][:, :self.prefix_tokens].mean(1)  # (B,D_lm)

        # ✅ FIX 핵심: hidden dtype을 StrategyHead dtype으로 강제 일치
        sh_dtype = next(self.strategy_head.parameters()).dtype
        hidden = hidden.to(dtype=sh_dtype)

        S, z = self.strategy_head(hidden)  # S:(B,K,256)

        tokens = {k: self.stage_proj[k](S) for k in self.stage_proj}
        pred = self.vetnet(inp, strategy_tokens=tokens)
        return pred, S, z

    @torch.no_grad()
    def generate_xai(self, clip: torch.Tensor, max_new_tokens=48) -> str:
        if clip.ndim == 1:
            clip = clip.unsqueeze(0)

        prefix = self._make_prefix(clip.to(device=next(self.parameters()).device))
        prompt = self._prompt_embeds(prefix.device).expand(prefix.size(0), -1, -1)
        embeds = torch.cat([prefix, prompt], dim=1)

        ids = self.llm.generate(
            inputs_embeds=embeds,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=0.7,
        )
        text = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        return text if len(text) > 0 else DEFAULT_XAI  # ✅ 반드시 출력 보장


# ============================================================
# Train
# ============================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    cache_root = "E:/VETNet_Pilot/preload_cache"
    folders = ["CSD", "DayRainDrop", "NightRainDrop", "rain100H", "RESIDE-6K"]

    ds = PreloadCacheDataset(cache_root, folders)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

    # Backbone frozen
    vetnet = VETNetBackbone().to(device).eval()
    for p in vetnet.parameters():
        p.requires_grad = False

    # LLM Core (Phi-3)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="cuda" if device.type == "cuda" else None,
    )

    # ✅ FIX ① LLM 완전 freeze
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False
    print("[LLM] frozen")

    clip_dim = int(ds[0]["clip"].numel())

    # StrategyHead는 학습되는 모듈이므로 float32 유지 권장
    strategy_head = StrategyHead(
        lm_dim=int(llm.config.hidden_size),
        strategy_dim=512,
        K=3,
        C=256
    ).to(device)  # (dtype는 float32)

    model = Phase3Model(vetnet, llm, tokenizer, strategy_head, clip_dim, prefix_tokens=8).to(device)

    # trainable params (LLM/VETNet 제외)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("[Trainable] params =", sum(p.numel() for p in trainable))

    opt = torch.optim.AdamW(trainable, lr=1e-4)

    results_dir = "E:/VETNet_Pilot/results/phase3_llm"
    os.makedirs(results_dir, exist_ok=True)

    it = 0
    for epoch in range(1, 21):
        for b in tqdm(dl, desc=f"Epoch {epoch:03d}/20"):
            it += 1
            inp = b["inp"].to(device, non_blocking=True)
            gt = b["gt"].to(device, non_blocking=True)
            clip = b["clip"].to(device, non_blocking=True)

            pred, S, z = model(inp, clip)
            pred = pred.clamp(0, 1)

            loss = F.l1_loss(pred, gt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if it % 100 == 0:
                # generate()는 그래프 분리 (no_grad)
                try:
                    xai = model.generate_xai(clip[0], max_new_tokens=48)
                except Exception as e:
                    # 실패해도 반드시 문장 출력
                    xai = DEFAULT_XAI + f" (gen failed: {type(e).__name__})"

                save_triplet_with_xai(
                    inp[0], pred[0], gt[0], xai,
                    os.path.join(results_dir, f"iter_{it:06d}.png"),
                    meta=f"epoch={epoch} iter={it} loss={loss.item():.4f}"
                )

    print("[DONE] Phase-3 complete")


if __name__ == "__main__":
    train()
