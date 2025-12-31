# ============================================================
# xai_lora_trainer.py  (Phase-4: XAI-LoRA TRAINING ONLY)
# ------------------------------------------------------------
# Goal:
#   Train ONLY a LoRA adapter on a 4-bit Mistral-7B to generate
#   natural-language XAI explanations from g_stage.
#
# Inputs:
#   - preload_cache/**/**_clip.pt  (cached CLIP vision embedding z_v)
#   - Phase-3 policy checkpoint (frozen) to produce g_stage from (z_v, z_t_fixed)
#
# Outputs:
#   - LoRA adapter only (no changes to backbone/gate/policy/CLIP)
#
# Notes:
#   - No inference mode in this script. TRAINING ONLY.
#   - We intentionally avoid Transformers Trainer to reduce side-effects.
#   - We include a Windows-safe workaround for broken TensorFlow installs:
#     Transformers in some versions imports tensorflow in unrelated modules.
#     We stub tensorflow to prevent DLL-load crashes.
# ============================================================

import os

# ---- must be set BEFORE importing transformers/peft ----
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import math
import time
import json
import glob
import random
import argparse
import types
import importlib.util
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Windows / broken TensorFlow workaround
# - Some Transformers builds import tensorflow in modules that
#   you never use (e.g., image_transforms). If TF is installed
#   but broken (DLL load fail), the import crashes immediately.
# - We stub a minimal tensorflow module so `import tensorflow as tf`
#   succeeds, but any actual TF usage will raise an error.
# ============================================================
def _install_tensorflow_stub_if_needed():
    try:
        spec = importlib.util.find_spec("tensorflow")
    except Exception:
        spec = None

    if spec is None:
        return  # TF not installed -> nothing to do

    # If TF is installed but broken, importing it will crash.
    # We prevent that by pre-inserting a stub module.
    if "tensorflow" in sys.modules:
        return

    tf_stub = types.ModuleType("tensorflow")
    tf_stub.__dict__["__version__"] = "0.0.stub"

    def _tf_stub_call(*args, **kwargs):
        raise RuntimeError("TensorFlow is stubbed to avoid DLL-load issues. Do not call TF APIs.")

    # Common namespaces referenced by some modules
    tf_stub.constant = _tf_stub_call
    tf_stub.convert_to_tensor = _tf_stub_call
    tf_stub.image = types.SimpleNamespace()
    tf_stub.math = types.SimpleNamespace()
    tf_stub.nn = types.SimpleNamespace()
    tf_stub.keras = types.SimpleNamespace()

    sys.modules["tensorflow"] = tf_stub
    sys.modules["tf_keras"] = types.ModuleType("tf_keras")

_install_tensorflow_stub_if_needed()

# ============================================================
# Add project ROOT
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
print("[xai_lora_trainer] ROOT =", ROOT)

# ============================================================
# Project imports (Phase-3 policy)
# ============================================================
from models.pilot.phase3_policy import Phase3GatePolicy

# ============================================================
# HF / PEFT / BNB imports
# ============================================================
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    CLIPTokenizer,
    CLIPTextModel,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_state_dict_flexible(ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        # Phase-3 usually saved as {"epoch":..., "policy": state_dict, ...}
        if "policy" in ckpt and isinstance(ckpt["policy"], dict):
            return ckpt["policy"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        # fallback
        return ckpt
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")

def list_clip_pt_files(cache_root: str) -> List[str]:
    # preload_cache/<dataset>/<6digits>_clip.pt
    pats = [
        os.path.join(cache_root, "*", "*_clip.pt"),
        os.path.join(cache_root, "*", "*clip.pt"),
    ]
    out = []
    for p in pats:
        out.extend(glob.glob(p))
    out = sorted(list(set(out)))
    return out

def parse_dataset_and_index_from_clip_path(clip_path: str) -> Tuple[str, str]:
    # .../preload_cache/CSD/000354_clip.pt
    ds_name = os.path.basename(os.path.dirname(clip_path))
    base = os.path.basename(clip_path)
    # "000354_clip.pt" -> "000354"
    idx = base.split("_")[0]
    return ds_name, idx

def safe_load_tensor(path: str) -> torch.Tensor:
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict) and "tensor" in t:
        t = t["tensor"]
    if not torch.is_tensor(t):
        raise ValueError(f"clip.pt is not a Tensor: {path}")
    # expected shape: [768] or [1,768]
    if t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim == 2 and t.size(0) != 1:
        # force [1,D]
        t = t.view(1, -1)
    elif t.ndim > 2:
        t = t.view(1, -1)
    return t.float()

def build_prompt_from_g(g_list: List[float]) -> str:
    g_txt = "[" + ", ".join([f"{v:.4f}" for v in g_list]) + "]"
    # prompt only (no hard-coded thresholds / no discrete strategy labels)
    return (
        "You are an image restoration expert.\n"
        "Given an internal stage-wise control signal (g_stage) used to modulate a restoration network, "
        "write a factual explanation of what this modulation implies.\n"
        "Constraints:\n"
        "- Do NOT guess scene content.\n"
        "- Do NOT mention datasets.\n"
        "- Describe relative emphasis/suppression across stages.\n"
        "- 2 to 4 sentences.\n\n"
        f"g_stage: {g_txt}\n\n"
        "XAI:"
    )

def build_pseudo_xai_from_g(g: List[float]) -> str:
    """
    'Rule-free' style: no discrete bins like "if mid<0.92".
    We describe continuous tendencies by comparing segments.
    """
    S = len(g)
    # robust segmenting for S=8 default, but works for any S>=3
    enc_end = min(3, S)
    mid_end = min(5, S)
    enc = g[:enc_end]
    mid = g[enc_end:mid_end]
    dec = g[mid_end:] if mid_end < S else []

    def mean(x): return sum(x) / max(1, len(x))
    enc_m = mean(enc)
    mid_m = mean(mid) if len(mid) else enc_m
    dec_m = mean(dec) if len(dec) else mid_m

    overall = mean(g)
    # relative differences (continuous)
    d_enc = enc_m - overall
    d_mid = mid_m - overall
    d_dec = dec_m - overall

    def phrase(d: float) -> str:
        # soft wording without fixed thresholds
        if d > 0.02:
            return "relatively emphasized"
        if d < -0.02:
            return "relatively suppressed"
        return "kept near baseline"

    enc_p = phrase(d_enc)
    mid_p = phrase(d_mid)
    dec_p = phrase(d_dec)

    # 2-4 factual sentences
    sent1 = (
        f"The early stages are {enc_p}, which controls how strongly low-level features are filtered or preserved."
    )
    sent2 = (
        f"The mid stages are {mid_p}, affecting how global structures and coherent artifacts are handled."
    )
    sent3 = (
        f"The later stages are {dec_p}, which influences refinement and detail reconstruction near the output."
    )

    # keep 3 sentences (paper-friendly, consistent)
    return " ".join([sent1, sent2, sent3])

def mask_prompt_labels(input_ids: torch.Tensor, prompt_len: int, pad_id: int) -> torch.Tensor:
    """
    labels = input_ids, but prompt tokens -> -100, padding -> -100
    """
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    labels[labels == pad_id] = -100
    return labels

# ============================================================
# Dataset
# ============================================================

class Phase4XaiDataset(Dataset):
    """
    Each item:
      - prompt (text)
      - target (pseudo XAI)
    We compute g_stage using frozen Phase-3 policy:
      g_stage = policy(z_v, z_t_fixed)["g_stage"].
    """
    def __init__(
        self,
        clip_pt_files: List[str],
        policy: Phase3GatePolicy,
        z_t_fixed: torch.Tensor,
        device: str,
    ):
        self.clip_pt_files = clip_pt_files
        self.policy = policy
        self.z_t_fixed = z_t_fixed
        self.device = device

        # sanity: infer vision_dim
        z0 = safe_load_tensor(self.clip_pt_files[0])
        self.vision_dim = int(z0.size(-1))

    def __len__(self):
        return len(self.clip_pt_files)

    @torch.no_grad()
    def _compute_g_stage(self, clip_path: str) -> List[float]:
        z_v = safe_load_tensor(clip_path).to(self.device)  # [1,D]
        out = self.policy(z_v, self.z_t_fixed)
        g = out["g_stage"][0].detach().float().cpu().tolist()
        return g

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clip_path = self.clip_pt_files[idx]
        g = self._compute_g_stage(clip_path)

        prompt = build_prompt_from_g(g)
        target = build_pseudo_xai_from_g(g)

        # keep metadata for debugging
        ds_name, item_id = parse_dataset_and_index_from_clip_path(clip_path)

        return {
            "prompt": prompt,
            "target": target,
            "g": g,
            "clip_path": clip_path,
            "dataset": ds_name,
            "item_id": item_id,
        }

# ============================================================
# Collate
# ============================================================

def make_collate_fn(tokenizer, max_length: int):
    pad_id = tokenizer.pad_token_id

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [b["prompt"] for b in batch]
        targets = [b["target"] for b in batch]

        # build full supervised text: prompt + space + target
        full_texts = [p + " " + t for p, t in zip(prompts, targets)]

        # tokenize prompt only to get prompt lengths
        prompt_tok = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        full_tok = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = full_tok["input_ids"]
        attn = full_tok["attention_mask"]

        # prompt lengths: count non-pad tokens in prompt_tok
        prompt_lens = prompt_tok["attention_mask"].sum(dim=1).tolist()

        # labels masking prompt portion
        labels = input_ids.clone()
        for i, pl in enumerate(prompt_lens):
            labels[i, :pl] = -100
        labels[labels == pad_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "meta": batch,
        }

    return collate

# ============================================================
# Train
# ============================================================

def train(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Phase4] device:", device)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "phase4_xai_lora_log.txt")
    print("[Phase4] log:", log_path)

    # --------------------------------------------------------
    # 1) Load CLIP text encoder (frozen) to get z_t_fixed
    # --------------------------------------------------------
    print("[Phase4] loading CLIP text encoder:", args.clip_text_model)
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_text_model)
    clip_text = CLIPTextModel.from_pretrained(args.clip_text_model).to(device)
    clip_text.eval()
    for p in clip_text.parameters():
        p.requires_grad = False

    inputs = clip_tokenizer([args.fixed_prompt], return_tensors="pt", padding=True, truncation=True)
    z_t = clip_text(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
    ).pooler_output  # [1,Dt]
    Dt = int(z_t.size(-1))
    print("[Phase4] fixed text_dim =", Dt)

    # --------------------------------------------------------
    # 2) Load Phase-3 policy (frozen)
    # --------------------------------------------------------
    print("[Phase4] loading Phase-3 policy ckpt:", args.phase3_ckpt)
    # infer vision_dim from a sample clip.pt
    clip_files_all = list_clip_pt_files(args.cache_root)
    if len(clip_files_all) == 0:
        raise FileNotFoundError(f"No *_clip.pt found under: {args.cache_root}")

    if args.max_items > 0:
        clip_files_all = clip_files_all[: args.max_items]
    print("[Phase4] clip.pt files =", len(clip_files_all))

    z0 = safe_load_tensor(clip_files_all[0])
    Dv = int(z0.size(-1))
    print("[Phase4] inferred vision_dim =", Dv)

    policy = Phase3GatePolicy(
        vision_dim=Dv,
        text_dim=Dt,
        num_stages=args.num_stages,
        g_min=args.g_min,
        hidden_dim=args.policy_hidden,
        dropout=0.0,
        num_strategies=0,
        init_gate_bias=2.0,
    ).to(device)
    sd = load_state_dict_flexible(args.phase3_ckpt)
    policy.load_state_dict(sd, strict=False)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    print("[Phase4] policy frozen:", True)

    # --------------------------------------------------------
    # 3) Dataset / Loader
    # --------------------------------------------------------
    ds = Phase4XaiDataset(
        clip_pt_files=clip_files_all,
        policy=policy,
        z_t_fixed=z_t,
        device=device,
    )
    print("[Phase4Dataset] items =", len(ds))

    # --------------------------------------------------------
    # 4) Load base LLM 4bit + attach LoRA (trainable only)
    # --------------------------------------------------------
    print("[Phase4] loading base LLM 4bit:", args.base_llm)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_llm, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_llm,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # k-bit training prep (enables grads on input norms etc)
    base = prepare_model_for_kbit_training(base)

    # LoRA config (paper-standard)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_targets.split(","),
    )
    model = get_peft_model(base, lora_cfg)

    # print trainable params
    trainable = 0
    total = 0
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"[Phase4] trainable params: {trainable/1e6:.2f} M / total {total/1e6:.2f} M ({100*trainable/total:.4f}%)")

    # --------------------------------------------------------
    # 5) Optimizer
    # --------------------------------------------------------
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # 6) Loader + collate
    # --------------------------------------------------------
    collate_fn = make_collate_fn(tokenizer, max_length=args.max_length)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True if args.batch_size > 1 else False,
        collate_fn=collate_fn,
    )
    print("[Phase4] steps/epoch =", len(dl))

    # --------------------------------------------------------
    # 7) Training loop (ALWAYS RUNS)
    # --------------------------------------------------------
    model.train()
    global_step = 0
    t_start = time.time()

    def save_lora(step_tag: str):
        save_dir = os.path.join(args.out_dir, step_tag)
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        # save training args
        with open(os.path.join(save_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        print(f"[Phase4] ✅ saved LoRA -> {save_dir}")

    print("[Phase4] start training ...")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[start] {time.ctime()}\n")
        f.write(json.dumps(vars(args), ensure_ascii=False) + "\n")

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        nstep = 0

        for batch in dl:
            global_step += 1

            input_ids = batch["input_ids"].to(next(model.parameters()).device)
            attention_mask = batch["attention_mask"].to(next(model.parameters()).device)
            labels = batch["labels"].to(next(model.parameters()).device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

            lv = float(loss.item())
            epoch_loss += lv
            nstep += 1

            if (global_step % args.log_every) == 0:
                elapsed = time.time() - t_start
                msg = f"[E{epoch:02d} | iter {global_step:06d}] loss={lv:.6f} avg={epoch_loss/max(1,nstep):.6f} time={elapsed:.1f}s"
                print(msg)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

            if (args.save_every > 0) and (global_step % args.save_every == 0):
                save_lora(f"lora_step_{global_step:06d}")

        # epoch end
        avg = epoch_loss / max(1, nstep)
        msg = f"[Epoch {epoch:02d}/{args.epochs}] avg_loss={avg:.6f}"
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        if args.save_each_epoch:
            save_lora(f"lora_epoch_{epoch:02d}")

    # final save
    save_lora("lora_final")
    print("[Phase4] Training completed.")

# ============================================================
# CLI
# ============================================================

def build_argparser():
    ap = argparse.ArgumentParser()

    # paths
    ap.add_argument("--out_dir", type=str, required=True, help="Output dir for LoRA adapters")
    ap.add_argument("--cache_root", type=str, required=True, help="E:/VETNet_Pilot/preload_cache")
    ap.add_argument("--phase3_ckpt", type=str, required=True, help="Phase-3 policy checkpoint (.pth)")

    # models
    ap.add_argument("--base_llm", type=str, default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--clip_text_model", type=str, default="openai/clip-vit-large-patch14")
    ap.add_argument("--fixed_prompt", type=str, default="Restore the degraded image.")

    # policy config (must match Phase-3)
    ap.add_argument("--num_stages", type=int, default=8)
    ap.add_argument("--g_min", type=float, default=0.1)
    ap.add_argument("--policy_hidden", type=int, default=512)

    # training
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_items", type=int, default=2000, help="Limit number of clip.pt files (0 = all)")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--save_each_epoch", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--bf16", action="store_true", help="Use bf16 compute dtype for 4bit (if supported)")

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target modules",
    )

    return ap

def main():
    args = build_argparser().parse_args()
    train(args)

if __name__ == "__main__":
    main()


# python e:/VETNet_Pilot/trainers/xai_lora_trainer.py --out_dir E:/VETNet_Pilot/checkpoints/xai_lora --phase3_ckpt E:/VETNet_Pilot/checkpoints/phase3_vlm_clean/epoch_004_L0.0274_P29.59_S0.9315.pth --cache_root E:/VETNet_Pilot/preload_cache --epochs 1 --batch_size 1 --max_items 2000 --lr 2e-4 --save_every 500
