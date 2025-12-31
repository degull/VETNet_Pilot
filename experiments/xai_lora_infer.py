# ============================================================
# xai_lora_infer.py
# ------------------------------------------------------------
# Inference for Phase-4 XAI-LoRA:
# - Load one cached vision feature (*.pt)
# - Compute g_stage using Phase-3 policy (fixed z_t)
# - Generate explanation using Mistral-7B 4bit + LoRA adapter
# ------------------------------------------------------------
# Path: E:\VETNet_Pilot\experiments\xai_lora_infer.py
# ============================================================

import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[xai_lora_infer] ROOT =", ROOT)

from models.pilot.phase3_policy import Phase3GatePolicy

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# -----------------------
# Config (edit as needed)
# -----------------------
CACHE_ROOT = r"E:\VETNet_Pilot\preload_cache"
PHASE3_CKPT = r"E:\VETNet_Pilot\checkpoints\phase3_vlm_clean\epoch_004_L0.0274_P29.59_S0.9315.pth"

CLIP_TEXT_MODEL = "openai/clip-vit-large-patch14"
FIXED_TEXT = "Restore the degraded image."

BASE_LM = "mistralai/Mistral-7B-v0.1"
LORA_ADAPTER_DIR = r"E:\VETNet_Pilot\checkpoints\xai_lora_mistral7b\lora_adapter"

# pick a specific cached vision file
# Example:
CLIP_PT_PATH = r"E:\VETNet_Pilot\preload_cache\CSD\000354_clip.pt"

MAX_NEW_TOKENS = 120
TEMPERATURE = 0.7
TOP_P = 0.9


# ============================================================
# Helpers
# ============================================================

def load_ckpt_flexible(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "policy" in ckpt:
            return ckpt["policy"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model" in ckpt:
            return ckpt["model"]
    return ckpt


def format_g_stage(g: torch.Tensor, decimals: int = 4) -> str:
    vals = [round(float(v.item()), decimals) for v in g]
    return "[" + ", ".join([f"{v:.{decimals}f}" for v in vals]) + "]"


def build_prompt(g_stage_1s: torch.Tensor) -> str:
    gvec = format_g_stage(g_stage_1s, decimals=4)
    prompt = (
        "You are an image restoration expert.\n"
        "Explain the restoration strategy implied by the following internal control signal.\n"
        "Do not guess the dataset or scene. Do not mention specific degradations.\n"
        "Focus on stage-wise modulation and what it implies mechanistically.\n\n"
        f"g_stage = {gvec}\n\n"
        "XAI explanation:"
    )
    return prompt


@torch.no_grad()
def get_fixed_clip_text_embed(model_name: str, text: str, device: str) -> torch.Tensor:
    from transformers import CLIPProcessor, CLIPTextModel

    processor = CLIPProcessor.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name).to(device)
    text_model.eval()
    for p in text_model.parameters():
        p.requires_grad = False

    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    out = text_model(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
    )
    return out.pooler_output  # [1, Dt]


# ============================================================
# Main
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Infer] device:", device)

    # 1) load cached z_v
    print("[Infer] load z_v:", CLIP_PT_PATH)
    z_v = torch.load(CLIP_PT_PATH, map_location="cpu")
    if z_v.ndim == 2 and z_v.size(0) == 1:
        z_v = z_v.squeeze(0)
    z_v = z_v.reshape(1, -1).to(device).float()
    Dv = int(z_v.size(-1))
    print("[Infer] vision_dim =", Dv)

    # 2) fixed z_t
    print("[Infer] compute fixed z_t from CLIP text encoder ...")
    z_t = get_fixed_clip_text_embed(CLIP_TEXT_MODEL, FIXED_TEXT, device=device)  # [1, Dt]
    Dt = int(z_t.size(-1))
    print("[Infer] text_dim =", Dt)

    # 3) load Phase-3 policy and compute g_stage
    print("[Infer] load Phase-3 policy:", PHASE3_CKPT)
    policy = Phase3GatePolicy(
        vision_dim=Dv,
        text_dim=Dt,
        num_stages=8,
        g_min=0.1,
        hidden_dim=512,
    ).to(device)
    sd = load_ckpt_flexible(PHASE3_CKPT)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    print("[Infer] policy loaded. Missing:", len(missing), "Unexpected:", len(unexpected))
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False

    with torch.no_grad():
        g_stage = policy(z_v, z_t)["g_stage"].detach().squeeze(0)  # [S]
    print("[Infer] g_stage =", format_g_stage(g_stage, decimals=4))

    # 4) load base LM 4bit + LoRA adapter
    print("[Infer] load Mistral 7B 4bit + LoRA adapter ...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_LM,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, LORA_ADAPTER_DIR)
    model.eval()

    # 5) generate explanation
    prompt = build_prompt(g_stage)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # show only generated part after "XAI explanation:"
    key = "XAI explanation:"
    if key in text:
        gen = text.split(key, 1)[-1].strip()
    else:
        gen = text.strip()

    print("\n" + "=" * 72)
    print("PROMPT:\n", prompt)
    print("\nLOLA-XAI OUTPUT:\n", gen)
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
