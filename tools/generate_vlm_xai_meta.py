# E:/VETNet_Pilot/tools/generate_vlm_xai_meta.py
import os
import json
import time
from typing import Dict, Any, List

import torch
from tqdm import tqdm
from PIL import Image

from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)

# ============================================================
# Config
# ============================================================
PAIRS_JSON = r"E:\VETNet_Pilot\data_cache\pairs_cache_phase2.json"
OUT_JSON   = r"E:\VETNet_Pilot\data_cache\meta_vlm_xai.json"

# ✅ instruction-following이 강한 VLM
VLM_NAME = "Salesforce/instructblip-flan-t5-xl"

# Save / Resume
SAVE_EVERY = 200

# Optional: limit for debugging (set None for all)
LIMIT = 10  # None for all

# Regenerate policy (기존 값이 이상하면 재생성)
REGEN_EMPTY = True
REGEN_BAD   = True

# ============================================================
# Generation (stable → retry)
# ============================================================
GEN_1 = dict(
    max_new_tokens=96,
    num_beams=4,
    do_sample=False,
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
    length_penalty=1.0,
    early_stopping=True,
)

GEN_2 = dict(
    max_new_tokens=140,
    num_beams=6,
    do_sample=False,
    repetition_penalty=1.25,
    no_repeat_ngram_size=4,
    length_penalty=1.05,
    early_stopping=True,
)

MIN_CHARS = 60  # 너무 짧은 출력은 불량으로 처리


# ============================================================
# Utils
# ============================================================
def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def infer_dataset_from_path(inp_path: str) -> str:
    p = inp_path.replace("/", "\\")
    key = "\\preload_cache\\"
    if key in p:
        tail = p.split(key, 1)[1]
        return tail.split("\\", 1)[0]
    return "unknown"


def load_pairs(pairs_json: str) -> List[Dict[str, str]]:
    with open(pairs_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"pairs json must be a list, got: {type(data)}")
    return data


def safe_open_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def clean_text(s: str) -> str:
    return " ".join((s or "").strip().split())


def is_degenerate(text: str) -> bool:
    t = clean_text(text)
    if len(t) == 0:
        return True
    if len(t) < MIN_CHARS:
        return True

    low = t.lower()

    # (1) 캡션 톤으로 튀는 경우를 강하게 잡음 (XAI가 아니라 캡션)
    caption_prefixes = (
        "a photo of", "a picture of", "an image of", "a close up of",
        "a cityscape", "a saturating image", "a man and his dog",
        "a view of", "a city street", "a snowy city"
    )
    if low.startswith(caption_prefixes):
        return True

    # (2) 반복 붕괴 / 리스트 붕괴 패턴
    if "a) a) a)" in low:
        return True
    if low.count(" a) ") >= 2 or low.count("a)") >= 6:
        return True

    # (3) 의미없는 반복 단어 (skyline skyline 등)
    if "skyline and a city skyline" in low:
        return True

    return False


# ============================================================
# Prompt (Conservative XAI)
# ============================================================
def build_vlm_prompt() -> str:
    # ✅ “캡션”이 아니라 “복원 관점의 설명”을 강제
    # ✅ low-level(노이즈/블러/커널) 단정 금지
    # ✅ 2문장 고정
    return (
        "You observe the image directly.\n"
        "Write an explainable image restoration description focused on visibility and structure.\n\n"
        "Rules:\n"
        "- Do NOT write a generic caption (do not start with 'a photo of' or 'a picture of').\n"
        "- Describe only visually evident phenomena (e.g., snow/rain/haze, reduced visibility, softened edges).\n"
        "- If the degradation type is not clearly observable, explicitly say it is not determinable from the image.\n"
        "- Do NOT speculate about noise levels, blur kernels, sensor artifacts, or causes.\n"
        "- Use spatial terms only if visually supported (foreground/background/top/bottom).\n"
        "- Do NOT mention datasets, models, methods, or tools.\n\n"
        "Output:\n"
        "- Exactly 2 sentences.\n"
        "- Final text only."
    )


# ============================================================
# VLM Inference (Image-conditioned)
# ============================================================
@torch.no_grad()
def generate_once(model, processor, image: Image.Image, prompt: str, device: torch.device, gen_cfg: dict) -> str:
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    out_ids = model.generate(
        **inputs,
        **gen_cfg,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    text = processor.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return clean_text(text)


@torch.no_grad()
def vlm_generate_xai(model, processor, image: Image.Image, device: torch.device) -> str:
    prompt = build_vlm_prompt()

    # try 1
    t1 = generate_once(model, processor, image, prompt, device, GEN_1)
    if not is_degenerate(t1):
        return t1

    # try 2
    t2 = generate_once(model, processor, image, prompt, device, GEN_2)
    if not is_degenerate(t2):
        return t2

    # fallback: 그래도 이상하면 가장 그럴듯한 쪽 반환
    # (둘 다 degenerate면 t2가 더 길 가능성이 높음)
    return t2 if len(t2) >= len(t1) else t1


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ensure_dir(OUT_JSON)

    pairs = load_pairs(PAIRS_JSON)
    if LIMIT is not None:
        pairs = pairs[: int(LIMIT)]

    print(f"[INFO] Loaded pairs: {len(pairs)}")
    if len(pairs) == 0:
        raise RuntimeError("pairs_cache_phase2.json is empty.")

    # resume
    meta: Dict[str, Any] = {}
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                print("[WARN] existing meta is not dict, resetting.")
                meta = {}
            else:
                print(f"[INFO] Resume from existing meta: {len(meta)} items")
        except Exception as e:
            print("[WARN] Failed to load existing meta, resetting.", type(e).__name__, str(e))
            meta = {}

    # load VLM
    print("[INFO] Loading VLM:", VLM_NAME)
    processor = InstructBlipProcessor.from_pretrained(VLM_NAME)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        VLM_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("[VLM] frozen")

    t0 = time.time()
    added = 0
    skipped = 0
    failed = 0
    regenerated = 0

    pbar = tqdm(pairs, desc="InstructBLIP XAI meta", total=len(pairs))
    for idx, item in enumerate(pbar, start=1):
        inp_path = item.get("inp", "")
        gt_path = item.get("gt", "")

        if not inp_path:
            failed += 1
            continue

        ds_name = infer_dataset_from_path(inp_path)

        # resume skip / regen policy
        if inp_path in meta and isinstance(meta[inp_path], dict) and "xai_llm" in meta[inp_path]:
            existing = clean_text(str(meta[inp_path].get("xai_llm", "") or ""))
            if len(existing) > 0 and not is_degenerate(existing):
                skipped += 1
                continue

            if (len(existing) == 0 and not REGEN_EMPTY) or (is_degenerate(existing) and not REGEN_BAD):
                skipped += 1
                continue

            regenerated += 1

        if not os.path.exists(inp_path):
            failed += 1
            meta[inp_path] = {
                "xai_llm": "",
                "gt": gt_path,
                "dataset": ds_name,
                "xai_source": "image-conditioned VLM (ViT + Q-Former + LLM)",
                "error": "inp_not_found",
            }
            continue

        try:
            img = safe_open_rgb(inp_path)
            xai = vlm_generate_xai(model, processor, img, device)

            meta[inp_path] = {
                "xai_llm": xai,
                "gt": gt_path,
                "dataset": ds_name,
                "xai_source": "image-conditioned VLM (ViT + Q-Former + LLM)",
                "prompt_hint": "InstructBLIP conservative XAI (no caption tone, no low-level speculation, 2 sentences)",
            }
            added += 1

        except Exception as e:
            failed += 1
            meta[inp_path] = {
                "xai_llm": "",
                "gt": gt_path,
                "dataset": ds_name,
                "xai_source": "image-conditioned VLM (ViT + Q-Former + LLM)",
                "error": f"{type(e).__name__}: {str(e)}",
            }

        # save periodically
        if (added + failed) % int(SAVE_EVERY) == 0:
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        if idx % 50 == 0:
            elapsed = time.time() - t0
            pbar.set_postfix(
                added=added, skipped=skipped, regen=regenerated, failed=failed, sec=f"{elapsed:.1f}"
            )

    # final save
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print("============================================================")
    print("[DONE] InstructBLIP XAI meta.json generated")
    print("Total pairs:", len(pairs))
    print("Saved items:", len(meta))
    print("Added:", added, "Skipped:", skipped, "Regenerated:", regenerated, "Failed:", failed)
    print("Saved to:", OUT_JSON)
    print("Elapsed(sec):", f"{elapsed:.1f}")
    print("============================================================")


if __name__ == "__main__":
    main()
