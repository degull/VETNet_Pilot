""" # G:/VETNet_pilot/models/pilot/tokenizer_utils.py
# Phase-2 LLM이 사용할 프롬프트 템플릿 생성기
# dataset_tag, scene_tag, degradation_type 등을 받아서
# LLM에 맞는 자연스러운 “Strategy Prompt” 문장을 만들어서 반환
import random
from typing import Optional, Dict


# 미리 이렇게 해두는게 맞는지?
# 같은 왜곡 종류라도 LLM이 이미지마다 전략 문장을 다르게 만듦?
# ---------------------------------------------------------
# 사전 정의된 장면(scene) / 열화(degradation) 설명들
# ---------------------------------------------------------
SCENE_TEMPLATES = [
    "an outdoor scene with buildings and streets",
    "a close-up object with fine edges",
    "a natural environment with trees and foliage",
    "a human face requiring careful contour preservation",
    "a vehicle with reflective metallic surfaces",
    "a signboard or text region with thin strokes",
]

DEGRADATION_TEMPLATES = {
    "Rain100H": "heavy rain streaks with motion blur",
    "Rain100L": "light rain streaks and mild blur",
    "Raindrop": "multiple transparent raindrops obstructing key image regions",
    "CSD": "snow particles and luminance fluctuations",
    "RESIDE": "dense haze with global illumination imbalance",
    "DayRainDrop": "raindrop blobs with bright outdoor background",
    "NightRainDrop": "raindrop blobs with low light noise",
}


# ---------------------------------------------------------
# 프롬프트 기본 템플릿
# (LLM에게 ‘이미지 복원 전문가’ 역할을 부여하는 시스템 프롬프트)
# ---------------------------------------------------------
BASE_SYSTEM_TEMPLATE = (
    "You are an expert in image restoration. "
    "Given a degraded input image, describe a restoration strategy that improves "
    "PSNR and SSIM while preserving important scene-dependent structures."
)


# ---------------------------------------------------------
# degradation + scene + dataset 기반 문장 생성
# ---------------------------------------------------------
def build_strategy_prompt(
    scene_desc: Optional[str] = None,
    dataset_tag: Optional[str] = None,
    extra_info: Optional[Dict] = None
):

    # --------------------------
    # 장면(scene) 자동 할당
    # --------------------------
    if scene_desc is None:
        scene_desc = random.choice(SCENE_TEMPLATES)

    # --------------------------
    # 열화 설명
    # --------------------------
    if dataset_tag in DEGRADATION_TEMPLATES:
        degradation_str = DEGRADATION_TEMPLATES[dataset_tag]
    else:
        degradation_str = "unknown degradation with structural distortions"

    # --------------------------
    # extra_info 추가 문장
    # --------------------------
    extra_sentence = ""
    if extra_info:
        if "severity" in extra_info:
            extra_sentence += f" The degradation severity appears {extra_info['severity']}."
        if "lighting" in extra_info:
            extra_sentence += f" The lighting condition is {extra_info['lighting']}."

    # --------------------------
    # 최종 Prompt 구성
    # --------------------------
    prompt = (
        f"{BASE_SYSTEM_TEMPLATE}\n\n"
        f"Scene description: {scene_desc}.\n"
        f"Observed degradation: {degradation_str}.\n"
        f"{extra_sentence}\n"
        f"Explain the restoration strategy focusing on:\n"
        f"- removing artifacts\n"
        f"- preserving structural edges\n"
        f"- maintaining correct colors\n"
        f"- maximizing PSNR/SSIM\n"
    )

    return prompt.strip()


# ---------------------------------------------------------
# Self-Test
# python models/pilot/tokenizer_utils.py 실행하면 테스트됨
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n[tokenizer_utils] Self-test 시작\n")

    example_scene = "a vehicle with reflective metal parts"
    prompt = build_strategy_prompt(
        scene_desc=example_scene,
        dataset_tag="Rain100H",
        extra_info={"severity": "heavy", "lighting": "low"}
    )

    print("=== Generated Prompt ===")
    print(prompt)
    print("\n[tokenizer_utils] Self-test 완료.\n")
 """

# ver2
# G:\VETNet_pilot\models\pilot\tokenizer_utils.py
import torch

# -----------------------------
# Prompts (Dual-head)
# -----------------------------
# Strategy prompt: used to stabilize LLM context for latent strategy regression (z)
STRATEGY_PROMPT = (
    "You are a restoration strategy controller. "
    "Given the visual embedding, form an internal restoration strategy representation."
)

# XAI prompt: used only when generating explanation text
XAI_PROMPT = (
    "Explain the restoration strategy for the current image.\n"
    "Write exactly 4 sentences.\n"
    "Sentence 1: overall degradation characteristics.\n"
    "Sentence 2: Stage 1 strategy.\n"
    "Sentence 3: Stage 2 strategy.\n"
    "Sentence 4: Stage 3 strategy.\n"
    "Do not include any headings, lists, examples, or quoted text."
)




def build_strategy_prompt(task_hint: str = "") -> str:
    if task_hint and len(task_hint.strip()) > 0:
        return STRATEGY_PROMPT + " Task hint: " + task_hint.strip()
    return STRATEGY_PROMPT


def build_xai_prompt(task_hint: str = "") -> str:
    if task_hint and len(task_hint.strip()) > 0:
        return XAI_PROMPT + " Task hint: " + task_hint.strip()
    return XAI_PROMPT


def ensure_pad_token(tokenizer):
    # Some causal LMs don't have pad_token by default
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def tokenize_prompt(tokenizer, prompt: str, device: torch.device, max_length: int = 128):
    """
    Tokenize text prompt for LLM.

    NOTE:
    - Training does NOT use generate(); prompt is a fixed context anchor.
    - For XAI generation, we can reuse this to provide a consistent conditioning prompt.
    """
    tok = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    return tok
