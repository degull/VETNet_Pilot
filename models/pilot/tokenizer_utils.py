# G:/VETNet_pilot/models/pilot/tokenizer_utils.py
# Phase-2 LLM이 사용할 프롬프트 템플릿 생성기
# dataset_tag, scene_tag, degradation_type 등을 받아서
# LLM에 맞는 자연스러운 “Strategy Prompt” 문장을 만들어서 반환
import random
from typing import Optional, Dict


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
    """
    Phase2에서 LLM 입력으로 사용되는 strategy prompt 문자열 생성.

    Args:
        scene_desc : CLIP으로 분석해 얻은 이미지 장면 설명(optional)
        dataset_tag : Rain100H / Raindrop / CSD 등
        extra_info : 선택적 dict (e.g., {'severity': 'heavy'})
    
    Returns:
        prompt : LLM에 주는 natural-language instruction
    """

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
