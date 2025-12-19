""" # G:/VETNet_pilot/models/pilot/llm_loader.py
# LLaMA/Mistral 같은 LLM을 4bit + LoRA로 불러오는 모듈
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

# 이 모듈은 "LLM (예: Mistral/LLaMA) + LoRA + 4bit" 로더 역할만 담당
# 실제 Strategy Text 생성/이미지 연동은 strategy_head.py에서 처리


@dataclass
class LLMConfig:

    base_model_name: str = "mistralai/Mistral-7B-v0.1"   # 네가 DPR-Net에서 쓰던 모델
    load_in_4bit: bool = True
    use_lora: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj")

    max_seq_len: int = 512

    # device_map은 "auto"로 두고, torch.device는 외부에서 처리
    device_map: str = "auto"


class LLMWithLoRA(nn.Module):


    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        # 필요 모듈 로드
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )
        except ImportError as e:
            raise ImportError(
                "[llm_loader] transformers 라이브러리가 필요합니다. "
                "설치: pip install transformers accelerate bitsandbytes peft"
            ) from e

        # 1) Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            # 일부 LLaMA 계열은 pad_token이 없음 → eos_token 재사용
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2) Model (4bit or full precision)
        quant_config = None
        if config.load_in_4bit:
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as e:
                print("[llm_loader WARNING] BitsAndBytesConfig 생성 실패, 4bit 비활성화:", e)
                quant_config = None
                config.load_in_4bit = False

        if config.load_in_4bit and quant_config is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                quantization_config=quant_config,
                device_map=config.device_map,
            )
        else:
            # fallback: FP16/FP32
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                torch_dtype=dtype,
                device_map=config.device_map,
            )

        # hidden_states 출력을 위해 설정
        self.model.config.output_hidden_states = True

        # 3) LoRA 적용 (원하면)
        if config.use_lora:
            try:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            except ImportError as e:
                raise ImportError(
                    "[llm_loader] peft 라이브러리가 필요합니다. "
                    "설치: pip install peft"
                ) from e

            if config.load_in_4bit:
                self.model = prepare_model_for_kbit_training(self.model)

            lora_cfg = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.lora_target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            print("[llm_loader] LoRA 적용 완료. Trainable params:",
                  sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        else:
            # LoRA 사용하지 않으면 대부분 freeze → Phase2에서는 LoRA 학습 권장
            for p in self.model.parameters():
                p.requires_grad = False
            print("[llm_loader] LoRA 비활성화 상태. LLM 파라미터는 freeze 됩니다.")

        # generate에서 pad_token_id 사용
        self.generation_kwargs = dict(
            max_new_tokens=64,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    # ---------------------------------------------------------
    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        return outputs

    # ---------------------------------------------------------
    @torch.no_grad()
    def generate_with_hidden(
        self,
        prompt: str,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 토크나이즈
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_len,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        # generate 호출 (hidden_states 얻기 위해 다시 한 번 forward)
        gen_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            **self.generation_kwargs,
        )

        # 텍스트 디코딩
        generated_text = self.tokenizer.batch_decode(
            gen_outputs,
            skip_special_tokens=True,
        )[0]

        # hidden_states 얻기: 전체 sequence를 다시 forward
        with torch.no_grad():
            outputs = self.model(
                input_ids=gen_outputs,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states  # tuple: [layer0, layer1, ..., layerN]
        last_hidden = hidden_states[-1]        # (B, L, C)

        return generated_text, last_hidden


# ============================================================
# 모듈 내 헬퍼 함수
# ============================================================
def load_llm(config: Optional[LLMConfig] = None) -> LLMWithLoRA:

    if config is None:
        config = LLMConfig()
    llm = LLMWithLoRA(config)
    return llm


# ============================================================
# Self-test: python models/pilot/llm_loader.py 로 실행
# ============================================================
if __name__ == "__main__":

    print("[llm_loader] Self-test 시작")

    try:
        cfg = LLMConfig(
            # 너무 큰 모델이 부담되면 여기서 더 작은 모델 이름으로 교체해도 됨
            base_model_name="mistralai/Mistral-7B-v0.1",
            load_in_4bit=True,
            use_lora=False,   # 처음 테스트 시에는 False로 두는 것도 가능
        )
        llm = load_llm(cfg)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[llm_loader] Device = {device}")

        test_prompt = (
            "You are an image restoration expert. "
            "Given a rainy and low-quality image, describe a restoration strategy "
            "that improves PSNR and SSIM while preserving important edges."
        )

        print("[llm_loader] Prompt:")
        print(test_prompt)

        text, hidden = llm.generate_with_hidden(test_prompt, device=device)

        print("\n[llm_loader] Generated Strategy Text:")
        print(text)
        print("\n[llm_loader] Last hidden state shape:", hidden.shape)

        print("\n[llm_loader] Self-test 완료 (성공).")

    except Exception as e:
        print("\n[llm_loader] Self-test 중 오류 발생:")
        print(repr(e))
        print(
            "\n→ transformers / peft / bitsandbytes 설치 여부와, "
            "base_model_name이 올바른지 확인해줘."
        )
 """

# ver2
# G:\VETNet_pilot\models\pilot\llm_loader.py
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_OK = True
except Exception as e:
    PEFT_OK = False


def _debug_env():
    print("[llm_loader] CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[llm_loader] GPU:", torch.cuda.get_device_name(0))
        print("[llm_loader] CUDA capability:", torch.cuda.get_device_capability(0))


def load_llm_with_lora(
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    device: str = "cuda",
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules=None,
):
    """
    Loads a causal LLM and applies LoRA.
    - For training stability, we DO NOT call generate() in training steps.
    - We'll use hidden states as a continuous regressor output.

    Requires:
      pip install transformers peft bitsandbytes accelerate
    """
    _debug_env()

    if target_modules is None:
        # Common target modules for LLaMA/Mistral-like architectures
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    print(f"[llm_loader] Loading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Ensure pad token exists
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token

    print(f"[llm_loader] Loading LLM: {model_name} (4bit={load_in_4bit})")

    kwargs = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if load_in_4bit:
        # bitsandbytes 4-bit quantization
        kwargs.update(dict(load_in_4bit=True))
    else:
        kwargs.update(dict(load_in_4bit=False))

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.train()

    if not PEFT_OK:
        raise RuntimeError(
            "[llm_loader] peft is not available. Install with: pip install peft"
        )

    # If k-bit training, prepare
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[llm_loader] LoRA attached. Trainable params: {trainable:,} / Total: {total:,} "
          f"({100.0*trainable/total:.4f}%)")

    return model, tok
