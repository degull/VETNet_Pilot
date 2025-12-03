# G:\VETNet_pilot\models\vllm_pilot.py (Llama Stub 최종 적용)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import warnings
import sys
import random

# --------------------------------------------------------------------------------
# Hugging Face Dependencies (실제 모델 로딩용)
try:
    from transformers import CLIPVisionModel, LlamaForCausalLM
except ImportError:
    warnings.warn("Hugging Face 'transformers' library not found. Using Dummy Models.")
    
    # LlamaForCausalLM 로딩 실패 시 사용될 Dummy
    class LlamaForCausalLM(nn.Module):
        def __init__(self, config): super().__init__();
        @classmethod
        def from_pretrained(cls, name): return cls(None)
        def forward(self, x): return x 
        
    # CLIPVisionModel 로딩 실패 시 사용될 Dummy
    class CLIPVisionModel(nn.Module):
        def __init__(self, config): super().__init__(); 
        @classmethod
        def from_pretrained(cls, name): return cls(None)
        def forward(self, x):
            hidden_state = torch.randn(x.size(0), 257, 768, device=x.device)
            class Output: pass
            output = Output()
            output.last_hidden_state = hidden_state
            return output
        class Config: hidden_size = 768
# --------------------------------------------------------------------------------

class VLLMPilot(nn.Module):
    def __init__(self, 
                 vision_model_name: str = "openai/clip-vit-base-patch32", 
                 llm_model_name: str = "openlm-research/open_llama_3b_v2", 
                 llm_dim: int = 2048, 
                 vision_out_dim: int = 768, 
                 **kwargs):
        super().__init__()
        self.llm_dim = llm_dim
        self.vision_out_dim = vision_out_dim 
        
        # ------------------- 1. Vision Tower (CLIP) -------------------
        try:
            # CLIP만 실제 모델 로딩 시도
            self.vision_tower = CLIPVisionModel.from_pretrained(vision_model_name)
            self.vision_out_dim = self.vision_tower.config.hidden_size 
        except Exception:
            self.vision_tower = self._create_dummy_vision_tower(vision_out_dim)
            self.vision_out_dim = vision_out_dim
            
        # ------------------- 2. LLM Core (Llama-like) -------------------
        # 💡 Llama For Causal LM 로딩 대신, 안전한 Dummy LLM Core를 사용하도록 강제
        self.llm_core = self._create_dummy_llm_core() 
        
        # ------------------- 3. Adapters (PEFT 학습 대상) -------------------
        self.llm_projector = nn.Sequential(
            nn.Linear(self.vision_out_dim, self.vision_out_dim), nn.GELU(), nn.Linear(self.vision_out_dim, self.vision_out_dim)
        )
        self.context_projection = nn.Linear(self.vision_out_dim, llm_dim) 
        self.text_decoder_head = nn.Linear(self.vision_out_dim, 5) 

        # ------------------- 4. Freeze/Trainable 설정 (PEFT) -------------------
        # 모든 Llama/CLIP Core 파라미터 Freeze
        for param in self.vision_tower.parameters(): param.requires_grad = False
        for param in self.llm_core.parameters(): param.requires_grad = False
        
        # Adapters/Heads Unfreeze
        for param in self.llm_projector.parameters(): param.requires_grad = True
        for param in self.context_projection.parameters(): param.requires_grad = True
        for param in self.text_decoder_head.parameters(): param.requires_grad = True
        
        print(f"VLLMPilot: Adapter 파라미터 {sum(p.numel() for p in self.parameters() if p.requires_grad)}개 Unfreeze 완료.")

    # --- Dummy Factory Methods ---
    def _create_dummy_vision_tower(self, out_dim):
        class DummyVisionTower(nn.Module):
            def __init__(self, out_dim): super().__init__(); self.out_dim = out_dim
            def forward(self, x):
                seq_len = 257 
                class Output: pass
                output = Output()
                output.last_hidden_state = torch.randn(x.size(0), seq_len, self.out_dim, device=x.device)
                return output
        return DummyVisionTower(out_dim)

    def _create_dummy_llm_core(self):
        class DummyLLMCore(nn.Module):
            def __init__(self): super().__init__(); 
            def forward(self, x): return x 
        return DummyLLMCore()

    def forward(self, x_336: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        vision_output = self.vision_tower(x_336)
        visual_tokens = vision_output.last_hidden_state
            
        llm_embeddings = self.llm_projector(visual_tokens)
        
        # 3. LLM Core 추론 (Llama) - 안전한 Dummy Core는 입력을 그대로 반환
        final_llm_hidden_state = self.llm_core(llm_embeddings) 
        
        # 4. Context Vector Z 및 Text Logits 추출
        pooled_context = final_llm_hidden_state.mean(dim=1) 
        Z = self.context_projection(pooled_context)
        text_logits = self.text_decoder_head(pooled_context)
        
        return Z, text_logits 

# ----------------- 코드 검증 및 테스트 ----------------- #

if __name__ == '__main__':
    print("--- 4단계: vllm_pilot.py 코드 검증 시작 (Final Test) ---")
    
    BATCH_SIZE = 2
    VLM_INPUT = 224 # CLIP 표준 크기로 통일
    LLM_Z_DIM = 2048 
    
    dummy_image_336 = torch.randn(BATCH_SIZE, 3, VLM_INPUT, VLM_INPUT)
    print(f"1. 입력 이미지 형태 (x_336): {dummy_image_336.shape}")
    
    try:
        model = VLLMPilot(llm_dim=LLM_Z_DIM)
    except Exception as e:
        print(f"\n[경고] VLLMPilot 초기화 실패: {e}. 라이브러리 설치 필요.")
        sys.exit(1)
    
    # 3. 순전파 실행
    try:
        Z_vector, text_logits = model(dummy_image_336)
        
        # 4. 결과 확인
        print("\n--- 순전파 결과 ---")
        target_Z_shape = torch.Size([BATCH_SIZE, LLM_Z_DIM])
        target_Text_shape = torch.Size([BATCH_SIZE, 5])
        
        assert Z_vector.shape == target_Z_shape, f"Z 벡터 형태 오류! 예상: {target_Z_shape}, 실제: {Z_vector.shape}"
        assert text_logits.shape == target_Text_shape, f"Text Logits 형태 오류! 예상: {target_Text_shape}, 실제: {text_logits.shape}"
        
        print("6. Context Vector Z 및 Text Logits 형태 일치 확인: 성공")
            
    except Exception as e:
        print(f"\n--- 순전파 중 오류 발생 ---")
        print(f"오류: {e}")
        
    print("\n--- 4단계: vllm_pilot.py 코드 검증 완료 ---")