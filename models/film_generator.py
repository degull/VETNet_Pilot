# # <-- Z -> (gamma, beta) 변환 (제어)
# G:\VETNet_pilot\models\film_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# ----------------- Core Component: FiLMGenerator ----------------- #

class FiLMGenerator(nn.Module):
    """
    Context Vector Z를 받아 VETNet의 4개 Encoder Stage에 필요한 
    FiLM 파라미터 (gamma, beta) 셋을 생성하는 Multi-Head MLP입니다.
    """
    def __init__(self, 
                 z_dim: int = 2048,                 # Context Vector Z의 입력 차원
                 base_dim: int = 48,                # VETNet의 기본 채널 크기 (C)
                 num_stages: int = 4,               # Encoder Stage의 개수 (1, 2, 3, 4)
                 shared_hidden_dim: int = 512):     # 공유 MLP의 중간 차원
        super().__init__()

        self.z_dim = z_dim
        self.num_stages = num_stages
        
        # VETNet 각 Stage의 채널 크기 (C, 2C, 4C, 8C)
        self.channel_dims = [base_dim * (2 ** i) for i in range(num_stages)]
        
        # 1. Shared MLP (공통 처리)
        self.shared_mlp = nn.Sequential(
            nn.Linear(z_dim, shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1) # 과적합 방지용
        )

        # 2. Multi-Head MLP (개별 Stage 제어 파라미터 생성)
        # 각 Head는 해당 Stage의 채널 수 Ci에 대해 2 * Ci 차원을 출력합니다 (gamma_i + beta_i).
        self.head_layers = nn.ModuleList()
        for c_dim in self.channel_dims:
            # 출력 차원: gamma (C_i) + beta (C_i) = 2 * C_i
            self.head_layers.append(
                nn.Linear(shared_hidden_dim, 2 * c_dim)
            )

    def forward(self, Z: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # Z shape: (B, z_dim) - Context Vector
        
        # 1. Shared MLP 통과
        Z_shared = self.shared_mlp(Z) # (B, shared_hidden_dim)
        
        film_params = []
        
        # 2. Multi-Head에서 각 Stage의 gamma, beta 생성
        for idx, head in enumerate(self.head_layers):
            # Head 출력: (B, 2 * C_i)
            output = head(Z_shared)
            
            # C_i 차원
            c_dim = self.channel_dims[idx]
            
            # Gamma (Scale) 및 Beta (Shift) 분리
            # output[:, :c_dim] -> gamma (C_i 차원)
            # output[:, c_dim:] -> beta (C_i 차원)
            gamma_flat = output[:, :c_dim]
            beta_flat = output[:, c_dim:]
            
            # FiLM 파라미터를 NCHW 포맷에 맞게 (B, C, 1, 1)로 변환 (Broadcasting 준비)
            gamma = gamma_flat.unsqueeze(-1).unsqueeze(-1)
            beta = beta_flat.unsqueeze(-1).unsqueeze(-1)
            
            film_params.append((gamma, beta))
            
        # film_params: [(gamma1, beta1), (gamma2, beta2), (gamma3, beta3), (gamma4, beta4)]
        return film_params

# ----------------- 코드 검증 및 테스트 ----------------- #

if __name__ == '__main__':
    print("--- 5단계: film_generator.py 코드 검증 시작 ---")
    
    # 설정 변수 (설계서 기반)
    Z_DIM = 2048        # VLLM Pilot의 출력 차원
    BASE_DIM = 48       # VETNet의 기본 채널 크기 (C)
    
    # 1. 더미 Context Vector Z 생성
    batch_size = 2
    dummy_Z = torch.randn(batch_size, Z_DIM)
    print(f"1. 입력 Context Vector Z 형태: {dummy_Z.shape}")
    
    # 2. FiLMGenerator 모델 인스턴스 생성
    model = FiLMGenerator(z_dim=Z_DIM, base_dim=BASE_DIM)
    
    # 3. 순전파 (Forward Pass) 실행
    try:
        film_params = model(dummy_Z)
        
        # 4. 결과 확인
        print("\n--- 순전파 결과 ---")
        
        # 예상 채널 차원
        expected_dims = [BASE_DIM * (2 ** i) for i in range(4)]
        
        # 5. 생성된 파라미터 셋 검증
        for i, (gamma, beta) in enumerate(film_params):
            expected_c = expected_dims[i]
            
            # NCHW 포맷 검증 (B, C, 1, 1)
            expected_shape = torch.Size([batch_size, expected_c, 1, 1])
            
            assert gamma.shape == expected_shape, f"Stage {i+1} gamma 형태 오류! 예상: {expected_shape}, 실제: {gamma.shape}"
            assert beta.shape == expected_shape, f"Stage {i+1} beta 형태 오류! 예상: {expected_shape}, 실제: {beta.shape}"
            
            print(f"5. Stage {i+1} 파라미터 형태 (C={expected_c}): {gamma.shape} 일치 확인")

        print("   -> FiLM 파라미터 셋 생성 및 형태 검증: 성공")

        # 6. 학습 가능 파라미터 수 검증 (이 모듈은 Phase 2에서 학습됨)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"6. FiLMGenerator 전체 파라미터 수: {total_params}")
        
    except Exception as e:
        print(f"\n--- 순전파 중 오류 발생 ---")
        print(f"오류: {e}")
        
    print("\n--- 5단계: film_generator.py 코드 검증 완료 ---")