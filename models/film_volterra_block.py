# # <-- FiLM 적용된 Volterra 블록
# G:\VETNet_pilot\models\film_volterra_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

# ----------------- Dependencies Stubs (Shape Consistency Check용) ----------------- #
# 실제 구현 시, 이 클래스들은 별도의 파일에서 정확히 임포트되어야 합니다.

# [의존성 1] volterra_layer.py에서 임포트되는 VolterraLayer2D
class VolterraLayer2D(nn.Module):
    """
    film_volterra_block.py 테스트를 위한 VolterraLayer2D 임시 스텁 (Shape Check용)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, rank=4, bias=True):
        super().__init__()
        # Volterra 연산 대신, 형태 유지를 위한 단순 Conv1x1을 사용합니다.
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias) 
    def forward(self, x):
        return self.conv(x)

# [의존성 2] Restormer의 MDTA (Multi-Dconv Head Attention) 임시 스텁
class MDTA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 1) # 단순 투영 (Attention 결과물)
    def forward(self, x):
        return self.proj(x)

# [의존성 3] Restormer의 GDFN (Gated D-conv FFN) 임시 스텁
class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.conv1 = nn.Conv2d(dim, hidden_dim * 2, 1)
        self.proj = nn.Conv2d(hidden_dim, dim, 1)
    def forward(self, x):
        x1, x2 = self.conv1(x).chunk(2, dim=1) 
        x = x1 * F.sigmoid(x2) # Gating 메커니즘 흉내
        return self.proj(x)

# ----------------- Core Component: FiLM_VolterraBlock ----------------- #

class FiLM_VolterraBlock(nn.Module):
    """
    FiLM 파라미터(gamma, beta)를 입력받아 LayerNorm 후 Feature를 변조하는
    적응형 Volterra Transformer Block입니다.
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False, volterra_rank=4):
        super().__init__()

        # Sub-modules for Attention Path
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = MDTA(dim=dim)
        # Attention Path 후 Volterra Layer 적용
        self.volterra1 = VolterraLayer2D(in_channels=dim, out_channels=dim, rank=volterra_rank, bias=bias)

        # Sub-modules for FFN Path
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.ffn = GDFN(dim=dim, ffn_expansion_factor=ffn_expansion_factor)
        # FFN Path 후 Volterra Layer 적용
        self.volterra2 = VolterraLayer2D(in_channels=dim, out_channels=dim, rank=volterra_rank, bias=bias)

    def forward(self, x, gamma, beta):
        # x: (B, C, H, W)
        # gamma, beta: (B, C, 1, 1) or (1, C, 1, 1)

        # 1. Attention Path (MDTA + Volterra)
        
        # LayerNorm 적용
        x_norm1 = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # NCHW -> NHWC -> NCHW
        
        # FiLM 변조 적용: F_mod = F_norm * gamma + beta
        x_mod1 = x_norm1 * gamma + beta 
        
        # MDTA 및 Volterra 연산
        x1 = self.volterra1(self.attn(x_mod1))
        x = x + x1 # Residual Connection

        # 2. FFN Path (GDFN + Volterra)

        # LayerNorm 적용
        x_norm2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # FiLM 변조 적용
        x_mod2 = x_norm2 * gamma + beta 
        
        # GDFN 및 Volterra 연산
        x2 = self.volterra2(self.ffn(x_mod2))
        x = x + x2 # Residual Connection
        
        return x

# ----------------- 코드 검증 및 테스트 ----------------- #

if __name__ == '__main__':
    print("--- FiLM_VolterraBlock 코드 검증 시작 ---")
    
    # 설정 변수
    batch_size = 4
    channels = 64  # dim (Feature Map Channel 수)
    height = 32
    width = 32
    num_heads = 4
    
    # 1. 더미 입력 데이터 생성 (NCHW 포맷)
    dummy_feature_map = torch.randn(batch_size, channels, height, width)
    print(f"1. 입력 Feature Map 형태 (x): {dummy_feature_map.shape}")
    
    # 2. FiLM 파라미터 (gamma, beta) 생성
    # FiLM 파라미터는 (Batch, Channel, 1, 1) 형태로 Broadcasting 가능해야 합니다.
    # 여기서는 Batch 독립적으로 (1, C, 1, 1)로 생성합니다.
    dummy_gamma = torch.ones(1, channels, 1, 1)
    dummy_beta = torch.zeros(1, channels, 1, 1)
    
    # gamma에 작은 값 변화를 줘서 FiLM이 작동하는지 확인
    dummy_gamma[0, 0:10, 0, 0] = 1.5 
    dummy_beta[0, 10:20, 0, 0] = 0.1
    
    print(f"2. FiLM 파라미터 형태 (gamma/beta): {dummy_gamma.shape}")
    
    # 3. FiLM_VolterraBlock 모델 인스턴스 생성
    model = FiLM_VolterraBlock(dim=channels, num_heads=num_heads, ffn_expansion_factor=2.0)
    print(f"3. FiLM_VolterraBlock 모델 초기화 완료.")
    
    # 4. 순전파 (Forward Pass) 실행
    try:
        output = model(dummy_feature_map, dummy_gamma, dummy_beta)
        
        # 5. 결과 확인
        print("\n--- 순전파 결과 ---")
        print(f"4. 출력 Feature Map 형태: {output.shape}")
        
        # 입력과 출력 형태가 동일해야 Residual Connection이 올바르게 작동함을 의미
        assert output.shape == dummy_feature_map.shape, "입력과 출력의 형태가 일치하지 않습니다!"
        
        # FiLM 변조가 실제로 적용되었는지 확인 (선형 항만 계산 시와 비교)
        # FiLM이 gamma=1, beta=0일 때 (변조 없음)와 비교하여 값이 달라지는지 확인
        output_neutral = model(dummy_feature_map, torch.ones_like(dummy_gamma), torch.zeros_like(dummy_beta))
        
        diff = torch.abs(output - output_neutral).sum()
        print(f"5. FiLM 변조 전/후 출력 차이 (L1 Sum): {diff.item():.4f}")
        
        # FiLM 파라미터에 변화를 주었으므로, 차이가 0보다 커야 합니다.
        if diff.item() > 0.01: 
            print("   -> FiLM 변조가 성공적으로 적용되어 출력에 변화를 주었습니다. (정상 작동)")
        else:
            print("   -> 경고: FiLM 변조가 적용되었으나 출력 차이가 미미합니다.")
            
    except Exception as e:
        print(f"\n--- 순전파 중 오류 발생 ---")
        print(f"오류: {e}")
        
    print("\n--- FiLM_VolterraBlock 코드 검증 완료 ---")