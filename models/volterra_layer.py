# <-- Volterra 비선형 필터 (기반 기술)
# G:\VETNet_pilot\models\volterra_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------- Utility Function ----------------- #

def circular_shift(x, shift_x, shift_y):
    """
    Feature Map을 2D 평면에서 순환 이동(Circular Shift)시키는 함수.
    Volterra Layer의 2차 항 계산을 위한 구성 요소로 사용됩니다.
    x: (B, C, H, W) 텐서
    shift_x: X축 이동량
    shift_y: Y축 이동량
    """
    # 텐서의 3번째 차원 (Height)과 4번째 차원 (Width)에 대해 이동을 적용
    return torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))

# ----------------- Core Layer ----------------- #

class VolterraLayer2D(nn.Module):
    """
    2차 볼테라 필터링을 구현하는 레이어.
    출력 = H1*x + H2*x^2 (선형 항 + 2차 비선형 항)
    제공된 설계서의 'use_lossless=False' 버전 구현을 따름 (Rank decomposition).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, rank=4, bias=True):
        super().__init__()
        self.rank = rank
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 1. 선형 항 (H1*x) - 일반적인 1차 컨볼루션
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.padding, bias=bias)

        # 2. 2차 비선형 항 (H2*x^2) - Rank Decomposition을 사용한 구현
        # 2차 볼테라 커널을 두 개의 랭크 R 컨볼루션으로 분해하여 근사합니다.
        # H2(x1, x2) ≈ sum_r (W2a_r * x1) * (W2b_r * x2)
        
        # W2a: Rank R의 컨볼루션 목록
        self.W2a = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.padding, bias=bias)
            for _ in range(rank)
        ])
        
        # W2b: Rank R의 컨볼루션 목록
        self.W2b = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.padding, bias=bias)
            for _ in range(rank)
        ])

    def forward(self, x):
        # 1. 선형 항 계산
        linear_term = self.conv1(x)
        
        # 2. 2차 비선형 항 초기화
        quadratic_term = 0
        
        # 3. 랭크 분해를 이용한 2차 항 계산
        for a, b in zip(self.W2a, self.W2b):
            # 활성화 함수 적용 (일반적으로 tanh나 sigmoid 같은 비선형성이 사용되지만,
            # 원 논문 구현을 따라 clamp를 사용하여 범위를 제한할 수 있습니다. 
            # 여기서는 Volterra Layer의 전형적인 tanh/ReLU 대신 clamping을 사용합니다.)
            
            # Note: Clamp 범위는 훈련의 안정성을 위해 -1.0 ~ 1.0으로 설정 (선택적)
            qa = torch.clamp(a(x), min=-1.0, max=1.0) 
            qb = torch.clamp(b(x), min=-1.0, max=1.0) 
            
            # 두 컨볼루션 출력의 요소별 곱셈 (비선형 결합)
            quadratic_term += qa * qb

        # 4. 최종 출력: 선형 항 + 2차 항
        out = linear_term + quadratic_term
        return out


# ----------------- 선택적 Lossless Volterra 구현 (참고용) ----------------- #

class VolterraLayer2D_Lossless(VolterraLayer2D):
    """
    'circular_shift' 유틸리티를 사용하여 Lossless Volterra 연산을 구현한 버전 (참고).
    FiLM_VolterraBlock은 기본적으로 VolterraLayer2D를 사용합니다.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        # Lossless 모드에서는 Rank가 필요 없습니다.
        super(VolterraLayer2D, self).__init__() 
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.padding, bias=bias)
        self.shifts = self._generate_shifts(kernel_size)
        
    def _generate_shifts(self, k):
        # 2차 항 계산을 위한 모든 가능한 시프트 쌍을 생성
        P = k // 2
        shifts = []
        for s1 in range(-P, P + 1):
            for s2 in range(-P, P + 1):
                # 대칭성 때문에 (0,0)과 중복 쌍 제외
                if s1 == 0 and s2 == 0: continue
                if (s1, s2) < (0, 0): continue
                shifts.append((s1, s2))
        return shifts

    def forward(self, x):
        linear_term = self.conv1(x)
        quadratic_term = 0
        
        for s1, s2 in self.shifts:
            x_shifted = circular_shift(x, s1, s2)
            prod = x * x_shifted
            prod = torch.clamp(prod, min=-1.0, max=1.0)
            quadratic_term += self.conv2(prod)
            
        return linear_term + quadratic_term

# ----------------- 테스트 코드 ----------------- #
if __name__ == '__main__':
    # 테스트를 위한 더미 입력 (Batch=1, Channel=3, Height=64, Width=64)
    dummy_input = torch.randn(1, 32, 64, 64)
    in_c = 32
    out_c = 32
    
    # Rank Decomposition Volterra Layer 테스트
    model_rank = VolterraLayer2D(in_channels=in_c, out_channels=out_c, rank=4)
    output_rank = model_rank(dummy_input)
    
    print(f"Rank-Decomposition Volterra Input Shape: {dummy_input.shape}")
    print(f"Rank-Decomposition Volterra Output Shape: {output_rank.shape}")
    
    # Lossless Volterra Layer 테스트 (선택적)
    model_lossless = VolterraLayer2D_Lossless(in_channels=in_c, out_channels=out_c)
    output_lossless = model_lossless(dummy_input)
    
    print(f"Lossless Volterra Input Shape: {dummy_input.shape}")
    print(f"Lossless Volterra Output Shape: {output_lossless.shape}")