# # <-- U-Net 구조 (몸체)
# G:\VETNet_pilot\models\vetnet_backbone.py
# G:\VETNet_pilot\models\vetnet_backbone.py (수정 버전)
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================================
# [의존성 1] film_volterra_block.py 에서 임포트되는 FiLM_VolterraBlock 스텁
# (FiLM 연산이 실제로 일어나도록 수정)
# ====================================================================
class FiLM_VolterraBlock(nn.Module):
    """ 테스트를 위한 FiLM_VolterraBlock 스텁 """
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False, volterra_rank=4):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 1) 
    
    def forward(self, x, gamma=None, beta=None):
        # 💡 수정된 부분: FiLM 변조가 Feature Map에 직접 적용되도록 함
        if gamma is not None and beta is not None:
             x_mod = x * gamma + beta # FiLM 연산 적용
        else:
             x_mod = x
             
        # Residual Connection
        return x + self.conv(x_mod) 

# ====================================================================
# [의존성 2] Restormer 기본 컴포넌트 (유지)
# ... Downsample, Upsample 클래스 유지 ...
# ====================================================================

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1), 
            nn.PixelShuffle(2)
        )
    def forward(self, x):
        return self.body(x)

# ====================================================================
# Core Component: Encoder / Decoder (FiLM 블록 시퀀스) (유지)
# ... Encoder, Decoder 클래스 유지 ...
# ====================================================================

class Encoder(nn.Module):
    """
    FiLM_VolterraBlock으로 구성된 인코더 시퀀스. 
    forward 시 gamma, beta를 필수로 받습니다.
    """
    def __init__(self, dim, depth, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            FiLM_VolterraBlock(dim, **kwargs) for _ in range(depth)
        ])

    def forward(self, x, gamma, beta): 
        for block in self.blocks:
            x = block(x, gamma, beta)
        return x

class Decoder(nn.Module):
    """
    FiLM_VolterraBlock으로 구성된 디코더 시퀀스. 
    내부적으로 중립 FiLM 파라미터(1, 0)를 생성하여 블록에 전달합니다.
    """
    def __init__(self, dim, depth, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            FiLM_VolterraBlock(dim, **kwargs) for _ in range(depth)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        gamma = torch.ones_like(x[:, 0:1, :, :]).repeat(1, C, 1, 1)
        beta = torch.zeros_like(x[:, 0:1, :, :]).repeat(1, C, 1, 1)

        for block in self.blocks:
            x = block(x, gamma, beta)
        return x


# ====================================================================
# VETNet-Pilot Backbone: Restormer-Volterra (U-Net) (유지)
# ... RestormerVolterra 클래스 유지 ...
# ====================================================================

class RestormerVolterra(nn.Module):
    """
    VETNet-Pilot의 메인 복원 네트워크 (Restormer 기반 U-Net 구조)
    """
    def __init__(self, in_channels=3, out_channels=3, dim=48, 
                 num_blocks=[4,6,6,8], heads=[1,2,4,8], **kwargs):
        super().__init__()
        
        self.dim = dim

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        
        # ----------------- Encoder -----------------
        self.encoder1 = Encoder(dim, num_blocks[0], num_heads=heads[0], **kwargs)
        self.down1 = Downsample(dim)
        
        self.encoder2 = Encoder(dim*2, num_blocks[1], num_heads=heads[1], **kwargs)
        self.down2 = Downsample(dim*2)
        
        self.encoder3 = Encoder(dim*4, num_blocks[2], num_heads=heads[2], **kwargs)
        self.down3 = Downsample(dim*4)

        # ----------------- Latent/Bottleneck -----------------
        self.latent = Encoder(dim*8, num_blocks[3], num_heads=heads[3], **kwargs)

        # ----------------- Decoder -----------------
        self.up3 = Upsample(dim*8, dim*4) 
        self.decoder3 = Decoder(dim*4, num_blocks[2], num_heads=heads[2], **kwargs)
        
        self.up2 = Upsample(dim*4, dim*2) 
        self.decoder2 = Decoder(dim*2, num_blocks[1], num_heads=heads[1], **kwargs)

        self.up1 = Upsample(dim*2, dim) 
        self.decoder1 = Decoder(dim, num_blocks[0], num_heads=heads[0], **kwargs)
        
        # ----------------- Refinement & Output -----------------
        self.refinement = Encoder(dim, num_blocks[0], num_heads=heads[0], **kwargs) 

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    def _pad_and_add(self, up_tensor, skip_tensor):
        if up_tensor.shape[-2:] != skip_tensor.shape[-2:]:
            up_tensor = F.interpolate(up_tensor, size=skip_tensor.shape[-2:], mode='bilinear', align_corners=False)
        return up_tensor + skip_tensor

    def forward(self, x, film_params):
        (gamma1, beta1), (gamma2, beta2), (gamma3, beta3), (gamma4, beta4) = film_params
        
        x_embed = self.patch_embed(x) 

        # 2. Encoder Path (FiLM 제어 적용)
        x2 = self.encoder1(x_embed, gamma1, beta1) 
        x3 = self.encoder2(self.down1(x2), gamma2, beta2) 
        x4 = self.encoder3(self.down2(x3), gamma3, beta3) 
        
        # 3. Latent/Bottleneck (FiLM 제어 적용)
        x5 = self.latent(self.down3(x4), gamma4, beta4) 

        # 4. Decoder Path (Decoder.forward에서 중립 파라미터 자동 처리)
        x6 = self.decoder3(self._pad_and_add(self.up3(x5), x4)) 
        x7 = self.decoder2(self._pad_and_add(self.up2(x6), x3)) 
        x8 = self.decoder1(self._pad_and_add(self.up1(x7), x2)) 
        
        # 5. Refinement (Encoder 클래스 사용. 중립 파라미터를 명시적으로 생성 및 전달)
        neutral_gamma_refine = torch.ones_like(x_embed[:, 0:self.dim, :, :])
        neutral_beta_refine = torch.zeros_like(x_embed[:, 0:self.dim, :, :])
        
        x9 = self.refinement(x8, neutral_gamma_refine, neutral_beta_refine)

        # 6. Output (Residual Learning 적용)
        out = self.output(x9 + x_embed)
        return out


# ====================================================================
# 코드 검증 및 테스트 (유지)
# ... create_dummy_film_params 함수 및 if __name__ == '__main__': 블록 유지 ...
# ====================================================================

def create_dummy_film_params(dim):
    dims = [dim, dim * 2, dim * 4, dim * 8]
    params = []
    
    for i, c in enumerate(dims):
        gamma = torch.ones(1, c, 1, 1) 
        beta = torch.zeros(1, c, 1, 1)
        
        gamma[0, :min(10, c), 0, 0] = 1.0 + 0.1 * (i + 1)
        beta[0, :min(10, c), 0, 0] = 0.05 * (i + 1)
        
        params.append((gamma, beta))
    return params

if __name__ == '__main__':
    print("--- 3단계: vetnet_backbone.py 코드 검증 시작 ---")
    
    in_channels = 3
    out_channels = 3
    base_dim = 48 
    input_height = 256
    input_width = 384
    
    dummy_image = torch.randn(1, in_channels, input_height, input_width)
    print(f"1. 입력 이미지 형태 (x): {dummy_image.shape}")

    dummy_film_params = create_dummy_film_params(base_dim)
    print(f"2. FiLM 파라미터 셋 생성 완료. (4쌍)")
    print(f"   -> E1 파라미터 채널 크기: {dummy_film_params[0][0].shape[1]}") 
    print(f"   -> E4 파라미터 채널 크기: {dummy_film_params[3][0].shape[1]}") 

    model = RestormerVolterra(in_channels=in_channels, out_channels=out_channels, dim=base_dim)
    print(f"3. RestormerVolterra 모델 초기화 완료. (Base Dim: {base_dim})")
    
    # 4. 순전파 (Forward Pass) 실행
    try:
        output = model(dummy_image, dummy_film_params)
        
        # 5. 결과 확인
        print("\n--- 순전파 결과 ---")
        print(f"4. 최종 출력 이미지 형태 (y_hat): {output.shape}")
        
        assert output.shape == dummy_image.shape, "입력과 최종 출력의 형태가 일치하지 않습니다! (해상도 오류 발생)"
        print("5. 입력과 출력 형태 일치 확인: 성공")

        # FiLM이 실제로 적용되었는지 간접 확인 (모든 FiLM을 중립으로 설정하고 비교)
        # Note: neutral_film_params를 다시 생성할 때, gamma/beta는 1/0으로 고정해야 함
        neutral_film_params = [(torch.ones_like(g), torch.zeros_like(b)) for g, b in dummy_film_params]
        output_neutral = model(dummy_image, neutral_film_params)
        
        diff = torch.abs(output - output_neutral).sum()
        print(f"6. FiLM 변조 전/후 출력 차이 (L1 Sum): {diff.item():.4f}")
        
        # FiLM 파라미터에 변화를 주었으므로, 차이가 0보다 커야 합니다.
        if diff.item() > 0.0: 
            print("   -> FiLM 제어 신호가 성공적으로 VETNet Backbone에 적용되었습니다. (정상 작동)")
        else:
             print("   -> 경고: FiLM 변조가 적용되었으나 출력 차이가 발생하지 않았습니다. (스텁의 문제일 가능성 높음)")
            
    except Exception as e:
        print(f"\n--- 순전파 중 오류 발생 ---")
        print(f"오류: {e}")
        
    print("\n--- 3단계: vetnet_backbone.py 코드 검증 완료 ---")