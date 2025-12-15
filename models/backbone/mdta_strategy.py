# G:/VETNet_pilot/models/backbone/mdta_strategy.py
# phase -1 (vetnet backbone)
""" import torch
import torch.nn as nn
import torch.nn.functional as F

class MDTA(nn.Module):

    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):

        B, C, H, W = x.shape
        N = H * W
        C_head = C // self.num_heads

        # 1. QKV 생성
        q, k, v = self.qkv(x).chunk(3, dim=1)  # (B, C, H, W)

        # 2. (B, C, H, W) → (B, Head, C_head, N)
        q = q.view(B, self.num_heads, C_head, N)
        k = k.view(B, self.num_heads, C_head, N)
        v = v.view(B, self.num_heads, C_head, N)

        # 3. 채널 방향(C_head)에서 어텐션 계산
        #    attn: (B, Head, C_head, C_head)
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)

        attn = torch.matmul(q_norm, k_norm.transpose(-1, -2)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 4. attn을 v에 적용 → (B, Head, C_head, N)
        out = torch.matmul(attn, v)

        # 5. (B, Head, C_head, N) → (B, C, H, W)
        out = out.view(B, C, H, W)
        out = self.project_out(out)
        return out


# -----------------------------------------------------------
# 아래는 Phase2용 Strategy Token 버전 (지금은 안 써도 됨)
# -----------------------------------------------------------

class MDTAWithStrategy(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.project_out = nn.Linear(dim, dim, bias=bias)

    def forward(self, X_flat, S_tokens=None):
        if S_tokens is not None:
            X_in = torch.cat([X_flat, S_tokens], dim=1)  # (B, N+K, C)
        else:
            X_in = X_flat

        B, N_tot, C = X_in.shape

        qkv = self.qkv(X_in)
        q, k, v = qkv.chunk(3, dim=-1)

        # 여기서는 간단한 token-wise attention (N_tot × N_tot)
        # → Phase2에서 windowing / 축소 고려
        q = q.view(B, N_tot, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, N_tot, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, N_tot, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)  # (B, H, N_tot, d)
        out = out.transpose(1, 2).contiguous().view(B, N_tot, C)
        out = self.project_out(out)

        # 이미지 토큰 부분만 반환 (앞의 N 개)
        return out[:, :X_flat.shape[1], :]


# ----------------- 테스트 코드 ----------------- #
if __name__ == "__main__":
    print("=== MDTA (Channel-wise Transposed) Test ===")
    x = torch.randn(1, 48, 256, 256)
    attn = MDTA(dim=48, num_heads=8)
    y = attn(x)
    print(f"Input  Shape: {x.shape}")
    print(f"Output Shape: {y.shape}")

    print("\n=== MDTAWithStrategy (Linear-based) Test ===")
    B, N, C = 2, 64, 48
    K = 4
    X_flat = torch.randn(B, N, C)
    S = torch.randn(B, K, C)
    attn_s = MDTAWithStrategy(dim=C)
    Y_flat = attn_s(X_flat, S_tokens=S)
    print(f"X_flat Shape: {X_flat.shape}")
    print(f"S_tokens Shape: {S.shape}")
    print(f"Y_flat Shape: {Y_flat.shape}")
 """


# G:/VETNet_pilot/models/backbone/mdta_strategy.py
# phase -2 (control bridge)
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Phase-1 MDTA (original: channel-wise transposed attention)
# ============================================================
class MDTA(nn.Module):
    """
    Multi-DConv Head Transposed Attention (channel-wise attention version)
    - input : (B, C, H, W)
    - attention computed in channel dimension (C_head x C_head) per head
    """
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        assert dim % num_heads == 0, f"[MDTA] dim({dim}) must be divisible by num_heads({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W
        C_head = C // self.num_heads

        q, k, v = self.qkv(x).chunk(3, dim=1)  # (B, C, H, W)

        # (B, C, H, W) -> (B, Head, C_head, N)
        q = q.view(B, self.num_heads, C_head, N)
        k = k.view(B, self.num_heads, C_head, N)
        v = v.view(B, self.num_heads, C_head, N)

        # normalize along token dimension (N)
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)

        # channel-wise attention: (B, Head, C_head, C_head)
        attn = torch.matmul(q_norm, k_norm.transpose(-1, -2)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)  # (B, Head, C_head, N)
        out = out.view(B, C, H, W)   # merge heads
        out = self.project_out(out)
        return out


# ============================================================
# Phase-2 MDTAWithStrategy (token concat + MDTA attention)
# ============================================================
class MDTAWithStrategy(nn.Module):
    """
    Phase-2 Strategy Token injection MDTA (token-wise attention).

    Required core logic:
        X ∈ R^{B×N×C}
        S ∈ R^{B×K×C}

        X_in = concat([X, S], dim=1)      # (B, N+K, C)
        Y = MDTA(X_in)
        Y_img = Y[:, :N, :]              # keep only image tokens

    Notes:
    - This module is used after feature flattening (H*W -> N tokens).
    - Strategy tokens "steer" attention but are not forwarded to next layers.
    """
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        assert dim % num_heads == 0, f"[MDTAWithStrategy] dim({dim}) must be divisible by num_heads({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # per-head temperature (broadcast to (B, H, N, N))
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.project_out = nn.Linear(dim, dim, bias=bias)

    def forward(self, X_flat: torch.Tensor, S_tokens: torch.Tensor = None) -> torch.Tensor:
        """
        X_flat : (B, N, C) image tokens
        S_tokens: (B, K, C) strategy tokens (optional)

        return : (B, N, C) image tokens only
        """
        if X_flat.dim() != 3:
            raise ValueError(f"[MDTAWithStrategy] X_flat must be (B,N,C). Got: {tuple(X_flat.shape)}")

        B, N, C = X_flat.shape
        if C != self.dim:
            raise ValueError(f"[MDTAWithStrategy] X_flat C({C}) != self.dim({self.dim})")

        # 1) concat strategy tokens (if provided)
        if S_tokens is not None:
            if S_tokens.dim() != 3:
                raise ValueError(f"[MDTAWithStrategy] S_tokens must be (B,K,C). Got: {tuple(S_tokens.shape)}")
            if S_tokens.size(0) != B or S_tokens.size(2) != C:
                raise ValueError(
                    f"[MDTAWithStrategy] S_tokens must be (B,K,C) with same B,C as X_flat. "
                    f"X_flat={tuple(X_flat.shape)} S_tokens={tuple(S_tokens.shape)}"
                )
            X_in = torch.cat([X_flat, S_tokens], dim=1)  # (B, N+K, C)
        else:
            X_in = X_flat                                # (B, N, C)

        N_tot = X_in.size(1)

        # 2) QKV
        qkv = self.qkv(X_in)               # (B, N_tot, 3C)
        q, k, v = qkv.chunk(3, dim=-1)     # each: (B, N_tot, C)

        # 3) reshape to multi-head: (B, H, N_tot, d)
        q = q.view(B, N_tot, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_tot, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_tot, self.num_heads, self.head_dim).transpose(1, 2)

        # 4) normalize (Restormer-style)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 5) attention over tokens: (B, H, N_tot, N_tot)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 6) apply attention: (B, H, N_tot, d)
        out = torch.matmul(attn, v)

        # 7) merge heads: (B, N_tot, C)
        out = out.transpose(1, 2).contiguous().view(B, N_tot, C)
        out = self.project_out(out)  # (B, N_tot, C)

        # 8) keep only image tokens
        Y_img = out[:, :N, :]
        return Y_img


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print("[mdta_strategy] Self-test 시작")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[mdta_strategy] Device =", device)

    # ---------------------------
    # Test A: Phase-1 MDTA (conv)
    # ---------------------------
    print("\n=== Test A: MDTA (Phase-1 conv) ===")
    x = torch.randn(1, 48, 64, 64, device=device)
    attn = MDTA(dim=48, num_heads=8).to(device)
    y = attn(x)
    print("Input :", tuple(x.shape))
    print("Output:", tuple(y.shape))

    # ---------------------------
    # Test B: Phase-2 MDTAWithStrategy (token concat)
    # ---------------------------
    print("\n=== Test B: MDTAWithStrategy (Phase-2 token concat) ===")
    B, N, C = 2, 1024, 64
    K = 4
    X_flat = torch.randn(B, N, C, device=device)
    S = torch.randn(B, K, C, device=device)

    mdta_s = MDTAWithStrategy(dim=C, num_heads=8).to(device)
    mdta_s.eval()

    with torch.no_grad():
        y_noS = mdta_s(X_flat, S_tokens=None)
        y_withS = mdta_s(X_flat, S_tokens=S)

    print("X_flat :", tuple(X_flat.shape))
    print("S      :", tuple(S.shape))
    print("Y(no S):", tuple(y_noS.shape))
    print("Y(with):", tuple(y_withS.shape))

    # strategy가 실제로 영향을 주는지 간단 체크
    diff = (y_withS - y_noS).abs().mean().item()
    print(f"[mdta_strategy] mean(|Y_withS - Y_noS|) = {diff:.6f} (0이 아니면 steering 영향 있음)")

    print("\n[mdta_strategy] Self-test 완료")
