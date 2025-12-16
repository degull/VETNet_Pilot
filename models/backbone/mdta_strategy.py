""" # G:/VETNet_pilot/models/backbone/mdta_strategy.py
# phase -1 (vetnet backbone)
import torch
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
# G:/VETNet_pilot/models/backbone/mdta_strategy.py
# Phase-1 MDTA (conv) + Phase-2 Strategy Steering (OOM-safe)
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# ROOT (디버그 편의)
# -------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))           # .../models/backbone
ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))               # .../VETNet_pilot
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ============================================================
# Phase-1 MDTA (original: channel-wise transposed attention)
# ============================================================
class MDTA(nn.Module):

    def __init__(self, dim: int, num_heads: int = 8, bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0, f"[MDTA] dim({dim}) must be divisible by num_heads({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

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
# Phase-2 MDTAWithStrategy v2 (OOM-safe steering)
# ============================================================
class MDTAWithStrategy(nn.Module):

    def __init__(self, dim: int, num_heads: int = 8, bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0, f"[MDTAWithStrategy] dim({dim}) must be divisible by num_heads({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # temperature per head (broadcast to (B, H, N, K))
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Q from image, K/V from strategy
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)

        self.project_out = nn.Linear(dim, dim, bias=bias)

        # S가 None일 때 fallback (아무 steering 없이 identity-ish)
        # - 계산을 완전히 생략하고 X_flat을 그대로 반환하면
        #   "with/without S" 차이가 너무 극단적이라 디버깅에 애매할 수 있음.
        # - 그래서 아주 가벼운 self-attn 대체(선택)도 가능하지만,
        #   여기서는 안전/단순 위해 "그대로 반환"을 기본으로 둠.
        self.return_identity_if_no_strategy = True

    def forward(self, X_flat: torch.Tensor, S_tokens: torch.Tensor | None = None) -> torch.Tensor:

        if X_flat.dim() != 3:
            raise ValueError(f"[MDTAWithStrategy] X_flat must be (B,N,C). Got: {tuple(X_flat.shape)}")
        B, N, C = X_flat.shape
        if C != self.dim:
            raise ValueError(f"[MDTAWithStrategy] X_flat C({C}) != self.dim({self.dim})")

        # No strategy: return X as-is (safe, fast)
        if S_tokens is None:
            if self.return_identity_if_no_strategy:
                return X_flat
            else:
                # (원하면 여기서 가벼운 변형을 넣을 수 있음)
                return self.project_out(self.q_proj(X_flat))

        if S_tokens.dim() != 3:
            raise ValueError(f"[MDTAWithStrategy] S_tokens must be (B,K,C). Got: {tuple(S_tokens.shape)}")
        if S_tokens.size(0) != B or S_tokens.size(2) != C:
            raise ValueError(
                f"[MDTAWithStrategy] S_tokens must be (B,K,C) with same B,C as X_flat. "
                f"X_flat={tuple(X_flat.shape)} S_tokens={tuple(S_tokens.shape)}"
            )
        K = S_tokens.size(1)
        if K == 0:
            return X_flat

        # 1) Q from image tokens
        q = self.q_proj(X_flat)  # (B, N, C)

        # 2) K,V from strategy tokens (only K tokens!)
        k = self.k_proj(S_tokens)  # (B, K, C)
        v = self.v_proj(S_tokens)  # (B, K, C)

        # 3) reshape to multi-head
        # q: (B, H, N, d), k/v: (B, H, K, d)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        # 4) normalize (Restormer-style)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 5) cross-attn score: (B, H, N, K)  ✅ O(N*K)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 6) apply: (B, H, N, d)
        out = torch.matmul(attn, v)

        # 7) merge heads: (B, N, C)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.project_out(out)

        # 8) residual-style steering (권장: 안정적)
        # - strategy가 "조향"이므로, 본문 feature를 덮어쓰기보다 delta로 주는 게 보통 안정적
        Y_img = X_flat + out
        return Y_img


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print("[mdta_strategy] Self-test 시작")
    print("[mdta_strategy] ROOT =", ROOT)

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
    # Test B: Phase-2 MDTAWithStrategy v2 (small tokens)
    # ---------------------------
    print("\n=== Test B: MDTAWithStrategy v2 (N=1024, K=4) ===")
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

    diff = (y_withS - y_noS).abs().mean().item()
    print(f"[mdta_strategy] mean(|Y_withS - Y_noS|) = {diff:.6f} (0이 아니면 steering 영향 있음)")

    # ---------------------------
    # Test C: Phase-2 MDTAWithStrategy v2 (large tokens: 256x256=65536)
    #   - 기존 v1은 여기서 N^2로 바로 터짐
    #   - v2는 N*K라서 보통 문제 없이 통과해야 함
    # ---------------------------
    print("\n=== Test C: MDTAWithStrategy v2 (N=65536, K=4) OOM-check ===")
    try:
        B, H, W, C = 1, 256, 256, 64
        N = H * W
        X_big = torch.randn(B, N, C, device=device)
        S_big = torch.randn(B, 4, C, device=device)
        with torch.no_grad():
            Y_big = mdta_s(X_big, S_tokens=S_big)
        print("X_big :", tuple(X_big.shape))
        print("Y_big :", tuple(Y_big.shape))
        print("[mdta_strategy] ✅ Large-token test passed (OOM 없이 통과)")
    except torch.cuda.OutOfMemoryError as e:
        print("[mdta_strategy] ❌ OOM 발생:", str(e))
        print("→ 그래도 v1 대비 메모리 폭발 위험은 훨씬 낮음. "
              "B/H/W/C를 줄이거나, AMP/grad-checkpoint를 고려.")
    except Exception as e:
        print("[mdta_strategy] ❌ 오류:", repr(e))

    print("\n[mdta_strategy] Self-test 완료")
