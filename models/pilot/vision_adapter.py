# G:/VETNet_pilot/models/pilot/vision_adapter.py
# CLIP 비전 임베딩 → LLM 입력 토큰
import torch
import torch.nn as nn
from typing import Tuple, Optional


class VisionAdapter(nn.Module):
    """
    CLIP 등 Vision Encoder에서 나온 embedding을
    LLM hidden space로 투영해서 'Strategy Prefix Tokens'로 만드는 모듈.

    입력:
        - vision_emb: Tensor, shape (B, Dv) 또는 (B, Nv, Dv)
          * (B, Dv): CLS 토큰이나 pooled feature 1개
          * (B, Nv, Dv): 패치/토큰 시퀀스

    출력:
        - strategy_tokens: (B, K, H)   # K: num_tokens, H: llm_hidden_dim
        - pooled_feat:     (B, H)      # 요약된 1개 벡터 (선택적, 이후에 쓸 수 있음)
    """

    def __init__(
        self,
        vision_dim: int,
        llm_hidden_dim: int,
        num_tokens: int = 4,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.num_tokens = num_tokens

        # Vision → hidden으로 가는 작은 MLP
        layers = [
            nn.Linear(vision_dim, llm_hidden_dim),
            nn.GELU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim),
        ]
        if use_layernorm:
            layers.append(nn.LayerNorm(llm_hidden_dim))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        # num_tokens 개의 prefix token을 만들기 위한 projection
        # pooled_hidden (B, H) → (B, K, H)
        self.token_proj = nn.Linear(llm_hidden_dim, num_tokens * llm_hidden_dim)

    def forward(self, vision_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vision_emb:
                - (B, Dv)  또는
                - (B, Nv, Dv)

        Returns:
            strategy_tokens: (B, K, H)
            pooled_hidden:   (B, H)
        """

        if vision_emb.dim() == 3:
            # (B, Nv, Dv) → mean pooling → (B, Dv)
            pooled = vision_emb.mean(dim=1)
        elif vision_emb.dim() == 2:
            pooled = vision_emb
        else:
            raise ValueError(
                f"vision_emb must be 2D or 3D tensor, got shape {vision_emb.shape}"
            )

        # Vision space → LLM hidden space
        pooled_hidden = self.mlp(pooled)      # (B, H)

        # 한 개 벡터에서 num_tokens 개 prefix token 만들기
        B, H = pooled_hidden.shape
        token_flat = self.token_proj(pooled_hidden)   # (B, K*H)
        strategy_tokens = token_flat.view(B, self.num_tokens, H)  # (B, K, H)

        return strategy_tokens, pooled_hidden


# ---------------------------------------------------------
# Self-Test
# python models/pilot/vision_adapter.py 로 단독 실행하면 테스트됨
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n[vision_adapter] Self-test 시작\n")

    # 가정:
    #   - CLIP ViT-L/14: Dv = 768 (또는 1024/1024+)
    #   - LLaMA hidden dim: H = 1024 (예시)
    vision_dim = 768
    llm_hidden_dim = 1024
    num_tokens = 4

    adapter = VisionAdapter(
        vision_dim=vision_dim,
        llm_hidden_dim=llm_hidden_dim,
        num_tokens=num_tokens,
        use_layernorm=True,
        dropout=0.0,
    )

    adapter.eval()

    # 케이스 1: CLS vector만 있을 때 → (B, Dv)
    B = 2
    dummy_vec = torch.randn(B, vision_dim)
    with torch.no_grad():
        tokens1, pooled1 = adapter(dummy_vec)

    print("[Case 1] Input shape:", dummy_vec.shape)
    print("         strategy_tokens:", tokens1.shape)   # (B, K, H)
    print("         pooled_hidden  :", pooled1.shape)   # (B, H)

    # 케이스 2: patch tokens 전체를 받을 때 → (B, Nv, Dv)
    B = 2
    Nv = 16   # 예: 16개의 패치 토큰
    dummy_seq = torch.randn(B, Nv, vision_dim)
    with torch.no_grad():
        tokens2, pooled2 = adapter(dummy_seq)

    print("\n[Case 2] Input shape:", dummy_seq.shape)
    print("          strategy_tokens:", tokens2.shape)  # (B, K, H)
    print("          pooled_hidden  :", pooled2.shape)  # (B, H)

    print("\n[vision_adapter] Self-test 완료.\n")
