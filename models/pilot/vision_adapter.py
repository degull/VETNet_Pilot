""" # G:/VETNet_pilot/models/pilot/vision_adapter.py
# CLIP 비전 임베딩 → LLM 입력 토큰
import torch
import torch.nn as nn
from typing import Tuple, Optional


class VisionAdapter(nn.Module):

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
 """

# ver2
# G:\VETNet_pilot\models\pilot\vision_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPProcessor


class CLIPVisionAdapter(nn.Module):
    """
    Frozen CLIP vision encoder that outputs a compact visual embedding.
    - Uses CLIPModel (vision + text) but we only use vision.
    - Returns pooled embedding (B, d_v) and optionally patch tokens.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(device)

        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self.clip.to(self.device)
        self.use_fp16 = use_fp16

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """
        images: (B,3,H,W) float tensor in [0,1]
        returns:
          pooled: (B, D)
          patch_tokens: (B, N, D) or None (depending on CLIP internals)
        """
        # CLIPProcessor expects PIL images or numpy, but it can also accept torch tensors
        # if we convert to list of PIL; however that's slow.
        # We'll normalize manually like CLIP expects using CLIP vision_model's config.
        # CLIP expects pixel_values normalized like processor does.
        # We'll reuse processor for normalization by converting to CPU and back only if needed.
        # To keep it simple and robust, use processor on CPU with minimal overhead for small batches.

        # NOTE: If you want pure-tensor pipeline, replace this with CLIP image normalization.
        imgs = images.detach().cpu()
        inputs = self.processor(images=imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        if self.use_fp16:
            pixel_values = pixel_values.half()

        vision_outputs = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
        # last_hidden_state: (B, N, D)
        patch_tokens = vision_outputs.last_hidden_state
        pooled = patch_tokens[:, 0]  # CLS token

        # L2 normalize (often beneficial)
        pooled = F.normalize(pooled, dim=-1)

        return pooled, patch_tokens
