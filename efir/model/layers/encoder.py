from typing import Optional
from torch import nn
import torch
from efir.model.layers.multi_head_attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, d: int, d_mlp: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, d=d)
        self.input_norm = nn.LayerNorm(d)
        self.middle_norm = nn.LayerNorm(d)
        self.middle_dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d, d_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.input_norm(x)
        y = self.mha(  # self-attention
            query=x,
            key=x,
            value=x,
            mask=mask,
        )
        z = self.middle_norm(x + self.middle_dropout(y))
        return z + self.mlp(z)  # TODO: dropout here?


class Encoder(nn.Module):
    """ Receives embeddings, and runs MHA on it """
    def __init__(self, num_blocks: int, num_heads: int, d: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(num_heads=num_heads, d=d) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: B, L, d
        # mask: B, L, d
        for block in self.blocks:
            x = block(x=x, mask=mask)
        return x

