from typing import Optional, Tuple

import torch
from torch import nn
from efir.model.layers.scaled_dot_product_attention import ScaledDotProductAttention, TContextAndAttention


class Head(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d
        self.query_projection = nn.Linear(d, d, bias=False)
        self.key_projection = nn.Linear(d, d, bias=False)
        self.value_projection = nn.Linear(d, d, bias=False)
        self.attention_layer = ScaledDotProductAttention(scale=1.0 / d)

    def forward(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> TContextAndAttention:
        return self.attention_layer(
            query=self.query_projection(query),
            key=self.key_projection(key),
            value=self.value_projection(value),
            mask=mask,
        )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d: int,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(d=d) for _ in range(num_heads)])
        self.w0 = nn.Linear(num_heads * d, d, bias=False)

    def forward(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        head_outputs = [head(query=query, key=key, value=value, mask=mask) for head in self.heads]
        return self.w0(torch.cat([head_output[0] for head_output in head_outputs], dim=-1))