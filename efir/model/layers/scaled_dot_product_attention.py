from typing import Optional, Tuple
from torch import nn
import torch


TContextAndAttention = Tuple[torch.Tensor, torch.Tensor]


class ScaledDotProductAttention(nn.Module):
    # Batch First
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(
        self, *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> TContextAndAttention:
        # query: B, S_q, d
        # key: B, S_k, d
        # value: B, S_k, d
        dot_product = torch.bmm(query, key.transpose(1, 2))  # B, S_q, S_k
        dot_product *= self.scale
        if mask is not None:
            # masked_fill_ is in-place?
            dot_product.masked_fill_(mask.view(dot_product.shape), -float("inf"))  # gets softmaxed to 0
        attention = torch.softmax(dot_product, dim=-1)  # B, S_q, S_k
        # ASSERT: S_k = S_v
        context = torch.bmm(attention, value)  # B, S_q, d
        return context, attention
