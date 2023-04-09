from typing import List
from torch import nn
import torch
from efir.model.layers.multi_head_attention import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(self, num_heads: int, d: int, d_mlp: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(num_heads=num_heads, d=d)
        self.self_attention_norm = nn.LayerNorm(d)
        self.self_attention_dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        y = self.self_attention_norm(x)
        y = x + self.self_attention_dropout(self.self_attention(query=x, key=x, value=x))
        return y + self.mlp(y)


class Patchify(nn.Module):
    def __init__(self, patch_height: int, patch_width: int) -> None:
        super().__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: b, c, h, w
        batch_size, num_channels, height, width = x.shape
        num_vertical_patches = height // self.patch_height
        num_horizontal_patches = height // self.patch_width
        x = x[:, :, :, None, :, None]  # b, c, h, 1, w, 1
        x = x.view(batch_size, num_channels, num_vertical_patches, self.patch_height, num_horizontal_patches, self.patch_height)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        # output: b, k, d
        return x.view(batch_size, num_vertical_patches * num_horizontal_patches, num_channels * self.patch_height * self.patch_width)


class MLPHead(nn.Module):
    def __init__(self, sizes: List[int]) -> None:
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(
                nn.Linear(sizes[i], sizes[i+1])
            )
            layers.append(nn.GELU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Transformer(nn.Module):
    def __init__(
        self,
        patch_height: int,
        patch_width: int,
        num_channels: int,
        d: int,
        d_mlp: int,
        num_heads: int,
        pool_size: int,
        d_out: int,
    ) -> None:
        super().__init__()
        self.patchify = Patchify(patch_height=patch_height, patch_width=patch_width)
        self.patch_embedder = nn.Linear(patch_height * patch_width * num_channels, d)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d=d, num_heads=num_heads, d_mlp=d_mlp)])
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        self.head = MLPHead(sizes=[d * pool_size, d_mlp, d_mlp, d_out])  # TODO: customizable sizes?

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: b, c, h, w
        # patched: b, k, d
        patched = self.patchify(x)
        x = self.patch_embedder(patched)
        for layer in self.decoder_blocks:
            x = layer(x)
        # x: b, k, d
        batch_size, seq_len, embedding_dim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = self.pool(x).view(batch_size, -1)
        # output: b, d_out
        return self.head(x)

