import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Dict
from efir.registry import AutoRegistrationBase


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bn: bool = True, leaky_relu: bool = False, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, **kwargs)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.has_bn = bn
        if leaky_relu:
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        return self.act(x)


class DeConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bn: bool = True, leaky_relu: bool = False, **kwargs) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, **kwargs)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.has_bn = bn
        if leaky_relu:
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        if self.has_bn:
            x = self.bn(x)
        return self.act(x)


class VAE(nn.Module, AutoRegistrationBase):
    def __init__(
            self,
            encoded_space_dim: int = 64,
            fc2_input_dim: int = 128,
            use_l1_loss: bool = False,
            bn: bool = False,
            leaky_relu: bool = False,
        ):
        super(VAE, self).__init__()
        # TODO: better architecture?
        relu_cls = nn.LeakyReLU if leaky_relu else nn.ReLU
        output_spatial_dim = 1
        self.use_l1_loss = use_l1_loss
        self.conv_encoder = nn.Sequential(
            # size=28
            ConvBNReLU(1, encoded_space_dim // 16, kernel_size=3, stride=2, padding=1, bn=bn, leaky_relu=leaky_relu),  # size= 14
            ConvBNReLU(encoded_space_dim // 16, encoded_space_dim // 8, kernel_size=3, stride=2, padding=1, bn=bn, leaky_relu=leaky_relu),  # 7
            ConvBNReLU(encoded_space_dim // 8, encoded_space_dim // 4, kernel_size=3, stride=2, padding=1, bn=bn, leaky_relu=leaky_relu),  # 4
            ConvBNReLU(encoded_space_dim // 4, encoded_space_dim, kernel_size=3, stride=2, padding=1, bn=bn, leaky_relu=leaky_relu),  # 2
        )
        self.enc_mlp = nn.Sequential(  # Wide and back
            nn.AdaptiveAvgPool2d(output_spatial_dim),
            nn.Flatten(),
            nn.Linear(output_spatial_dim * output_spatial_dim * encoded_space_dim, fc2_input_dim),
            relu_cls(),
            nn.Linear(fc2_input_dim, encoded_space_dim * 2)  # mu and logvar
        )
        self.dec_mlp = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            relu_cls(),
            nn.Linear(fc2_input_dim, output_spatial_dim * output_spatial_dim * encoded_space_dim),
            relu_cls(),
            nn.Unflatten(dim=1, unflattened_size=(encoded_space_dim, output_spatial_dim, output_spatial_dim)),
        )
        self.convt_decoder = nn.Sequential(
            # stride=2 => h_out = h*s -s -2p + k + op
            # size = 1
            DeConvBNReLU(encoded_space_dim, encoded_space_dim // 4, kernel_size=5, stride=2, padding=0, bn=bn, leaky_relu=leaky_relu),  # 5
            DeConvBNReLU(encoded_space_dim // 4, encoded_space_dim // 8, kernel_size=5, stride=2, padding=0, bn=bn, leaky_relu=leaky_relu),  # 13
            ConvBNReLU(encoded_space_dim // 8, encoded_space_dim // 8, kernel_size=3, stride=1, padding=1, bn=bn, leaky_relu=leaky_relu),  # 13
            DeConvBNReLU(encoded_space_dim // 8, encoded_space_dim // 16, kernel_size=4, stride=2, padding=0, bn=bn, leaky_relu=leaky_relu),  # 28
            ConvBNReLU(encoded_space_dim // 16, 1, kernel_size=3, stride=1, padding=1, bn=False, leaky_relu=False),  # 28
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_encoder(x)  # 2, 2
        x = self.enc_mlp(x)  # d
        return x

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.dec_mlp(z)  # 1, 1
        z = self.convt_decoder(z)  # 31, 31
        return z

    def encode_decode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encode(x)
        mu, log_var = x.chunk(2, 1)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        yhat, mu, log_var = self.encode_decode(x)
        regularization = -0.5 * torch.mean(  # batch
            torch.sum(  # elements
                1 + log_var - mu ** 2 - log_var.exp(),
                dim=1,
            ),
            dim=0,
        )
        outputs = {
            "regularization": regularization,
            "yhat": yhat,
            "mu": mu,
            "log_var": log_var,
        }
        if self.use_l1_loss:
            outputs["l1_loss"] = F.l1_loss(yhat, x)
        else:
            outputs["mse_loss"] = F.mse_loss(yhat, x)
        return outputs