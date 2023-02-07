import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict
from efir.registry import AutoRegistrationBase


class AE(nn.Module, AutoRegistrationBase):
    def __init__(self, encoded_space_dim: int = 64, fc2_input_dim: int = 128):
        super(AE, self).__init__()
        self.encpder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(3 * 3 * 32, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 3 * 3 * 32),
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),
            nn.ConvTranspose2d(32, 16, 3,
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,
            padding=1, output_padding=1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encpder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        return self.decode(x)

    def losses(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        yhat = self.forward(x)
        return {"mse_loss": F.mse_loss(yhat, x), "yhat": yhat}