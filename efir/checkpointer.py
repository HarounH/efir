from torch import nn
from typing import Any, Optional, Dict
import torch
import os
from efir.utils import CodeBlock
import wandb
import logging

logger = logging.getLogger()


class Checkpointer:
    def __init__(self, dir: str, max_epoch_digits: int = 5) -> None:
        self.dir = dir
        self.max_epoch_digits = max_epoch_digits
        with CodeBlock(f"Checkpointer creating output {dir=}", logger=logger):
            os.makedirs(dir, exist_ok=True)

    def get_path(self, epoch: int) -> str:
        return os.path.join(self.dir, f"model{epoch:0{self.max_epoch_digits}d}.pt")

    def __call__(self, model: nn.Module, epoch: int, metadata: Optional[Dict] = None) -> str:
        output_path = self.get_path(epoch)
        with CodeBlock(f"Saving checkpoint at {epoch=} to {output_path=}", logger=logger):
            torch.save(model.state_dict(), output_path)
            artifact = wandb.Artifact(f"model_checkpoint_{epoch}.pt", type='model', metadata=(metadata or {}))
            artifact.add_file(output_path)
            wandb.run.log_artifact(artifact)  # type: ignore
        return output_path

    def load(self, model: nn.Module, epoch: int) -> nn.Module:
        output_path = self.get_path(epoch)
        model.load_state_dict(torch.load(output_path))
        return model

    def get_inference_results_path(self, epoch: int) -> str:
        return os.path.join(self.dir, f"inference_results_{epoch:0{self.max_epoch_digits}d}.npz")