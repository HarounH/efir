import torch
from torchvision import datasets, transforms
import argparse
import wandb
from efir.utils import CodeBlock, setup_logger, load_config, cfg_node_to_dict
import logging
from efir.model.vae import VAE
from efir.model.ae import AE
from torch import nn
from efir.registry import Registry
from yacs.config import CfgNode
import os
from efir.checkpointer import Checkpointer
from tqdm import tqdm
from typing import Optional
import random


setup_logger()

logger = logging.getLogger()


@torch.inference_mode()
def validate(model: nn.Module, cfg_node: CfgNode, test_epoch: int, device: str) -> float:
    with CodeBlock("initializing testing dataset and dataloader", logger):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Download the training data and apply the transformations
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )

        # Create the data loader
        test_loader = torch.utils.data.DataLoader(  # type: ignore
            test_dataset, batch_size=cfg.DATA.TEST_DATALOADER.batch_size, shuffle=False
        )
    total_val_loss = 0.0
    with CodeBlock(f"Running inference for {test_epoch=}", logger):
        for batch_idx, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            outputs = model.losses(data)  # type: ignore
            losses = {
                ("test/" + loss_name): weight * outputs.get(loss_name)
                for loss_name, weight in loss_weights.items()
                if loss_name in outputs
            }
            loss: torch.Tensor = sum(
                losses.values(),
                torch.tensor(0, dtype=torch.float, device=device)
            )
            log_kwargs = {  # log infrequently to limit bandwidth
                "test/inputs_and_reconstructions": wandb.Image(
                    torch.cat(
                        (
                            data.detach()[:viz_count, ...].cpu(),
                            outputs["yhat"].detach()[:viz_count, ...].cpu(),
                        ),
                        dim=-1
                    ),
                    classes=[{"id": i, "name": x} for i, x in enumerate(labels.detach()[:viz_count].cpu().tolist())],  # type: ignore
                ),
            }
            wandb.log({
                **({k: v.detach().cpu().item() for k, v in losses.items()}),
                "test/total_loss": loss.detach().cpu().item(),
                "test/batch_idx": batch_idx,
                "test/epoch": test_epoch,
                **log_kwargs,
            })
            total_val_loss += loss.detach().cpu().item()
    return total_val_loss

if __name__ == "__main__":
    # wandb.login()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="efir/configs/mnist_ae.yaml")
    parser.add_argument("-lo", "--leave_out", type=int, default=None)
    args = parser.parse_args()
    leave_out = args.leave_out
    cfg = load_config(args.config)
    device = cfg.DEVICE
    output_dir = cfg.OUTPUT_DIR
    cfg_dict = cfg_node_to_dict(cfg)
    cfg_dict["leave_out"] = leave_out
    wandb.init(
        project="efir",
        config=cfg_dict,
        group="efir_mnist",
        tags=[f"leave_out_{leave_out}"],
        # mode="offline",
        # TODO: set group
    )
    run_name = wandb.run.name  # type: ignore
    checkpointer = Checkpointer(os.path.join(output_dir, run_name))
    logger.info(f"Starting run with {run_name=}, and {checkpointer.dir=}")

    model = Registry.build_from_cfg(cfg.MODEL).to(device)
    wandb.watch(model)

    optimizer = Registry.build_from_cfg(cfg.OPTIMIZER, params=model.parameters())
    # Traditional training
    n_epochs = cfg.TRAINING_LOOP.n_epochs
    validation_frequency = cfg.TRAINING_LOOP.validation_frequency
    viz_count = cfg.TRAINING_LOOP.get("viz_count", 4)
    loss_weights = cfg.LOSS_WEIGHTS

    # Create dataset
    with CodeBlock("initializing train dataset and dataloader", logger):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Download the training data and apply the transformations
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        if leave_out is not None:
            indices = (train_dataset.targets != leave_out).nonzero().reshape(-1)
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
        # Create the data loader
        train_loader = torch.utils.data.DataLoader(  # type: ignore
            train_dataset, batch_size=cfg.DATA.TRAIN_DATALOADER.batch_size, shuffle=True
        )
    n_batches = len(train_loader)
    # Invoke training
    with CodeBlock("Training Loop", logger):
        for epoch in range(n_epochs):
            for batch_idx, (data, labels) in enumerate(tqdm(train_loader)):
                # Do something with the data
                data = data.to(device)
                outputs = model.losses(data)
                losses = {
                    loss_name: weight * outputs.get(loss_name)
                    for loss_name, weight in loss_weights.items()
                    if loss_name in outputs
                }
                loss: torch.Tensor = sum(
                    losses.values(),
                    torch.tensor(0, dtype=torch.float, device=device)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                log_kwargs = {  # log infrequently to limit bandwidth
                    "inputs_and_reconstructions": wandb.Image(
                        torch.cat(
                            (
                                data.detach()[:viz_count, ...].cpu(),
                                outputs["yhat"].detach()[:viz_count, ...].cpu(),
                            ),
                            dim=-1
                        ),
                        classes=[{"id": i, "name": x} for i, x in enumerate(labels.detach()[:viz_count].cpu().tolist())],  # type: ignore
                    ),
                } if (batch_idx % (n_batches // validation_frequency)) == 0 else {}
                wandb.log({
                    **({k: v.detach().cpu().item() for k, v in losses.items()}),
                    "total_loss": loss.detach().cpu().item(),
                    "epoch": epoch,
                    "batch": batch_idx,
                    **log_kwargs,
                })
            ## Invoke validation
            if epoch % validation_frequency == 0:
                with CodeBlock(f"Validating on {epoch=}", logger):
                    val_loss = validate(model, cfg, test_epoch=epoch, device=device)
                with CodeBlock(f"Checkpointing on {epoch=}", logger):
                    checkpointer(
                        model,
                        epoch,
                        {"val_loss": val_loss}
                    )
    # Invoke test
    with CodeBlock(f"Testing at the end", logger):
        val_loss = validate(model, cfg, test_epoch=(n_epochs + 1), device=device)

    with CodeBlock(f"Checkpointing at the end", logger):
        checkpointer(
            model,
            (n_epochs + 1),
            {"val_loss": val_loss}
        )
    wandb.finish()