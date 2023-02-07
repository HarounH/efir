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


setup_logger()


n_epochs = 10
validation_frequency = 5
batch_size = 64
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
            test_dataset, batch_size=batch_size, shuffle=False
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
            wandb.log({
                **({k: v.detach().cpu().item() for k, v in losses.items()}),
                "test/total_loss": loss.detach().cpu().item(),
                "test/epoch": test_epoch,
                "inputs": wandb.Image(
                    data.detach()[:viz_count, ...].cpu(),
                    classes=[{"id": i, "name": x} for i, x in enumerate(labels.detach()[:viz_count].cpu().tolist())],  # type: ignore
                ),
                "recontructions": wandb.Image(outputs["yhat"].detach()[:viz_count, ...].cpu())
            })
            total_val_loss += loss.detach().cpu().item()
    return total_val_loss

if __name__ == "__main__":
    # wandb.login()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="efir/configs/mnist_ae.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = cfg.DEVICE
    output_dir = cfg.OUTPUT_DIR
    cfg_dict = cfg_node_to_dict(cfg)

    wandb.init(
        project="efir",
        config=cfg_dict,
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

        # Create the data loader
        train_loader = torch.utils.data.DataLoader(  # type: ignore
            train_dataset, batch_size=batch_size, shuffle=True
        )

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
                wandb.log({
                    **({k: v.detach().cpu().item() for k, v in losses.items()}),
                    "total_loss": loss.detach().cpu().item(),
                    "epoch": epoch,
                    "batch": batch_idx,
                    "inputs": wandb.Image(
                        data.detach()[:viz_count, ...].cpu(),
                        classes=[{"id": i, "name": x} for i, x in enumerate(labels.detach()[:viz_count].cpu().tolist())],  # type: ignore
                    ),
                    "recontructions": wandb.Image(outputs["yhat"].detach()[:viz_count, ...].cpu())
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
