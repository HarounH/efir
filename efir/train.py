import torch
from torchvision import datasets, transforms
import argparse
import wandb
from efir.utils import CodeBlock, get_on_trace_ready, log_memories, setup_logger, load_config, cfg_node_to_dict
import logging
from efir.model.vae import VAE
from efir.model.ae import AE
from torch import nn
from efir.registry import Registry
from yacs.config import CfgNode
import os
from efir.checkpointer import Checkpointer
from tqdm import tqdm
from typing import Optional, Callable
import random
from torch.utils.data import Subset, DataLoader, Dataset
from torch.profiler import profile,  record_function, ProfilerActivity


setup_logger()

logger = logging.getLogger()


@torch.inference_mode()
def validate(
    model: nn.Module,
    cfg_node: CfgNode,
    test_epoch: int,
    device: str,
    prefix: str = "test",
    profile: bool = False,
) -> float:
    with CodeBlock(
        "initializing testing dataset and dataloader",
        logger,
        profile=profile,
        profile_kwargs=CodeBlock.ProfileKwargs(
            schedule=None,
            on_trace_ready=get_on_trace_ready(f"{prefix}_{test_epoch}_test_dataset_creation"),
        )
    ):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((cfg.DATA.INPUT.mean,), (cfg.DATA.INPUT.std,)),
            ]
        )
        # Download the training data and apply the transformations
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        if prefix == "unseen_test":
            indices = (
                (torch.isin(test_dataset.targets, torch.tensor(include_targets), invert=True))
                .nonzero()
                .reshape(-1)
            )
            logger.info(f"Found {indices.shape} valid indices")
            test_dataset = Subset(test_dataset, indices.tolist())

        # Create the data loader
        test_loader = DataLoader(  # type: ignore
            test_dataset, batch_size=cfg.DATA.TEST_DATALOADER.batch_size, shuffle=False
        )

    log_memories({"test_dataset": test_dataset}, logger)

    total_val_loss = 0.0
    with CodeBlock(
        f"Running inference for {test_epoch=}",
        logger,
        profile=profile,
        profile_kwargs=CodeBlock.ProfileKwargs(
            on_trace_ready=get_on_trace_ready(f"{prefix}_{test_epoch}_inference"),
        ),
    ) as prof:
        for batch_idx, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            outputs = model.losses(data)  # type: ignore
            losses = {
                (f"{prefix}/" + loss_name): weight * outputs.get(loss_name)
                for loss_name, weight in loss_weights.items()
                if loss_name in outputs
            }
            loss: torch.Tensor = sum(
                losses.values(), torch.tensor(0, dtype=torch.float, device=device)
            )
            log_kwargs = {  # log infrequently to limit bandwidth
                f"{prefix}/inputs_and_reconstructions": wandb.Image(
                    torch.cat(
                        (
                            data.detach()[:viz_count, ...].cpu(),
                            outputs["yhat"].detach()[:viz_count, ...].cpu(),
                        ),
                        dim=-1,
                    ),
                    classes=[{"id": i, "name": x} for i, x in enumerate(labels.detach()[:viz_count].cpu().tolist())],  # type: ignore
                ),
            }
            wandb.log(
                {
                    **({k: v.detach().cpu().item() for k, v in losses.items()}),
                    f"{prefix}/total_loss": loss.detach().cpu().item(),
                    f"{prefix}/batch_idx": batch_idx,
                    f"{prefix}/epoch": test_epoch,
                    **log_kwargs,
                }
            )
            if prof is not None:
                prof.step()
            total_val_loss += loss.detach().cpu().item()
    return total_val_loss


if __name__ == "__main__":
    # wandb.login()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="efir/configs/mnist_ae.yaml")
    parser.add_argument("-it", "--include_targets", type=int, default=None, nargs="*")
    args = parser.parse_args()
    include_targets = args.include_targets
    cfg = load_config(args.config)
    device = cfg.DEVICE
    output_dir = cfg.OUTPUT_DIR
    cfg_dict = cfg_node_to_dict(cfg)
    cfg_dict["include_targets"] = include_targets
    wandb.init(
        project="efir",
        config=cfg_dict,
        group="efir_mnist",
        tags=[f"include_targets_{'_'.join(str(x) for x in include_targets)}"],
        # mode="offline",
        # TODO: set group
    )
    run_name = wandb.run.name  # type: ignore
    checkpointer = Checkpointer(os.path.join(output_dir, run_name))
    logger.info(f"Starting run with {run_name=}, and {checkpointer.dir=}")

    model = Registry.build_from_cfg(cfg.MODEL).to(device)
    print(model)
    wandb.watch(model)

    optimizer = Registry.build_from_cfg(cfg.OPTIMIZER, params=model.parameters())
    if use_scheduler := cfg.enable_scheduler:
        scheduler = Registry.build_from_cfg(cfg.SCHEDULER, optimizer=optimizer)
    # Traditional training
    n_epochs = cfg.TRAINING_LOOP.n_epochs
    validation_frequency = cfg.TRAINING_LOOP.validation_frequency
    viz_count = cfg.TRAINING_LOOP.get("viz_count", 4)
    loss_weights = cfg.LOSS_WEIGHTS

    # Create dataset
    with CodeBlock("initializing train dataset and dataloader", logger):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=2, fill=0),
                transforms.ToTensor(),
                # transforms.Normalize((cfg.DATA.INPUT.mean,), (cfg.DATA.INPUT.std,)),  # -1 -> 1
            ]
        )

        # Download the training data and apply the transformations
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        if include_targets is not None:
            indices = (
                (torch.isin(train_dataset.targets, torch.tensor(include_targets)))
                .nonzero()
                .reshape(-1)
            )
            logger.info(f"Found {indices.shape} valid indices")
            train_dataset = Subset(train_dataset, indices.tolist())
        # Create the data loader
        train_loader = DataLoader(  # type: ignore
            train_dataset, batch_size=cfg.DATA.TRAIN_DATALOADER.batch_size, shuffle=True
        )
    n_batches = len(train_loader)
    # Invoke training
    with CodeBlock(
        "Training Loop",
        logger,
        profile=True,
        profile_kwargs=CodeBlock.ProfileKwargs(
            on_trace_ready=get_on_trace_ready(f"training_loop"),
        ),
    ) as prof:
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
                    losses.values(), torch.tensor(0, dtype=torch.float, device=device)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if use_scheduler:
                    scheduler.step()
                log_kwargs = (
                    {  # log infrequently to limit bandwidth
                        "inputs_and_reconstructions": wandb.Image(
                            torch.cat(
                                (
                                    data.detach()[:viz_count, ...].cpu(),
                                    outputs["yhat"].detach()[:viz_count, ...].cpu(),
                                ),
                                dim=-1,
                            ),
                            classes=[{"id": i, "name": x} for i, x in enumerate(labels.detach()[:viz_count].cpu().tolist())],  # type: ignore
                        ),
                    }
                    if (batch_idx % (n_batches // validation_frequency)) == 0
                    else {}
                )
                wandb.log(
                    {
                        **({k: v.detach().cpu().item() for k, v in losses.items()}),
                        "total_loss": loss.detach().cpu().item(),
                        "epoch": epoch,
                        "batch": batch_idx,
                        **log_kwargs,
                    }
                )
                prof.step()
            ## Invoke validation
            if epoch % validation_frequency == 0:
                with CodeBlock(f"Validating on {epoch=}", logger):
                    val_loss = validate(model, cfg, test_epoch=epoch, device=device, profile=False)
                with CodeBlock(f"Checkpointing on {epoch=}", logger):
                    checkpointer(model, epoch, {"val_loss": val_loss})
    # Invoke test
    with CodeBlock(f"Testing at the end", logger):
        val_loss = validate(model, cfg, test_epoch=(n_epochs + 1), device=device, profile=True)

    with CodeBlock(f"Testing Unseen class at the end", logger):
        val_loss = validate(model, cfg, test_epoch=(n_epochs + 1), device=device, prefix="unseen_test")

    with CodeBlock(f"Checkpointing at the end", logger):
        checkpointer(model, (n_epochs + 1), {"val_loss": val_loss})
    wandb.finish()
