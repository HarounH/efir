import argparse
from collections import defaultdict
from efir.utils import CodeBlock, setup_logger, load_config, cfg_node_to_dict
import torch
from torch import nn
from yacs.config import CfgNode
from efir.registry import Registry
from efir.checkpointer import Checkpointer
import logging
from torchvision import datasets, transforms
import numpy as np
import os
from efir.model.vae import VAE
from efir.model.ae import AE
from torch.utils.data import Subset, DataLoader, Dataset
from tqdm import tqdm

setup_logger()

logger = logging.getLogger()


@torch.inference_mode()
def inference(
    model: nn.Module,
    cfg_node: CfgNode,
    device: str,
    output_path: str,
) -> None:
    loss_weights = cfg_node.LOSS_WEIGHTS
    results = defaultdict(list)
    for train in [True, False]:
        with CodeBlock(
            f"initializing dataset and dataloader with {train=}",
            logger,
        ):
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((cfg.DATA.INPUT.mean,), (cfg.DATA.INPUT.std,)),
                ]
            )
            # Download the training data and apply the transformations
            test_dataset = datasets.MNIST(
                root="./data", train=train, download=True, transform=transform
            )

            # Create the data loader
            test_loader = DataLoader(  # type: ignore
                test_dataset, batch_size=cfg.DATA.TEST_DATALOADER.batch_size, shuffle=False
            )

        with CodeBlock(
            "running inference",
            logger,
        ):
            for batch_idx, (data, labels) in enumerate(tqdm(test_loader)):
                data = data.to(device)
                outputs = model(data)
                results[f"data_t{train}"].append(data.to("cpu").detach().numpy())
                results[f"labels_t{train}"].append(labels.to("cpu").detach().numpy())
                for k, v in outputs.items():
                    results[k + f"_t{train}"].append(outputs[k].to("cpu").detach().numpy())

    with CodeBlock(
        "saving results",
        logger,
    ):
        np.savez(output_path, **results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="efir/configs/mnist_ae.yaml")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=21)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.DEVICE
    output_dir = cfg.OUTPUT_DIR
    checkpointer = Checkpointer(os.path.join(output_dir, args.run_name))

    cfg_dict = cfg_node_to_dict(cfg)


    model = checkpointer.load(
        Registry.build_from_cfg(cfg.MODEL),
        epoch=args.epoch,
    ).to(device)
    output_path = checkpointer.get_inference_results_path(args.epoch)
    inference(
        model=model,
        cfg_node=cfg,
        device=device,
        output_path=output_path,
    )