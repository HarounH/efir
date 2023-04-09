import torch
from efir.model.layers.decoder import Transformer
from yacs.config import CfgNode
from efir.registry import Registry
from torchvision import datasets, transforms
import wandb
from efir.utils import CodeBlock, cfg_node_to_dict, setup_logger, load_config
from torch.utils.data import Subset, DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn
import argparse
import logging


setup_logger()
logger = logging.getLogger(__name__)


def evaluate_model(model: nn.Module, epoch: int):
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = datasets.MNIST(root='../data/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().cpu().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute metrics
    accuracy = 100 * correct / total
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    wandb.log({
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/f1": f1,
        "test/epoch": epoch,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../efir/configs/mnist_transformer_clf.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    device = config.DEVICE
    wandb.init(project= "efir", group="train_mnist_transformer_classifier", config=cfg_node_to_dict(config))

    with CodeBlock("initializing things", logger=logger):
        model = Transformer(**(config.MODEL))
        model = model.to(device)

        optimizer = Registry.build_from_cfg(cfg_node=config.OPTIMIZER, params=model.parameters())
        if use_scheduler := config.get("SCHEDULER", {}).get("type", None) is not None:
            scheduler = Registry.build_from_cfg(cfg_node=config.SCHEDULER, optimizer=optimizer)
        else:
            scheduler = None

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((cfg.DATA.INPUT.mean,), (cfg.DATA.INPUT.std,)),
            ]
        )

        train_dataset = datasets.MNIST(
            root="../data/", train=True, download=True, transform=transform
        )


        train_loader = DataLoader(  # type: ignore
            train_dataset, batch_size=config.DATA.TRAIN_DATALOADER.batch_size, shuffle=True,
        )


    criterion = nn.CrossEntropyLoss()
    wandb.watch(model, log="all")
    for epoch in range(config.TRAINING_LOOP.n_epochs):
        model.train()
        for idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            yhat = model(data)

            loss = criterion(yhat, labels)
            loss.backward()
            optimizer.step()
            if use_scheduler and scheduler is not None:
                scheduler.step()
            wandb.log({
                "loss": loss.detach().cpu().item(),
            })
        if epoch % config.TRAINING_LOOP.validation_frequency == 0:
            with CodeBlock(f"Validating on {epoch=}", logger):
                evaluate_model(model, epoch=epoch)
    evaluate_model(model, epoch=epoch)