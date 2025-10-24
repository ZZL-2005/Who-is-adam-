import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from datasets.cifar10 import DataConfig, create_cifar10_dataloaders
from models import build_model
from scripts.train import build_experiment_config, load_config


def _build_loss(loss_name: str) -> nn.Module:
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss_name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss function '{loss_name}'")


def _compute_loss(outputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        num_classes = outputs.size(1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
        predictions = torch.nn.functional.softmax(outputs, dim=1)
        return criterion(predictions, targets_one_hot)
    return criterion(outputs, targets)


@torch.no_grad()
def _evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    loss_name: str,
    device: torch.device,
) -> Any:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = _compute_loss(outputs, targets, criterion, loss_name)
        total_loss += loss.item() * targets.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        count += targets.size(0)

    return {
        "loss": total_loss / max(count, 1),
        "accuracy": correct / max(count, 1),
        "samples": count,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on CIFAR-10.")
    parser.add_argument("--cfg", type=str, required=True, help="Configuration file used during training.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (epoch_XXX.pt).")
    parser.add_argument("--loss", type=str, required=True, help="Loss function name (cross_entropy or mse).")
    parser.add_argument("--device", type=str, help="Override device (cpu/cuda).")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to evaluate.")
    parser.add_argument("--pretrained", action="store_true", help="Initialize model with pretrained weights before loading checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.cfg).resolve()
    raw_cfg = load_config(config_path)

    if args.device:
        raw_cfg["device"] = args.device

    experiment_cfg = build_experiment_config(raw_cfg)
    data_cfg: DataConfig = experiment_cfg.data_cfg

    device = torch.device(experiment_cfg.device if torch.cuda.is_available() or experiment_cfg.device == "cpu" else "cpu")
    loaders = create_cifar10_dataloaders(data_cfg)
    if args.split not in loaders or loaders[args.split] is None:
        raise ValueError(f"Requested split '{args.split}' not available.")

    model = build_model(experiment_cfg.model, num_classes=10, pretrained=args.pretrained).to(device)
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    criterion = _build_loss(args.loss).to(device)
    metrics = _evaluate_model(model, loaders[args.split], criterion, args.loss, device)

    print(
        f"Evaluation on {args.split} split | loss={metrics['loss']:.4f} "
        f"accuracy={metrics['accuracy']*100:.2f}% ({metrics['samples']} samples)"
    )


if __name__ == "__main__":
    main()
