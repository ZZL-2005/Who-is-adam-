import argparse
import copy
import csv
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml

from datasets.cifar10 import DataConfig, create_cifar10_dataloaders, set_global_seed
from models import build_model

LossName = str


@dataclass
class ExperimentConfig:
    model: str
    device: str
    learning_rates: Iterable[float]
    loss_functions: Iterable[LossName]
    optimizer_config: Dict[str, Any]
    epochs: int
    log_interval: int
    val_interval: int
    checkpoint_interval: int
    max_checkpoints: Optional[int]
    output_dir: Path
    seed: int
    weight_decay: float
    data_cfg: DataConfig


def _ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    runs_dir = output_dir / "runs"
    aggregate_logs = output_dir / "logs"
    figures = output_dir / "figures"
    for directory in (output_dir, runs_dir, aggregate_logs, figures):
        directory.mkdir(parents=True, exist_ok=True)
    return {"runs": runs_dir, "aggregate_logs": aggregate_logs, "figures": figures}


def _prepare_run_directory(base_dir: Path, run_name: str) -> Dict[str, Path]:
    run_dir = base_dir / run_name
    checkpoints = run_dir / "checkpoints"
    for directory in (run_dir, checkpoints):
        directory.mkdir(parents=True, exist_ok=True)
    return {"root": run_dir, "checkpoints": checkpoints}


def _save_run_configs(
    run_dir: Path,
    base_config_path: Path,
    run_config: Dict[str, Any],
) -> None:
    if base_config_path.is_file():
        shutil.copy2(base_config_path, run_dir / "base_config.yaml")
    config_path = run_dir / "config.yaml"
    with config_path.open("w") as fp:
        yaml.safe_dump(run_config, fp, sort_keys=False)


def _one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(targets, num_classes=num_classes).float()


def _compute_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    loss_name: LossName,
) -> torch.Tensor:
    if loss_name == "mse":
        num_classes = outputs.size(1)
        targets_one_hot = _one_hot(targets, num_classes)
        predictions = torch.nn.functional.softmax(outputs, dim=1)
        return criterion(predictions, targets_one_hot)
    return criterion(outputs, targets)


def _accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _build_loss(loss_name: LossName) -> nn.Module:
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss_name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss function '{loss_name}'")


def _build_optimizer(model: nn.Module, cfg: Dict[str, Any], lr_override: Optional[float] = None) -> torch.optim.Optimizer:
    name = cfg.get("name", "").lower()
    lr = lr_override if lr_override is not None else cfg.get("lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 0.0)

    if name == "adam":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        eps = cfg.get("eps", 1e-8)
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "sgd":
        momentum = cfg.get("momentum", 0.0)
        nesterov = cfg.get("nesterov", False)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer '{name}'")


def _train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    loss_name: LossName,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = _compute_loss(outputs, targets, criterion, loss_name)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += batch_size

    mean_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return mean_loss, accuracy


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    loss_name: LossName,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = _compute_loss(outputs, targets, criterion, loss_name)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += batch_size

    mean_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return mean_loss, accuracy


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    run_dir: Path,
    max_to_keep: Optional[int],
    saved_paths: List[Path],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    checkpoint_path = run_dir / f"epoch_{epoch:03d}.pt"
    torch.save(checkpoint, checkpoint_path)
    saved_paths.append(checkpoint_path)

    if max_to_keep is None or max_to_keep <= 0:
        return

    while len(saved_paths) > max_to_keep:
        oldest = saved_paths.pop(0)
        if oldest.exists():
            oldest.unlink()


def _write_run_log(records: List[Dict[str, Any]], log_path: Path) -> None:
    if not records:
        return

    fieldnames = ["run_name", "epoch", "split", "loss", "accuracy"]
    with log_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _append_results(summary: Dict[str, Any], results_path: Path) -> None:
    fieldnames = sorted(summary.keys())
    write_header = not results_path.exists()

    with results_path.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(summary)


def _format_run_name(model: str, optimizer: str, loss_name: str, lr: float, timestamp: str) -> str:
    lr_str = f"{lr:.4f}".rstrip("0").rstrip(".")
    return f"{timestamp}_{model}_{optimizer}_{loss_name}_lr{lr_str}"


def run_experiments(
    config: ExperimentConfig,
    raw_cfg: Dict[str, Any],
    config_source: Path,
) -> None:
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    if device.type != config.device:
        print(f"[WARN] Requested device '{config.device}' not available. Falling back to '{device}'.")

    dirs = _ensure_dirs(config.output_dir)
    results_path = dirs["aggregate_logs"] / "results.csv"

    loaders = create_cifar10_dataloaders(config.data_cfg)
    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    test_loader = loaders["test"]

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    optimizer_name = config.optimizer_config.get("name", "")
    for optimizer_name in [optimizer_name]:
        for loss_name in config.loss_functions:
            for lr in config.learning_rates:
                run_name = _format_run_name(config.model, optimizer_name, loss_name, lr, timestamp)
                print(f"\n=== Running experiment: {run_name} ===")

                set_global_seed(config.seed)
                model = build_model(config.model, num_classes=10, pretrained=False).to(device)
                optimizer_cfg = dict(config.optimizer_config)
                optimizer_cfg["lr"] = lr
                optimizer_cfg["weight_decay"] = config.weight_decay
                optimizer = _build_optimizer(model, optimizer_cfg, lr_override=lr)
                criterion = _build_loss(loss_name).to(device)

                run_paths = _prepare_run_directory(dirs["runs"], run_name)
                run_dir = run_paths["root"]
                run_checkpoint_dir = run_paths["checkpoints"]
                run_log_path = run_dir / "metrics.csv"

                run_config_dict = copy.deepcopy(raw_cfg)
                run_config_dict["run_name"] = run_name
                run_config_dict["timestamp"] = timestamp
                run_config_dict["model"] = config.model
                run_config_dict["device"] = config.device
                run_config_dict["epochs"] = config.epochs
                run_config_dict["seed"] = config.seed
                run_config_dict["learning_rate"] = lr
                run_config_dict["learning_rates"] = [lr]
                run_config_dict["loss_function"] = loss_name
                run_config_dict["loss_functions"] = [loss_name]
                run_config_dict["optimizer"] = copy.deepcopy(run_config_dict.get("optimizer", {}))
                run_config_dict["optimizer"]["name"] = optimizer_name
                run_config_dict["optimizer"]["lr"] = lr
                run_config_dict["optimizer"]["weight_decay"] = config.weight_decay
                run_config_dict["data_config"] = asdict(config.data_cfg)
                run_config_dict["output_dir"] = str(config.output_dir)
                run_config_dict["artifacts"] = {
                    "run_dir": str(run_dir),
                    "checkpoints": str(run_checkpoint_dir),
                    "log": str(run_log_path),
                }

                _save_run_configs(run_dir, config_source, run_config_dict)

                epoch_records: List[Dict[str, Any]] = []
                saved_checkpoints: List[Path] = []

                best_val_acc = -math.inf
                best_test_acc = -math.inf
                best_epoch = 0

                for epoch in range(1, config.epochs + 1):
                    train_loss, train_acc = _train_one_epoch(model, train_loader, optimizer, criterion, loss_name, device)
                    epoch_records.append(
                        {
                            "run_name": run_name,
                            "epoch": epoch,
                            "split": "train",
                            "loss": train_loss,
                            "accuracy": train_acc,
                        }
                    )
                    print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}%")

                    evaluate_now = val_loader is not None and (epoch % config.val_interval == 0)
                    if evaluate_now:
                        val_loss, val_acc = _evaluate(model, val_loader, criterion, loss_name, device)
                        epoch_records.append(
                            {
                                "run_name": run_name,
                                "epoch": epoch,
                                "split": "val",
                                "loss": val_loss,
                                "accuracy": val_acc,
                            }
                        )
                        best_val_acc = max(best_val_acc, val_acc)
                        best_epoch = epoch if val_acc >= best_val_acc else best_epoch
                        print(f"             val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

                    test_due = epoch % config.val_interval == 0
                    if test_due:
                        test_loss, test_acc = _evaluate(model, test_loader, criterion, loss_name, device)
                        epoch_records.append(
                            {
                                "run_name": run_name,
                                "epoch": epoch,
                                "split": "test",
                                "loss": test_loss,
                                "accuracy": test_acc,
                            }
                        )
                        best_test_acc = max(best_test_acc, test_acc)
                        print(f"             test_loss={test_loss:.4f} test_acc={test_acc*100:.2f}%")

                    if epoch % config.checkpoint_interval == 0:
                        _save_checkpoint(
                            model,
                            optimizer,
                            epoch,
                            run_checkpoint_dir,
                            config.max_checkpoints,
                            saved_checkpoints,
                        )

                _write_run_log(epoch_records, run_log_path)

                summary = {
                    "run_name": run_name,
                    "model": config.model,
                    "optimizer": optimizer_name,
                    "loss": loss_name,
                    "lr": lr,
                    "epochs": config.epochs,
                    "best_val_acc": best_val_acc if val_loader is not None else "",
                    "best_test_acc": best_test_acc,
                    "best_epoch": best_epoch if val_loader is not None else "",
                    "log_path": str(run_log_path),
                    "checkpoint_dir": str(run_checkpoint_dir),
                    "config_path": str(run_dir / "config.yaml"),
                }
                _append_results(summary, results_path)


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_path(path: str, base_dir: Path) -> Path:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = (base_dir / path_obj).resolve()
    return path_obj


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r") as fp:
        cfg = yaml.safe_load(fp) or {}

    if "inherit" in cfg:
        base_path = _resolve_path(cfg["inherit"], config_path.parent)
        base_cfg = load_config(base_path)
        cfg.pop("inherit")
        return _deep_update(base_cfg, cfg)
    return cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models on CIFAR-10 with configurable setup.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--device", type=str, help="Override device (cpu/cuda).")
    parser.add_argument("--output-dir", type=str, help="Override output directory.")
    return parser.parse_args()


def build_experiment_config(raw_cfg: Dict[str, Any]) -> ExperimentConfig:
    data_cfg = DataConfig(
        data_root=raw_cfg.get("data_root", "./data"),
        batch_size=raw_cfg.get("batch_size", 128),
        num_workers=raw_cfg.get("num_workers", 4),
        val_split=raw_cfg.get("val_split", 5000),
        seed=raw_cfg.get("seed", 42),
        download=raw_cfg.get("download", True),
    )

    output_dir = Path(raw_cfg.get("output_dir", "./output")).resolve()

    return ExperimentConfig(
        model=raw_cfg.get("model", "ResNet18"),
        device=raw_cfg.get("device", "cuda"),
        learning_rates=raw_cfg.get("learning_rates", [raw_cfg.get("optimizer", {}).get("lr", 1e-3)]),
        loss_functions=raw_cfg.get("loss_functions", ["cross_entropy"]),
        optimizer_config=raw_cfg.get("optimizer", {}),
        epochs=raw_cfg.get("epochs", 30),
        log_interval=raw_cfg.get("log_interval", 50),
        val_interval=raw_cfg.get("val_interval", 1),
        checkpoint_interval=raw_cfg.get("checkpoint_interval", 1),
        max_checkpoints=raw_cfg.get("max_checkpoints"),
        output_dir=output_dir,
        seed=raw_cfg.get("seed", 42),
        weight_decay=raw_cfg.get("weight_decay", raw_cfg.get("optimizer", {}).get("weight_decay", 0.0)),
        data_cfg=data_cfg,
    )


def main() -> None:
    args = _parse_args()
    config_path = Path(args.cfg).resolve()
    raw_cfg = load_config(config_path)

    if args.epochs is not None:
        raw_cfg["epochs"] = args.epochs
    if args.device is not None:
        raw_cfg["device"] = args.device
    if args.output_dir is not None:
        raw_cfg["output_dir"] = args.output_dir

    experiment_cfg = build_experiment_config(raw_cfg)
    run_experiments(experiment_cfg, raw_cfg, config_path)


if __name__ == "__main__":
    main()
