import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def _read_metrics(log_path: Path) -> Dict[str, List[float]]:
    epochs: List[int] = []
    splits: Dict[str, Dict[int, float]] = {"train": {}, "val": {}, "test": {}}

    with log_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            epoch = int(row["epoch"])
            split = row["split"]
            loss = float(row["loss"])
            if split in splits:
                splits[split][epoch] = loss
                if epoch not in epochs:
                    epochs.append(epoch)

    epochs.sort()
    series = {"epoch": epochs}
    for split, values in splits.items():
        series[split] = [values.get(epoch) for epoch in epochs] if values else []
    return series


def _plot(series: Dict[str, List[float]], title: str, output: Path, show: bool) -> None:
    epochs = series["epoch"]
    if not epochs:
        raise ValueError("No data available to plot.")

    plt.figure(figsize=(10, 6))
    for split in ("train", "val", "test"):
        values = series[split]
        if values:
            plt.plot(epochs, values, label=f"{split} loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss curves from training logs.")
    parser.add_argument("--log", type=str, required=True, help="Path to a run log CSV file.")
    parser.add_argument("--output", type=str, help="Path to save plot image. Defaults to figures/<run_name>_loss.png.")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log).resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    series = _read_metrics(log_path)
    run_name = log_path.parent.name
    title = f"Loss Curves - {run_name}"

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = log_path.parent / "figures" / "loss.png"

    _plot(series, title, output_path, args.show)
    print(f"Saved loss plot to: {output_path}")


if __name__ == "__main__":
    main()
