import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def _load_results(results_path: Path) -> List[Dict[str, str]]:
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with results_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        return list(reader)


def _summarize(rows: List[Dict[str, str]]) -> List[Tuple[str, str, float, float, int]]:
    buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        optimizer = row.get("optimizer", "unknown")
        loss = row.get("loss", "unknown")
        best_test_acc = _parse_float(row.get("best_test_acc", "nan"))
        if best_test_acc == best_test_acc:  # nan check
            buckets[(optimizer, loss)].append(best_test_acc)

    summary = []
    for (optimizer, loss), values in buckets.items():
        avg = sum(values) / len(values) if values else float("nan")
        best = max(values) if values else float("nan")
        summary.append((optimizer, loss, avg, best, len(values)))
    summary.sort(key=lambda item: item[2], reverse=True)
    return summary


def _print_summary(summary: List[Tuple[str, str, float, float, int]]) -> None:
    if not summary:
        print("No results to summarize.")
        return

    header = f"{'Optimizer':<12} {'Loss':<15} {'Avg Test Acc (%)':>16} {'Best Test Acc (%)':>18} {'Runs':>6}"
    print(header)
    print("-" * len(header))
    for optimizer, loss, avg, best, count in summary:
        avg_pct = avg * 100 if avg == avg else float("nan")
        best_pct = best * 100 if best == best else float("nan")
        print(f"{optimizer:<12} {loss:<15} {avg_pct:16.2f} {best_pct:18.2f} {count:6d}")


def _print_top_runs(rows: List[Dict[str, str]], top_k: int) -> None:
    scored = []
    for row in rows:
        best_test = _parse_float(row.get("best_test_acc", "nan"))
        if best_test == best_test:
            scored.append((best_test, row))
    scored.sort(key=lambda item: item[0], reverse=True)

    print(f"\nTop {min(top_k, len(scored))} runs by test accuracy:")
    for idx, (score, row) in enumerate(scored[:top_k], start=1):
        print(
            f"{idx:2d}. {row.get('run_name')} | optimizer={row.get('optimizer')} | loss={row.get('loss')} "
            f"| lr={row.get('lr')} | best_test_acc={score*100:.2f}%"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze aggregated experiment results.")
    parser.add_argument(
        "--results",
        type=str,
        default="output/logs/results.csv",
        help="Path to results CSV file (default: output/logs/results.csv).",
    )
    parser.add_argument("--top", type=int, default=5, help="Show top-k runs by test accuracy.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results_path = Path(args.results).resolve()
    rows = _load_results(results_path)

    summary = _summarize(rows)
    _print_summary(summary)
    if args.top > 0:
        _print_top_runs(rows, args.top)


if __name__ == "__main__":
    main()
