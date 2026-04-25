#!/usr/bin/env python3
"""
Plot training / validation loss vs epoch from one or more training_loss_log.csv files.

Useful for learning curves and comparing runs (e.g. Ours vs a reproduced DINO-WM) when
data path, seed, and budget are matched — see docs/baseline_comparison.md.

Examples:
  # Single run, mark budget checkpoints (2/10/50/100) on the x-axis, save figure:
  python scripts/plot_learning_curve.py \\
    --csv outputs/2026-04-21/12-00-00/training_loss_log.csv --out learning_curve.png

  # Overlay val_loss for two models (label=path, repeatable):
  python scripts/plot_learning_curve.py --metric val_loss --out both.png \\
    --run "Ours=outputs/2026-04-21/runA/training_loss_log.csv" \\
    --run "DINO-WM repro=outputs/2026-04-21/runB/training_loss_log.csv"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        print(f"Missing CSV: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        print(f"Expected 'epoch' column in {path}", file=sys.stderr)
        sys.exit(1)
    return df.sort_values("epoch").reset_index(drop=True)


def _val_at_or_before(df: pd.DataFrame, col: str, max_epoch: int) -> Optional[float]:
    sub = df[df["epoch"] <= max_epoch]
    if sub.empty or col not in sub.columns:
        return None
    return float(sub[col].iloc[-1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Single run: path to training_loss_log.csv (use with --out).",
    )
    p.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeatable. Compare multiple CSVs, e.g. Ours=path/to/training_loss_log.csv",
    )
    p.add_argument(
        "--metric",
        default="val_loss",
        help="Column to plot (default: val_loss). Common: train_loss, val_loss",
    )
    p.add_argument(
        "--mark-epochs",
        default="2,10,50,100",
        help="Comma-separated epochs for vertical lines (set empty to disable). Default: 2,10,50,100",
    )
    p.add_argument(
        "--print-checkpoints",
        action="store_true",
        help="Print metric value at the last row with epoch <= each mark-epoch (per --run or --csv).",
    )
    p.add_argument("--out", type=Path, default=Path("learning_curve.png"), help="Output PNG path")
    p.add_argument("--title", type=str, default="", help="Optional plot title")
    args = p.parse_args()

    runs: List[Tuple[str, Path]] = []
    if args.run:
        for spec in args.run:
            if "=" not in spec:
                p.error(f"--run expects LABEL=path, got: {spec!r}")
            label, path = spec.split("=", 1)
            runs.append((label.strip(), Path(path.strip())))
    elif args.csv:
        runs.append(("run", args.csv))
    else:
        p.error("Provide --csv or at least one --run LABEL=path")

    mark_epochs: List[int] = []
    if args.mark_epochs.strip():
        for part in args.mark_epochs.split(","):
            part = part.strip()
            if part:
                mark_epochs.append(int(part))
    mark_epochs = sorted(set(mark_epochs))

    fig, ax = plt.subplots(figsize=(9, 5))
    y_max: Optional[float] = None
    y_min: Optional[float] = None

    for label, cpath in runs:
        df = _load_csv(cpath)
        if args.metric not in df.columns:
            print(f"Column {args.metric!r} not in {cpath}", file=sys.stderr)
            sys.exit(1)
        ax.plot(df["epoch"], df[args.metric], label=label, marker="o", markersize=2, linewidth=1.2)
        if args.print_checkpoints and mark_epochs:
            for me in mark_epochs:
                v = _val_at_or_before(df, args.metric, me)
                if v is not None:
                    print(f"{label}\t{args.metric}\tepoch<={me}\t{v:.6f}")
        valid = df[args.metric].dropna()
        if not valid.empty:
            y_max = max(y_max, float(valid.max())) if y_max is not None else float(valid.max())
            y_min = min(y_min, float(valid.min())) if y_min is not None else float(valid.min())

    for me in mark_epochs:
        ax.axvline(me, color="0.5", linestyle="--", alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(args.metric)
    ax.set_title(args.title or f"{args.metric} vs epoch")
    if y_min is not None and y_max is not None and y_max > 0 and y_min > 0 and y_max / y_min > 50:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
