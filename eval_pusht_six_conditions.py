#!/usr/bin/env python3
"""
Run PushT planning once per visual condition (NC, SC, C, LC, LCG, D) on a single trained
checkpoint, then write JSON + a simple summary.

Typical use after training (from repo root, with checkpoints under <ckpt_base_path>/outputs/):

  export DATASET_DIR=/path/to/parent/datasets/data
  python eval_pusht_six_conditions.py \\
    --model-name 2026-04-21/18-44-53 \\
    --ckpt-base-path ./

`model_name` is the Hydra run folder under `outputs/`, e.g. `2026-04-21/18-44-53` (date/time).

The six conditions match `env.visual_conditions.VISUAL_COLUMNS` and the paper’s sim render presets.
For multi–checkpoint tables, use `evaluate_visual_grid.py` with a JSON config instead.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep in sync with `env.visual_conditions.VISUAL_COLUMNS` (do not import `env`: triggers heavy deps)
VISUAL_COLUMNS = ("NC", "SC", "C", "LC", "LCG", "D")  # same as env.visual_conditions.VISUAL_COLUMNS


def _sanitize_tag(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return s or "model"


def extract_planning_result_dir(stdout: str) -> Optional[str]:
    for line in stdout.split("\n"):
        if "Planning result saved dir:" in line:
            return line.split("Planning result saved dir:")[-1].strip()
    return None


def parse_final_success_rate(logs_json: Path) -> Optional[float]:
    if not logs_json.is_file():
        return None
    final_rate: Optional[float] = None
    for line in logs_json.read_text().strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "final_eval/success_rate" in data:
            final_rate = float(data["final_eval/success_rate"])
    return final_rate


def run_plan_pusht(
    *,
    ckpt_base_path: str,
    model_name: str,
    visual_condition: str,
    config_name: str,
    extra_overrides: List[str],
    timeout_s: int,
    dry_run: bool,
) -> subprocess.CompletedProcess:
    cmd: List[str] = [
        sys.executable,
        str(REPO_ROOT / "plan.py"),
        "--config-name",
        config_name,
        f"ckpt_base_path={ckpt_base_path}",
        f"model_name={model_name}",
        "wandb_logging=false",
        f"pusht_env.visual_condition={visual_condition}",
    ]
    cmd.extend(extra_overrides)
    if dry_run:
        print("Would run:\n  " + " \\\n  ".join(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    envp = os.environ.copy()
    envp["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + envp.get("PYTHONPATH", "")
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=envp,
    )


def find_checkpoint_dir(ckpt_base_path: str, model_name: str) -> Path:
    return (Path(ckpt_base_path).resolve() / "outputs" / model_name).resolve()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate one PushT model under 6 sim visual conditions (planning success)."
    )
    p.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Subpath under <ckpt_base_path>/outputs/, e.g. 2026-04-21/18-44-53",
    )
    p.add_argument(
        "--ckpt-base-path",
        type=str,
        default="./",
        help="Parent of `outputs/<model_name>/` (default: current dir).",
    )
    p.add_argument(
        "--config-name",
        type=str,
        default="plan_pusht_local",
        help="Hydra planning config (default: plan_pusht_local).",
    )
    p.add_argument(
        "--conditions",
        type=str,
        nargs="*",
        default=list(VISUAL_COLUMNS),
        help=f"Subset or order; default: all {list(VISUAL_COLUMNS)}",
    )
    p.add_argument(
        "--extra",
        dest="extra_hydra",
        nargs="*",
        default=[],
        help="Extra args forwarded to plan.py, e.g. --extra n_evals=20 goal_H=5",
    )
    p.add_argument("--timeout", type=int, default=7200, help="Per-condition subprocess timeout (s).")
    p.add_argument("--dry-run", action="store_true", help="Print plan commands only.")
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Default: plan_outputs/sixcond_<tag>_<ts>.json",
    )
    p.add_argument(
        "--no-check-ckpt",
        action="store_true",
        help="Do not require outputs/<model_name> to exist before running.",
    )
    args = p.parse_args()
    norm_conds: List[str] = []
    for c in args.conditions:
        u = c.strip().upper()
        if u not in VISUAL_COLUMNS:
            p.error(f"Unknown condition {c!r}; expected one of {list(VISUAL_COLUMNS)}")
        norm_conds.append(u)
    args.conditions = norm_conds
    model_name = args.model_name.strip().strip("/")
    ckpt_dir = find_checkpoint_dir(args.ckpt_base_path, model_name)
    if not args.no_check_ckpt and not ckpt_dir.is_dir():
        p.error(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            f"  Set --ckpt-base-path so that {{ckpt}}/outputs/{model_name} exists, or pass --no-check-ckpt."
        )

    rates: Dict[str, Optional[float]] = {c: None for c in args.conditions}
    plan_out_dirs: Dict[str, Optional[str]] = {c: None for c in args.conditions}
    failed: List[str] = []
    for cond in args.conditions:
        print(f"=== Condition {cond} ===", flush=True)
        proc = run_plan_pusht(
            ckpt_base_path=args.ckpt_base_path,
            model_name=model_name,
            visual_condition=cond,
            config_name=args.config_name,
            extra_overrides=args.extra_hydra,
            timeout_s=args.timeout,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue
        if proc.returncode != 0:
            print(proc.stdout[-2000:] if proc.stdout else "", file=sys.stderr)
            print(proc.stderr[-2000:] if proc.stderr else "", file=sys.stderr)
            print(f"FAILED {cond} rc={proc.returncode}", file=sys.stderr)
            failed.append(cond)
            continue
        out_dir = extract_planning_result_dir(proc.stdout or "")
        plan_out_dirs[cond] = out_dir
        if not out_dir:
            print("Could not parse 'Planning result saved dir:' from plan.py stdout", file=sys.stderr)
            failed.append(cond)
            continue
        sr = parse_final_success_rate(Path(out_dir) / "logs.json")
        rates[cond] = sr
        print(f"  success_rate: {sr}  (plan result dir: {out_dir})")

    if args.dry_run:
        return

    out_json = args.output_json
    if out_json is None:
        tag = _sanitize_tag(model_name)
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        out_dir = REPO_ROOT / "plan_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"sixcond_{tag}_{ts}.json"

    payload: Dict[str, Any] = {
        "model_name": model_name,
        "ckpt_base_path": str(Path(args.ckpt_base_path).resolve()),
        "ckpt_dir": str(ckpt_dir),
        "config_name": args.config_name,
        "conditions": list(args.conditions),
        "rates": rates,
        "plan_result_dirs": plan_out_dirs,
        "failed": failed,
        "ts": datetime.now().isoformat(),
    }
    out_json = out_json.resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_json}")

    print()
    print("| " + " | ".join(args.conditions) + " |")
    print("|" + "|".join("---" for _ in args.conditions) + "|")
    print(
        "| "
        + " | ".join("---" if rates[c] is None else f"{rates[c]:.2f}" for c in args.conditions)
        + " |"
    )


if __name__ == "__main__":
    main()
