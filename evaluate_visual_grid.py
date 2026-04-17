#!/usr/bin/env python3
"""
Run planning for wall or pusht over the six visual conditions (NC … D) and a grid of
trained checkpoints. Aggregates success rates and writes LaTeX tables.

Example:
  python evaluate_visual_grid.py --config evaluate_visual_grid_config.json

Requires valid checkpoints under ckpt_base_path/outputs/<model_name>/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent

VISUAL_COLUMNS = ("NC", "SC", "C", "LC", "LCG", "D")


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


def latex_escape_model_name(name: str) -> str:
    return (
        name.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def write_latex_table(
    path: Path,
    title: str,
    row_labels: List[str],
    col_labels: List[str],
    rates: Dict[str, Dict[str, Optional[float]]],
) -> None:
    lines = [
        f"% {title} — success rate (mean over eval rollouts).",
        r"% Columns: " + ", ".join(col_labels),
        r"\begin{tabular}{l" + "c" * len(col_labels) + "}",
        r"\hline",
        "Model & " + " & ".join(col_labels) + r" \\",
        r"\hline",
    ]
    for row in row_labels:
        cells = [latex_escape_model_name(row)]
        for c in col_labels:
            v = rates.get(row, {}).get(c)
            cells.append("---" if v is None else f"{v:.2f}")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def run_plan(
    *,
    ckpt_base_path: str,
    model_name: str,
    visual_condition: str,
    env: str,
    config_name: str,
    extra_overrides: List[str],
    timeout_s: int,
    dry_run: bool,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "plan.py"),
        "--config-name",
        config_name,
        f"ckpt_base_path={ckpt_base_path}",
        f"model_name={model_name}",
        "wandb_logging=false",
    ]
    if env == "wall":
        cmd.append(f"wall_env.visual_condition={visual_condition}")
    elif env == "pusht":
        cmd.append(f"pusht_env.visual_condition={visual_condition}")
    else:
        raise ValueError(f"env must be wall or pusht, got {env!r}")
    cmd.extend(extra_overrides)

    print("Running:\n  " + " \\\n  ".join(cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=REPO_ROOT / "evaluate_visual_grid_config.json")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--timeout", type=int, default=7200)
    args = p.parse_args()

    cfg = load_config(args.config)
    ckpt_base_path = cfg["ckpt_base_path"]
    env = cfg["env"]
    if env not in ("wall", "pusht"):
        raise SystemExit("config.env must be 'wall' or 'pusht'")
    config_name = cfg.get("config_name", "plan_wall" if env == "wall" else "plan_pusht")
    row_labels: List[str] = cfg["row_labels"]
    models: Dict[str, str] = cfg["models"]
    if set(models.keys()) != set(row_labels):
        print(
            "Warning: models keys should match row_labels; using row_labels order with models[row]",
            file=sys.stderr,
        )
    conditions: List[str] = cfg.get("conditions", list(VISUAL_COLUMNS))
    extra_overrides: List[str] = cfg.get("extra_hydra_overrides", [])

    rates: Dict[str, Dict[str, Optional[float]]] = {r: {c: None for c in conditions} for r in row_labels}

    for row_label in row_labels:
        model_name = models[row_label]
        for cond in conditions:
            proc = run_plan(
                ckpt_base_path=ckpt_base_path,
                model_name=model_name,
                visual_condition=cond,
                env=env,
                config_name=config_name,
                extra_overrides=extra_overrides,
                timeout_s=args.timeout,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                continue
            if proc.returncode != 0:
                print(proc.stdout[-2000:], file=sys.stderr)
                print(proc.stderr[-2000:], file=sys.stderr)
                print(f"FAILED {row_label} {cond} rc={proc.returncode}", file=sys.stderr)
                continue
            out_dir = extract_planning_result_dir(proc.stdout)
            if not out_dir:
                print("Could not parse output dir from stdout", file=sys.stderr)
                continue
            sr = parse_final_success_rate(Path(out_dir) / "logs.json")
            rates[row_label][cond] = sr
            print(f"{row_label} / {cond}: {sr}")

    out_json = REPO_ROOT / f"evaluate_visual_grid_results_{env}.json"
    if not args.dry_run:
        with out_json.open("w") as f:
            json.dump({"rates": rates, "config": cfg}, f, indent=2)
        print(f"Wrote {out_json}")

    tex_path = REPO_ROOT / ("result_maze.tex" if env == "wall" else "result_pushT.tex")
    title = "Wall (dot navigation)" if env == "wall" else "PushT"
    write_latex_table(tex_path, title, row_labels, conditions, rates)
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
