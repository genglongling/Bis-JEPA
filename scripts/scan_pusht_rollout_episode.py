#!/usr/bin/env python3
"""Scan val episodes for PushT rollout figures (WM fail + Bisim success, NC)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_rollout_cache import (  # noqa: E402
    PUSHT_BISIM_MODEL,
    PUSHT_WM_MODEL,
    build_pusht_plan_targets,
)
from plan import planning_main  # noqa: E402

PUSHT_WM = PUSHT_WM_MODEL
PUSHT_BISIM = PUSHT_BISIM_MODEL


def _read_success(run_dir: Path) -> tuple[float, float]:
    logs_path = run_dir / "logs.json"
    logs = [
        json.loads(line)
        for line in logs_path.read_text().strip().split("\n")
        if line.strip()
    ]
    final = logs[-1]
    sr = final.get("final_eval/success_rate", final.get("mpc/success_rate"))
    sd = final.get("final_eval/mean_state_dist", final.get("mpc/mean_state_dist"))
    return float(sr), float(sd)


def run_planning_nc(
    *,
    model_name: str,
    targets_path: Path,
    run_dir: Path,
    ckpt_base: Path,
    n_steps: int,
    num_samples: int,
    opt_steps: int,
) -> tuple[float, float]:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    cfg_dict = {
        "saved_folder": str(run_dir.resolve()),
        "ckpt_base_path": str(ckpt_base.resolve()),
        "model_name": model_name,
        "model_epoch": "latest",
        "seed": 99,
        "n_evals": 1,
        "goal_source": "file",
        "goal_file_path": str(targets_path.resolve()),
        "goal_H": n_steps,
        "n_plot_samples": 1,
        "rollout_frames_dir": None,
        "debug_dset_init": False,
        "wandb_logging": False,
        "objective": {
            "_target_": "planning.objectives.create_objective_fn",
            "alpha": 1,
            "base": 2,
            "mode": "last",
        },
        "planner": {
            "_target_": "planning.mpc.MPCPlanner",
            "max_iter": n_steps,
            "n_taken_actions": 1,
            "sub_planner": {
                "target": "planning.cem.CEMPlanner",
                "horizon": n_steps,
                "topk": 20,
                "num_samples": num_samples,
                "var_scale": 1,
                "opt_steps": opt_steps,
                "eval_every": 1,
            },
            "name": "mpc_cem",
        },
        "pusht_env": {"visual_condition": "NC"},
    }

    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        planning_main(cfg_dict)
    finally:
        os.chdir(cwd)
    return _read_success(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=6)
    parser.add_argument("--ckpt-base", type=Path, default=REPO / "anderson_archive")
    parser.add_argument("--work-dir", type=Path, default=REPO / "tmp_scan_pusht_ep")
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--opt-steps", type=int, default=15)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--max-episode", type=int, default=20)
    parser.add_argument("--out-json", type=Path, default=REPO / "tmp_scan_pusht_ep/results.json")
    args = parser.parse_args()

    os.environ["DATASET_DIR"] = str((REPO / "datasets" / "data").resolve())
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if "PLAN_DEVICE" not in os.environ:
        os.environ["PLAN_DEVICE"] = "mps"

    args.work_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    candidates: list[int] = []

    for ep in range(args.start_episode, args.max_episode + 1):
        targets_path = args.work_dir / f"targets_ep{ep}.pkl"
        build_pusht_plan_targets(
            episode=ep,
            goal_h=args.n_steps,
            out_path=targets_path,
        )
        wm_sr, wm_sd = run_planning_nc(
            model_name=PUSHT_WM,
            targets_path=targets_path,
            run_dir=args.work_dir / f"wm_ep{ep}",
            ckpt_base=args.ckpt_base,
            n_steps=args.n_steps,
            num_samples=args.num_samples,
            opt_steps=args.opt_steps,
        )
        bi_sr, bi_sd = run_planning_nc(
            model_name=PUSHT_BISIM,
            targets_path=targets_path,
            run_dir=args.work_dir / f"bisim_ep{ep}",
            ckpt_base=args.ckpt_base,
            n_steps=args.n_steps,
            num_samples=args.num_samples,
            opt_steps=args.opt_steps,
        )
        row = {
            "episode": ep,
            "wm_success": wm_sr,
            "wm_state_dist": wm_sd,
            "bisim_success": bi_sr,
            "bisim_state_dist": bi_sd,
        }
        results.append(row)
        print(
            f"ep {ep:2d}: WM sr={wm_sr} sd={wm_sd:.1f} | "
            f"Bisim sr={bi_sr} sd={bi_sd:.1f}"
        )
        if wm_sr == 0.0 and bi_sr == 1.0:
            candidates.append(ep)
            print("  >>> CANDIDATE (WM fail, Bisim success)")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"candidates": candidates, "results": results}
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out_json}")
    if candidates:
        print(f"Best candidates: {candidates}")
    else:
        print("No strict WM-fail + Bisim-success episodes found.")


if __name__ == "__main__":
    main()
