#!/usr/bin/env python3
"""Build rollout figure cache from planning (PushT) and dataset trajectories (PointMaze)."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from env.visual_conditions import LEGACY_SIX_COLUMNS

CACHE_ROOT = REPO / "docs" / "figures" / "rollout_examples" / "cache"
POINTMAZE_SUCCESS_EP = 52
POINTMAZE_FAIL_EP = 550
PUSHT_VAL_EPISODE = 3  # override via --pusht-episode after scan_pusht_rollout_episode.py
PUSHT_WM_MODEL = "2026-07-01/16-35-19"
PUSHT_BISIM_MODEL = "2026-06-26/23-30-32"  # local archive; 2026-07-01/16-30-24 has no ckpts synced


def _subsample_indices(length: int, n_steps: int) -> np.ndarray:
    if length <= n_steps:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, n_steps, dtype=int)


def build_pointmaze_cache(
    *,
    n_steps: int = 6,
    out_root: Path = CACHE_ROOT,
) -> None:
    states = torch.load(
        REPO / "datasets" / "data" / "point_maze" / "states.pth",
        map_location="cpu",
        weights_only=False,
    )
    goal_xy = np.array([1.0, 1.0], dtype=np.float32)
    specs = {
        "pointmaze_dinowm_unsuccessful": POINTMAZE_FAIL_EP,
        "pointmaze_dinobisim_successful": POINTMAZE_SUCCESS_EP,
    }
    for figure_name, ep_idx in specs.items():
        traj = states[ep_idx, :, :2].numpy()
        start_xy = traj[0].astype(np.float32)
        idxs = _subsample_indices(len(traj), n_steps)
        figure_dir = out_root / figure_name
        figure_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "source": "point_maze_dataset",
            "episode": int(ep_idx),
            "start_xy": start_xy.tolist(),
            "target_xy": goal_xy.tolist(),
            "agent_xy_steps": [traj[i].tolist() for i in idxs],
            "step_indices": idxs.tolist(),
        }
        with open(figure_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote PointMaze cache meta -> {figure_dir / 'meta.json'}")


def build_pusht_plan_targets(
    episode: int = PUSHT_VAL_EPISODE,
    goal_h: int = 6,
    out_path: Path | None = None,
) -> Path:
    from env.pusht.pusht_wrapper import PushTWrapper

    val_dir = REPO / "datasets" / "data" / "pusht_noise" / "val"
    states = torch.load(val_dir / "states.pth", map_location="cpu", weights_only=False)
    actions = torch.load(val_dir / "rel_actions.pth", map_location="cpu", weights_only=False)
    seq_lengths = pickle.load(open(val_dir / "seq_lengths.pkl", "rb"))
    velocities = torch.load(val_dir / "velocities.pth", map_location="cpu", weights_only=False)

    ep_len = int(seq_lengths[episode])
    state_traj = torch.cat(
        [states[episode, :ep_len], velocities[episode, :ep_len]],
        dim=-1,
    ).numpy()
    init_state = state_traj[0].astype(np.float32)
    goal_state = state_traj[-1].astype(np.float32)

    env = PushTWrapper(with_velocity=True, with_target=True, visual_condition="NC")
    obs_0, _ = env.prepare(99, init_state)
    obs_g, _ = env.prepare(99, goal_state)
    obs_0 = {k: v[np.newaxis, np.newaxis, ...] for k, v in obs_0.items()}
    obs_g = {k: v[np.newaxis, np.newaxis, ...] for k, v in obs_g.items()}

    act = actions[episode, : goal_h * 5]
    gt_actions = act.reshape(goal_h, 5 * act.shape[-1]).numpy()[None, ...]

    payload = {
        "obs_0": obs_0,
        "obs_g": obs_g,
        "state_0": init_state[None, ...],
        "state_g": goal_state[None, ...],
        "gt_actions": gt_actions,
        "goal_H": goal_h,
        "goal_pose": env.goal_pose.tolist(),
        "val_episode": int(episode),
    }
    out_path = out_path or REPO / "docs" / "figures" / "rollout_examples" / "pusht_plan_targets.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Wrote PushT plan targets -> {out_path}")
    return out_path


def capture_pusht_planning_rollouts(
    *,
    n_steps: int = 6,
    ckpt_base_path: Path = REPO / "anderson_archive",
    targets_path: Path | None = None,
    out_root: Path = CACHE_ROOT,
    device: str | None = None,
    figures: tuple[str, ...] | None = None,
) -> None:
    from plan import planning_main

    targets_path = targets_path or build_pusht_plan_targets()
    all_specs = {
        "pusht_dinowm_unsuccessful": PUSHT_WM_MODEL,
        "pusht_dinobisim_successful": PUSHT_BISIM_MODEL,
    }
    if figures is None:
        specs = all_specs
    else:
        specs = {k: all_specs[k] for k in figures if k in all_specs}
    work_root = REPO / "docs" / "figures" / "rollout_examples" / "_plan_runs"
    work_root.mkdir(parents=True, exist_ok=True)

    targets = pickle.load(open(targets_path, "rb"))
    goal_pose = targets.get("goal_pose")

    for figure_name, model_name in specs.items():
        figure_meta: dict | None = None
        for cond in LEGACY_SIX_COLUMNS:
            run_dir = work_root / f"{figure_name}_{cond}"
            run_dir.mkdir(parents=True, exist_ok=True)
            frames_dir = run_dir / "frames"
            if frames_dir.exists():
                shutil.rmtree(frames_dir)

            cfg_dict = {
                "saved_folder": str(run_dir),
                "ckpt_base_path": str(ckpt_base_path),
                "model_name": model_name,
                "model_epoch": "latest",
                "seed": 99,
                "n_evals": 1,
                "goal_source": "file",
                "goal_file_path": str(targets_path.resolve()),
                "goal_H": n_steps,
                "n_plot_samples": 1,
                "rollout_frames_dir": str(frames_dir),
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
                        "num_samples": 120,
                        "var_scale": 1,
                        "opt_steps": 15,
                        "eval_every": 1,
                    },
                    "name": "mpc_cem",
                },
                "pusht_env": {"visual_condition": cond},
            }
            if device is not None:
                os.environ["PLAN_DEVICE"] = device

            print(f"Planning {figure_name} condition={cond} model={model_name}")
            cwd = os.getcwd()
            os.chdir(run_dir)
            try:
                planning_main(cfg_dict)
            finally:
                os.chdir(cwd)

            src_dir = frames_dir / "output_final_eval0"
            if not src_dir.is_dir():
                raise FileNotFoundError(f"Missing rollout frames: {src_dir}")
            step_files = sorted(src_dir.glob("step_*.png"))
            idxs = _subsample_indices(len(step_files), n_steps)

            dest_root = out_root / figure_name / cond
            dest_root.mkdir(parents=True, exist_ok=True)
            for out_i, src_i in enumerate(idxs):
                shutil.copy2(step_files[src_i], dest_root / f"step_{out_i:02d}.png")

            print(f"Cached {len(idxs)} frames -> {dest_root}")

        nc_run_dir = work_root / f"{figure_name}_NC"
        nc_meta_src = nc_run_dir / "frames" / "output_final_meta.json"
        nc_logs_src = nc_run_dir / "logs.json"
        if nc_meta_src.is_file():
            run_meta = json.loads(nc_meta_src.read_text())
            traj = run_meta.get("agent_xy_traj")
            step_files = sorted(
                (nc_run_dir / "frames" / "output_final_eval0").glob("step_*.png")
            )
            idxs = _subsample_indices(len(step_files), n_steps)
            figure_meta = {
                "source": "planning",
                "model_name": model_name,
                "state_0": run_meta["state_0"],
                "state_g": run_meta["state_g"],
                "frameskip": run_meta.get("frameskip", 5),
                "filename": run_meta.get("filename", "output_final"),
                "subsample_indices": idxs.tolist(),
            }
            if "val_episode" in targets:
                figure_meta["val_episode"] = targets["val_episode"]
            if goal_pose is not None:
                figure_meta["goal_pose"] = goal_pose
            elif "goal_pose" in run_meta:
                figure_meta["goal_pose"] = run_meta["goal_pose"]
            if traj is not None:
                figure_meta["agent_xy_steps"] = [traj[i] for i in idxs]
            if nc_logs_src.is_file():
                final_log = json.loads(nc_logs_src.read_text().strip().split("\n")[-1])
                sr = final_log.get(
                    "final_eval/success_rate",
                    final_log.get("mpc/success_rate"),
                )
                sd = final_log.get(
                    "final_eval/mean_state_dist",
                    final_log.get("mpc/mean_state_dist"),
                )
                figure_meta["planning_success"] = bool(sr)
                figure_meta["mean_state_dist"] = sd
            meta_path = out_root / figure_name / "meta.json"
            meta_path.write_text(json.dumps(figure_meta, indent=2))
            print(f"Wrote figure meta -> {meta_path} (success={figure_meta.get('planning_success')})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=6)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)
    parser.add_argument("--pointmaze-only", action="store_true")
    parser.add_argument("--pusht-only", action="store_true")
    parser.add_argument(
        "--wm-only",
        action="store_true",
        help="PushT planning: DINO-WM figure only.",
    )
    parser.add_argument(
        "--bisim-only",
        action="store_true",
        help="PushT planning: DINO-Bisim figure only.",
    )
    parser.add_argument(
        "--skip-pusht-planning",
        action="store_true",
        help="Only write PushT plan targets, do not run MPC (slow on CPU).",
    )
    parser.add_argument("--ckpt-base", type=Path, default=REPO / "anderson_archive")
    parser.add_argument(
        "--pusht-episode",
        type=int,
        default=PUSHT_VAL_EPISODE,
        help="Val episode index for shared PushT init/goal (plan targets).",
    )
    args = parser.parse_args()

    os.environ["DATASET_DIR"] = str((REPO / "datasets" / "data").resolve())
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    run_pm = not args.pusht_only
    run_pt = not args.pointmaze_only

    if run_pm:
        build_pointmaze_cache(n_steps=args.n_steps, out_root=args.cache_dir)
    if run_pt:
        targets = build_pusht_plan_targets(
            episode=args.pusht_episode,
            goal_h=args.n_steps,
        )
        if not args.skip_pusht_planning:
            figure_filter = None
            if args.wm_only:
                figure_filter = ("pusht_dinowm_unsuccessful",)
            elif args.bisim_only:
                figure_filter = ("pusht_dinobisim_successful",)
            capture_pusht_planning_rollouts(
                n_steps=args.n_steps,
                ckpt_base_path=args.ckpt_base,
                targets_path=targets,
                out_root=args.cache_dir,
                figures=figure_filter,
            )


if __name__ == "__main__":
    main()
