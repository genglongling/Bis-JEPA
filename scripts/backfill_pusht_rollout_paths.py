#!/usr/bin/env python3
"""Backfill real agent paths into PushT rollout figure cache meta."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO / "docs" / "figures" / "rollout_examples" / "cache"
PLAN_ROOT = REPO / "docs" / "figures" / "rollout_examples" / "_plan_runs"
TARGETS_PATH = REPO / "docs" / "figures" / "rollout_examples" / "pusht_plan_targets.pkl"
PUSHT_RENDER_SCALE = 512 / 224

FIGURES = (
    "pusht_dinowm_unsuccessful",
    "pusht_dinobisim_successful",
)


def _subsample_indices(length: int, n_steps: int) -> np.ndarray:
    if length <= n_steps:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, n_steps, dtype=int)


def _detect_agent_px(img_rgb: np.ndarray) -> np.ndarray | None:
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    mask = (r < 100) & (g > 80) & (g < 140) & (b > 180)
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    contour = max(cnts, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float32,
    )


def _agent_xy_from_frames(frame_dir: Path) -> list[list[float]]:
    step_files = sorted(frame_dir.glob("step_*.png"))
    if not step_files:
        raise FileNotFoundError(f"No rollout frames in {frame_dir}")
    traj: list[list[float]] = []
    for step_path in step_files:
        img = np.asarray(Image.open(step_path).convert("RGB"))
        px = _detect_agent_px(img)
        if px is None:
            raise RuntimeError(f"Could not detect agent in {step_path}")
        world = (px * PUSHT_RENDER_SCALE).tolist()
        traj.append(world)
    return traj


def backfill_figure(
    figure_name: str,
    *,
    cache_root: Path = CACHE_ROOT,
    plan_root: Path = PLAN_ROOT,
    n_steps: int = 6,
) -> None:
    meta_path = cache_root / figure_name / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing cache meta: {meta_path}")

    meta = json.loads(meta_path.read_text())
    nc_frames = plan_root / f"{figure_name}_NC" / "frames" / "output_final_eval0"
    nc_meta_path = plan_root / f"{figure_name}_NC" / "frames" / "output_final_meta.json"

    if nc_meta_path.is_file():
        run_meta = json.loads(nc_meta_path.read_text())
        if "agent_xy_traj" in run_meta:
            traj = run_meta["agent_xy_traj"]
            source = "planning export meta"
        else:
            traj = _agent_xy_from_frames(nc_frames)
            source = "vision backfill from NC frames"
    else:
        traj = _agent_xy_from_frames(nc_frames)
        source = "vision backfill from NC frames"

    idxs = meta.get("subsample_indices")
    if idxs is None:
        idxs = _subsample_indices(len(traj), n_steps).tolist()
    meta["subsample_indices"] = idxs
    meta["agent_xy_steps"] = [traj[int(i)] for i in idxs]

    if TARGETS_PATH.is_file():
        targets = pickle.load(open(TARGETS_PATH, "rb"))
        if "goal_pose" in targets:
            meta["goal_pose"] = targets["goal_pose"]

    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Updated {figure_name} ({source}) -> {meta_path}")
    print(f"  start={meta['state_0'][0][:2]} steps={len(meta['agent_xy_steps'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)
    parser.add_argument("--plan-root", type=Path, default=PLAN_ROOT)
    parser.add_argument("--n-steps", type=int, default=6)
    parser.add_argument("--figure", choices=FIGURES, action="append")
    args = parser.parse_args()

    figures = args.figure or FIGURES
    for figure_name in figures:
        backfill_figure(
            figure_name,
            cache_root=args.cache_dir,
            plan_root=args.plan_root,
            n_steps=args.n_steps,
        )


if __name__ == "__main__":
    main()
