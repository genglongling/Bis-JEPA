#!/usr/bin/env python3
"""Render PushT / PointMaze visual-condition figures for papers and README."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from env.pusht.pusht_env import PushTEnv
from env.visual_conditions import LEGACY_SIX_COLUMNS

DEFAULT_OUT = REPO / "docs" / "figures" / "pusht_visual_conditions"
EVAL_MONTAGE_OUT = REPO / "docs" / "figures" / "eval_montage_4x6.png"
EVAL_MONTAGE_ICLR_PDF = REPO / "docs" / "figures" / "eval_montage_4x6_iclr.pdf"
EVAL_MONTAGE_ICLR_PNG = REPO / "docs" / "figures" / "eval_montage_4x6_iclr.png"

# ICLR full-width figure (~\\textwidth); square cells, 1 row × 6 conditions.
ICLR_FIGWIDTH = 6.75
ICLR_NCOLS = 6
ICLR_NROWS = 1
ICLR_FIGSIZE = (ICLR_FIGWIDTH, ICLR_FIGWIDTH / ICLR_NCOLS)

MONTAGE_ROW_KINDS_ICLR = ("pusht_success",)
MONTAGE_ROW_KINDS_FULL = ("pm_fail", "pm_success", "pusht_fail", "pusht_success")
ICLR_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

PUSHT_GOAL_POSE = np.array([256.0, 256.0, np.pi / 4])
PUSHT_FAIL_STATE = np.array([100.0, 420.0, 150.0, 350.0, 0.5, 0.0, 0.0])
PUSHT_SUCCESS_STATE = np.array(
    [
        PUSHT_GOAL_POSE[0] + 35.0,
        PUSHT_GOAL_POSE[1] + 55.0,
        PUSHT_GOAL_POSE[0],
        PUSHT_GOAL_POSE[1],
        PUSHT_GOAL_POSE[2],
        0.0,
        0.0,
    ]
)

ROLLOUT_FIGURES_DIR = REPO / "docs" / "figures" / "rollout_examples"
ROLLOUT_CACHE_DIR = ROLLOUT_FIGURES_DIR / "cache"
ROLLOUT_NUM_STEPS = 6
PUSHT_CANVAS_SIZE = 512.0
# Affine world (qpos) -> pixel for fixed top-down MuJoCo camera in maze_xml.
POINTMAZE_W2P_A = np.array(
    [
        [35.29775376, 0.31605118],
        [2.94078676, -38.57210938],
        [42.38619506, 179.63132262],
    ],
    dtype=np.float64,
)
TRAJECTORY_COLOR = (210, 45, 45)
TRAJECTORY_THICKNESS = 2
# Shared episode for PointMaze figs 1–2: same start + early steps, then diverge.
POINTMAZE_SHARED_WAYPOINTS = [
    np.array([1.2, 3.2], dtype=np.float32),
    np.array([1.2, 2.9], dtype=np.float32),
    np.array([1.2, 2.5], dtype=np.float32),
]
POINTMAZE_FAIL_WAYPOINTS = POINTMAZE_SHARED_WAYPOINTS + [
    np.array([1.5, 2.2], dtype=np.float32),
    np.array([2.0, 2.4], dtype=np.float32),
    np.array([2.2, 2.5], dtype=np.float32),
]
POINTMAZE_SUCCESS_WAYPOINTS = POINTMAZE_SHARED_WAYPOINTS + [
    np.array([1.2, 2.0], dtype=np.float32),
    np.array([2.0, 2.0], dtype=np.float32),
    np.array([1.2, 1.2], dtype=np.float32),
]
# Shared episode for PushT figs 3–4: same init, same planner horizon, different outcome.
PUSHT_ROLLOUT_INIT = np.array([200.0, 400.0, 200.0, 310.0, 0.15, 0.0, 0.0])

ROW_LABELS = [
    "PointMaze unsuccessful",
    "PointMaze successful",
    "PushT unsuccessful",
    "PushT successful",
]

def _load_maze_xml():
    """Import maze_xml without loading env.pointmaze (mujoco-py)."""
    name = "env.pointmaze.maze_xml"
    if name in sys.modules:
        return sys.modules[name]
    pkg_path = REPO / "env" / "pointmaze"
    for pkg_name, path in [("env", REPO / "env"), ("env.pointmaze", pkg_path)]:
        if pkg_name not in sys.modules:
            mod = types.ModuleType(pkg_name)
            mod.__path__ = [str(path)]
            sys.modules[pkg_name] = mod
    spec = importlib.util.spec_from_file_location(name, pkg_path / "maze_xml.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _setup_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _upscale(img: np.ndarray, size: int, *, smooth: bool = False) -> np.ndarray:
    pil = Image.fromarray(_to_uint8_rgb(img))
    resample = Image.Resampling.LANCZOS if smooth else Image.Resampling.NEAREST
    return np.array(pil.resize((size, size), resample))


def _apply_pointmaze_color_condition(img: np.ndarray, visual_condition: str) -> np.ndarray:
    """Post-render tint so PointMaze columns match PushT lighting-color shifts."""
    cond = visual_condition.upper()
    out = img.astype(np.float32)
    if cond == "LCG":
        h, w = out.shape[:2]
        for x in range(w):
            t = x / max(w - 1, 1)
            tint = np.array(
                [200.0 + 55.0 * t, 220.0 + 35.0 * (1.0 - t), 255.0],
                dtype=np.float32,
            ) / 255.0
            out[:, x] = np.clip(out[:, x] * tint, 0, 255)
        return out.astype(np.uint8)

    tints = {
        "NC": np.array([255, 255, 255], dtype=np.float32),
        "SC": np.array([245, 248, 255], dtype=np.float32),
        "C": np.array([230, 240, 255], dtype=np.float32),
        "LC": np.array([255, 200, 160], dtype=np.float32),
        "D": np.array([255, 255, 255], dtype=np.float32),
        "D1": np.array([255, 255, 255], dtype=np.float32),
    }
    tint = tints.get(cond, tints["NC"])
    blend = 0.55 if cond != "NC" else 0.0
    if blend > 0:
        overlay = np.ones_like(out) * tint
        out = np.clip((1.0 - blend) * out + blend * overlay, 0, 255)
    return out.astype(np.uint8)


def _pointmaze_xy_to_px(xy: np.ndarray, size: int) -> tuple[int, int]:
    del size  # calibration is for render_size=224; scale linearly if needed
    hom = np.array([float(xy[0]), float(xy[1]), 1.0], dtype=np.float64)
    px, py = hom @ POINTMAZE_W2P_A
    return int(round(px)), int(round(py))


def _pusht_xy_to_px(xy: np.ndarray, size: int) -> tuple[int, int]:
    scale = size / PUSHT_CANVAS_SIZE
    return int(round(float(xy[0]) * scale)), int(round(float(xy[1]) * scale))


def _draw_trajectory_path(
    img: np.ndarray,
    path_xy: list[np.ndarray],
    *,
    to_px,
) -> np.ndarray:
    if len(path_xy) < 2:
        return img
    out = img.copy()
    pts = np.array([to_px(np.asarray(p, dtype=np.float32)) for p in path_xy], dtype=np.int32)
    cv2.polylines(
        out,
        [pts],
        isClosed=False,
        color=TRAJECTORY_COLOR,
        thickness=TRAJECTORY_THICKNESS,
        lineType=cv2.LINE_AA,
    )
    return out


def _apply_distractors(img: np.ndarray, distractor_seed: int = 0, frame_idx: int = 0) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    rs = np.random.RandomState((frame_idx + distractor_seed) % (2**32 - 1))
    for _ in range(6):
        cx = int(rs.randint(0, w))
        cy = int(rs.randint(0, h))
        rad = int(rs.randint(3, 10))
        col = tuple(int(x) for x in rs.randint(40, 255, size=3))
        cv2.circle(out, (cx, cy), rad, col, -1)
    cx = int(w // 2 + (w // 3) * np.sin(0.15 * frame_idx))
    cy = int(h // 2 + (h // 3) * np.cos(0.12 * frame_idx))
    cv2.circle(out, (cx, cy), 12, (255, 40, 40), -1)
    return out


def _draw_pusht_markers(
    img: np.ndarray,
    *,
    start_xy: np.ndarray | None,
    target_xy: np.ndarray | None,
) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    scale = w / PUSHT_CANVAS_SIZE

    def _to_px(xy: np.ndarray) -> tuple[int, int]:
        return int(xy[0] * scale), int(xy[1] * scale)

    marker_size = max(8, int(10 * scale))
    thickness = max(1, int(2 * scale))
    if target_xy is not None:
        cv2.drawMarker(
            out,
            _to_px(target_xy),
            color=(30, 180, 30),
            markerType=cv2.MARKER_STAR,
            markerSize=marker_size,
            thickness=thickness,
        )
    if start_xy is not None:
        cv2.drawMarker(
            out,
            _to_px(start_xy),
            color=(30, 120, 255),
            markerType=cv2.MARKER_TRIANGLE_UP,
            markerSize=marker_size,
            thickness=thickness,
        )
    return out


def render_pusht_frame(
    visual_condition: str,
    state: np.ndarray,
    distractor_seed: int = 0,
    *,
    start_xy: np.ndarray | None = None,
    target_xy: np.ndarray | None = None,
    goal_pose: np.ndarray | None = None,
) -> np.ndarray:
    env = PushTEnv(
        visual_condition=visual_condition,
        with_velocity=True,
        with_target=True,
        distractor_seed=distractor_seed,
    )
    env.goal_pose = (goal_pose if goal_pose is not None else PUSHT_GOAL_POSE).copy()
    env.reset_to_state = state.copy()
    obs, _ = env.reset()
    img = _to_uint8_rgb(obs["visual"])
    if target_xy is None and goal_pose is not None:
        target_xy = goal_pose[:2]
    return _draw_pusht_markers(img, start_xy=start_xy, target_xy=target_xy)


def render_pointmaze_frame(
    visual_condition: str,
    agent_xy: np.ndarray,
    *,
    start_xy: np.ndarray | None = None,
    target_xy: np.ndarray | None = None,
    distractor_seed: int = 0,
) -> np.ndarray:
    maze_xml = _load_maze_xml()
    img = maze_xml.render_pointmaze_rgb(
        agent_xy,
        target_xy if target_xy is not None else maze_xml.POINTMAZE_GOAL_XY,
        start_xy=start_xy,
        visual_condition=visual_condition,
        render_size=224,
    )
    img = _apply_pointmaze_color_condition(_to_uint8_rgb(img), visual_condition)
    cond = visual_condition.upper()
    if cond in ("D", "D1"):
        img = _apply_distractors(img, distractor_seed=distractor_seed)
    return img


def render_condition_grid(
    conditions: tuple[str, ...] = LEGACY_SIX_COLUMNS,
    out_dir: Path = DEFAULT_OUT,
    prefix: str = "bisim_full",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for cond in conditions:
        img = render_pusht_frame(cond, PUSHT_FAIL_STATE)
        path = out_dir / f"{prefix}_{cond}.png"
        Image.fromarray(img).save(path)
        frames.append(img)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, cond, img in zip(axes.ravel(), conditions, frames):
        ax.imshow(img)
        ax.set_title(cond, fontsize=14)
        ax.axis("off")
    fig.tight_layout()
    montage_path = out_dir / f"{prefix}_six_conditions.png"
    fig.savefig(montage_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {len(frames)} panels + montage -> {montage_path}")


def _render_montage_row(
    row_kind: str,
    conditions: tuple[str, ...],
    maze_xml,
    *,
    cell_size: int,
    distractor_seed: int,
    smooth_upscale: bool,
) -> list[np.ndarray]:
    row_imgs = []
    for cond in conditions:
        if row_kind == "pm_fail":
            agent = maze_xml.POINTMAZE_FAIL_AGENT
            start = maze_xml.POINTMAZE_FAIL_AGENT
            img = render_pointmaze_frame(
                cond,
                agent,
                start_xy=start,
                target_xy=maze_xml.POINTMAZE_GOAL_XY,
                distractor_seed=distractor_seed,
            )
            img = _draw_trajectory_path(
                img,
                [start, agent],
                to_px=lambda xy: _pointmaze_xy_to_px(xy, img.shape[0]),
            )
        elif row_kind == "pm_success":
            agent = maze_xml.POINTMAZE_SUCCESS_AGENT
            start = maze_xml.POINTMAZE_FAIL_AGENT
            img = render_pointmaze_frame(
                cond,
                agent,
                start_xy=start,
                target_xy=maze_xml.POINTMAZE_GOAL_XY,
                distractor_seed=distractor_seed,
            )
            img = _draw_trajectory_path(
                img,
                [start, agent],
                to_px=lambda xy: _pointmaze_xy_to_px(xy, img.shape[0]),
            )
        elif row_kind == "pusht_fail":
            agent = PUSHT_FAIL_STATE[:2]
            start = PUSHT_ROLLOUT_INIT[:2]
            img = render_pusht_frame(
                cond,
                PUSHT_FAIL_STATE,
                distractor_seed=distractor_seed,
                start_xy=start,
                target_xy=PUSHT_GOAL_POSE[:2],
            )
            img = _draw_trajectory_path(
                img,
                [start, agent],
                to_px=lambda xy: _pusht_xy_to_px(xy, img.shape[0]),
            )
        elif row_kind == "pusht_success":
            agent = PUSHT_SUCCESS_STATE[:2]
            start = PUSHT_ROLLOUT_INIT[:2]
            img = render_pusht_frame(
                cond,
                PUSHT_SUCCESS_STATE,
                distractor_seed=distractor_seed,
                start_xy=start,
                target_xy=PUSHT_GOAL_POSE[:2],
            )
            img = _draw_trajectory_path(
                img,
                [start, agent],
                to_px=lambda xy: _pusht_xy_to_px(xy, img.shape[0]),
            )
        else:
            raise ValueError(f"Unknown montage row kind: {row_kind}")
        row_imgs.append(_upscale(img, cell_size, smooth=smooth_upscale))
    return row_imgs


def _collect_montage_rows(
    conditions: tuple[str, ...],
    row_kinds: tuple[str, ...],
    *,
    cell_size: int,
    distractor_seed: int,
    smooth_upscale: bool,
) -> list[list[np.ndarray]]:
    maze_xml = _load_maze_xml()
    return [
        _render_montage_row(
            kind,
            conditions,
            maze_xml,
            cell_size=cell_size,
            distractor_seed=distractor_seed,
            smooth_upscale=smooth_upscale,
        )
        for kind in row_kinds
    ]


def _style_panel_axes(ax: plt.Axes, *, iclr: bool) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    if iclr:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.45)
            spine.set_edgecolor("0.82")
    else:
        ax.axis("off")


def _lerp_pusht_states(init: np.ndarray, target: np.ndarray, n_steps: int) -> list[np.ndarray]:
    alphas = np.linspace(0.0, 1.0, n_steps)
    states = []
    for a in alphas:
        s = init.copy()
        s[:4] = init[:4] * (1.0 - a) + target[:4] * a
        d_angle = target[4] - init[4]
        d_angle = (d_angle + np.pi) % (2.0 * np.pi) - np.pi
        s[4] = init[4] + a * d_angle
        s[5:] = 0.0
        states.append(s)
    return states


def _render_pointmaze_step_grid(
    waypoints: list[np.ndarray],
    conditions: tuple[str, ...],
    *,
    cell_size: int,
    distractor_seed: int,
    start_xy: np.ndarray | None = None,
    target_xy: np.ndarray | None = None,
) -> list[list[np.ndarray]]:
    rows: list[list[np.ndarray]] = []
    origin = start_xy if start_xy is not None else waypoints[0]
    for step_idx, agent_xy in enumerate(waypoints):
        path = [origin] + waypoints[: step_idx + 1]
        row_imgs = []
        for cond in conditions:
            img = render_pointmaze_frame(
                cond,
                agent_xy,
                start_xy=start_xy,
                target_xy=target_xy,
                distractor_seed=distractor_seed + step_idx,
            )
            img = _draw_trajectory_path(
                img,
                path,
                to_px=lambda xy: _pointmaze_xy_to_px(xy, img.shape[0]),
            )
            row_imgs.append(_upscale(img, cell_size, smooth=True))
        rows.append(row_imgs)
    return rows


def _render_pusht_step_grid(
    states: list[np.ndarray],
    conditions: tuple[str, ...],
    *,
    cell_size: int,
    distractor_seed: int,
    start_xy: np.ndarray | None = None,
    target_xy: np.ndarray | None = None,
    goal_pose: np.ndarray | None = None,
) -> list[list[np.ndarray]]:
    rows: list[list[np.ndarray]] = []
    origin = start_xy if start_xy is not None else states[0][:2]
    for step_idx, state in enumerate(states):
        path = [origin] + [s[:2] for s in states[: step_idx + 1]]
        row_imgs = []
        for cond in conditions:
            img = render_pusht_frame(
                cond,
                state,
                distractor_seed=distractor_seed + step_idx,
                start_xy=start_xy,
                target_xy=target_xy,
                goal_pose=goal_pose,
            )
            img = _draw_trajectory_path(
                img,
                path,
                to_px=lambda xy: _pusht_xy_to_px(xy, img.shape[0]),
            )
            row_imgs.append(_upscale(img, cell_size, smooth=True))
        rows.append(row_imgs)
    return rows


def _load_rollout_cache_meta(figure_name: str, cache_dir: Path) -> dict | None:
    meta_path = cache_dir / figure_name / "meta.json"
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text())


def _rows_from_pointmaze_cache(
    figure_name: str,
    conditions: tuple[str, ...],
    *,
    cache_dir: Path,
    cell_size: int,
    distractor_seed: int,
    n_steps: int,
) -> list[list[np.ndarray]] | None:
    meta = _load_rollout_cache_meta(figure_name, cache_dir)
    if meta is None:
        return None
    start_xy = np.asarray(meta["start_xy"], dtype=np.float32)
    target_xy = np.asarray(meta["target_xy"], dtype=np.float32)
    waypoints = [
        np.asarray(xy, dtype=np.float32) for xy in meta["agent_xy_steps"][:n_steps]
    ]
    return _render_pointmaze_step_grid(
        waypoints,
        conditions,
        cell_size=cell_size,
        distractor_seed=distractor_seed,
        start_xy=start_xy,
        target_xy=target_xy,
    )


def _rows_from_pusht_cache(
    figure_name: str,
    conditions: tuple[str, ...],
    *,
    cache_dir: Path,
    cell_size: int,
    n_steps: int,
    fallback_states: list[np.ndarray] | None = None,
    distractor_seed: int = 0,
) -> list[list[np.ndarray]] | None:
    meta = _load_rollout_cache_meta(figure_name, cache_dir)
    figure_dir = cache_dir / figure_name
    if meta is None or not figure_dir.is_dir():
        return None

    start_xy = np.asarray(meta["state_0"][0][:2], dtype=np.float32)
    goal_pose = np.asarray(meta.get("goal_pose", PUSHT_GOAL_POSE), dtype=np.float32)
    target_xy = goal_pose[:2]
    waypoints = [
        np.asarray(xy, dtype=np.float32)
        for xy in meta.get("agent_xy_steps", [])[:n_steps]
    ]

    rows: list[list[np.ndarray]] = []
    used_cache = False
    for step_idx in range(n_steps):
        if waypoints and step_idx < len(waypoints):
            path = [start_xy] + waypoints[: step_idx + 1]
        elif fallback_states is not None and step_idx < len(fallback_states):
            path = [start_xy] + [s[:2] for s in fallback_states[: step_idx + 1]]
        else:
            path = [start_xy, target_xy]
        row_imgs = []
        for cond_idx, cond in enumerate(conditions):
            frame_path = figure_dir / cond / f"step_{step_idx:02d}.png"
            if frame_path.is_file():
                img = np.array(Image.open(frame_path).convert("RGB"))
                img = _draw_pusht_markers(img, start_xy=start_xy, target_xy=target_xy)
                img = _draw_trajectory_path(
                    img,
                    path,
                    to_px=lambda xy: _pusht_xy_to_px(xy, img.shape[0]),
                )
                used_cache = True
            elif fallback_states is not None and step_idx < len(fallback_states):
                img = render_pusht_frame(
                    cond,
                    fallback_states[step_idx],
                    distractor_seed=distractor_seed + step_idx + cond_idx,
                    start_xy=start_xy,
                    target_xy=target_xy,
                    goal_pose=goal_pose,
                )
                img = _draw_trajectory_path(
                    img,
                    path,
                    to_px=lambda xy: _pusht_xy_to_px(xy, img.shape[0]),
                )
            else:
                return None
            row_imgs.append(_upscale(img, cell_size, smooth=True))
        rows.append(row_imgs)
    return rows if used_cache else None


def _save_iclr_mosaic(rows: list[list[np.ndarray]], stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = _compose_iclr_strip_figure(rows)
    pdf_path = out_dir / f"{stem}_iclr.pdf"
    png_path = out_dir / f"{stem}_iclr.png"
    fig.savefig(pdf_path, pad_inches=0, facecolor="white")
    fig.savefig(png_path, pad_inches=0, facecolor="white")
    plt.close(fig)
    print(f"Wrote {len(rows)}×{len(rows[0])} rollout figure -> {pdf_path}")


def render_rollout_example_figures(
    out_dir: Path = ROLLOUT_FIGURES_DIR,
    *,
    cache_dir: Path = ROLLOUT_CACHE_DIR,
    cell_size: int = 256,
    n_steps: int = ROLLOUT_NUM_STEPS,
    distractor_seed: int = 0,
    use_cache: bool = True,
) -> None:
    """Four step×condition grids: PM fail (WM), PM succ (Bisim), PT fail (WM), PT succ (Bisim)."""
    conditions = LEGACY_SIX_COLUMNS
    maze_xml = _load_maze_xml()
    pm_start = maze_xml.POINTMAZE_FAIL_AGENT
    pm_target = maze_xml.POINTMAZE_GOAL_XY

    pm_fail_rows = None
    pm_success_rows = None
    pt_fail_rows = None
    pt_success_rows = None
    pt_fail_fallback = _lerp_pusht_states(PUSHT_ROLLOUT_INIT, PUSHT_FAIL_STATE, n_steps)
    pt_success_fallback = _lerp_pusht_states(PUSHT_ROLLOUT_INIT, PUSHT_SUCCESS_STATE, n_steps)
    if use_cache:
        pm_fail_rows = _rows_from_pointmaze_cache(
            "pointmaze_dinowm_unsuccessful",
            conditions,
            cache_dir=cache_dir,
            cell_size=cell_size,
            distractor_seed=distractor_seed,
            n_steps=n_steps,
        )
        pm_success_rows = _rows_from_pointmaze_cache(
            "pointmaze_dinobisim_successful",
            conditions,
            cache_dir=cache_dir,
            cell_size=cell_size,
            distractor_seed=distractor_seed,
            n_steps=n_steps,
        )
        pt_fail_rows = _rows_from_pusht_cache(
            "pusht_dinowm_unsuccessful",
            conditions,
            cache_dir=cache_dir,
            cell_size=cell_size,
            n_steps=n_steps,
            fallback_states=pt_fail_fallback,
            distractor_seed=distractor_seed,
        )
        pt_success_rows = _rows_from_pusht_cache(
            "pusht_dinobisim_successful",
            conditions,
            cache_dir=cache_dir,
            cell_size=cell_size,
            n_steps=n_steps,
            fallback_states=pt_success_fallback,
            distractor_seed=distractor_seed,
        )

    if pm_fail_rows is None:
        pm_fail_rows = _render_pointmaze_step_grid(
            POINTMAZE_FAIL_WAYPOINTS[:n_steps],
            conditions,
            cell_size=cell_size,
            distractor_seed=distractor_seed,
            start_xy=pm_start,
            target_xy=pm_target,
        )
    if pm_success_rows is None:
        pm_success_rows = _render_pointmaze_step_grid(
            POINTMAZE_SUCCESS_WAYPOINTS[:n_steps],
            conditions,
            cell_size=cell_size,
            distractor_seed=distractor_seed,
            start_xy=pm_start,
            target_xy=pm_target,
        )

    if pt_fail_rows is None:
        pt_fail_rows = _render_pusht_step_grid(
            pt_fail_fallback,
            conditions,
            cell_size=cell_size,
            distractor_seed=distractor_seed,
            start_xy=PUSHT_ROLLOUT_INIT[:2],
            target_xy=PUSHT_GOAL_POSE[:2],
            goal_pose=PUSHT_GOAL_POSE,
        )
    if pt_success_rows is None:
        pt_success_rows = _render_pusht_step_grid(
            pt_success_fallback,
            conditions,
            cell_size=cell_size,
            distractor_seed=distractor_seed,
            start_xy=PUSHT_ROLLOUT_INIT[:2],
            target_xy=PUSHT_GOAL_POSE[:2],
            goal_pose=PUSHT_GOAL_POSE,
        )

    _save_iclr_mosaic(pm_fail_rows, "pointmaze_dinowm_unsuccessful", out_dir)
    _save_iclr_mosaic(pm_success_rows, "pointmaze_dinobisim_successful", out_dir)
    _save_iclr_mosaic(pt_fail_rows, "pusht_dinowm_unsuccessful", out_dir)
    _save_iclr_mosaic(pt_success_rows, "pusht_dinobisim_successful", out_dir)


def _compose_iclr_strip_figure(rows: list[list[np.ndarray]]) -> plt.Figure:
    """Stitch panels edge-to-edge (no gutters, borders, or margins)."""
    with plt.rc_context(ICLR_RC):
        strips = [np.concatenate(row, axis=1) for row in rows]
        mosaic = np.concatenate(strips, axis=0) if len(strips) > 1 else strips[0]
        height, width = mosaic.shape[:2]
        fig_w = ICLR_FIGWIDTH
        fig_h = fig_w * height / width
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(mosaic, interpolation="lanczos")
        ax.axis("off")
        return fig


def _compose_eval_montage_figure(
    rows: list[list[np.ndarray]],
    conditions: tuple[str, ...],
    *,
    iclr: bool,
) -> plt.Figure:
    nrows, ncols = len(rows), len(conditions)
    if iclr:
        return _compose_iclr_strip_figure(rows)

    fig_w = ncols * 3.2 + 2.0
    fig_h = nrows * 3.2 + 1.2
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.imshow(rows[r][c])
            _style_panel_axes(ax, iclr=False)
            if r == 0:
                ax.set_title(f"{c + 1}. {conditions[c]}", fontsize=16, pad=8)
            if c == 0:
                ax.text(
                    -0.08,
                    0.5,
                    ROW_LABELS[r],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=14,
                )
    fig.suptitle(
        "Visual conditions (NC → D): PointMaze fail/success, PushT fail/success",
        fontsize=18,
        y=0.98,
    )
    fig.subplots_adjust(left=0.12, right=0.99, top=0.93, bottom=0.02, wspace=0.05, hspace=0.08)
    return fig


def render_eval_montage_4x6(
    out_path: Path = EVAL_MONTAGE_OUT,
    *,
    cell_size: int = 280,
    distractor_seed: int = 0,
    iclr: bool = False,
    also_write_iclr: bool = False,
) -> Path:
    conditions = LEGACY_SIX_COLUMNS
    smooth = iclr or also_write_iclr

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _save_iclr(rows: list[list[np.ndarray]], pdf_path: Path) -> None:
        fig = _compose_eval_montage_figure(rows, conditions, iclr=True)
        png_path = pdf_path.with_suffix(".png")
        fig.savefig(pdf_path, pad_inches=0, facecolor="white")
        fig.savefig(png_path, pad_inches=0, facecolor="white")
        plt.close(fig)
        written.extend([pdf_path, png_path])

    if iclr:
        rows = _collect_montage_rows(
            conditions,
            MONTAGE_ROW_KINDS_ICLR,
            cell_size=cell_size,
            distractor_seed=distractor_seed,
            smooth_upscale=smooth,
        )
        pdf_path = out_path if out_path.suffix == ".pdf" else out_path.with_name(out_path.stem + "_iclr.pdf")
        _save_iclr(rows, pdf_path)
    else:
        rows = _collect_montage_rows(
            conditions,
            MONTAGE_ROW_KINDS_FULL,
            cell_size=cell_size,
            distractor_seed=distractor_seed,
            smooth_upscale=smooth,
        )
        fig = _compose_eval_montage_figure(rows, conditions, iclr=False)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(out_path)

    if also_write_iclr and not iclr:
        iclr_rows = _collect_montage_rows(
            conditions,
            MONTAGE_ROW_KINDS_ICLR,
            cell_size=cell_size,
            distractor_seed=distractor_seed,
            smooth_upscale=smooth,
        )
        _save_iclr(iclr_rows, EVAL_MONTAGE_ICLR_PDF)

    for path in written:
        tag = "1×6 ICLR" if "iclr" in path.stem else "4×6"
        print(f"Wrote {tag} eval montage -> {path}")
    return written[0]


def main() -> None:
    _setup_headless()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--montage",
        action="store_true",
        help="Render PushT-only 2×3 montage (legacy six conditions).",
    )
    parser.add_argument(
        "--eval-montage",
        action="store_true",
        help="Render 4×6 PointMaze + PushT success/fail montage (default if no flags).",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=EVAL_MONTAGE_OUT)
    parser.add_argument("--prefix", type=str, default="bisim_full")
    parser.add_argument("--cell-size", type=int, default=280)
    parser.add_argument(
        "--iclr",
        action="store_true",
        help="ICLR paper layout: serif fonts, full text width, PDF+PNG, no suptitle.",
    )
    parser.add_argument(
        "--also-iclr",
        action="store_true",
        help="When rendering default montage, also write ICLR PDF/PNG variants.",
    )
    parser.add_argument(
        "--rollout-figures",
        action="store_true",
        help="Render 4 step×condition rollout grids (PM/PT × WM-fail/Bisim-succ).",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=ROLLOUT_NUM_STEPS,
        help="Number of planning steps (rows) per rollout figure.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore rollout cache and use synthetic fallback trajectories.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROLLOUT_CACHE_DIR,
        help="Directory with captured planning / dataset rollout cache.",
    )
    args = parser.parse_args()

    if args.rollout_figures:
        render_rollout_example_figures(
            out_dir=ROLLOUT_FIGURES_DIR,
            cache_dir=args.cache_dir,
            cell_size=args.cell_size,
            n_steps=args.rollout_steps,
            use_cache=not args.no_cache,
        )
    elif args.montage:
        render_condition_grid(out_dir=args.out_dir, prefix=args.prefix)
    elif args.eval_montage or not args.montage:
        out = args.out
        if args.iclr and out == EVAL_MONTAGE_OUT:
            out = EVAL_MONTAGE_ICLR_PDF
        render_eval_montage_4x6(
            out_path=out,
            cell_size=args.cell_size,
            iclr=args.iclr,
            also_write_iclr=args.also_iclr,
        )


if __name__ == "__main__":
    main()
