"""MuJoCo XML builders for PointMaze (no mujoco-py / gym dependency)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np

from .dynamic_mjc import MJCModel

WALL = 10
EMPTY = 11
GOAL = 12

U_MAZE = (
    "#####\\"
    + "#GOO#\\"
    + "###O#\\"
    + "#OOO#\\"
    + "#####"
)


def parse_maze(maze_str: str) -> np.ndarray:
    lines = maze_str.strip().split("\\")
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == "#":
                maze_arr[w, h] = WALL
            elif tile == "G":
                maze_arr[w, h] = GOAL
            elif tile in (" ", "O", "0"):
                maze_arr[w, h] = EMPTY
            else:
                raise ValueError(f"Unknown tile type: {tile}")
    return maze_arr


def point_maze(
    maze_str: str,
    background_builtin: str = "checker",
    background_rgb1: str = "0.2 0.3 0.4",
    background_rgb2: str = "0.1 0.2 0.3",
) -> MJCModel:
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel("point_maze")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited="false")
    default.geom(
        friction=".5 .1 .1",
        density="1000",
        margin="0.002",
        condim="1",
        contype="2",
        conaffinity="1",
    )

    asset = mjcmodel.root.asset()
    asset.texture(
        type="2d",
        name="groundplane",
        builtin=background_builtin,
        rgb1=background_rgb1,
        rgb2=background_rgb2,
        width=100,
        height=100,
    )
    asset.texture(
        name="skybox",
        type="skybox",
        builtin="flat",
        rgb1="0.95 0.95 0.95",
        width="800",
        height="800",
    )
    asset.material(name="groundplane", texture="groundplane", texrepeat="20 20")
    asset.material(name="wall", rgba=".7 .5 .3 1")
    asset.material(name="target", rgba=".6 .3 .3 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4", diffuse=".8 .8 .8", specular="0.1 0.1 0.1")
    visual.map(znear=".01")
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(
        name="ground",
        size="40 40 0.25",
        pos="0 0 -0.1",
        type="plane",
        contype=1,
        conaffinity=0,
        material="groundplane",
    )

    particle = worldbody.body(name="particle", pos=[1.2, 1.2, 0])
    particle.geom(name="particle_geom", type="sphere", size=0.1, rgba="0.0 0.0 1.0 0.0", contype=1)
    particle.site(name="particle_site", pos=[0.0, 0.0, 0], size=0.2, rgba="0.3 0.6 0.3 1")
    particle.joint(name="ball_x", type="slide", pos=[0, 0, 0], axis=[1, 0, 0])
    particle.joint(name="ball_y", type="slide", pos=[0, 0, 0], axis=[0, 1, 0])

    worldbody.site(
        name="target_site",
        pos=[0.0, 0.0, 0],
        size=0.18,
        rgba="0.2 0.85 0.25 1",
    )
    worldbody.site(
        name="start_site",
        pos=[0.0, 0.0, 0],
        size=0.16,
        rgba="1 0.55 0 1",
    )

    width, height = maze_arr.shape
    for w in range(width):
        for h in range(height):
            if maze_arr[w, h] == WALL:
                worldbody.geom(
                    conaffinity=1,
                    type="box",
                    name=f"wall_{w}_{h}",
                    material="wall",
                    pos=[w + 1.0, h + 1.0, 0],
                    size=[0.5, 0.5, 0.2],
                )

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    return mjcmodel


def write_model_xml(mjcmodel: MJCModel) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    mjcmodel.root.write(tmp)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


# U-maze goal tile (maze grid indices) and typical start / success agent positions.
POINTMAZE_GOAL_XY = np.array([1.0, 1.0], dtype=np.float32)
POINTMAZE_FAIL_AGENT = np.array([1.2, 3.2], dtype=np.float32)
POINTMAZE_SUCCESS_AGENT = np.array([1.2, 1.2], dtype=np.float32)

# Floor colors aligned with PushT canvas fills (NC/SC/C/LC/LCG/D).
POINTMAZE_BG = {
    "NC": dict(
        background_builtin="checker",
        background_rgb1="1.0 1.0 1.0",
        background_rgb2="0.94 0.94 0.94",
    ),
    "SC": dict(
        background_builtin="checker",
        background_rgb1="0.96 0.97 1.0",
        background_rgb2="0.90 0.93 0.98",
    ),
    "C": dict(
        background_builtin="checker",
        background_rgb1="0.90 0.94 1.0",
        background_rgb2="0.84 0.90 0.98",
    ),
    "LC": dict(
        background_builtin="checker",
        background_rgb1="1.0 0.78 0.63",
        background_rgb2="0.95 0.72 0.58",
    ),
    "LCG": dict(
        background_builtin="gradient",
        background_rgb1="0.78 0.86 1.0",
        background_rgb2="1.0 0.78 0.63",
    ),
    "D": dict(
        background_builtin="checker",
        background_rgb1="1.0 1.0 1.0",
        background_rgb2="0.94 0.94 0.94",
    ),
    "D1": dict(
        background_builtin="checker",
        background_rgb1="1.0 1.0 1.0",
        background_rgb2="0.94 0.94 0.94",
    ),
}


def render_pointmaze_rgb(
    agent_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    start_xy: np.ndarray | None = None,
    visual_condition: str = "NC",
    render_size: int = 224,
) -> np.ndarray:
    """Render a top-down PointMaze (U-maze) frame with mujoco>=3."""
    import mujoco

    cond = visual_condition.upper()
    if cond == "D":
        cond = "D"
    bg = POINTMAZE_BG.get(cond, POINTMAZE_BG["NC"])

    mjcmodel = point_maze(U_MAZE, **bg)
    xml_path = write_model_xml(mjcmodel)
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        data.qpos[0] = float(agent_xy[0])
        data.qpos[1] = float(agent_xy[1])
        target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        data.site_xpos[target_id] = np.array(
            [float(target_xy[0]), float(target_xy[1]), 0.0],
            dtype=np.float64,
        )
        start_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "start_site")
        if start_xy is not None:
            data.site_xpos[start_id] = np.array(
                [float(start_xy[0]), float(start_xy[1]), 0.0],
                dtype=np.float64,
            )
        else:
            model.site_size[start_id] = 0.0
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, render_size, render_size)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.azimuth = 90.0
        camera.elevation = -90.0
        camera.distance = 7.0
        camera.lookat[:] = np.array([3.0, 3.0, 0.0])
        renderer.update_scene(data, camera)
        img = renderer.render()
        return np.asarray(img, dtype=np.uint8)
    finally:
        xml_path.unlink(missing_ok=True)
