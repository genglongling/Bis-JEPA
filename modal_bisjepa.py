#!/usr/bin/env python3
"""
Bis-JEPA on Modal: 4×A100 DDP train + 4×A100 other-11 planning eval.

Avoids GCP A100 quota/capacity. Uses HuggingFace Accelerate (DDP under the hood).
Matches Anderson resume_baselines_gpu012 (live DINO, steps_per_epoch=105340, 50 epochs)
except batch_size=40 for 4-GPU.

Setup (once):
  pip install modal
  modal token new
  modal run modal_bisjepa.py --action upload-data
  # optional check:
  modal run modal_bisjepa.py --action verify

Train (both jobs on one app — prefer deployed):
  modal deploy modal_bisjepa.py
  modal run modal_bisjepa.py --action resume-deployed --roll both

Or ephemeral single app with both:
  modal run --detach modal_bisjepa.py --action train-both

Or one roll on deployed app:
  modal run modal_bisjepa.py --action resume-deployed --roll nfull

Eval subset (T1 T2 V1 V2): one A100:1 container per (model, condition).
  5 models × 4 conditions = 20 containers (dinobisim_n1000 added after upload):
  modal deploy modal_bisjepa.py
  modal run modal_bisjepa.py --action eval-other11

Dim ablation n1000 (4× containers × 4×A100 each, one workspace):
  wm32 / wm128 (DINO-WM + MLP) + bisim128 / bisim384 (DINO-Bisim) — Anderson settings
  modal deploy modal_bisjepa.py
  modal run --detach modal_bisjepa.py --action train-dim-n1000

Full pipeline (verify then both trains on same app):
  modal run --detach modal_bisjepa.py --action pipeline

Resume: re-run the same --roll; hydra.run.dir is fixed under the ckpt volume.
Override epochs:  modal run --detach modal_bisjepa.py --action train --roll n5000 --epochs 10

Logs:
  Modal dashboard app logs
  modal volume get bis-jepa-ckpts logs/modal ./modal_logs

Volumes:
  bis-jepa-data   → /data          (pusht_noise)
  bis-jepa-ckpts  → /ckpts         (outputs/by_tag, logs, plan_outputs)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

APP_NAME = "bis-jepa"
REPO_REMOTE = "/root/Bis-JEPA"
DATA_MOUNT = "/data"
CKPT_MOUNT = "/ckpts"
NUM_GPUS = 4
# Modal max function timeout is 24h; n5000/nfull @ 50 epochs may need resume.
DEFAULT_TIMEOUT_S = 24 * 60 * 60

# Active eval set: Texture 1/2 + mild Viewpoint 1/2 (was full other-11).
OTHER11_CONDITIONS: Tuple[str, ...] = (
    "T1",
    "T2",
    "V1",
    "V2",
)
# Five models ready on Modal now (dinobisim_n1000 uploaded later).
EVAL_OTHER11_MODELS: Tuple[str, ...] = (
    "dinowm_n1000",
    "dinowm_n5000",
    "dinowm_nfull",
    "dinobisim_n5000",
    "dinobisim_nfull",
)
GATED_UNTIL_TRAINED: Tuple[str, ...] = ()

# n1000: one 4×A100 container per tag (Anderson configs).
# WM = train_dinowm_local + wm_latent_dim (same patch MLP as bisim).
# Bisim = train_local + bisim_latent_dim.
# (tag, method, latent_dim)
DIM_ABLATION_N1000: Tuple[Tuple[str, str, int], ...] = (
    ("wm32_n1000", "dinowm", 32),
    ("bisim128_n1000", "bisim", 128),
    ("wm128_n1000", "dinowm", 128),
    ("bisim384_n1000", "bisim", 384),
)

data_vol = modal.Volume.from_name("bis-jepa-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("bis-jepa-ckpts", create_if_missing=True)

# Train + PushT planning deps. Skip mujoco/jax/d4rl stack.
_pip = [
    "accelerate==0.26.1",
    "hydra-core==1.2.0",
    "omegaconf",
    "einops==0.4.1",
    "wandb",
    "tqdm",
    "numpy",
    "pillow",
    "opencv-python-headless",
    "h5py",
    "scipy",
    "decord",
    "timm",
    "torchvision",
    # Planning / PushT env
    "gym==0.23.1",
    "pygame==2.5.2",
    "pymunk==6.8.0",
    "shapely==2.0.3",
    "scikit-image==0.19.3",
    "cloudpickle==2.1.0",
    "imageio",
    "imageio-ffmpeg==0.4.9",
    "matplotlib==3.5.3",
    "psutil==5.9.8",
]

_REPO_ROOT = Path(__file__).resolve().parent
_CODE_DIRS = (
    "models",
    "conf",
    "datasets",
    "env",
    "metrics",
    "loss_history",
    "distributed_fn",
    "planning",
)
_CODE_FILES = (
    "train.py",
    "multi_training.py",
    "utils.py",
    "preprocessor.py",
    "custom_resolvers.py",
    "plan.py",
    "eval_pusht_six_conditions.py",
)
_DIR_IGNORE = [
    "**/__pycache__",
    "**/*.pyc",
    ".DS_Store",
    "data",  # datasets/data lives on Modal volume
    "**/data/**",
]


def _build_image() -> modal.Image:
    img = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install(
            "git",
            "ffmpeg",
            "libgl1",
            "libglib2.0-0",
            "libsdl2-2.0-0",
            "libsdl2-image-2.0-0",
            "libsdl2-mixer-2.0-0",
            "libsdl2-ttf-2.0-0",
        )
        .pip_install(
            "torch==2.4.1",
            "torchvision==0.19.1",
            index_url="https://download.pytorch.org/whl/cu121",
        )
        .pip_install(*_pip)
        # Bake DINOv2 into the image so 4 ranks don't race torch.hub downloads.
        .run_commands(
            "python -c \"import torch; "
            "torch.hub._validate_not_a_forked_repo=lambda a,b,c: True; "
            "torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14', trust_repo=True); "
            "print('dinov2 cached OK')\""
        )
        .env(
            {
                "PYTHONUNBUFFERED": "1",
                "WANDB_MODE": "disabled",
                "SDL_VIDEODRIVER": "dummy",
                # expandable_segments triggers CUDA launch failures on Modal A100s
                "NCCL_P2P_DISABLE": "0",
                "NCCL_IB_DISABLE": "1",
                "DATASET_DIR": DATA_MOUNT,
                "CKPT_BASE": CKPT_MOUNT,
                "TMPDIR": "/dev/shm/bis-jepa",
                "TORCH_HOME": "/root/.cache/torch",
            }
        )
    )
    # Only ship small code dirs — never the 90GB+ local datasets / archives.
    for d in _CODE_DIRS:
        img = img.add_local_dir(
            _REPO_ROOT / d,
            remote_path=f"{REPO_REMOTE}/{d}",
            copy=True,
            ignore=_DIR_IGNORE,
        )
    for f in _CODE_FILES:
        img = img.add_local_file(
            _REPO_ROOT / f,
            remote_path=f"{REPO_REMOTE}/{f}",
            copy=True,
        )
    return img.workdir(REPO_REMOTE)


image = _build_image()

app = modal.App(APP_NAME, image=image)



def _run(cmd: list[str], cwd: Optional[str] = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or REPO_REMOTE, check=True)


@app.function(
    volumes={DATA_MOUNT: data_vol},
    timeout=60 * 60,
    cpu=4,
    memory=16384,
)
def remote_check_data() -> str:
    root = Path(DATA_MOUNT) / "pusht_noise" / "train"
    if not root.is_dir():
        return f"MISSING {root} — run: modal run modal_bisjepa.py --action upload-data"
    n = sum(1 for _ in root.rglob("*") if _.is_file())
    data_vol.reload()
    return f"OK {root} files≈{n}"


@app.function(
    gpu=f"A100:{NUM_GPUS}",
    timeout=30 * 60,
    volumes={CKPT_MOUNT: ckpt_vol},
    memory=65536,
)
def verify_ddp(
    continue_pipeline: bool = False,
    epochs: int = 50,
    batch_size: int = 40,
    steps_per_epoch: int = 105340,
) -> str:
    """Explicit DistributedDataParallel smoke test (multi_training.py)."""
    # Clear bad allocator flag if inherited from older image layers.
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    Path("/dev/shm/bis-jepa").mkdir(parents=True, exist_ok=True)
    log_dir = Path(CKPT_MOUNT) / "logs" / "modal"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "multi_training_ddp_verify.log"
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={NUM_GPUS}",
        "multi_training.py",
        f"--expect-gpus={NUM_GPUS}",
        "--steps",
        "5",
    ]
    with open(log, "w") as f:
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=REPO_REMOTE, stdout=f, stderr=subprocess.STDOUT)
    ckpt_vol.commit()
    if proc.returncode != 0:
        # One retry — Modal NCCL can flake on first collective.
        print("DDP verify failed once; retrying...", flush=True)
        with open(log, "a") as f:
            f.write("\n=== RETRY ===\n")
            f.flush()
            proc = subprocess.run(cmd, cwd=REPO_REMOTE, stdout=f, stderr=subprocess.STDOUT)
        ckpt_vol.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"DDP verify failed; see {log}")
    msg = f"DDP verify OK ({NUM_GPUS}×A100) log={log}"
    print(msg, flush=True)
    # Spawn both trains via coordinator so --detach keeps them on one app.
    if continue_pipeline:
        train_both_coordinator.spawn(
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
        )
        print("spawned train_both_coordinator (n5000 + nfull)", flush=True)
    return msg


@app.function(
    # CPU parent keeps the Modal app alive under --detach; children are the two 4×A100 jobs.
    timeout=DEFAULT_TIMEOUT_S,
    cpu=1,
    memory=2048,
)
def train_both_coordinator(
    epochs: int = 50,
    batch_size: int = 40,
    steps_per_epoch: int = 105340,
) -> str:
    """Spawn n5000 + nfull on this same app and wait for both (required for --detach)."""
    print(
        f"coordinator: spawning n5000 + nfull epochs={epochs} batch={batch_size} "
        f"steps_per_epoch={steps_per_epoch}",
        flush=True,
    )
    c1 = train_dinobisim.spawn(
        roll="n5000",
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        continue_pipeline=False,
    )
    c2 = train_dinobisim.spawn(
        roll="nfull",
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        continue_pipeline=False,
    )
    print(f"coordinator: n5000={c1.object_id} nfull={c2.object_id}; waiting...", flush=True)
    # Block so --detach keeps this function (and its children) on one app.
    r1 = c1.get()
    r2 = c2.get()
    msg = f"BOTH DONE | {r1} | {r2}"
    print(msg, flush=True)
    return msg


@app.function(
    timeout=DEFAULT_TIMEOUT_S,
    cpu=1,
    memory=2048,
)
def train_dim_ablation_n1000_coordinator(
    epochs: int = 50,
    batch_size: int = 40,
    steps_per_epoch: int = 105340,
) -> str:
    """
    Spawn 4× A100:4 containers on this app (Anderson configs):
      wm32_n1000 / wm128_n1000     — DINO-WM + wm_latent_dim MLP
      bisim128_n1000 / bisim384_n1000 — DINO-Bisim + bisim_latent_dim
    Wait for all so --detach keeps one workspace alive.
    """
    print(
        f"dim-ablation coordinator: n1000 × {len(DIM_ABLATION_N1000)} "
        f"epochs={epochs} batch={batch_size} steps_per_epoch={steps_per_epoch}",
        flush=True,
    )
    calls = []
    for tag, method, dim in DIM_ABLATION_N1000:
        call = train_variant.spawn(
            tag=tag,
            method=method,
            roll="n1000",
            latent_dim=dim,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
        )
        calls.append((tag, method, dim, call))
        print(
            f"  spawned {tag} method={method} dim={dim} id={call.object_id}",
            flush=True,
        )
    results = []
    for tag, method, dim, call in calls:
        r = call.get()
        results.append(f"{tag}({method},d={dim}): {r}")
        print(f"  finished {tag}: {r}", flush=True)
    msg = "DIM_ABLATION_N1000 DONE | " + " | ".join(results)
    print(msg, flush=True)
    return msg


def _run_train_variant(
    *,
    tag: str,
    method: str = "bisim",
    roll: str = "n1000",
    latent_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 40,
    steps_per_epoch: int = 105340,
    val_steps: int = 200,
    num_workers: int = 4,
    bisim_pos_encoding: str = "grid_mlp",
) -> str:
    """Shared train body for Modal GPU containers (accelerate DDP)."""
    method_l = method.strip().lower()
    if method_l in ("dinowm", "wm"):
        config_name = "train_dinowm_local"
        is_bisim = False
    elif method_l in ("bisim", "dinobisim"):
        config_name = "train_local"
        is_bisim = True
    else:
        raise ValueError(f"method must be dinowm|bisim, got {method!r}")

    if roll not in ("n1000", "n5000", "nfull"):
        raise ValueError(f"roll must be n1000|n5000|nfull, got {roll}")
    if batch_size % NUM_GPUS != 0:
        raise ValueError(f"batch_size={batch_size} not divisible by {NUM_GPUS}")
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be > 0, got {latent_dim}")

    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    data_vol.reload()
    train_root = Path(DATA_MOUNT) / "pusht_noise" / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(
            f"missing {train_root}; run modal run modal_bisjepa.py --action upload-data"
        )

    Path("/dev/shm/bis-jepa").mkdir(parents=True, exist_ok=True)
    out_dir = Path(CKPT_MOUNT) / "outputs" / "by_tag" / tag
    log_dir = Path(CKPT_MOUNT) / "logs" / "modal"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{tag}_4gpu.log"

    os.environ["DATASET_DIR"] = DATA_MOUNT
    os.environ["CKPT_BASE"] = CKPT_MOUNT
    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "disabled")
    os.environ["WANDB_DIR"] = str(Path(CKPT_MOUNT) / "wandb")
    os.environ["TMPDIR"] = "/dev/shm/bis-jepa"
    os.environ["PYTHONUNBUFFERED"] = "1"

    rollout_args: list[str] = []
    if roll == "n1000":
        rollout_args = ["env.dataset.n_rollout=1000"]
    elif roll == "n5000":
        rollout_args = ["env.dataset.n_rollout=5000"]

    # Anderson: dinowm vs bisim configs; dim via wm_latent_dim or bisim_latent_dim.
    if is_bisim:
        extra = [
            f"bisim_latent_dim={latent_dim}",
            f"bisim_pos_encoding={bisim_pos_encoding}",
        ]
        dim_note = f"bisim_latent_dim={latent_dim}"
    else:
        extra = [
            f"wm_latent_dim={latent_dim}",
            f"bisim_pos_encoding={bisim_pos_encoding}",
            f"bisim_hidden_dim=256",
        ]
        dim_note = f"wm_latent_dim={latent_dim}"

    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        str(NUM_GPUS),
        "--multi_gpu",
        "--mixed_precision",
        "no",
        "train.py",
        "--config-name",
        config_name,
        f"training.epochs={epochs}",
        f"training.batch_size={batch_size}",
        f"training.steps_per_epoch={steps_per_epoch}",
        f"training.val_steps_per_epoch={val_steps}",
        "training.skip_openloop_rollout=false",
        "training.save_every_x_epoch=1",
        "training.log_image_mosaics_to_wandb=false",
        "use_precomputed_dino=false",
        f"env.num_workers={num_workers}",
        *rollout_args,
        *extra,
        f"wandb_display_name={tag}",
        f"ckpt_base_path={CKPT_MOUNT}",
        f"hydra.run.dir={out_dir}",
    ]

    print(
        f"START {tag} method={method_l} {dim_note} config={config_name} "
        f"epochs={epochs} batch={batch_size} steps_per_epoch={steps_per_epoch} "
        f"live_dino -> {log}",
        flush=True,
    )
    with open(log, "a") as f:
        f.write(
            f"\n=== START {tag} method={method_l} {dim_note} config={config_name} "
            f"epochs={epochs} batch={batch_size} steps_per_epoch={steps_per_epoch} ===\n"
        )
        f.write("+ " + " ".join(cmd) + "\n")
        f.flush()
        ckpt_vol.commit()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_REMOTE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        line_i = 0
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
            line_i += 1
            if line_i % 25 == 0:
                f.flush()
                ckpt_vol.commit()
        rc = proc.wait()
        f.flush()
    ckpt_vol.commit()
    if rc != 0:
        raise RuntimeError(f"train {tag} failed rc={rc}; see volume log {log}")
    msg = f"DONE {tag} method={method_l} {dim_note} -> {out_dir}"
    print(msg, flush=True)
    return msg


@app.function(
    gpu=f"A100:{NUM_GPUS}",
    timeout=DEFAULT_TIMEOUT_S,
    volumes={DATA_MOUNT: data_vol, CKPT_MOUNT: ckpt_vol},
    memory=65536,
)
def train_variant(
    tag: str,
    method: str = "bisim",
    roll: str = "n1000",
    latent_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 40,
    steps_per_epoch: int = 105340,
    val_steps: int = 200,
    num_workers: int = 4,
    bisim_pos_encoding: str = "grid_mlp",
) -> str:
    """Train one named variant: method=dinowm|bisim (Anderson configs)."""
    return _run_train_variant(
        tag=tag,
        method=method,
        roll=roll,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps,
        num_workers=num_workers,
        bisim_pos_encoding=bisim_pos_encoding,
    )


@app.function(
    gpu=f"A100:{NUM_GPUS}",
    timeout=DEFAULT_TIMEOUT_S,
    volumes={DATA_MOUNT: data_vol, CKPT_MOUNT: ckpt_vol},
    memory=65536,
)
def train_dinobisim(
    roll: str = "n5000",
    epochs: int = 50,
    batch_size: int = 40,
    steps_per_epoch: int = 105340,
    val_steps: int = 200,
    num_workers: int = 4,
    bisim_pos_encoding: str = "grid_mlp",
    continue_pipeline: bool = False,
) -> str:
    """
    Train one dinobisim_* baseline with accelerate multi-GPU (DDP).
    roll: n1000 | n5000 | nfull
    Anderson parity: live DINO, steps_per_epoch=105340, epochs=50 (batch_size may differ).
    """
    _ = continue_pipeline  # kept for API compatibility with older callers
    return _run_train_variant(
        tag=f"dinobisim_{roll}",
        method="bisim",
        roll=roll,
        latent_dim=32,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps,
        num_workers=num_workers,
        bisim_pos_encoding=bisim_pos_encoding,
    )


def _model_ready(tag: str) -> bool:
    """True if model_50 (preferred) or model_latest exists under by_tag/<tag>."""
    ckpt = Path(CKPT_MOUNT) / "outputs" / "by_tag" / tag / "checkpoints"
    hydra = Path(CKPT_MOUNT) / "outputs" / "by_tag" / tag / "hydra.yaml"
    if not hydra.is_file():
        return False
    return (ckpt / "model_50.pth").is_file() or (ckpt / "model_latest.pth").is_file()


def _progress_json(tag: str) -> Path:
    return Path(CKPT_MOUNT) / "plan_outputs" / f"sixcond_{tag}_other11_modal.json"


def _part_json(tag: str, cond: str) -> Path:
    d = Path(CKPT_MOUNT) / "plan_outputs" / "other11_parts"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}__{cond}.json"


def _load_progress(tag: str) -> Dict[str, Any]:
    path = _progress_json(tag)
    if not path.is_file():
        return {
            "model_name": f"by_tag/{tag}",
            "conditions": list(OTHER11_CONDITIONS),
            "rates": {c: None for c in OTHER11_CONDITIONS},
            "plan_result_dirs": {c: None for c in OTHER11_CONDITIONS},
            "failed": [],
            "ts": None,
        }
    with open(path) as f:
        return json.load(f)


def _save_progress(tag: str, payload: Dict[str, Any]) -> None:
    path = _progress_json(tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["ts"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _record_condition_result(
    tag: str,
    cond: str,
    *,
    ok: bool,
    sr: Optional[float],
    plan_dir: Optional[str],
) -> None:
    """Write per-condition part file + merge into model progress JSON (flock-safe)."""
    import fcntl

    part = {
        "tag": tag,
        "cond": cond,
        "ok": ok,
        "sr": sr,
        "plan_dir": plan_dir,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    part_path = _part_json(tag, cond)
    part_path.write_text(json.dumps(part, indent=2))

    prog_path = _progress_json(tag)
    prog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prog_path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read().strip()
            if raw:
                prog = json.loads(raw)
            else:
                prog = {
                    "model_name": f"by_tag/{tag}",
                    "conditions": list(OTHER11_CONDITIONS),
                    "rates": {c: None for c in OTHER11_CONDITIONS},
                    "plan_result_dirs": {c: None for c in OTHER11_CONDITIONS},
                    "failed": [],
                }
            rates = prog.setdefault("rates", {})
            dirs = prog.setdefault("plan_result_dirs", {})
            failed = prog.setdefault("failed", [])
            if ok and sr is not None:
                rates[cond] = float(sr)
                dirs[cond] = plan_dir
                if cond in failed:
                    failed.remove(cond)
            else:
                if cond not in failed:
                    failed.append(cond)
            prog["model_name"] = f"by_tag/{tag}"
            prog["ckpt_base_path"] = CKPT_MOUNT
            prog["conditions"] = list(OTHER11_CONDITIONS)
            prog["ts"] = datetime.now(timezone.utc).isoformat()
            f.seek(0)
            f.truncate()
            json.dump(prog, f, indent=2)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _condition_done(tag: str, cond: str) -> bool:
    """True if this (tag, cond) already has a successful rate."""
    part_path = _part_json(tag, cond)
    if part_path.is_file():
        try:
            part = json.loads(part_path.read_text())
            if part.get("ok") and part.get("sr") is not None:
                return True
        except json.JSONDecodeError:
            pass
    prog = _load_progress(tag)
    rates = prog.get("rates") or {}
    return rates.get(cond) is not None


def _pending_jobs(models: Tuple[str, ...]) -> List[Tuple[str, str]]:
    """(tag, condition) pairs still needing a successful rate."""
    jobs: List[Tuple[str, str]] = []
    for tag in models:
        if not _model_ready(tag):
            print(f"[eval] skip {tag}: checkpoint/hydra missing on volume", flush=True)
            continue
        for cond in OTHER11_CONDITIONS:
            if not _condition_done(tag, cond):
                jobs.append((tag, cond))
    return jobs


def _run_plan_condition(tag: str, cond: str) -> Dict[str, Any]:
    """Run one plan.py condition on the single visible GPU in this container."""
    os.environ["DATASET_DIR"] = DATA_MOUNT
    os.environ["CKPT_BASE"] = CKPT_MOUNT
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["PYTHONUNBUFFERED"] = "1"
    Path("/dev/shm/bis-jepa").mkdir(parents=True, exist_ok=True)

    log_dir = Path(CKPT_MOUNT) / "logs" / "modal" / "eval_other11"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}_{cond}.log"
    prefix = f"[{tag}|{cond}]"

    cmd = [
        sys.executable,
        "plan.py",
        "--config-name",
        "plan_pusht_local",
        f"ckpt_base_path={CKPT_MOUNT}",
        f"model_name=by_tag/{tag}",
        "wandb_logging=false",
        f"pusht_env.visual_condition={cond}",
        "model_epoch=50",
        "n_evals=50",
        "goal_H=5",
        "goal_source=dset",
        "planner.max_iter=5",
        "planner.sub_planner.num_samples=300",
    ]
    t0 = time.time()
    print(f"{prefix} START model_epoch=50 n_evals=50 goal_H=5", flush=True)
    print(f"{prefix} log_file={log_path}", flush=True)
    print(f"{prefix} cmd={' '.join(cmd)}", flush=True)

    # Stream plan.py stdout/stderr to Modal logs AND the volume log file.
    interesting = (
        "[CEM",
        "[GD]",
        "[PLAN]",
        "MPC iter",
        "self.is_success",
        "Success rate",
        "Loaded ",
        "Resuming from",
        "Planning result",
        "Remapping",
        "Error",
        "Traceback",
        "Dumped plan",
        "eval_seed",
    )
    line_i = 0
    with open(log_path, "w") as logf:
        logf.write(f"+ {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_REMOTE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            line_i += 1
            s = line.rstrip()
            if s and any(k in s for k in interesting):
                print(f"{prefix} {s}", flush=True)
            elif line_i % 200 == 0:
                # heartbeat so silent stretches still show activity
                elapsed = (time.time() - t0) / 60.0
                print(f"{prefix} ... still running elapsed={elapsed:.1f}min lines={line_i}", flush=True)
            if line_i % 50 == 0:
                logf.flush()
                try:
                    ckpt_vol.commit()
                except Exception:
                    pass
        rc = proc.wait()
        logf.flush()
    try:
        ckpt_vol.commit()
    except Exception:
        pass

    elapsed_min = (time.time() - t0) / 60.0
    if rc != 0:
        # Tail last error lines into Modal logs for quick diagnosis.
        try:
            tail = log_path.read_text(errors="ignore").splitlines()[-30:]
            print(f"{prefix} FAIL rc={rc} after {elapsed_min:.1f}min — last lines:", flush=True)
            for tl in tail:
                print(f"{prefix} | {tl}", flush=True)
        except Exception:
            print(f"{prefix} FAIL rc={rc} after {elapsed_min:.1f}min log={log_path}", flush=True)
        return {
            "tag": tag,
            "cond": cond,
            "ok": False,
            "sr": None,
            "plan_dir": None,
            "elapsed_min": elapsed_min,
            "log": str(log_path),
        }

    plan_dir = None
    sr = None
    try:
        text = log_path.read_text(errors="ignore")
        for line in text.splitlines():
            if "Planning result saved dir:" in line:
                plan_dir = line.split("Planning result saved dir:", 1)[1].strip()
        if plan_dir:
            logs_json = Path(plan_dir) / "logs.json"
            if logs_json.is_file():
                for line in logs_json.read_text().strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "final_eval/success_rate" in data:
                        sr = float(data["final_eval/success_rate"])
    except Exception as exc:  # noqa: BLE001
        print(f"{prefix} parse error: {exc}", flush=True)

    print(
        f"{prefix} DONE sr={sr} took={elapsed_min:.1f}min plan_dir={plan_dir}",
        flush=True,
    )
    return {
        "tag": tag,
        "cond": cond,
        "ok": sr is not None,
        "sr": sr,
        "plan_dir": plan_dir,
        "elapsed_min": elapsed_min,
        "log": str(log_path),
    }


@app.function(
    gpu="A100:1",
    timeout=DEFAULT_TIMEOUT_S,
    volumes={DATA_MOUNT: data_vol, CKPT_MOUNT: ckpt_vol},
    memory=65536,
    cpu=16,
)
def eval_one_condition(tag: str, cond: str) -> str:
    """One container = one (model, condition)."""
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("WANDB_MODE", "disabled")
    Path("/dev/shm/bis-jepa").mkdir(parents=True, exist_ok=True)

    data_vol.reload()
    ckpt_vol.reload()

    if not _model_ready(tag):
        raise FileNotFoundError(
            f"missing ckpt/hydra for {tag} under {CKPT_MOUNT}/outputs/by_tag/{tag}"
        )
    if cond not in OTHER11_CONDITIONS:
        raise ValueError(f"unknown condition {cond!r}; expected {OTHER11_CONDITIONS}")

    if _condition_done(tag, cond):
        msg = f"SKIP {tag} {cond}: already done"
        print(msg, flush=True)
        return msg

    result = _run_plan_condition(tag, cond)
    _record_condition_result(
        tag,
        cond,
        ok=bool(result.get("ok")),
        sr=result.get("sr"),
        plan_dir=result.get("plan_dir"),
    )
    ckpt_vol.commit()
    status = "OK" if result.get("ok") else "FAIL"
    msg = (
        f"{status} {tag} {cond} sr={result.get('sr')} "
        f"took={result.get('elapsed_min'):.1f}min"
    )
    print(msg, flush=True)
    return msg


@app.function(
    timeout=60 * 5,
    cpu=1,
    memory=2048,
    volumes={DATA_MOUNT: data_vol, CKPT_MOUNT: ckpt_vol},
)
def eval_other11_status(tail_lines: int = 8) -> str:
    """Summarize per-condition progress + last lines of each condition log."""
    ckpt_vol.reload()
    log_dir = Path(CKPT_MOUNT) / "logs" / "modal" / "eval_other11"
    lines: List[str] = []
    done = 0
    failed = 0
    running = 0
    pending = 0
    for tag in EVAL_OTHER11_MODELS:
        prog = _load_progress(tag)
        rates = prog.get("rates") or {}
        failed_list = set(prog.get("failed") or [])
        lines.append(f"=== {tag} ===")
        for cond in OTHER11_CONDITIONS:
            log_path = log_dir / f"{tag}_{cond}.log"
            sr = rates.get(cond)
            if sr is not None:
                status = f"DONE sr={sr}"
                done += 1
            elif log_path.is_file():
                try:
                    text = log_path.read_text(errors="ignore")
                    tail = text.splitlines()[-tail_lines:] if tail_lines > 0 else []
                    looks_alive = (
                        "[CEM]" in text
                        or "MPC iter" in text
                        or "Success rate" in text
                        or "[PLAN] Model loaded" in text
                        or "Dumped plan targets" in text
                    )
                    looks_dead = (
                        "Could not override" in text
                        or "unexpected keyword argument" in text
                        or "Error executing job" in text
                    ) and not looks_alive
                except Exception:
                    text = ""
                    tail = []
                    looks_alive = False
                    looks_dead = False
                if looks_dead:
                    status = "FAIL"
                    failed += 1
                else:
                    # Log present and not clearly dead — treat as running even if
                    # an earlier attempt put this cond in failed[].
                    status = "RUNNING/LOG"
                    running += 1
            elif cond in failed_list:
                status = "FAIL"
                failed += 1
            else:
                status = "PENDING"
                pending += 1
            lines.append(
                f"  {cond:4s} {status}  log={log_path.name if log_path.is_file() else '-'}"
            )
            if log_path.is_file() and tail_lines > 0:
                try:
                    for tl in tail:
                        lines.append(f"         | {tl}")
                except Exception as exc:  # noqa: BLE001
                    lines.append(f"         | (tail error: {exc})")
    summary = (
        f"summary: done={done} running_or_logged={running} "
        f"failed={failed} pending={pending} "
        f"/ total={len(EVAL_OTHER11_MODELS) * len(OTHER11_CONDITIONS)}"
    )
    lines.insert(0, summary)
    msg = "\n".join(lines)
    print(msg, flush=True)
    return msg


@app.function(
    timeout=60 * 30,
    cpu=1,
    memory=2048,
    volumes={DATA_MOUNT: data_vol, CKPT_MOUNT: ckpt_vol},
)
def eval_other11_fanout() -> str:
    """
    Spawn one A100:1 container per pending (model, condition).
    For the 5 ready models: up to 5×4 = 20 containers. Fire-and-forget spawn;
    re-run to resume any still-pending pairs.
    """
    ckpt_vol.reload()
    jobs = _pending_jobs(EVAL_OTHER11_MODELS)
    print(
        f"[fanout] pending={len(jobs)} / max={len(EVAL_OTHER11_MODELS) * len(OTHER11_CONDITIONS)} "
        f"models={list(EVAL_OTHER11_MODELS)} conditions={list(OTHER11_CONDITIONS)}",
        flush=True,
    )
    if not jobs:
        msg = "fanout: nothing pending"
        print(msg, flush=True)
        return msg

    ids = []
    for tag, cond in jobs:
        call = eval_one_condition.spawn(tag=tag, cond=cond)
        ids.append(f"{tag}:{cond}={call.object_id}")
        print(f"[fanout] spawned {tag} {cond} id={call.object_id}", flush=True)

    msg = f"fanout: spawned {len(ids)} containers (1 GPU each)"
    print(msg, flush=True)
    # Do not wait — each condition can take ~10–16h; parent would hit 24h limit.
    return msg


@app.local_entrypoint()
def main(
    action: str = "pipeline",
    roll: str = "n5000",
    epochs: int = 50,
    batch_size: int = 40,
    steps_per_epoch: int = 105340,
    local_data: str = "",
):
    """
    Actions:
      upload-data  — push local pusht_noise into Modal volume bis-jepa-data
      check-data   — remote sanity check
      verify       — 4×A100 DDP smoke test
      train        — one roll on this (ephemeral) app
      train-both   — spawn n5000 + nfull on this same app (--detach)
      train-dim-n1000 — 4× A100:4: wm32, bisim128, wm128, bisim384 n1000 (--detach)
      resume-deployed — spawn onto permanent deployed bis-jepa app (same workspace app)
      eval-other11 — spawn up to 20× A100:1 (5 models × T1 T2 V1 V2)
      eval-status  — per-condition DONE/FAIL/RUNNING + log tails
      pipeline     — verify → n5000 + nfull in parallel (use --detach)
    """

    action = action.strip().lower()
    repo = Path(__file__).resolve().parent
    default_data = repo / "datasets" / "data" / "pusht_noise"
    src = Path(local_data) if local_data else default_data

    if action == "upload-data":
        if not (src / "train").is_dir():
            raise SystemExit(f"missing {src}/train — set --local-data or place pusht_noise under datasets/data/")
        print(f"Uploading {src} → volume bis-jepa-data:pusht_noise ...")
        subprocess.run(
            [
                "modal",
                "volume",
                "put",
                "bis-jepa-data",
                str(src),
                "pusht_noise",
            ],
            check=True,
        )
        print(remote_check_data.remote())
        return

    if action == "check-data":
        print(remote_check_data.remote())
        return

    if action == "verify":
        print(verify_ddp.remote())
        return

    if action == "train":
        call = train_dinobisim.spawn(
            roll=roll,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            continue_pipeline=False,
        )
        print(f"spawned train_dinobisim({roll}) call_id={call.object_id}")
        print("Monitor: modal app logs  |  volume: bis-jepa-ckpts/logs/modal/")
        return

    if action == "train-both":
        # One parent under --detach; parent waits on both GPU children so they share this app.
        call = train_both_coordinator.spawn(
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
        )
        print(f"spawned train_both_coordinator call_id={call.object_id}")
        print("n5000 + nfull run as children of this same Modal app.")
        print("Monitor: modal app logs  |  volume: bis-jepa-ckpts/logs/modal/")
        return

    if action in ("train-dim-n1000", "train_dim_n1000", "train-dim-ablation"):
        # Prefer deployed app so the 4× A100:4 jobs share one permanent workspace.
        try:
            fn = modal.Function.from_name(APP_NAME, "train_dim_ablation_n1000_coordinator")
            call = fn.spawn(
                epochs=epochs,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
            )
            print(f"deployed-app dim-ablation coordinator call_id={call.object_id}")
        except Exception as exc:
            print(f"deployed lookup failed ({exc}); spawning on this app", flush=True)
            call = train_dim_ablation_n1000_coordinator.spawn(
                epochs=epochs,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
            )
            print(f"spawned dim-ablation coordinator call_id={call.object_id}")
        print(
            "4 containers × A100:4 — wm32_n1000, bisim128_n1000, "
            "wm128_n1000, bisim384_n1000"
        )
        print(f"Monitor: https://modal.com/apps/gll2027/main/deployed/{APP_NAME}")
        print("Logs: bis-jepa-ckpts/logs/modal/{tag}_4gpu.log")
        return

    if action == "resume-deployed":
        roll_l = roll.strip().lower()
        if roll_l in ("n1000", "n5000", "nfull"):
            fn = modal.Function.from_name(APP_NAME, "train_dinobisim")
            call = fn.spawn(
                roll=roll_l,
                epochs=epochs,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                continue_pipeline=False,
            )
            print(f"deployed-app train_dinobisim({roll_l}) call_id={call.object_id}")
        elif roll_l in ("both", "all"):
            fn = modal.Function.from_name(APP_NAME, "train_both_coordinator")
            call = fn.spawn(
                epochs=epochs,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
            )
            print(f"deployed-app coordinator call_id={call.object_id}")
        elif roll_l in ("dim-n1000", "dim_n1000", "dims"):
            fn = modal.Function.from_name(APP_NAME, "train_dim_ablation_n1000_coordinator")
            call = fn.spawn(
                epochs=epochs,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
            )
            print(f"deployed-app dim-ablation coordinator call_id={call.object_id}")
        else:
            raise SystemExit(
                f"resume-deployed --roll must be n1000|n5000|nfull|both|dim-n1000, got {roll!r}"
            )
        print(f"Monitor: https://modal.com/apps/gll2027/main/deployed/{APP_NAME}")
        return

    if action in ("eval-other11", "eval_other11"):
        # Prefer deployed app: spawn one A100:1 per pending (model, condition).
        try:
            fn = modal.Function.from_name(APP_NAME, "eval_other11_fanout")
            call = fn.spawn()
            print(f"deployed-app eval_other11_fanout call_id={call.object_id}")
        except Exception as exc:
            print(f"deployed lookup failed ({exc}); spawning on this app", flush=True)
            call = eval_other11_fanout.spawn()
            print(f"spawned eval_other11_fanout call_id={call.object_id}")
        print(
            "Spawns 5×4 = 20 A100:1 containers (one model×condition each): "
            "dinowm_{n1000,n5000,nfull}, dinobisim_{n5000,nfull} × "
            "T1 T2 V1 V2. Re-run to resume unfinished pairs."
        )
        print(f"Monitor: https://modal.com/apps/gll2027/main/deployed/{APP_NAME}")
        print("Progress: bis-jepa-ckpts/plan_outputs/sixcond_*_other11_modal.json")
        print("Parts: bis-jepa-ckpts/plan_outputs/other11_parts/")
        print("Per-condition logs: bis-jepa-ckpts/logs/modal/eval_other11/{tag}_{cond}.log")
        print("Status: modal run modal_bisjepa.py --action eval-status")
        return

    if action in ("eval-status", "eval_status", "status"):
        try:
            fn = modal.Function.from_name(APP_NAME, "eval_other11_status")
            print(fn.remote(tail_lines=8))
        except Exception as exc:
            print(f"deployed lookup failed ({exc}); running on this app", flush=True)
            print(eval_other11_status.remote(tail_lines=8))
        return

    if action == "pipeline":
        call = verify_ddp.spawn(
            continue_pipeline=True,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
        )
        print(f"spawned pipeline verify→n5000+nfull(parallel)  call_id={call.object_id}")
        print("Monitor: modal app logs  |  volume: bis-jepa-ckpts/logs/modal/")
        return

    raise SystemExit(
        f"unknown action={action!r}; use upload-data|check-data|verify|train|train-both|"
        "train-dim-n1000|resume-deployed|eval-other11|eval-status|pipeline"
    )

