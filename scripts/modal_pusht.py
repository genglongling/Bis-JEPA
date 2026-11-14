"""Train and evaluate the PushT DINO-Bisim fix on Modal.

n=1000 runs first. The default GPU job spawns n=5000 and the full set only
when the six-condition mean success rate is above the previous DINO-Bisim
n=1000 result (0.36, 0.36, 0.36, 0.32, 0.32, 0.30). The four-A100 variants
train n=1000, n=5000, and the full set at the same time, each in its own
checkpoint directory.

  modal run scripts/modal_pusht.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

import modal

APP_NAME = os.environ.get("DINOBISIM_APP_NAME", "dinobisim-pusht")
VOLUME_NAME = "dinobisim-pusht"
REPO_REMOTE = "/root/bisjepa"
DATA_URL = "https://osf.io/download/k2d8w/?view_only=a56a296ce3b24cceaf408383a175ce28"

# Previous DINO-Bisim PushT success at 1000 rollouts (NC SC C LC LCG D).
PREVIOUS_N1000 = {
    "NC": 0.36,
    "SC": 0.36,
    "C": 0.36,
    "LC": 0.32,
    "LCG": 0.32,
    "D": 0.30,
}
CONDITIONS = ("NC", "SC", "C", "LC", "LCG", "D")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "unzip", "libsdl2-2.0-0")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "hydra-core==1.3.2",
        "antlr4-python3-runtime==4.9.3",
        "omegaconf==2.3.0",
        "accelerate==0.26.1",
        "wandb==0.17.0",
        "einops==0.4.1",
        "tqdm",
        "numpy==1.26.4",
        "pillow",
        "gym==0.23.1",
        "pymunk==6.8.0",
        "pygame==2.5.2",
        "shapely",
        "scikit-image",
        "opencv-python-headless",
        "matplotlib",
        "cloudpickle",
        "imageio",
        "huggingface_hub",
    )
    .run_commands(
        "pip install 'decord==0.6.0' || pip install 'eva-decord==0.6.1'",
        "pip install 'numpy==1.26.4'",
    )
    .add_local_dir(
        ".",
        remote_path=REPO_REMOTE,
        ignore=[
            ".git",
            "outputs",
            "outputs(old)",
            "plan_outputs",
            "released_checkpoints",
            "assets",
            "**/__pycache__",
        ],
    )
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _mean(rates: dict) -> float:
    return sum(rates[k] for k in CONDITIONS) / len(CONDITIONS)


def _improved(rates: dict) -> bool:
    if any(rates.get(k) is None for k in CONDITIONS):
        return False
    return _mean(rates) > _mean(PREVIOUS_N1000)


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error = None
    for attempt in range(1, 6):
        try:
            print(f"Downloading {url} (attempt {attempt})", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "dinobisim-pusht"})
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp.replace(dest)
            print(f"Downloaded {dest.stat().st_size} bytes", flush=True)
            return
        except Exception as exc:
            last_error = exc
            print(f"Download failed: {exc}", flush=True)
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {url}") from last_error


def _run(cmd: list[str], cwd: str) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _ensure_dataset(data_root: Path) -> None:
    marker = data_root / "pusht_noise" / "train" / "states.pth"
    if marker.exists():
        print(f"Dataset already present at {marker.parent}", flush=True)
        return
    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / "pusht_noise.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000_000:
        _download(DATA_URL, zip_path)
    _run(["unzip", "-q", "-o", str(zip_path), "-d", str(data_root / "unzipped")], cwd=str(data_root))
    unzipped = data_root / "unzipped"
    candidates = list(unzipped.rglob("states.pth"))
    train_states = [p for p in candidates if p.parent.name == "train"]
    if not train_states:
        raise FileNotFoundError(f"No train/states.pth under {unzipped}")
    pusht_root = train_states[0].parent.parent
    dest = data_root / "pusht_noise"
    if pusht_root.resolve() != dest.resolve():
        if dest.exists():
            raise FileExistsError(dest)
        dest.symlink_to(pusht_root, target_is_directory=True)
    if not marker.exists():
        raise FileNotFoundError(marker)


def _gpu_count() -> int:
    import torch

    count = int(torch.cuda.device_count())
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    print(f"Using {count} GPU(s): {names}", flush=True)
    return max(count, 1)


def _eval_six(ckpt_base: Path, run_name: str) -> None:
    """Run the six visual conditions, one per GPU, in as many waves as needed."""
    rates = {}
    plan_dirs = {}
    failed = []
    conds = list(CONDITIONS)
    width = _gpu_count()
    for start in range(0, len(conds), width):
        chunk = conds[start : start + width]
        procs = []
        for gpu, cond in enumerate(chunk):
            out_json = ckpt_base / f"sixcond_{cond}.json"
            plan_dir = ckpt_base / "plan_outputs" / cond
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = REPO_REMOTE + os.pathsep + env.get("PYTHONPATH", "")
            cmd = [
                "python",
                "eval_pusht_six_conditions.py",
                "--model-name",
                run_name,
                "--ckpt-base-path",
                str(ckpt_base),
                "--config-name",
                "plan_pusht_local",
                "--no-wandb",
                "--conditions",
                cond,
                "--output-json",
                str(out_json),
                "--extra",
                "n_evals=50",
                "goal_H=5",
                "goal_source=dset",
                "planner.max_iter=5",
                f"hydra.run.dir={plan_dir}",
            ]
            print("Running:", " ".join(cmd), f"CUDA_VISIBLE_DEVICES={gpu}", flush=True)
            procs.append((cond, out_json, subprocess.Popen(cmd, cwd=REPO_REMOTE, env=env)))
        for cond, out_json, proc in procs:
            rc = proc.wait()
            if rc != 0 or not out_json.is_file():
                failed.append(cond)
                rates[cond] = None
                continue
            payload = json.loads(out_json.read_text())
            rates[cond] = payload.get("rates", {}).get(cond)
            plan_dirs[cond] = (payload.get("plan_result_dirs") or {}).get(cond)
            if rates[cond] is None:
                failed.append(cond)
    merged = {
        "model_name": run_name,
        "ckpt_base_path": str(ckpt_base),
        "conditions": list(CONDITIONS),
        "rates": {cond: rates.get(cond) for cond in CONDITIONS},
        "plan_result_dirs": plan_dirs,
        "failed": failed,
    }
    (ckpt_base / "sixcond.json").write_text(json.dumps(merged, indent=2))
    print(json.dumps(merged["rates"], indent=2), flush=True)


def _eval_already_done(out_json: Path) -> bool:
    if not out_json.is_file():
        return False
    try:
        rates = json.loads(out_json.read_text()).get("rates") or {}
    except json.JSONDecodeError:
        return False
    return all(rates.get(cond) is not None for cond in CONDITIONS)


def _tag(n_rollout: int) -> str:
    return "nfull" if n_rollout < 0 else f"n{n_rollout}"


def _commit_volume_until(stop: threading.Event) -> None:
    while not stop.wait(600):
        try:
            volume.commit()
        except Exception as exc:
            print(f"volume commit failed: {exc}", flush=True)


@app.function(
    image=image,
    # Cheapest first. A single GPU is what Modal has been able to place immediately.
    gpu=["T4", "L4", "A10", "L40S", "A100", "H100"],
    timeout=24 * 60 * 60,
    cpu=(2, 8),
    memory=(8192, 32768),
    volumes={"/vol": volume},
)
def train_and_eval(n_rollout: int = 1000) -> dict:
    """Train DINO-Bisim at this rollout cap, evaluate six PushT conditions, maybe continue."""
    return _launch_train(n_rollout, spawn_larger=True)


@app.function(
    image=image,
    gpu="T4:4",
    timeout=24 * 60 * 60,
    cpu=(8, 16),
    memory=(32768, 65536),
    volumes={"/vol": volume},
)
def train_and_eval_t4x4(n_rollout: int = 1000) -> dict:
    """Same n=1000 run on four T4s, stored apart from the single-GPU job."""
    tag = _tag(n_rollout)
    return _launch_train(
        n_rollout,
        run_tag=f"{tag}-d384-t4x4",
        seed_tag=None,
        spawn_larger=False,
    )


@app.function(
    image=image,
    gpu="L4:4",
    timeout=24 * 60 * 60,
    cpu=(8, 16),
    memory=(32768, 65536),
    volumes={"/vol": volume},
)
def train_and_eval_l4x4(n_rollout: int = 1000) -> dict:
    """Same n=1000 run on four L4s, stored apart from the other jobs."""
    tag = _tag(n_rollout)
    return _launch_train(
        n_rollout,
        run_tag=f"{tag}-d384-l4x4",
        seed_tag=None,
        spawn_larger=False,
    )


@app.function(
    image=image,
    gpu="A100:4",
    timeout=24 * 60 * 60,
    cpu=16,
    memory=65536,
    volumes={"/vol": volume},
)
def train_and_eval_a100x4(n_rollout: int = 1000) -> dict:
    """One 384-d rollout cap on four A100s. Launch n=1000, n=5000, and full together."""
    return _launch_a100x4(n_rollout)


def _launch_a100x4(n_rollout: int) -> dict:
    tag = _tag(n_rollout)
    return _launch_train(
        n_rollout,
        run_tag=f"{tag}-d384-a100x4",
        seed_tag=None,
        spawn_larger=False,
    )


@app.function(
    image=image,
    gpu="A10:4",
    timeout=24 * 60 * 60,
    cpu=(8, 16),
    memory=(32768, 65536),
    volumes={"/vol": volume},
)
def train_and_eval_a10x4(n_rollout: int = 1000) -> dict:
    """Same 384-d n=1000 run on four A10s, stored apart from the other jobs."""
    tag = _tag(n_rollout)
    return _launch_train(
        n_rollout,
        run_tag=f"{tag}-d384-a10x4",
        seed_tag=None,
        spawn_larger=False,
    )


def _launch_train(
    n_rollout: int,
    run_tag: str | None = None,
    seed_tag: str | None = None,
    spawn_larger: bool = True,
) -> dict:
    os.environ["DATASET_DIR"] = "/vol/data"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "/vol/wandb" if run_tag is None else f"/vol/wandb/{run_tag}"
    os.environ["TORCH_HOME"] = "/vol/torch"
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    Path("/vol/torch").mkdir(parents=True, exist_ok=True)

    _ensure_dataset(Path("/vol/data"))
    volume.commit()

    stop_commit = threading.Event()
    committer = threading.Thread(target=_commit_volume_until, args=(stop_commit,), daemon=True)
    committer.start()

    tag = run_tag or _tag(n_rollout)
    run_name = f"dinobisim_{tag}"
    ckpt_base = Path("/vol/runs") / tag
    ckpt_base.mkdir(parents=True, exist_ok=True)
    run_dir = ckpt_base / "outputs" / run_name
    n_override = "null" if n_rollout < 0 else str(n_rollout)
    seed_run_dir = None
    if seed_tag is not None:
        seed_run_dir = Path("/vol/runs") / seed_tag / "outputs" / f"dinobisim_{seed_tag}"

    try:
        summary = _train_eval_once(
            ckpt_base, run_dir, run_name, n_override, n_rollout, seed_run_dir=seed_run_dir
        )
    finally:
        stop_commit.set()
        volume.commit()

    if spawn_larger and n_rollout == 1000 and summary["improved_vs_previous_n1000"]:
        print("n=1000 improved. Spawning n=5000 and n=full.", flush=True)
        train_and_eval.spawn(5000)
        train_and_eval.spawn(-1)
    elif spawn_larger and n_rollout == 1000 and not summary["improved_vs_previous_n1000"]:
        print("n=1000 did not improve on the previous DINO-Bisim mean. Stopping.", flush=True)
    return summary


def _epochs_left(run_dir: Path, target: int = 50) -> int:
    ckpt_dir = run_dir / "checkpoints"
    done = 0
    if ckpt_dir.is_dir():
        for path in ckpt_dir.glob("model_*.pth"):
            suffix = path.stem.removeprefix("model_")
            if suffix.isdigit():
                done = max(done, int(suffix))
    left = max(target - done, 0)
    print(f"Checkpoint epoch {done}; {left} epochs left to reach {target}", flush=True)
    return left


def _seed_checkpoints(run_dir: Path, source_run_dir: Path) -> None:
    """Copy the newest numbered checkpoint so a second job can resume without sharing files."""
    dest = run_dir / "checkpoints"
    if any(dest.glob("model_*.pth")):
        print(f"Keeping existing checkpoints in {dest}", flush=True)
        return
    src = source_run_dir / "checkpoints"
    if not src.is_dir():
        print(f"No source checkpoints at {src}", flush=True)
        return
    best: tuple[int, Path] | None = None
    for path in src.glob("model_*.pth"):
        suffix = path.stem.removeprefix("model_")
        if suffix.isdigit() and (best is None or int(suffix) > best[0]):
            best = (int(suffix), path)
    if best is None:
        print(f"No numbered checkpoints in {src}", flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best[1], dest / best[1].name)
    shutil.copy2(best[1], dest / "model_latest.pth")
    print(f"Seeded epoch {best[0]} from {best[1]}", flush=True)


def _train_eval_once(
    ckpt_base: Path,
    run_dir: Path,
    run_name: str,
    n_override: str,
    n_rollout: int,
    seed_run_dir: Path | None = None,
) -> dict:
    # One download. Four accelerate processes would otherwise race on the hub cache.
    _run(
        [
            "python",
            "-c",
            "import torch; torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14', trust_repo=True)",
        ],
        cwd=REPO_REMOTE,
    )
    _run(
        [
            "python",
            "scripts/precompute_dino_features.py",
            "--data-path",
            "/vol/data/pusht_noise",
            f"--n-rollout={n_rollout}",
            "--batch-size",
            "64",
        ],
        cwd=REPO_REMOTE,
    )
    volume.commit()
    # The exhaustive run wrote a partial checkpoint into this directory. Do not resume it.
    marker = ckpt_base / "fast_protocol.json"
    if run_dir.exists() and not marker.exists():
        shutil.rmtree(run_dir)
    marker.write_text(json.dumps({"steps_per_epoch": 1000, "val_steps_per_epoch": 200, "precomputed_dino": True}))
    if seed_run_dir is not None:
        _seed_checkpoints(run_dir, seed_run_dir)
    epochs_left = _epochs_left(run_dir)
    if epochs_left == 0:
        print("Training already reached 50 epochs. Evaluating the saved checkpoint.", flush=True)
    else:
        n_gpu = _gpu_count()
        _run(
        [
            "accelerate",
            "launch",
            "--num_processes",
            str(n_gpu),
            "--num_machines",
            "1",
            "--mixed_precision",
            "no",
            "--dynamo_backend",
            "no",
            "train.py",
            "--config-name",
            "train_local",
            f"ckpt_base_path={ckpt_base}",
            f"hydra.run.dir={run_dir}",
            f"hydra.sweep.dir={run_dir}",
            f"env.dataset.n_rollout={n_override}",
            "env.dataset.use_precomputed_dino=true",
            "env.num_workers=2",
            f"wandb_display_name={run_name}",
            "bisim_latent_dim=384",
            "training.seed=0",
            f"training.batch_size={5 * n_gpu}",
            f"training.epochs={epochs_left}",
            "training.steps_per_epoch=1000",
            "training.val_steps_per_epoch=200",
            "training.log_image_mosaics_to_wandb=false",
            "training.log_ground_truth_mosaic=false",
        ],
        cwd=REPO_REMOTE,
    )
    volume.commit()
    out_json = ckpt_base / "sixcond.json"
    if _eval_already_done(out_json):
        print(f"Six-condition eval already complete at {out_json}. Skipping.", flush=True)
    else:
        _eval_six(ckpt_base, run_name)
        plan_src = Path(REPO_REMOTE) / "plan_outputs"
        plan_dest = ckpt_base / "plan_outputs"
        if plan_src.is_dir():
            subprocess.run(
                ["cp", "-a", str(plan_src), str(plan_dest)],
                check=False,
            )
    rates = json.loads(out_json.read_text())["rates"]
    summary = {
        "tag": run_name,
        "n_rollout": None if n_rollout < 0 else n_rollout,
        "rates": rates,
        "mean": None if any(rates.get(k) is None for k in CONDITIONS) else _mean(rates),
        "previous_n1000_mean": _mean(PREVIOUS_N1000),
        "improved_vs_previous_n1000": _improved(rates),
    }
    (ckpt_base / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(variant: str = "default") -> None:
    fn = {
        "t4x4": train_and_eval_t4x4,
        "l4x4": train_and_eval_l4x4,
        "a100x4": train_and_eval_a100x4,
        "a100x4-n5000": train_and_eval_a100x4,
        "a100x4-nfull": train_and_eval_a100x4,
        "a10x4": train_and_eval_a10x4,
    }.get(variant, train_and_eval)
    n_rollout = {"a100x4-n5000": 5000, "a100x4-nfull": -1}.get(variant, 1000)
    summary = fn.remote(n_rollout)
    print(json.dumps(summary, indent=2))
