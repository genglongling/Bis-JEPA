"""Train and evaluate the PushT DINO-Bisim fix on Modal.

n=1000 runs first. n=5000 and the full set are spawned only when the
six-condition mean success rate is above the previous DINO-Bisim n=1000
result (0.36, 0.36, 0.36, 0.32, 0.32, 0.30).

  modal run scripts/modal_pusht.py
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import urllib.request
from pathlib import Path

import modal

APP_NAME = "dinobisim-pusht"
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


def _eval_six(ckpt_base: Path, run_name: str) -> None:
    """Run the six visual conditions four at a time, one GPU each."""
    rates = {}
    plan_dirs = {}
    failed = []
    conds = list(CONDITIONS)
    for start in range(0, len(conds), 4):
        chunk = conds[start : start + 4]
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
    gpu="A100:4",
    timeout=24 * 60 * 60,
    cpu=16,
    memory=65536,
    volumes={"/vol": volume},
)
def train_and_eval(n_rollout: int = 1000) -> dict:
    """Train DINO-Bisim at this rollout cap, evaluate six PushT conditions, maybe continue."""
    os.environ["DATASET_DIR"] = "/vol/data"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "/vol/wandb"
    os.environ["TORCH_HOME"] = "/vol/torch"
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    Path("/vol/wandb").mkdir(parents=True, exist_ok=True)
    Path("/vol/torch").mkdir(parents=True, exist_ok=True)

    _ensure_dataset(Path("/vol/data"))
    volume.commit()

    stop_commit = threading.Event()
    committer = threading.Thread(target=_commit_volume_until, args=(stop_commit,), daemon=True)
    committer.start()

    tag = _tag(n_rollout)
    run_name = f"dinobisim_{tag}"
    ckpt_base = Path("/vol/runs") / tag
    ckpt_base.mkdir(parents=True, exist_ok=True)
    run_dir = ckpt_base / "outputs" / run_name
    n_override = "null" if n_rollout < 0 else str(n_rollout)

    try:
        summary = _train_eval_once(ckpt_base, run_dir, run_name, n_override, n_rollout)
    finally:
        stop_commit.set()
        volume.commit()

    if n_rollout == 1000 and summary["improved_vs_previous_n1000"]:
        print("n=1000 improved. Spawning n=5000 and n=full.", flush=True)
        train_and_eval.spawn(5000)
        train_and_eval.spawn(-1)
    elif n_rollout == 1000:
        print("n=1000 did not improve on the previous DINO-Bisim mean. Stopping.", flush=True)
    return summary


def _train_eval_once(
    ckpt_base: Path, run_dir: Path, run_name: str, n_override: str, n_rollout: int
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
            "accelerate",
            "launch",
            "--num_processes",
            "4",
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
            "env.num_workers=2",
            f"wandb_display_name={run_name}",
            "training.seed=0",
            "training.batch_size=20",
            "training.log_image_mosaics_to_wandb=false",
        ],
        cwd=REPO_REMOTE,
    )
    volume.commit()
    _eval_six(ckpt_base, run_name)
    out_json = ckpt_base / "sixcond.json"
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
def main() -> None:
    summary = train_and_eval.remote(1000)
    print(json.dumps(summary, indent=2))
