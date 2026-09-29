#!/usr/bin/env python3
"""
Run PushT planning once per visual condition on a single trained checkpoint.

Default: category **color_d** (C1–C7 + D1) unless overridden.
All 17 conditions: `--category all` or list `--conditions` explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from subprocess import TimeoutExpired
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_visual_conditions():
    import importlib.util

    path = REPO_ROOT / "env" / "visual_conditions.py"
    spec = importlib.util.spec_from_file_location("visual_conditions", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_vc = _load_visual_conditions()
VISUAL_COLUMNS_DEFAULT = _vc.VISUAL_COLUMNS
VISUAL_CATEGORIES = _vc.VISUAL_CATEGORIES
COLOR_CONDITIONS = _vc.COLOR_CONDITIONS
LEGACY_SIX_COLUMNS = _vc.LEGACY_SIX_COLUMNS


def resolve_eval_conditions(
    conditions: Optional[List[str]],
    category: Optional[str],
) -> List[str]:
    """Default sweep: C1–C7 + D1 via EVAL_CATEGORY=color_d."""
    if conditions:
        return conditions
    if category:
        if category == "all":
            return list(VISUAL_COLUMNS_DEFAULT)
        if category not in VISUAL_CATEGORIES:
            raise ValueError(
                f"Unknown category {category!r}; expected one of "
                f"{list(VISUAL_CATEGORIES)} or 'all'"
            )
        return list(VISUAL_CATEGORIES[category])
    cat = os.environ.get("EVAL_CATEGORY", "color_d")
    if cat == "all":
        return list(VISUAL_COLUMNS_DEFAULT)
    if cat in VISUAL_CATEGORIES:
        return list(VISUAL_CATEGORIES[cat])
    return list(COLOR_CONDITIONS)


def _sanitize_tag(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return s or "model"


def extract_planning_result_dir(stdout: str) -> Optional[str]:
    for line in stdout.split("\n"):
        if "Planning result saved dir:" in line:
            return line.split("Planning result saved dir:")[-1].strip()
    return None


def find_latest_plan_output_dir(model_name: str, since_t: float) -> Optional[Path]:
    """
    When plan.py streams to the terminal we cannot read stdout; pick the newest
    plan_outputs/* directory created after this run started. Names often contain
    replace_slash(model_name) from Hydra.
    """
    root = REPO_ROOT / "plan_outputs"
    if not root.is_dir():
        return None
    tag = _sanitize_tag(model_name.replace("/", "_"))
    candidates: List[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m < since_t - 2.0:
            continue
        candidates.append(p)
    if not candidates:
        return None
    # Prefer dirs whose name echoes the model subpath
    preferred = [p for p in candidates if tag in p.name or model_name.replace("/", "_") in p.name]
    pool = preferred if preferred else candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def resolve_plan_output_dir(repo_root: Path, model_name: str, since_t: float) -> Optional[Path]:
    """
    When plan.py stdout is not captured, find the Hydra run dir written by subprocess.run:
    newest plan_outputs/*/logs.json with a finished final_eval line and mtime after this run began.
    """
    root = repo_root / "plan_outputs"
    if not root.is_dir():
        return None
    tag_slash = model_name.replace("/", "_")
    sanitized = _sanitize_tag(model_name)
    candidates: List[tuple[bool, float, Path]] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        lj = p / "logs.json"
        if not lj.is_file():
            continue
        try:
            log_mtime = lj.stat().st_mtime
        except OSError:
            continue
        if log_mtime < since_t - 120:
            continue
        if parse_final_success_rate(lj) is None:
            continue
        name_hit = sanitized in p.name or tag_slash in p.name
        candidates.append((name_hit, log_mtime, p))
    if not candidates:
        # Fall back: any fresh logs.json from this run window (parallel eval dirs).
        for p in root.iterdir():
            if not p.is_dir():
                continue
            lj = p / "logs.json"
            if not lj.is_file():
                continue
            try:
                log_mtime = lj.stat().st_mtime
            except OSError:
                continue
            if log_mtime < since_t - 120:
                continue
            if parse_final_success_rate(lj) is None:
                continue
            candidates.append((False, log_mtime, p))
    if not candidates:
        return None
    return max(candidates, key=lambda x: (x[0], x[1]))[2]


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
    capture_output: bool = False,
    wandb_logging: bool = True,
) -> subprocess.CompletedProcess:
    cmd: List[str] = [
        sys.executable,
        str(REPO_ROOT / "plan.py"),
        "--config-name",
        config_name,
        f"ckpt_base_path={ckpt_base_path}",
        f"model_name={model_name}",
        f"wandb_logging={str(wandb_logging).lower()}",
        f"pusht_env.visual_condition={visual_condition}",
    ]
    cmd.extend(extra_overrides)
    if dry_run:
        print("Would run:\n  " + " \\\n  ".join(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    envp = os.environ.copy()
    envp["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + envp.get("PYTHONPATH", "")
    envp.setdefault("PYTHONUNBUFFERED", "1")
    envp.setdefault(
        "DATASET_DIR",
        str(REPO_ROOT / "datasets" / "data"),
    )
    # Default: stream child stdout/stderr so long MPC runs do not look "stuck"
    # (capture_output=True hides all output until the process exits).
    kw: dict = dict(
        cwd=str(REPO_ROOT),
        env=envp,
        text=True,
    )
    if timeout_s and timeout_s > 0:
        kw["timeout"] = timeout_s
    if capture_output:
        kw["capture_output"] = True
    else:
        kw["stdout"] = None
        kw["stderr"] = None
    return subprocess.run(cmd, **kw)


def find_checkpoint_dir(ckpt_base_path: str, model_name: str) -> Path:
    return (Path(ckpt_base_path).resolve() / "outputs" / model_name).resolve()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate one PushT model under sim visual conditions (planning success)."
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
        "--category",
        type=str,
        default=None,
        choices=[*VISUAL_CATEGORIES.keys(), "all"],
        help="Evaluate one category preset (default: EVAL_CATEGORY env or 'color_d' = C1–C7 + D1). "
        "Use 'all' for all 17 conditions.",
    )
    p.add_argument(
        "--conditions",
        type=str,
        nargs="*",
        default=None,
        help="Explicit condition list (overrides --category). "
        f"All 17: {list(VISUAL_COLUMNS_DEFAULT)}",
    )
    p.add_argument(
        "--extra",
        dest="extra_hydra",
        nargs="*",
        default=[],
        help="Extra args forwarded to plan.py, e.g. --extra n_evals=20 goal_H=5",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-condition subprocess timeout in seconds. 0 = no limit (recommended for long MPC). "
        "Default 0; use e.g. 14400 to cap a single plan run at 4h.",
    )
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
    p.add_argument(
        "--capture-output",
        action="store_true",
        help="Buffer plan.py output (only for debugging). Default is to stream logs live.",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Do not log planning runs to Weights & Biases (default: W&B on).",
    )
    args = p.parse_args()
    try:
        args.conditions = resolve_eval_conditions(args.conditions, args.category)
    except ValueError as e:
        p.error(str(e))
    norm_conds: List[str] = []
    for c in args.conditions:
        u = c.strip().upper()
        if u == "D":
            u = "D1"
        if u not in VISUAL_COLUMNS_DEFAULT:
            p.error(f"Unknown condition {c!r}; expected one of {list(VISUAL_COLUMNS_DEFAULT)}")
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
    n_cond = len(args.conditions)
    t_all = time.time()
    print(
        f"[EVAL {time.strftime('%H:%M:%S')}] start model={model_name} "
        f"conditions={n_cond} {args.conditions} capture_output={args.capture_output}",
        flush=True,
    )
    for ci, cond in enumerate(args.conditions, 1):
        print(
            f"[EVAL {time.strftime('%H:%M:%S')}] condition {ci}/{n_cond} {cond} starting",
            flush=True,
        )
        if not args.dry_run:
            print(
                "  (CEM+MPC: 50 eps × 5 MPC × 30 CEM steps; 20–60+ min/condition is normal. "
                "Look for [CEM]/[MPC] heartbeats every ~30s.)\n",
                flush=True,
            )
        t0 = time.time()
        try:
            proc = run_plan_pusht(
                ckpt_base_path=args.ckpt_base_path,
                model_name=model_name,
                visual_condition=cond,
                config_name=args.config_name,
                extra_overrides=args.extra_hydra,
                timeout_s=args.timeout,
                dry_run=args.dry_run,
                capture_output=args.capture_output,
                wandb_logging=not args.no_wandb,
            )
        except TimeoutExpired as exc:
            print(
                f"[EVAL {time.strftime('%H:%M:%S')}] FAILED {cond} timeout "
                f"after {(time.time() - t0) / 60:.1f}min: {exc}",
                file=sys.stderr,
                flush=True,
            )
            failed.append(cond)
            continue
        if args.dry_run:
            continue
        if proc.returncode != 0:
            combined = (proc.stdout or "") + (proc.stderr or "")
            print(combined[-4000:], file=sys.stderr)
            print(
                f"[EVAL {time.strftime('%H:%M:%S')}] FAILED {cond} rc={proc.returncode} "
                f"after {(time.time() - t0) / 60:.1f}min",
                file=sys.stderr,
                flush=True,
            )
            failed.append(cond)
            continue
        out_dir = extract_planning_result_dir(proc.stdout or "")
        if not out_dir and proc.stderr:
            out_dir = extract_planning_result_dir(proc.stderr or "")
        if not out_dir:
            guessed = resolve_plan_output_dir(REPO_ROOT, model_name, t0)
            if guessed is None:
                guessed = find_latest_plan_output_dir(model_name, t0)
            out_dir = str(guessed) if guessed else None
        plan_out_dirs[cond] = out_dir
        if not out_dir:
            print(
                "Could not find plan result dir (stdout parse + plan_outputs/ scan).",
                file=sys.stderr,
            )
            failed.append(cond)
            continue
        sr = parse_final_success_rate(Path(out_dir) / "logs.json")
        rates[cond] = sr
        took = time.time() - t0
        elapsed = time.time() - t_all
        eta = (elapsed / ci) * (n_cond - ci)
        print(
            f"[EVAL {time.strftime('%H:%M:%S')}] condition {ci}/{n_cond} {cond} done "
            f"sr={sr} took={took / 60:.1f}min elapsed={elapsed / 60:.1f}min "
            f"eta={eta / 60:.1f}min  (plan result dir: {out_dir})",
            flush=True,
        )

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
    print(
        f"[EVAL {time.strftime('%H:%M:%S')}] wrote {out_json} "
        f"elapsed={(time.time() - t_all) / 60:.1f}min failed={failed or 'none'}",
        flush=True,
    )
    print(f"Wrote {out_json}", flush=True)

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
