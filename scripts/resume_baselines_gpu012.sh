#!/usr/bin/env bash
# Resume/retrain live-DINO baselines (outputs/by_tag/<tag>/),
# evaluate legacy_six after each model; extra mild_tv (T1 T2 V1 V2) via eval_readme_tables_all.sh.
#
# One job per GPU (sequential): WM → Bisim → VicReg
#   GPU 0: n1000  — dinowm → dinobisim → dinovicreg
#   GPU 1: n5000  — dinowm → dinobisim → dinovicreg
#   GPU 2: nfull  — dinowm → dinobisim → dinovicreg
#
# Partial by_tag checkpoints are resumed (remaining epochs = EPOCHS - saved_epoch).
# Set FRESH_TRAIN=1 to wipe by_tag and retrain from scratch.
#
# Train: live DINO, n1000-sized epochs (105340 samples ≈ 5267 batches @ bs20)
# Bisim/VicReg: bisim_pos_encoding=grid_mlp
# Eval:  live DINO (use_precomputed_dino_eval=false)
# WANDB: disabled (avoids offline BrokenPipe crashes when co-tenanting / long runs)
#
# Usage (Anderson):
#   bash scripts/resume_baselines_gpu012.sh
#
# From Mac:
#   bash scripts/resume_baselines_gpu012.sh --remote
set -euo pipefail

REPO="${REPO:-$HOME/Bis-JEPA}"
REMOTE="${REMOTE:-anderson-accelerator}"

if [[ "${1:-}" == "--remote" ]]; then
  LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
  RHOST="${REMOTE:-anderson-accelerator}"
  RREPO="${REMOTE_REPO:-~/Bis-JEPA}"
  scp "$LOCAL_REPO/eval_pusht_six_conditions.py" "$LOCAL_REPO/plan.py" \
    "$LOCAL_REPO/preprocessor.py" "$RHOST:${RREPO}/"
  scp "$LOCAL_REPO/planning/base_planner.py" "$LOCAL_REPO/planning/cem.py" \
    "$LOCAL_REPO/planning/mpc.py" "$LOCAL_REPO/planning/evaluator.py" \
    "$LOCAL_REPO/planning/gd.py" "$RHOST:${RREPO}/planning/"
  scp "$LOCAL_REPO/conf/plan_pusht_local.yaml" \
    "$LOCAL_REPO/conf/train_local.yaml" \
    "$LOCAL_REPO/conf/train_dinowm_local.yaml" \
    "$RHOST:${RREPO}/conf/"
  scp "$LOCAL_REPO/env/visual_conditions.py" "$RHOST:${RREPO}/env/"
  scp "$LOCAL_REPO/env/pusht/pusht_env.py" "$RHOST:${RREPO}/env/pusht/"
  scp "$LOCAL_REPO/scripts/resume_baselines_gpu012.sh" \
    "$LOCAL_REPO/scripts/parallel_rollout_pipeline.sh" \
    "$LOCAL_REPO/scripts/eval_readme_tables_all.sh" \
    "$LOCAL_REPO/scripts/precomputed_baselines_manifest.json" \
    "$LOCAL_REPO/scripts/update_readme_pusht_results.py" \
    "$RHOST:${RREPO}/scripts/"
  ssh -o BatchMode=yes "$RHOST" "bash ${RREPO}/scripts/resume_baselines_gpu012.sh"
  exit 0
fi

CKPT_BASE="${CKPT_BASE:-/mnt/t7shield/gloria/bis-jepa}"
T7_SCRATCH="${T7_SCRATCH:-$CKPT_BASE/scratch}"
# Logs on T7 to avoid filling root "/"
LOG_DIR="${LOG_DIR:-$CKPT_BASE/logs/rollout_sweep}"
MANIFEST="${MANIFEST:-$REPO/scripts/precomputed_baselines_manifest.json}"
RESULTS_JSON="${RESULTS_JSON:-$T7_SCRATCH/plan_outputs/rollout_sweep_results.json}"
PLAN_OUTPUTS="${PLAN_OUTPUTS:-$T7_SCRATCH/plan_outputs}"

GPU_N1000="${GPU_N1000:-0}"
GPU_N5000="${GPU_N5000:-1}"
GPU_NFULL="${GPU_NFULL:-2}"

export CKPT_BASE LOG_DIR MANIFEST RESULTS_JSON PLAN_OUTPUTS
export SAVE_EVERY="${SAVE_EVERY:-1}" TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
# disabled avoids wandb BrokenPipe that killed co-tenant runs
export WANDB_MODE="${WANDB_MODE:-disabled}"
export USE_PRECOMPUTED_DINO=false
export USE_PRECOMPUTED_DINO_EVAL=false
# Resume partial by_tag runs; do not wipe unless FRESH_TRAIN=1
export FRESH_TRAIN="${FRESH_TRAIN:-0}"
export FORCE_TRAIN="${FORCE_TRAIN:-0}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-20}"
export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-n1000}"
export STEPS_PER_EPOCH_N1000="${STEPS_PER_EPOCH_N1000:-105340}"
export VAL_STEPS_PER_EPOCH="${VAL_STEPS_PER_EPOCH:-200}"
export SKIP_OPENLOOP_ROLLOUT="${SKIP_OPENLOOP_ROLLOUT:-0}"
export BISIM_POS_ENCODING="${BISIM_POS_ENCODING:-grid_mlp}"
# Co-tenants often leave <20GiB free; 12GiB is enough to start one trainer
export MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-12000}"
export EPOCHS="${EPOCHS:-50}" PURGE_AFTER_PULL=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DATASET_DIR="${DATASET_DIR:-$REPO/datasets/data}"
export EVAL_CATEGORY="${EVAL_CATEGORY:-legacy_six}"
export CEM_EPISODE_BATCH_SIZE="${CEM_EPISODE_BATCH_SIZE:-1}"
export CEM_NUM_SAMPLES="${CEM_NUM_SAMPLES:-300}"
export SKIP_ARCHIVE="${SKIP_ARCHIVE:-1}"
# /dev/shm for PyTorch DataLoader shared memory (T7 is exfat — breaks torch_shm_manager)
export TMPDIR="${TMPDIR:-/dev/shm/gloria-bisjepa}"
export WANDB_DIR="${WANDB_DIR:-$CKPT_BASE/wandb}"
export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/resume_gpu012.log"; }

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-dino_wm310}"
mkdir -p "$LOG_DIR" "$PLAN_OUTPUTS" "$TMPDIR" "$REPO/plan_outputs" "$CKPT_BASE/outputs/by_tag"
cd "$REPO"

stop_conflicting_jobs() {
  log "Stopping prior train/eval/worker jobs..."
  local self=$$
  local parent=$PPID
  for pat in \
    eval_precomputed_baselines_gpu012.sh \
    retrain_eval_all_baselines_gpu012.sh \
    retrain_n1000_baselines_gpu012.sh \
    "parallel_rollout_pipeline.sh worker" \
    "parallel_rollout_pipeline.sh eval_only" \
    eval_pusht_six_conditions.py \
    "python plan.py" \
    "python train.py"
  do
    while read -r pid; do
      [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]] || continue
      [[ "$pid" == "$self" || "$pid" == "$parent" ]] && continue
      kill "$pid" 2>/dev/null || true
    done < <(pgrep -f "$pat" 2>/dev/null || true)
  done
  for tag in dinowm_n1000 dinowm_n5000 dinowm_nfull \
    dinovicreg_n1000 dinovicreg_n5000 dinovicreg_nfull \
    dinobisim_n1000 dinobisim_n5000 dinobisim_nfull; do
    while read -r pid; do
      [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]] || continue
      [[ "$pid" == "$self" || "$pid" == "$parent" ]] && continue
      kill "$pid" 2>/dev/null || true
    done < <(pgrep -f "wandb_display_name=${tag}" 2>/dev/null || true)
  done
  sleep 5
  if command -v nvidia-smi >/dev/null 2>&1; then
    while read -r pid _; do
      [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]] || continue
      [[ "$pid" == "$self" || "$pid" == "$parent" ]] && continue
      owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d " ")
      [[ "$owner" == "$USER" ]] || continue
      cmd=$(ps -o args= -p "$pid" 2>/dev/null || true)
      [[ "$cmd" == *python* ]] || continue
      log "  kill GPU pid=$pid"
      kill -9 "$pid" 2>/dev/null || true
    done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
  fi
  sleep 3
  rm -f "$LOG_DIR/parallel.lock" "$LOG_DIR/parallel_gpu0.lock" \
        "$LOG_DIR/parallel_gpu1.lock" "$LOG_DIR/parallel_gpu2.lock"
  nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv 2>/dev/null | tee -a "$LOG_DIR/resume_gpu012.log" || true
}

launch_chain() {
  local gpu="$1"
  shift
  local logfile="$LOG_DIR/baseline_gpu${gpu}_retrain_eval.log"
  : > "$logfile"
  log "LAUNCH GPU $gpu sequential: $* -> $logfile"
  nohup env \
    CKPT_BASE="$CKPT_BASE" MANIFEST="$MANIFEST" RESULTS_JSON="$RESULTS_JSON" \
    PLAN_OUTPUTS="$PLAN_OUTPUTS" LOG_DIR="$LOG_DIR" \
    USE_PRECOMPUTED_DINO=false USE_PRECOMPUTED_DINO_EVAL=false \
    FORCE_TRAIN="$FORCE_TRAIN" FRESH_TRAIN="$FRESH_TRAIN" \
    SKIP_OPENLOOP_ROLLOUT="$SKIP_OPENLOOP_ROLLOUT" \
    EPOCHS="$EPOCHS" SAVE_EVERY="$SAVE_EVERY" \
    STEPS_PER_EPOCH="$STEPS_PER_EPOCH" STEPS_PER_EPOCH_N1000="$STEPS_PER_EPOCH_N1000" \
    VAL_STEPS_PER_EPOCH="$VAL_STEPS_PER_EPOCH" \
    TRAIN_NUM_WORKERS="$TRAIN_NUM_WORKERS" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    BISIM_POS_ENCODING="$BISIM_POS_ENCODING" \
    MIN_GPU_FREE_MIB="$MIN_GPU_FREE_MIB" \
    EVAL_CATEGORY="$EVAL_CATEGORY" \
    CEM_EPISODE_BATCH_SIZE="$CEM_EPISODE_BATCH_SIZE" \
    CEM_NUM_SAMPLES="$CEM_NUM_SAMPLES" \
    SKIP_ARCHIVE="$SKIP_ARCHIVE" \
    DATASET_DIR="$DATASET_DIR" TMPDIR="$TMPDIR" \
    WANDB_MODE="$WANDB_MODE" PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    bash "$REPO/scripts/parallel_rollout_pipeline.sh" worker_multi "$gpu" "$@" \
    >> "$logfile" 2>&1 &
  log "  PID=$!"
}

log "=== resume_baselines_gpu012 (LIVE DINO, 1 job/GPU sequential) start ==="
log "FRESH_TRAIN=$FRESH_TRAIN FORCE_TRAIN=$FORCE_TRAIN WANDB_MODE=$WANDB_MODE"
log "EVAL_CATEGORY=$EVAL_CATEGORY (main); mild_tv T1/T2/V1/V2 via eval_readme_tables_all.sh"
log "steps/epoch=$STEPS_PER_EPOCH (n1000=${STEPS_PER_EPOCH_N1000} samples ≈ $((STEPS_PER_EPOCH_N1000 / TRAIN_BATCH_SIZE)) batches @ bs$TRAIN_BATCH_SIZE) epochs=$EPOCHS workers=$TRAIN_NUM_WORKERS"
log "bisim_pos_encoding=$BISIM_POS_ENCODING min_gpu_free_mib=$MIN_GPU_FREE_MIB"
log "Manifest=$MANIFEST Results=$RESULTS_JSON"

mkdir -p "$LOG_DIR"
printf '%s\n' "$EVAL_CATEGORY" > "$LOG_DIR/EVAL_CATEGORY"

stop_conflicting_jobs

if [[ "$FRESH_TRAIN" == "1" ]]; then
  log "FRESH_TRAIN=1 — wiping by_tag checkpoints"
  rm -rf "$CKPT_BASE/outputs/by_tag"
  mkdir -p "$CKPT_BASE/outputs/by_tag"
else
  log "Keeping by_tag checkpoints (resume partial runs)"
  for t in dinowm_n1000 dinowm_n5000 dinowm_nfull dinobisim_n1000 dinobisim_n5000 dinobisim_nfull; do
    d="$CKPT_BASE/outputs/by_tag/$t/checkpoints"
    if [[ -d "$d" ]]; then
      n=$(ls -1 "$d"/model_[0-9]*.pth 2>/dev/null | wc -l)
      latest=$(ls -1t "$d"/model_[0-9]*.pth 2>/dev/null | head -1 | xargs -I{} basename {} 2>/dev/null || echo none)
      log "  $t: $n ckpts latest=$latest"
    else
      log "  $t: MISSING (will train fresh)"
    fi
  done
fi

# Keep pending eval status; do not wipe rates for unfinished tags only
python3 - <<PY
import json
from pathlib import Path
results = Path("$RESULTS_JSON")
data = json.loads(results.read_text()) if results.is_file() else {"entries": []}
for e in data.get("entries", []):
    if not e.get("rates"):
        e["eval_status"] = "pending"
        e["ckpt_base_path"] = "$CKPT_BASE"
results.parent.mkdir(parents=True, exist_ok=True)
results.write_text(json.dumps(data, indent=2) + "\n")
print(f"Updated {results}")
PY

# One job per GPU: WM → Bisim → VicReg
launch_chain "$GPU_N1000" \
  "dinowm_n1000:dinowm:1000" \
  "dinobisim_n1000:bisim:1000" \
  "dinovicreg_n1000:bisim:1000"

launch_chain "$GPU_N5000" \
  "dinowm_n5000:dinowm:5000" \
  "dinobisim_n5000:bisim:5000" \
  "dinovicreg_n5000:bisim:5000"

launch_chain "$GPU_NFULL" \
  "dinowm_nfull:dinowm:full" \
  "dinobisim_nfull:bisim:full" \
  "dinovicreg_nfull:bisim:full"

log "=== launched 3 GPU sequential chains (1 train job/GPU; resume from by_tag) ==="
log "Monitor: tail -f $LOG_DIR/baseline_gpu*_retrain_eval.log"
log "Results: $RESULTS_JSON"
